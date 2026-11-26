"""
Train a vanilla GAN (BCE) or a WGAN-GP on the abnormal beats of the training patients.

Vanilla GAN : Adam lr 2e-4, betas (0.5, 0.999) for G and D; one-sided label
              smoothing (real label 0.9); non-saturating generator loss.
WGAN-GP     : same generator; critic without sigmoid; Adam lr 1e-4, betas (0.0, 0.9)
              (Gulrajani et al.); 5 critic steps per generator step; gradient
              penalty lambda = 10 on interpolates x_hat = eps*x + (1-eps)*x_fake.

Every epoch logs losses and (vanilla) discriminator accuracy on real/fake, or
(WGAN) the Wasserstein estimate and gradient penalty. Every `--eval-every` epochs:
Frechet distance in the Workshop 2 ResNet feature space, plausibility, diversity,
and a 4x4 grid of generated beats (runs/workshop5/<run>/samples/).
The checkpoint with the lowest Frechet distance is the "best" generator.

Usage (from workshop_5/):
    python src/train_gan.py --model vanilla --smoke
    python src/train_gan.py --model vanilla
    python src/train_gan.py --model wgan
    python src/train_gan.py --model wgan --conditional      # optional cGAN
"""
import argparse
import json
import random
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_gan import CLASSES, W5_DIR, abnormal_train_beats, load_w2_classifier
from gan import Z_DIM, Discriminator, Generator, count_parameters
from gan_metrics import diversity, frechet_distance, w2_features_and_probs

RUNS = W5_DIR / "runs" / "workshop5"
CKPT = W5_DIR / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vanilla", "wgan"], default="vanilla")
    p.add_argument("--conditional", action="store_true", help="cGAN on the S/V/F class")
    p.add_argument("--epochs", type=int, default=None, help="default 150 vanilla, 300 wgan")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default 2e-4 vanilla, 1e-4 wgan")
    p.add_argument("--label-smooth", type=float, default=0.9)
    p.add_argument("--d-noise", type=float, default=0.0, help="std of noise added to D inputs")
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gradient_penalty(critic, real, fake, labels=None):
    """lambda * E[(||grad_xhat C(xhat)||_2 - 1)^2] is applied by the caller."""
    eps = torch.rand(real.size(0), 1, 1)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(x_hat, labels)
    grads = torch.autograd.grad(out.sum(), x_hat, create_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def generate(G, n, n_classes, seed=0, labels=None):
    G.eval()
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, Z_DIM, generator=g)
    if n_classes and labels is None:
        labels = torch.randint(0, n_classes, (n,), generator=g)
    x = G(z, labels).squeeze(1).numpy()
    G.train()
    return x, labels


def save_grid(beats, path, title, labels=None):
    fig, axes = plt.subplots(4, 4, figsize=(10, 6), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(beats[k], lw=1, color="C3")
        if labels is not None:
            ax.set_title(CLASSES[int(labels[k])], fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs is None:
        args.epochs = 150 if args.model == "vanilla" else 300
    if args.lr is None:
        args.lr = 2e-4 if args.model == "vanilla" else 1e-4
    if args.run_name is None:
        args.run_name = ("cgan_" if args.conditional else "gan_") + args.model
    if args.smoke:
        args.epochs, args.eval_every, args.run_name = 2, 1, args.run_name + "_smoke"
    set_seed(args.seed)

    X, labels = abnormal_train_beats()
    if args.smoke:
        X, labels = X[:1024], labels[:1024]
    n_classes = len(CLASSES) if args.conditional else 0
    loader = DataLoader(TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(labels)),
                        batch_size=args.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    G = Generator(n_classes=n_classes)
    D = Discriminator(critic=args.model == "wgan", n_classes=n_classes)
    betas = (0.5, 0.999) if args.model == "vanilla" else (0.0, 0.9)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=betas)

    # Reference for the metrics: Workshop 2 features of the real training beats
    w2, w2_thr = load_w2_classifier()
    real_feats, real_probs = w2_features_and_probs(w2, X)
    real_div = diversity(X)
    print(f"{args.model.upper()}{' (conditional)' if n_classes else ''}: {len(X):,} abnormal "
          f"training beats | G {count_parameters(G):,} params | "
          f"{'critic' if args.model == 'wgan' else 'D'} {count_parameters(D):,} params")
    print(f"Real beats: W2 says abnormal {100 * (real_probs >= w2_thr).mean():.1f}%, "
          f"diversity {real_div:.3f}")

    run_dir = RUNS / args.run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    save_grid(X[:16], run_dir / "samples" / "real.png", "Real abnormal training beats",
              labels[:16] if n_classes else None)

    if args.model == "vanilla":
        print(f"{'ep':>4} {'D_loss':>7} {'G_loss':>7} {'D(x)':>6} {'D(G(z))':>7} "
              f"{'acc_r':>6} {'acc_f':>6} {'FD':>8} {'abn%':>5} {'div':>6} {'time':>5}")
    else:
        print(f"{'ep':>4} {'C_loss':>8} {'W_dist':>7} {'GP':>6} {'G_loss':>8} "
              f"{'FD':>8} {'abn%':>5} {'div':>6} {'time':>5}")

    history, best_fd, g_steps = [], np.inf, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        logs = {k: [] for k in ["d_loss", "g_loss", "d_real", "d_fake", "acc_real", "acc_fake",
                                "w_dist", "gp"]}
        for i, (x, y) in enumerate(loader):
            y = y if n_classes else None
            z = torch.randn(x.size(0), Z_DIM)
            if args.model == "vanilla":
                fake = G(z, y)
                x_in = x + args.d_noise * torch.randn_like(x) if args.d_noise else x
                f_in = fake.detach() + args.d_noise * torch.randn_like(x) if args.d_noise else fake.detach()
                d_real, d_fake = D(x_in, y), D(f_in, y)
                d_loss = (F.binary_cross_entropy(d_real, torch.full_like(d_real, args.label_smooth))
                          + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                d_gen = D(fake, y)                                   # non-saturating G loss
                g_loss = F.binary_cross_entropy(d_gen, torch.ones_like(d_gen))
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_steps += 1
                logs["g_loss"].append(g_loss.item())
                logs["d_real"].append(d_real.mean().item())
                logs["d_fake"].append(d_fake.mean().item())
                logs["acc_real"].append((d_real > 0.5).float().mean().item())
                logs["acc_fake"].append((d_fake < 0.5).float().mean().item())
            else:
                fake = G(z, y).detach()
                c_real, c_fake = D(x, y).mean(), D(fake, y).mean()
                gp = gradient_penalty(D, x, fake, y)
                d_loss = c_fake - c_real + args.gp_lambda * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                logs["w_dist"].append((c_real - c_fake).item())
                logs["gp"].append(gp.item())
                if i % args.n_critic == 0:
                    g_loss = -D(G(torch.randn(x.size(0), Z_DIM), y), y).mean()
                    opt_g.zero_grad()
                    g_loss.backward()
                    opt_g.step()
                    g_steps += 1
                    logs["g_loss"].append(g_loss.item())
            logs["d_loss"].append(d_loss.item())

        row = {"epoch": epoch, "g_steps": g_steps,
               **{k: float(np.mean(v)) for k, v in logs.items() if v}}
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(f"train/{k}", v, epoch)

        mark = ""
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            gen, gen_labels = generate(G, 2000, n_classes)
            gen_feats, gen_probs = w2_features_and_probs(w2, gen)
            row["fd"] = frechet_distance(real_feats, gen_feats)
            row["abnormal_pct"] = float(100 * (gen_probs >= w2_thr).mean())
            row["diversity"] = diversity(gen)
            for k in ["fd", "abnormal_pct", "diversity"]:
                writer.add_scalar(f"quality/{k}", row[k], epoch)
            save_grid(gen, run_dir / "samples" / f"epoch_{epoch:03d}.png",
                      f"{args.run_name}, epoch {epoch} (FD {row['fd']:.1f})",
                      gen_labels.numpy() if n_classes else None)
            if row["fd"] < best_fd:
                best_fd = row["fd"]
                torch.save({"G_state": G.state_dict(), "D_state": D.state_dict(),
                            "model": args.model, "n_classes": n_classes, "epoch": epoch,
                            "fd": best_fd, "config": vars(args)}, CKPT / f"{args.run_name}_best.pt")
                mark = " *"
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        fd_txt = (f"{row['fd']:>8.2f} {row['abnormal_pct']:>5.1f} {row['diversity']:>6.3f}"
                  if "fd" in row else f"{'':>8} {'':>5} {'':>6}")
        if args.model == "vanilla":
            print(f"{epoch:>4} {row['d_loss']:>7.3f} {row['g_loss']:>7.3f} {row['d_real']:>6.3f} "
                  f"{row['d_fake']:>7.3f} {row['acc_real']:>6.2f} {row['acc_fake']:>6.2f} "
                  f"{fd_txt} {time.time() - t0:>4.0f}s{mark}")
        else:
            print(f"{epoch:>4} {row['d_loss']:>8.3f} {row['w_dist']:>7.3f} {row['gp']:>6.3f} "
                  f"{row['g_loss']:>8.3f} {fd_txt} {time.time() - t0:>4.0f}s{mark}")

    writer.close()
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    torch.save({"G_state": G.state_dict(), "D_state": D.state_dict(), "model": args.model,
                "n_classes": n_classes, "epoch": args.epochs, "config": vars(args)},
               CKPT / f"{args.run_name}_last.pt")

    # Training curves
    ev = hist.dropna(subset=["fd"]) if "fd" in hist else hist.iloc[0:0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if args.model == "vanilla":
        axes[0].plot(hist.epoch, hist.d_loss, label="D loss")
        axes[0].plot(hist.epoch, hist.g_loss, label="G loss")
        axes[0].set_title("Losses (BCE)")
        axes[1].plot(hist.epoch, hist.acc_real, label="D accuracy on real")
        axes[1].plot(hist.epoch, hist.acc_fake, label="D accuracy on fake")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("Discriminator accuracy")
    else:
        axes[0].plot(hist.epoch, hist.d_loss, label="critic loss")
        axes[0].plot(hist.epoch, hist.g_loss, label="G loss")
        axes[0].set_title("Losses")
        axes[1].plot(hist.epoch, hist.w_dist, label="Wasserstein estimate")
        axes[1].plot(hist.epoch, hist.gp, label="gradient penalty")
        axes[1].set_title("Critic: E[C(real)] - E[C(fake)]")
    axes[2].plot(ev.epoch, ev.fd, "o-", color="C3")
    axes[2].set_title("Frechet distance (W2 features, lower = better)")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    best = torch.load(CKPT / f"{args.run_name}_best.pt", map_location="cpu")
    if not args.smoke and not args.conditional:
        target = CKPT / f"gan_{args.model}_best.pt"
        if CKPT / f"{args.run_name}_best.pt" != target:
            shutil.copy(CKPT / f"{args.run_name}_best.pt", target)
        print(f"Copied best generator to {target}")
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"run_name": args.run_name, "best_epoch": best["epoch"], "best_fd": best["fd"],
                   "generator_steps": g_steps, "config": vars(args)}, f, indent=4)
    print(f"\nBest epoch {best['epoch']}: Frechet distance {best['fd']:.2f}")


if __name__ == "__main__":
    main()
