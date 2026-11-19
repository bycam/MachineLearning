"""
Train an AE or a VAE on the NORMAL beats of the training patients.

- Adam optimizer, fixed number of epochs (default 30)
- VAE: KL annealing, beta grows linearly from 0 to the target over the
  first `--anneal-epochs` epochs (prevents posterior collapse)
- Every epoch logs train / validation reconstruction loss (validation = normal
  beats of the validation patients) and, for monitoring only, the validation
  anomaly AUC-ROC (reconstruction error as score, all validation beats)
- Checkpoint = epoch with the lowest validation loss on NORMAL beats (no labels
  of abnormal beats are used to choose it); for the VAE only epochs after the
  annealing are eligible and the loss uses the target beta

Outputs: runs/workshop4/<run>/ (TensorBoard, history.csv, curves.png, summary.json)
         checkpoints/<run>_best.pt

Usage (from workshop_4/):
    python src/train_ae.py --model ae --z-dim 32 --smoke
    python src/train_ae.py --model ae --z-dim 32
    python src/train_ae.py --model vae --z-dim 32 --beta 1.0
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from autoencoder import build_model, count_parameters, kl_divergence, loss_fn, recon_error
from data_ae import W4_DIR, load_splits, to_loader

RUNS = W4_DIR / "runs" / "workshop4"
CKPT = W4_DIR / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ae", "vae"], default="ae")
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0, help="VAE only")
    p.add_argument("--anneal-epochs", type=int, default=10, help="VAE only")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def score_beats(model, X, batch_size=2048):
    """Per-beat reconstruction MSE (VAE: decoding mu) and KL (0 for the AE)."""
    model.eval()
    mse, kl = [], []
    for (x,) in to_loader(X, batch_size):
        if model.kind == "vae":
            out = model(x, sample=False)
            kl.append(kl_divergence(out["mu"], out["logvar"]).numpy())
        else:
            out = model(x)
            kl.append(np.zeros(len(x), dtype=np.float32))
        mse.append(recon_error(out["recon"], x).numpy())
    return np.concatenate(mse), np.concatenate(kl)


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = (f"ae_z{args.z_dim}" if args.model == "ae"
                         else f"vae_z{args.z_dim}_b{args.beta:g}")
    if args.smoke:
        args.epochs, args.anneal_epochs, args.run_name = 2, 1, args.run_name + "_smoke"
    set_seed(args.seed)

    data = load_splits()
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    X_tr_n, X_va_n = X_tr[y_tr == 0], X_va[y_va == 0]
    if args.smoke:
        X_tr_n = X_tr_n[:5000]
    train_loader = to_loader(X_tr_n, args.batch_size, shuffle=True, seed=args.seed)
    print(f"Training on {len(X_tr_n):,} normal beats | validation: {len(X_va_n):,} normal, "
          f"{len(X_va):,} total for the AUC")

    model = build_model(args.model, args.z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"{args.model.upper()} z_dim={args.z_dim} ({count_parameters(model):,} parameters)"
          + (f", beta={args.beta}, KL annealing over {args.anneal_epochs} epochs"
             if args.model == "vae" else ""))

    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    ckpt_path = CKPT / f"{args.run_name}_best.pt"

    print(f"{'ep':>3} {'beta':>5} {'tr_loss':>9} {'tr_mse':>7} {'tr_kl':>7} "
          f"{'va_loss':>9} {'va_mse':>7} {'va_kl':>7} {'va_AUC':>6} {'time':>5}")
    history, best = [], np.inf
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        beta = (args.beta * min(1.0, (epoch - 1) / args.anneal_epochs)
                if args.model == "vae" else 0.0)

        model.train()
        tot, tot_mse, tot_kl, n = 0.0, 0.0, 0.0, 0
        for (x,) in train_loader:
            out = model(x)
            loss, logs = loss_fn(model, out, x, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(x)
            tot_mse += logs["mse"] * len(x)
            tot_kl += logs["kl"] * len(x)
            n += len(x)

        # Validation on NORMAL beats, with the target beta (comparable across epochs)
        mse_n, kl_n = score_beats(model, X_va_n)
        va_loss = (mse_n.mean() if args.model == "ae"
                   else (mse_n * 216 + args.beta * kl_n).mean())
        # Monitoring only: anomaly AUC on all validation beats
        mse_all, _ = score_beats(model, X_va)
        va_auc = roc_auc_score(y_va, mse_all)

        row = {"epoch": epoch, "beta": beta, "train_loss": tot / n, "train_mse": tot_mse / n,
               "train_kl": tot_kl / n, "val_loss": float(va_loss), "val_mse": float(mse_n.mean()),
               "val_kl": float(kl_n.mean()), "val_auc_recon": float(va_auc)}
        history.append(row)
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)

        eligible = args.model == "ae" or epoch > args.anneal_epochs or args.smoke
        mark = ""
        if eligible and va_loss < best:
            best = float(va_loss)
            torch.save({"model_state": model.state_dict(),
                        "model_kwargs": {"kind": args.model, "z_dim": args.z_dim},
                        "epoch": epoch, "val_loss": best, "val_mse": float(mse_n.mean()),
                        "val_auc_recon": float(va_auc), "config": vars(args)}, ckpt_path)
            mark = " *"
        print(f"{epoch:>3} {beta:>5.2f} {row['train_loss']:>9.4f} {row['train_mse']:>7.4f} "
              f"{row['train_kl']:>7.2f} {va_loss:>9.4f} {row['val_mse']:>7.4f} "
              f"{row['val_kl']:>7.2f} {va_auc:>6.3f} {time.time() - t0:>4.0f}s{mark}")

    writer.close()
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)

    fig, axes = plt.subplots(1, 3 if args.model == "vae" else 2, figsize=(14, 3.8))
    axes[0].plot(hist.epoch, hist.train_mse, label="train (normal)")
    axes[0].plot(hist.epoch, hist.val_mse, label="validation (normal)")
    axes[0].set_title("Reconstruction MSE per sample")
    axes[1].plot(hist.epoch, hist.val_auc_recon, color="C2")
    axes[1].set_title("Validation anomaly AUC-ROC (monitoring)")
    if args.model == "vae":
        axes[2].plot(hist.epoch, hist.train_kl, label="train KL")
        axes[2].plot(hist.epoch, hist.val_kl, label="val KL")
        ax2 = axes[2].twinx()
        ax2.plot(hist.epoch, hist.beta, color="gray", ls="--", label="beta")
        axes[2].set_title("KL divergence (beta dashed)")
        axes[2].legend(loc="center right")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=150)
    plt.close(fig)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    summary = {"run_name": args.run_name, "best_epoch": ckpt["epoch"],
               "best_val_loss": ckpt["val_loss"], "val_mse_at_best": ckpt["val_mse"],
               "val_auc_recon_at_best": ckpt["val_auc_recon"],
               "parameters": count_parameters(model), "config": vars(args)}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nBest epoch {ckpt['epoch']}: val MSE (normal) {ckpt['val_mse']:.4f}, "
          f"val AUC {ckpt['val_auc_recon']:.4f}")
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
