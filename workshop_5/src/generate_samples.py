"""
Generate synthetic abnormal beats with the best vanilla GAN and WGAN-GP
generators, save them, and draw visual comparisons with real beats.

Outputs:
  data/synthetic_vanilla.npy, data/synthetic_wgan.npy   (n, 216) in [-1, 1]
  reports/sample_grids.png       4x4 real | vanilla GAN | WGAN-GP, side by side
  reports/morphology_overlay.png 100 beats overlaid + mean, per source
  reports/training_progress.png  generated beats at epochs 10, 50, 100, best

Usage (from workshop_5/):  python src/generate_samples.py --n 5000
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_gan import W5_DIR, abnormal_train_beats
from gan import Z_DIM, Generator

CKPT = W5_DIR / "checkpoints"
DATA = W5_DIR / "data"
REPORTS = W5_DIR / "reports"
RUNS = W5_DIR / "runs" / "workshop5"
MODELS = {"vanilla": "Vanilla GAN", "wgan": "WGAN-GP"}
T_MS = np.arange(216) / 360 * 1000 - 200          # ms relative to the R-peak


def load_generator(path):
    ckpt = torch.load(path, map_location="cpu")
    G = Generator(n_classes=ckpt["n_classes"])
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    return G, ckpt


@torch.no_grad()
def sample(G, n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, 1000):
        z = torch.randn(min(1000, n - i), Z_DIM, generator=g)
        out.append(G(z).squeeze(1).numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    real, _ = abnormal_train_beats()
    rng = np.random.default_rng(0)
    sets = {"Real (train, abnormal)": real[rng.choice(len(real), 16, replace=False)]}
    overlay = {"Real (train, abnormal)": real[rng.choice(len(real), 100, replace=False)]}

    for key, name in MODELS.items():
        G, ckpt = load_generator(CKPT / f"gan_{key}_best.pt")
        X = sample(G, args.n, args.seed)
        np.save(DATA / f"synthetic_{key}.npy", X.astype(np.float32))
        print(f"{name}: best epoch {ckpt['epoch']} (FD {ckpt['fd']:.2f}) -> "
              f"{len(X):,} beats saved to data/synthetic_{key}.npy")
        sets[name] = X[:16]
        overlay[name] = X[:100]

    # 4x4 grids side by side
    fig = plt.figure(figsize=(16, 5.2))
    outer = fig.add_gridspec(1, 3, wspace=0.08)
    for k, (name, beats) in enumerate(sets.items()):
        inner = outer[k].subgridspec(4, 4, wspace=0.05, hspace=0.08)
        for j in range(16):
            ax = fig.add_subplot(inner[j // 4, j % 4])
            ax.plot(beats[j], lw=0.9, color="black" if k == 0 else ("C0" if k == 1 else "C3"))
            ax.set_ylim(-1.1, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(name, fontsize=11, loc="left")
    fig.savefig(REPORTS / "sample_grids.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overlay of 100 beats per source
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, (name, beats), c in zip(axes, overlay.items(), ["black", "C0", "C3"]):
        ax.plot(T_MS, beats.T, color=c, alpha=0.08, lw=0.8)
        ax.plot(T_MS, beats.mean(0), color=c, lw=2.2, label="mean")
        ax.set_title(f"{name}: 100 beats")
        ax.set_xlabel("ms from R-peak")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("amplitude (scaled)")
    fig.tight_layout()
    fig.savefig(REPORTS / "morphology_overlay.png", dpi=150)
    plt.close(fig)

    # Training progress: saved sample grids at several epochs
    picks = []
    for key, name in MODELS.items():
        best = torch.load(CKPT / f"gan_{key}_best.pt", map_location="cpu")["epoch"]
        for ep in [10, 50, 100, best]:
            f = RUNS / f"gan_{key}" / "samples" / f"epoch_{ep:03d}.png"
            if f.exists():
                picks.append((name, ep, f))
    if picks:
        cols = 4
        rows = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows))
        for ax in np.atleast_1d(axes).flat:
            ax.axis("off")
        for ax, (name, ep, f) in zip(np.atleast_1d(axes).flat, picks):
            ax.imshow(plt.imread(f))
            ax.set_title(f"{name}, epoch {ep}", fontsize=10)
        fig.tight_layout()
        fig.savefig(REPORTS / "training_progress.png", dpi=150)
        plt.close(fig)
    print(f"Saved figures to {REPORTS}")


if __name__ == "__main__":
    main()
