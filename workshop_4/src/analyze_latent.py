"""
Latent space analysis: what structure do the representations capture?

Encodes TEST beats with
  - the AE encoder (z),
  - the VAE encoder (mu, not a sampled z),
  - the Workshop 2 ResNet1D, frozen (64 features before its final layer),
then:
  1. Silhouette score on the latent vectors (full dimension), by
     normal/abnormal, by AAMI class (N/S/V/F) and by patient record.
  2. t-SNE and UMAP 2D maps colored by AAMI class and by patient record.
  3. VAE: KL per latent dimension (how many dimensions are actually used).

A balanced subsample is used (all abnormal test beats + 3,000 normal ones)
because t-SNE and the silhouette score scale quadratically.

Outputs in reports/. Usage (from workshop_4/):  python src/analyze_latent.py
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_ae import REPO, W4_DIR, load_splits, to_loader
from evaluate_ae import load_model

sys.path.append(str(REPO / "workshop_2" / "src"))
from model import ECGResNet1D   # noqa: E402  (Workshop 2 architecture)

REPORTS = W4_DIR / "reports"
CKPT = W4_DIR / "checkpoints"
W2_CKPT = REPO / "workshop_2" / "checkpoints" / "model_best.pt"
COLORS = {"N": "#b0b7c3", "S": "#e07a1f", "V": "#c0392b", "F": "#6c3483"}


@torch.no_grad()
def encode_ae(model, X):
    out = []
    for (x,) in to_loader(X, 2048):
        out.append((model.encode(x)[0] if model.kind == "vae" else model.encode(x)).numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_w2(X):
    """Frozen Workshop 2 ResNet1D: output of global average pooling (input of the classifier)."""
    ckpt = torch.load(W2_CKPT, map_location="cpu")
    net = ECGResNet1D(**ckpt["model_kwargs"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    out = []
    for (x,) in to_loader(X, 2048):
        h = x
        for layer in [net.stem, net.stage1, net.stage2, net.stage3, net.stage4, net.pool]:
            h = layer(h)
        out.append(h.flatten(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def vae_kl_per_dim(vae, X):
    kls = []
    for (x,) in to_loader(X, 2048):
        mu, logvar = vae.encode(x)
        kls.append((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).numpy())
    return np.concatenate(kls).mean(0)


def scatter(ax, emb, labels, title, palette=None):
    if palette is None:                               # categorical by record
        cats = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        palette = {c: cmap(i % 10) for i, c in enumerate(cats)}
    for c in palette:
        m = labels == c
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=3 if c == "N" else 6, alpha=0.5 if c == "N" else 0.8,
                       color=palette[c], label=f"{c} ({m.sum()})", rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-normal", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    REPORTS.mkdir(exist_ok=True)

    X, y, meta = load_splits()["test"]
    rng = np.random.default_rng(args.seed)
    normal = np.where(y == 0)[0]
    keep = np.sort(np.concatenate([np.where(y == 1)[0],
                                   rng.choice(normal, args.n_normal, replace=False)]))
    X, y, meta = X[keep], y[keep], meta.iloc[keep].reset_index(drop=True)
    cls = meta.aami_class.to_numpy()
    rec = meta.record_id.to_numpy()
    print(f"Test subsample: {len(y):,} beats ({int(y.sum())} abnormal, {int((y == 0).sum())} normal)")

    ae, _ = load_model(CKPT / "ae_best.pt")
    vae, _ = load_model(CKPT / "vae_best.pt")
    reps = {f"AE (z={ae.z_dim})": encode_ae(ae, X),
            f"VAE mu (z={vae.z_dim})": encode_ae(vae, X),
            "W2 ResNet1D (64, supervised)": encode_w2(X)}

    # 1. Silhouette on the latent vectors (standardized so no dimension dominates)
    rows = []
    for name, Z in reps.items():
        Zs = StandardScaler().fit_transform(Z)
        rows.append({"representation": name, "dimension": Z.shape[1],
                     "silhouette normal/abnormal": silhouette_score(Zs, y),
                     "silhouette AAMI class": silhouette_score(Zs, cls),
                     "silhouette patient record": silhouette_score(Zs, rec)})
    sil = pd.DataFrame(rows)
    print("\nSilhouette scores on the latent vectors (higher = better separated):")
    print(sil.round(4).to_string(index=False))
    sil.to_csv(REPORTS / "latent_silhouette.csv", index=False)

    # 2. t-SNE and UMAP
    embeddings = {}
    for name, Z in reps.items():
        Zs = StandardScaler().fit_transform(Z)
        print(f"t-SNE and UMAP for {name}...")
        embeddings[(name, "t-SNE")] = TSNE(n_components=2, perplexity=30, init="pca",
                                           random_state=args.seed).fit_transform(Zs)
        embeddings[(name, "UMAP")] = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                               random_state=args.seed).fit_transform(Zs)

    sil_by = sil.set_index("representation")
    for color_by, labels, palette, fname in [
            ("AAMI class", cls, COLORS, "latent_by_class.png"),
            ("patient record", rec, None, "latent_by_patient.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for j, name in enumerate(reps):
            col = "silhouette AAMI class" if color_by == "AAMI class" else "silhouette patient record"
            for i, method in enumerate(["t-SNE", "UMAP"]):
                scatter(axes[i, j], embeddings[(name, method)], labels,
                        f"{name} - {method}\nsilhouette ({color_by}) = {sil_by.loc[name, col]:.3f}",
                        palette)
        axes[0, 0].legend(fontsize=8, markerscale=3, loc="best")
        fig.suptitle(f"Test-set latent representations colored by {color_by}", fontsize=12)
        fig.tight_layout()
        fig.savefig(REPORTS / fname, dpi=150)
        plt.close(fig)

    # 3. VAE active dimensions
    kl_dim = vae_kl_per_dim(vae, X[y == 0])
    active = int((kl_dim > 0.01).sum())
    print(f"\nVAE KL per latent dimension (normal beats): {np.round(kl_dim, 2).tolist()}")
    print(f"Active dimensions (KL > 0.01): {active} of {len(kl_dim)}")
    pd.DataFrame({"dimension": range(len(kl_dim)), "mean_KL": kl_dim}).to_csv(
        REPORTS / "vae_kl_per_dimension.csv", index=False)
    print(f"\nSaved latent_by_class.png, latent_by_patient.png and tables to {REPORTS}")


if __name__ == "__main__":
    main()
