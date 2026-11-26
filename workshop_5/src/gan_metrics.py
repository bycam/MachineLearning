"""
Generation-quality metrics for ECG beats, shared by training and evaluation.

The Inception Score / FID use an ImageNet network, which is meaningless for ECG.
We use the frozen Workshop 2 ResNet1D (trained on real beats) instead:

- Frechet distance (FD) between the 64-dim ResNet features of real and generated
  beats (same formula as FID): lower = distributions closer.
- Plausibility: share of generated beats the ResNet classifies as abnormal
  (the GANs are trained on abnormal beats).
- Diversity: mean L2 distance between random pairs of beats; much lower than for
  real beats = mode collapse.
- Memorization: distance from each generated beat to its nearest real training beat,
  compared with the same distance for held-out real beats.
All beats here are in the [-1, 1] scale of data_gan.scale_beats.
"""
import numpy as np
import torch
from scipy import linalg

from data_gan import zscore_beats


@torch.no_grad()
def w2_features_and_probs(net, X, batch=2048):
    """64-dim features (global average pooling output) and P(abnormal) of the W2 ResNet1D."""
    feats, probs = [], []
    Xz = zscore_beats(np.asarray(X, dtype=np.float32))
    for i in range(0, len(Xz), batch):
        h = torch.from_numpy(Xz[i:i + batch]).unsqueeze(1)
        for layer in [net.stem, net.stage1, net.stage2, net.stage3, net.stage4, net.pool]:
            h = layer(h)
        feats.append(h.flatten(1).numpy())
        probs.append(torch.sigmoid(net.head(h)).squeeze(1).numpy())
    return np.concatenate(feats).astype(np.float64), np.concatenate(probs)


def frechet_distance(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):          # older SciPy versions return (sqrtm, error)
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def diversity(X, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), n_pairs), rng.integers(0, len(X), n_pairs)
    keep = i != j
    return float(np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1).mean())


def nearest_neighbor_distance(X, reference, chunk=512):
    """Distance from each beat of X to its nearest beat in `reference` (L2)."""
    ref = torch.from_numpy(np.asarray(reference, dtype=np.float32))
    out = []
    for i in range(0, len(X), chunk):
        x = torch.from_numpy(np.asarray(X[i:i + chunk], dtype=np.float32))
        out.append(torch.cdist(x, ref).min(dim=1).values.numpy())
    return np.concatenate(out)
