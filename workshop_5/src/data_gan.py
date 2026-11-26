"""
Data for Workshop 5: same 101,418 beats and patient splits as Workshops 1-4.

GANs use beats min-max scaled to [-1, 1] per beat (matches the Tanh output of
the generator). The Workshop 2 classifier z-scores every beat internally, so a
beat in [-1, 1] and the original beat have exactly the same shape for it:
synthetic beats can be fed to it directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W5_DIR = Path(__file__).resolve().parents[1]
REPO = W5_DIR.parent
W1_DATA = REPO / "workshop_1" / "data"
W2_SRC = REPO / "workshop_2" / "src"
W2_CKPT = REPO / "workshop_2" / "checkpoints" / "model_best.pt"
SPLIT_KEYS = {"train": "train_indices", "val": "validation_indices", "test": "test_indices"}
CLASSES = ["S", "V", "F"]                 # abnormal AAMI classes (conditional GAN labels)


def scale_beats(X):
    """Per-beat min-max to [-1, 1]."""
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return (2 * (X - lo) / (hi - lo + 1e-6) - 1).astype(np.float32)


def zscore_beats(X):
    """Per-beat z-score, identical to Workshop 2 normalize_beats."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + 1e-6)).astype(np.float32)


def load_splits():
    """{"train" | "val" | "test": (X raw (n, 216), y (n,), metadata)}"""
    X = np.load(W1_DATA / "processed" / "X.npy").astype(np.float32)
    meta = pd.read_csv(W1_DATA / "processed" / "metadata.csv",
                       dtype={"record_id": str, "subject_group": str})
    y = meta["label"].to_numpy().astype(np.int64)
    with open(W1_DATA / "splits.json") as f:
        splits = json.load(f)
    out = {}
    for name, key in SPLIT_KEYS.items():
        idx = np.array(splits[key])
        out[name] = (X[idx], y[idx], meta.iloc[idx].reset_index(drop=True))
    return out


def abnormal_train_beats():
    """GAN training data: abnormal beats of the training patients, scaled to [-1, 1]."""
    X, y, meta = load_splits()["train"]
    mask = y == 1
    cls = meta["aami_class"].to_numpy()[mask]
    labels = np.array([CLASSES.index(c) for c in cls], dtype=np.int64)
    return scale_beats(X[mask]), labels


def load_w2_classifier():
    """Frozen Workshop 2 ResNet1D (used as feature extractor and judge)."""
    import torch
    if str(W2_SRC) not in sys.path:
        sys.path.append(str(W2_SRC))
    from model import ECGResNet1D
    ckpt = torch.load(W2_CKPT, map_location="cpu")
    net = ECGResNet1D(**ckpt["model_kwargs"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt["threshold"]
