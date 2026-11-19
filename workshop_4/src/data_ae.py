"""
Data for Workshop 4: the same 101,418 heartbeats and patient splits as
Workshops 1-3, z-scored per beat exactly as in Workshop 2.

The autoencoders are trained on the NORMAL beats of the training patients
only, so that abnormal beats are "unknown" to them and reconstruct badly.
Validation and test keep both classes for anomaly detection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

W4_DIR = Path(__file__).resolve().parents[1]
REPO = W4_DIR.parent
W1_DATA = REPO / "workshop_1" / "data"
SPLIT_KEYS = {"train": "train_indices", "val": "validation_indices", "test": "test_indices"}


def normalize_beats(X):
    """Per-beat z-score (same as Workshop 2): keeps the shape, removes gain and offset."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + 1e-6)).astype(np.float32)


def load_splits():
    """{"train" | "val" | "test": (X (n, 216) float32, y (n,), metadata DataFrame)}"""
    X = normalize_beats(np.load(W1_DATA / "processed" / "X.npy"))
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


def to_loader(X, batch_size=512, shuffle=False, seed=42):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)
