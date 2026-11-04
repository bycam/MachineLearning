"""
Data loading for Workshop 2.

Reuses the processed heartbeats and the patient-grouped splits from
Workshop 1, so the deep learning model is evaluated on exactly the same
test patients as the Random Forest baseline.

Usage (from workshop_2/):  python src/data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
W1_DIR = REPO_ROOT / "workshop_1"
PROCESSED_DIR = W1_DIR / "data" / "processed"
SPLITS_PATH = W1_DIR / "data" / "splits.json"

SPLIT_KEYS = {
    "train": "train_indices",
    "val": "validation_indices",
    "test": "test_indices",
}


def normalize_beats(X):
    """
    Z-score each heartbeat on its own (zero mean, unit std).

    Amplitude and baseline offset change a lot between patients and
    electrodes. Normalizing per beat makes the model focus on the SHAPE
    of the beat, which is what separates normal from abnormal beats.
    It uses no statistics from other samples, so it cannot leak
    information from validation or test into training.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + 1e-6)).astype(np.float32)


def load_splits():
    """
    Return {"train": (X, y, metadata), "val": ..., "test": ...}.
    """
    if not (PROCESSED_DIR / "X.npy").exists():
        raise FileNotFoundError(
            "Processed data not found. Run: python src/prepare_data.py"
        )

    X = np.load(PROCESSED_DIR / "X.npy")
    metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    y = metadata["label"].to_numpy().astype(np.int64)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    X = normalize_beats(X)

    data = {}
    for name, key in SPLIT_KEYS.items():
        idx = np.array(splits[key])
        data[name] = (X[idx], y[idx], metadata.iloc[idx].reset_index(drop=True))
    return data


class ECGBeatDataset(Dataset):
    """
    One item = (beat tensor of shape (1, 216), label 0/1).

    The leading 1 is the channel dimension that Conv1d expects
    (like the RGB channels of an image, but we only have one ECG lead).
    """

    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        beat = self.X[i].copy()
        if self.augment:
            beat = augment_beat(beat)
        return torch.from_numpy(beat).unsqueeze(0), torch.tensor(self.y[i])


def augment_beat(beat, rng=np.random):
    """
    Light, physiologically plausible augmentations (training only):
      - amplitude scaling: electrode contact and body shape change gain
      - small time shift: R-peak annotations are not perfectly aligned
      - Gaussian noise: muscle activity and electrical noise
      - baseline wander: breathing makes the baseline drift slowly
    None of them change the beat type, so the label stays valid.
    """
    n = len(beat)

    beat = beat * rng.uniform(0.9, 1.1)

    shift = rng.randint(-5, 6)
    if shift > 0:
        beat = np.concatenate([np.full(shift, beat[0]), beat[:-shift]])
    elif shift < 0:
        beat = np.concatenate([beat[-shift:], np.full(-shift, beat[-1])])

    beat = beat + rng.normal(0, 0.05, n)

    t = np.linspace(0, 1, n)
    beat = beat + 0.1 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t
                               + rng.uniform(0, 2 * np.pi))

    return beat.astype(np.float32)


def make_dataloaders(batch_size=256, seed=42):
    """
    Build train / val / test DataLoaders.

    Only the training loader shuffles and augments. Validation and test
    stay fixed so the metrics are comparable between epochs.
    num_workers=0 because multiprocessing is unreliable on Windows.
    """
    data = load_splits()
    generator = torch.Generator().manual_seed(seed)

    loaders = {
        "train": DataLoader(
            ECGBeatDataset(*data["train"][:2], augment=True),
            batch_size=batch_size, shuffle=True, generator=generator,
            num_workers=0,
        ),
        "val": DataLoader(
            ECGBeatDataset(*data["val"][:2]),
            batch_size=batch_size, shuffle=False, num_workers=0,
        ),
        "test": DataLoader(
            ECGBeatDataset(*data["test"][:2]),
            batch_size=batch_size, shuffle=False, num_workers=0,
        ),
    }
    return loaders, data


if __name__ == "__main__":
    loaders, data = make_dataloaders()

    print(f"{'Split':<6}{'Beats':>8}{'Normal':>9}{'Abnormal':>10}{'% Abn':>8}"
          f"{'Patients':>10}")
    for name, (X, y, meta) in data.items():
        print(f"{name:<6}{len(y):>8}{(y == 0).sum():>9}{(y == 1).sum():>10}"
              f"{100 * y.mean():>7.1f}%{meta['subject_group'].nunique():>10}")

    xb, yb = next(iter(loaders["train"]))
    print(f"\nOne training batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
    print(f"Beat after normalization: mean={xb.mean():.3f}, std={xb.std():.3f}")

    n_pos = data["train"][1].sum()
    n_neg = len(data["train"][1]) - n_pos
    print(f"pos_weight for the loss (normal/abnormal in train): {n_neg / n_pos:.2f}")