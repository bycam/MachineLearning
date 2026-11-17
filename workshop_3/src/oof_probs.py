"""
Out-of-fold ResNet1D probabilities for the TRAIN patients.

Problem (see diagnose_shift.py): the Workshop 2 ResNet1D was trained on the
train patients, so its probabilities are more reliable there than on new
patients. An agent trained on them learns to trust mid-range probabilities
and over-alarms on validation.

Fix: split the 34 train patients into 4 groups; train the same ResNet1D
(Workshop 2 final configuration) on 3 groups and predict the 4th, rotating.
Every train beat then gets a probability from a network that never saw that
patient, i.e. behaving like on validation/test patients.

Validation and test beats keep the Workshop 2 model probabilities.
The original train probabilities are kept in the column `p_w2`.

Each fold is saved to data/oof_folds/ as soon as it finishes; if the run is
interrupted, running the script again skips the folds already computed.

Usage (from workshop_3/):  python src/oof_probs.py
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

W3_DIR = Path(__file__).resolve().parents[1]
REPO = W3_DIR.parent
W1_DATA = REPO / "workshop_1" / "data"
BEATS_PATH = W3_DIR / "data" / "beats.csv.gz"

sys.path.append(str(REPO / "workshop_2" / "src"))
from data import ECGBeatDataset, normalize_beats   # noqa: E402
from model import ECGResNet1D                      # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_fold(X_tr, y_tr, epochs, seed):
    """Workshop 2 final configuration (run 2), fixed number of epochs."""
    set_seed(seed)
    loader = DataLoader(ECGBeatDataset(X_tr, y_tr, augment=True), batch_size=256,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = ECGResNet1D(base_channels=16, dropout=0.5)
    pos_weight = np.sqrt((y_tr == 0).sum() / (y_tr == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            loss = criterion(model(x), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        scheduler.step()
        print(f"    epoch {epoch}: loss {total / len(y_tr):.4f} ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 2048):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + 2048]).unsqueeze(1))).numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X = normalize_beats(np.load(W1_DATA / "processed" / "X.npy"))
    meta = pd.read_csv(W1_DATA / "processed" / "metadata.csv", dtype={"subject_group": str})
    with open(W1_DATA / "splits.json") as f:
        train_idx = np.array(json.load(f)["train_indices"])

    y = meta["label"].to_numpy()[train_idx]
    groups = meta["subject_group"].to_numpy()[train_idx]
    X_train = X[train_idx]
    oof = np.zeros(len(train_idx), dtype=np.float32)

    fold_dir = W3_DIR / "data" / "oof_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for k, (fit, held) in enumerate(GroupKFold(n_splits=args.folds).split(X_train, y, groups)):
        fold_file = fold_dir / f"fold{k + 1}_of{args.folds}_ep{args.epochs}.npy"
        if fold_file.exists():                       # resume after an interruption
            oof[held] = np.load(fold_file)
            print(f"Fold {k + 1}/{args.folds}: already done, loaded {fold_file.name}")
            continue
        print(f"Fold {k + 1}/{args.folds}: train on {len(set(groups[fit]))} patients, "
              f"predict {len(set(groups[held]))} unseen patients ({len(held):,} beats)")
        model = train_fold(X_train[fit], y[fit], args.epochs, args.seed + k)
        oof[held] = predict(model, X_train[held])
        np.save(fold_file, oof[held])                # saved immediately after each fold
        print(f"    PR-AUC on the unseen patients: "
              f"{average_precision_score(y[held], oof[held]):.4f}")

    print(f"\nOut-of-fold PR-AUC over all train patients: {average_precision_score(y, oof):.4f}")

    beats = pd.read_csv(BEATS_PATH, dtype={"record_id": str, "subject_group": str})
    if "p_w2" not in beats:
        beats["p_w2"] = beats["p"]
    oof_by_index = pd.Series(oof, index=train_idx)
    is_train = beats["split"] == "train"
    beats.loc[is_train, "p"] = oof_by_index.loc[beats.loc[is_train, "beat_index"]].to_numpy().round(6)
    beats.to_csv(BEATS_PATH, index=False)
    print(f"Updated train probabilities in {BEATS_PATH}")

    bins = [0, 0.2, 0.4, 0.6, 0.784, 0.9, 1.0001]
    tr = beats[is_train]
    table = pd.DataFrame({
        "train (W2 model)": tr.groupby(pd.cut(tr.p_w2, bins, right=False), observed=False).label.mean(),
        "train (out-of-fold)": tr.groupby(pd.cut(tr.p, bins, right=False), observed=False).label.mean(),
        "val": beats[beats.split == "val"].pipe(
            lambda b: b.groupby(pd.cut(b.p, bins, right=False), observed=False).label.mean()),
    })
    print("\nShare of abnormal beats per probability bin:")
    print((100 * table).round(1).astype(str) + "%")


if __name__ == "__main__":
    main()
