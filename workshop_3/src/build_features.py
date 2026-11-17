"""
Precompute the per-beat information used by the RL environment.

For every heartbeat of the Workshop 1 dataset (101,418 beats):
  - p        : abnormal probability from the Workshop 2 ResNet1D
  - rr_pre   : seconds since the previous R-peak
  - rr_post  : seconds until the next R-peak
  - rr_avg   : mean of the previous 10 RR intervals (local rhythm of the patient)
  - label, aami_class, record, split (from Workshop 1)

RR intervals are computed from ALL annotated beats of the raw recording
(including paced or unclassifiable beats that were excluded from the dataset),
so the rhythm is measured on the real sequence of heartbeats.

Output: workshop_3/data/beats.csv.gz

Usage (from workshop_3/):  python src/build_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import f1_score, precision_score, recall_score

W3_DIR = Path(__file__).resolve().parents[1]
REPO = W3_DIR.parent
W1_DATA = REPO / "workshop_1" / "data"
RAW_DIR = W1_DATA / "raw" / "mitdb"
W2_CHECKPOINT = REPO / "workshop_2" / "checkpoints" / "model_best.pt"
OUT_PATH = W3_DIR / "data" / "beats.csv.gz"

# Reuse the exact Workshop 2 model and per-beat normalization
sys.path.append(str(REPO / "workshop_2" / "src"))
from data import normalize_beats   # noqa: E402
from model import ECGResNet1D      # noqa: E402

FS = 360
# Every annotation symbol that marks a heartbeat (PhysioNet convention)
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def resnet_probabilities(X):
    ckpt = torch.load(W2_CHECKPOINT, map_location="cpu")
    model = ECGResNet1D(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            x = torch.from_numpy(normalize_beats(X[i:i + 2048])).unsqueeze(1)
            probs.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(probs), ckpt["threshold"]


def rr_features(record_id, samples):
    """RR intervals around each of our beats, using every annotated beat of the record."""
    ann = wfdb.rdann(str(RAW_DIR / str(record_id)), "atr")
    peaks = np.array([s for s, sym in zip(ann.sample, ann.symbol)
                      if sym in BEAT_SYMBOLS])
    rr = np.diff(peaks) / FS                     # rr[i] = peaks[i+1] - peaks[i]

    pos = np.searchsorted(peaks, samples)
    if not np.array_equal(peaks[pos], samples):
        raise ValueError(f"Record {record_id}: beat positions not found in annotations")

    rr_pre = np.where(pos > 0, rr[np.clip(pos - 1, 0, len(rr) - 1)], rr[0])
    rr_post = np.where(pos < len(peaks) - 1, rr[np.clip(pos, 0, len(rr) - 1)], rr[-1])

    # Mean of the 10 RR intervals that end at the current beat
    csum = np.concatenate([[0.0], np.cumsum(rr)])
    end = np.clip(pos, 1, len(rr))               # number of intervals available
    start = np.clip(end - 10, 0, None)
    rr_avg = (csum[end] - csum[start]) / (end - start)
    return rr_pre, rr_post, rr_avg


def main():
    X = np.load(W1_DATA / "processed" / "X.npy")
    meta = pd.read_csv(W1_DATA / "processed" / "metadata.csv")
    with open(W1_DATA / "splits.json") as f:
        splits = json.load(f)

    split = np.empty(len(meta), dtype=object)
    for name, key in [("train", "train_indices"), ("val", "validation_indices"),
                      ("test", "test_indices")]:
        split[splits[key]] = name

    print(f"Computing ResNet1D probabilities for {len(X):,} beats...")
    p, w2_threshold = resnet_probabilities(X)

    print("Computing RR intervals from the raw annotations...")
    rr = np.zeros((len(meta), 3))
    for record_id, idx in meta.groupby("record_id").indices.items():
        rr[idx] = np.column_stack(rr_features(record_id, meta["sample"].to_numpy()[idx]))

    beats = pd.DataFrame({
        "beat_index": np.arange(len(meta)),
        "record_id": meta["record_id"].astype(str),
        "subject_group": meta["subject_group"].astype(str),
        "split": split,
        "sample": meta["sample"],
        "aami_class": meta["aami_class"],
        "label": meta["label"],
        "p": p.round(6),
        "rr_pre": rr[:, 0].round(4),
        "rr_post": rr[:, 1].round(4),
        "rr_avg": rr[:, 2].round(4),
    }).sort_values(["record_id", "sample"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    beats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(beats):,} beats to {OUT_PATH}\n")

    # ---- Sanity checks ----
    print(beats.groupby("split").agg(beats=("label", "size"),
                                     abnormal=("label", "sum"),
                                     records=("record_id", "nunique")))

    test = beats[beats.split == "test"]
    pred = (test.p >= w2_threshold).astype(int)
    print(f"\nWorkshop 2 on test with threshold {w2_threshold:.3f} "
          f"(must match the Workshop 2 report):")
    print(f"  precision {precision_score(test.label, pred):.4f} | "
          f"recall {recall_score(test.label, pred):.4f} | "
          f"F1 {f1_score(test.label, pred):.4f}")

    beats["rr_pre_ratio"] = beats.rr_pre / beats.rr_avg
    beats["rr_post_ratio"] = beats.rr_post / beats.rr_avg
    print("\nMedian features per AAMI class (all splits):")
    print(beats.groupby("aami_class")[["p", "rr_pre_ratio", "rr_post_ratio"]]
          .median().round(3))


if __name__ == "__main__":
    main()