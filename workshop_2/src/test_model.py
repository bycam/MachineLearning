"""
Load the saved checkpoint and classify heartbeats from NEW data.

Three ways to use it (from workshop_2/):

  1) A WFDB ECG record (.dat + .hea, 360 Hz). If an .atr annotation file
     exists, its beat positions are used and the true labels are shown;
     otherwise R-peaks are detected automatically with XQRS.
        python src/test_model.py --record ..\\workshop_1\\data\\raw\\mitdb\\108

  2) A .npy file with already-cut beats, shape (n_beats, 216):
        python src/test_model.py --npy my_beats.npy

  3) No arguments: a quick demo on 10 random beats of the test split.
        python src/test_model.py

Predictions are saved to reports/predictions_<name>.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from wfdb import processing

from data import PROCESSED_DIR, SPLITS_PATH, normalize_beats
from model import ECGResNet1D

W2_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = W2_DIR / "checkpoints" / "model_best.pt"
REPORTS_DIR = W2_DIR / "reports"

FS = 360
BEFORE, AFTER = int(0.2 * FS), int(0.4 * FS)   # 72 + 144 = 216 samples
WINDOW = BEFORE + AFTER

# Same AAMI grouping as Workshop 1 (Q / paced beats are not classified)
AAMI = {"N": "NLRej", "S": "AaJS", "V": "VE", "F": "F"}
SYMBOL_TO_CLASS = {s: c for c, symbols in AAMI.items() for s in symbols}


def load_model(path=CHECKPOINT):
    ckpt = torch.load(path, map_location="cpu")
    model = ECGResNet1D(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["threshold"]


@torch.no_grad()
def predict(model, beats_raw, threshold):
    """beats_raw: (n, 216) raw ECG in mV. Returns probabilities and 0/1 labels."""
    x = torch.from_numpy(normalize_beats(beats_raw)).unsqueeze(1)
    probs = torch.sigmoid(model(x)).numpy()
    return probs, (probs >= threshold).astype(int)


def beats_from_record(record_path, channel=0):
    """Cut 216-sample windows around each beat of a WFDB record."""
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, channel]
    if record.fs != FS:
        signal, _ = processing.resample_sig(signal, record.fs, FS)

    true_class = None
    if Path(record_path + ".atr").exists():
        ann = wfdb.rdann(record_path, "atr")
        keep = [(s, sym) for s, sym in zip(ann.sample, ann.symbol)
                if sym in SYMBOL_TO_CLASS]
        peaks = np.array([s for s, _ in keep])
        true_class = np.array([SYMBOL_TO_CLASS[sym] for _, sym in keep])
        if record.fs != FS:
            peaks = (peaks * FS / record.fs).astype(int)
        print(f"Using {len(peaks)} annotated beats from {record_path}.atr")
    else:
        peaks = processing.xqrs_detect(signal, fs=FS, verbose=False)
        print(f"No annotations found: detected {len(peaks)} R-peaks with XQRS")

    valid = (peaks - BEFORE >= 0) & (peaks + AFTER <= len(signal))
    peaks = peaks[valid]
    if true_class is not None:
        true_class = true_class[valid]
    beats = np.stack([signal[p - BEFORE:p + AFTER] for p in peaks]).astype(np.float32)
    return beats, peaks, true_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=str, help="WFDB record path without extension")
    p.add_argument("--npy", type=str, help=".npy file with shape (n_beats, 216)")
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    args = p.parse_args()

    model, threshold = load_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} (threshold {threshold:.3f})")

    true_class = None
    if args.record:
        beats, peaks, true_class = beats_from_record(args.record)
        out = pd.DataFrame({"sample": peaks, "time_s": np.round(peaks / FS, 2)})
        name = Path(args.record).name
    elif args.npy:
        beats = np.load(args.npy).astype(np.float32)
        if beats.ndim != 2 or beats.shape[1] != WINDOW:
            sys.exit(f"Expected shape (n_beats, {WINDOW}), got {beats.shape}")
        out = pd.DataFrame({"beat": np.arange(len(beats))})
        name = Path(args.npy).stem
    else:
        # Demo: 10 random raw beats from the test split, with their true labels
        with open(SPLITS_PATH) as f:
            test_idx = np.array(json.load(f)["test_indices"])
        rng = np.random.default_rng(0)
        labels = pd.read_csv(PROCESSED_DIR / "metadata.csv")["label"].to_numpy()
        pick = np.concatenate([
            rng.choice(test_idx[labels[test_idx] == c], size=5, replace=False)
            for c in (0, 1)])
        beats = np.load(PROCESSED_DIR / "X.npy")[pick]
        meta = pd.read_csv(PROCESSED_DIR / "metadata.csv").iloc[pick]
        true_class = meta["aami_class"].to_numpy()
        out = pd.DataFrame({"record": meta["record_id"].to_numpy(),
                            "sample": meta["sample"].to_numpy()})
        name = "demo"

    probs, preds = predict(model, beats, threshold)
    out["p_abnormal"] = probs.round(4)
    out["prediction"] = np.where(preds == 1, "Abnormal", "Normal")
    if true_class is not None:
        out["true_aami"] = true_class
        out["true_label"] = np.where(true_class == "N", "Normal", "Abnormal")
        out["correct"] = out["prediction"] == out["true_label"]

    print(f"\n{len(out)} beats -> {int(preds.sum())} predicted abnormal "
          f"({100 * preds.mean():.1f}%)")
    if true_class is not None:
        print(f"Accuracy against annotations: {100 * out['correct'].mean():.1f}%")
    print(out.head(15).to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"predictions_{name}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved all predictions to {path}")


if __name__ == "__main__":
    main()