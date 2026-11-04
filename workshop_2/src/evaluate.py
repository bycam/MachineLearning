"""
Evaluate a trained checkpoint and compare it with the Workshop 1 baseline.

- Same metrics as Workshop 1 (accuracy, precision, recall, F1, PR-AUC) + ROC-AUC
- Uses the decision threshold tuned on VALIDATION (stored in the checkpoint)
- Error analysis: per AAMI class (N, S, V, F) and per patient record
- On the test split, it also reloads the Workshop 1 Random Forest and
  recomputes its metrics and confusion matrix on the SAME test beats

Outputs go to workshop_2/reports/.

Usage (from workshop_2/):
    python src/evaluate.py                 # test set (run once, at the end)
    python src/evaluate.py --split val     # validation set (for diagnosis)
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from data import PROCESSED_DIR, SPLIT_KEYS, SPLITS_PATH, W1_DIR, load_splits
from model import ECGResNet1D

W2_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = W2_DIR / "reports"
RF_PATH = W1_DIR / "models" / "final_random_forest.joblib"
RF_CONFIG_PATH = W1_DIR / "models" / "model_config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=str(W2_DIR / "checkpoints" / "model_best.pt"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    return p.parse_args()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ECGResNet1D(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model, X, device, batch_size=1024):
    probs = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).unsqueeze(1).to(device)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
        "F1-score": f1_score(y, preds, zero_division=0),
        "PR-AUC": average_precision_score(y, probs),
        "ROC-AUC": roc_auc_score(y, probs),
        "Threshold": threshold,
    }


def per_class_table(meta, preds):
    """
    Share of beats of each AAMI class predicted as abnormal.
    For N that is the false-alarm rate; for S, V, F it is the detection rate.
    """
    rows = []
    for cls in ["N", "S", "V", "F"]:
        mask = (meta["aami_class"] == cls).to_numpy()
        if mask.sum() == 0:
            continue
        flagged = preds[mask].mean()
        rows.append({
            "AAMI class": cls,
            "Beats": int(mask.sum()),
            "Predicted abnormal (%)": round(100 * flagged, 2),
            "Meaning": "false alarms" if cls == "N" else "detected (recall)",
        })
    return pd.DataFrame(rows)


def per_record_table(meta, y, preds):
    df = pd.DataFrame({"record": meta["record_id"].astype(str), "y": y, "p": preds})
    rows = []
    for rec, g in df.groupby("record"):
        tp = int(((g.y == 1) & (g.p == 1)).sum())
        fp = int(((g.y == 0) & (g.p == 1)).sum())
        fn = int(((g.y == 1) & (g.p == 0)).sum())
        rows.append({
            "Record": rec, "Beats": len(g), "Abnormal": int(g.y.sum()),
            "TP": tp, "FP": fp, "FN": fn,
            "Recall": round(tp / (tp + fn), 3) if tp + fn else np.nan,
            "Precision": round(tp / (tp + fp), 3) if tp + fp else np.nan,
        })
    return pd.DataFrame(rows)


def plot_confusion_matrices(results, y, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y, r["preds"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13)
        ax.set_xticks([0, 1], ["Normal", "Abnormal"])
        ax.set_yticks([0, 1], ["Normal", "Abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} (thr={r['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results, y, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y, r["probs"])
        ap = average_precision_score(y, r["probs"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_random_forest(split):
    """Workshop 1 Random Forest, evaluated on the raw (un-normalized) beats it was trained on."""
    if not RF_PATH.exists():
        return None
    rf = joblib.load(RF_PATH)
    with open(RF_CONFIG_PATH) as f:
        threshold = json.load(f)["threshold"]
    with open(SPLITS_PATH) as f:
        idx = np.array(json.load(f)[SPLIT_KEYS[split]])
    X_raw = np.load(PROCESSED_DIR / "X.npy")[idx]
    return rf.predict_proba(X_raw)[:, 1], threshold


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_splits()[args.split]
    model, ckpt = load_model(args.checkpoint, device)
    threshold = ckpt["threshold"]
    probs = predict_proba(model, X, device)

    results = {"ResNet1D": {"probs": probs, "threshold": threshold,
                            "preds": (probs >= threshold).astype(int)}}

    try:
        rf_out = load_random_forest(args.split)
    except Exception as e:       # e.g. scikit-learn version mismatch
        print(f"[warning] Could not load the Random Forest: {e}")
        rf_out = None
    if rf_out is not None:
        rf_probs, rf_thr = rf_out
        results = {"Random Forest (W1)": {
            "probs": rf_probs, "threshold": rf_thr,
            "preds": (rf_probs >= rf_thr).astype(int)}} | results

    tag = args.split
    print(f"\n=== {tag.upper()} set: {len(y):,} beats, {int(y.sum()):,} abnormal "
          f"({100 * y.mean():.1f}%) ===")
    print(f"Checkpoint: {args.checkpoint} (best epoch {ckpt['best_epoch']})\n")

    comparison = pd.DataFrame(
        {name: compute_metrics(y, r["probs"], r["threshold"])
         for name, r in results.items()}
    ).T
    print("Metrics:")
    print(comparison.round(4).to_string())
    comparison.to_csv(REPORTS_DIR / f"{tag}_metrics.csv")

    for name, r in results.items():
        cm = confusion_matrix(y, r["preds"])
        print(f"\nConfusion matrix - {name}  [rows: actual N/A, cols: predicted N/A]")
        print(cm)
        pd.DataFrame(cm, index=["Actual_Normal", "Actual_Abnormal"],
                     columns=["Predicted_Normal", "Predicted_Abnormal"]).to_csv(
            REPORTS_DIR / f"{tag}_confusion_{name.split()[0].lower()}.csv")

    print("\nPer AAMI class:")
    class_tables = []
    for name, r in results.items():
        t = per_class_table(meta, r["preds"])
        t.insert(0, "Model", name)
        class_tables.append(t)
    class_df = pd.concat(class_tables, ignore_index=True)
    print(class_df.to_string(index=False))
    class_df.to_csv(REPORTS_DIR / f"{tag}_per_class.csv", index=False)

    rec_df = per_record_table(meta, y, results["ResNet1D"]["preds"])
    print("\nPer patient record (ResNet1D):")
    print(rec_df.to_string(index=False))
    rec_df.to_csv(REPORTS_DIR / f"{tag}_per_record.csv", index=False)

    plot_confusion_matrices(results, y, REPORTS_DIR / f"{tag}_confusion_matrices.png")
    plot_pr_curves(results, y, REPORTS_DIR / f"{tag}_pr_curves.png")
    print(f"\nSaved tables and figures to {REPORTS_DIR}")


if __name__ == "__main__":
    main()