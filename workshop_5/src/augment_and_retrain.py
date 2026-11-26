"""
Data augmentation experiment: retrain the Workshop 2 ResNet1D on
  (a) real data only                       (baseline)
  (b) real + synthetic abnormal beats from the vanilla GAN
  (c) real + synthetic abnormal beats from the WGAN-GP

Fair comparison: same architecture and hyperparameters as the final Workshop 2
model (base 16 channels, dropout 0.5, AdamW lr 3e-4, weight decay 1e-2, cosine
schedule, batch 256, the same signal augmentations), the SAME pos_weight in the
three conditions (computed on the real data only, so only the data changes),
the same number of epochs, the same validation/test patients. The epoch with
the best validation PR-AUC is kept and its decision threshold is tuned on
validation (max F1). Several seeds per condition give mean +/- std, because the
differences may be smaller than the run-to-run variability.

Outputs: reports/augmentation_runs.csv (one row per run),
         reports/augmentation_summary.csv, reports/augmentation_results.png

Usage (from workshop_5/):
    python src/augment_and_retrain.py --smoke
    python src/augment_and_retrain.py                       # 3 seeds x 3 conditions
    python src/augment_and_retrain.py --seeds 42 --epochs 8 # faster
    python src/augment_and_retrain.py --epochs 6 --conditions oversample
        # control: add 4,000 duplicated REAL abnormal beats instead of synthetic ones;
        # rows are appended to the existing augmentation_runs.csv
"""
import argparse
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from data_gan import W2_SRC, W5_DIR, load_splits, zscore_beats

sys.path.append(str(W2_SRC))
from data import ECGBeatDataset    # noqa: E402  (Workshop 2 dataset with its augmentations)
from model import ECGResNet1D      # noqa: E402

DATA = W5_DIR / "data"
REPORTS = W5_DIR / "reports"
CKPT = W5_DIR / "checkpoints"
CONDITIONS = {"real": "(a) Real only",
              "vanilla": "(b) Real + vanilla GAN",
              "wgan": "(c) Real + WGAN-GP",
              "oversample": "(d) Real + oversampling (control)"}
SHORT = {"(a) Real only": "real", "(b) Real + vanilla GAN": "+ vanilla",
         "(c) Real + WGAN-GP": "+ WGAN-GP", "(d) Real + oversampling (control)": "+ oversampling"}
COLORS = ["#8d99ae", "#2a6f97", "#c0392b", "#e0a800"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).unsqueeze(1))).numpy())
    return np.concatenate(out)


def run(cond, seed, data, synth, args, pos_weight):
    set_seed(seed)
    X_tr, y_tr = data["train"]
    if cond == "oversample":
        # control: same number of extra abnormal beats, but duplicated REAL ones
        # (the W2 signal augmentations make each copy slightly different)
        idx = np.random.default_rng(seed).choice(np.flatnonzero(y_tr == 1), args.n_synthetic)
        X_tr = np.concatenate([X_tr, X_tr[idx]])
        y_tr = np.concatenate([y_tr, np.ones(len(idx), dtype=np.int64)])
    elif cond != "real":
        X_tr = np.concatenate([X_tr, synth[cond]])
        y_tr = np.concatenate([y_tr, np.ones(len(synth[cond]), dtype=np.int64)])
    loader = DataLoader(ECGBeatDataset(X_tr, y_tr, augment=True), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = ECGResNet1D(base_channels=16, dropout=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X_va, y_va = data["val"]
    best_auc, best_state, best_epoch = -1, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (x, y) in enumerate(loader):
            if args.smoke and i >= 20:
                break
            loss = criterion(model(x), y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        p_va = predict(model, X_va)
        auc = average_precision_score(y_va, p_va)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    p_va = predict(model, X_va)
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[np.argmax(f1[:-1])])

    X_te, y_te = data["test"]
    p_te = predict(model, X_te)
    pred = (p_te >= threshold).astype(int)
    cls = data["test_classes"]
    row = {"condition": CONDITIONS[cond], "seed": seed, "train_beats": len(y_tr),
           "train_abnormal": int(y_tr.sum()), "best_epoch": best_epoch,
           "val PR-AUC": best_auc, "threshold": threshold,
           "test PR-AUC": average_precision_score(y_te, p_te), "test ROC-AUC": roc_auc_score(y_te, p_te),
           "test precision": precision_score(y_te, pred), "test recall": recall_score(y_te, pred),
           "test F1": f1_score(y_te, pred)}
    for c in ["S", "V", "F"]:
        row[f"detected {c} (%)"] = 100 * pred[cls == c].mean()
    row["false alarms N (%)"] = 100 * pred[cls == "N"].mean()
    if seed == args.seeds[0] and not args.smoke:
        torch.save({"model_state": model.state_dict(), "threshold": threshold, "condition": cond,
                    "seed": seed}, CKPT / f"w2_retrained_{cond}.pt")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-synthetic", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                   default=["real", "vanilla", "wgan"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, [42]
    REPORTS.mkdir(exist_ok=True)

    splits = load_splits()
    data = {s: (zscore_beats(splits[s][0]), splits[s][1]) for s in ["train", "val", "test"]}
    data["test_classes"] = splits["test"][2]["aami_class"].to_numpy()
    synth = {k: zscore_beats(np.load(DATA / f"synthetic_{k}.npy")[:args.n_synthetic])
             for k in ["vanilla", "wgan"]}
    y_tr = data["train"][1]
    pos_weight = float(np.sqrt((y_tr == 0).sum() / (y_tr == 1).sum()))
    print(f"Real train: {len(y_tr):,} beats ({int(y_tr.sum()):,} abnormal) | synthetic added: "
          f"{args.n_synthetic:,} abnormal | pos_weight fixed at {pos_weight:.2f} | "
          f"{args.epochs} epochs | seeds {args.seeds}\n")

    rows = []
    runs_csv = REPORTS / "augmentation_runs.csv"
    if args.conditions != ["real", "vanilla", "wgan"] and runs_csv.exists() and not args.smoke:
        # keep the runs of the other conditions already in the file
        old = pd.read_csv(runs_csv)
        new_labels = [CONDITIONS[c] for c in args.conditions]
        rows = old[~old.condition.isin(new_labels)].to_dict("records")
    for seed in args.seeds:
        for cond in args.conditions:
            t0 = time.time()
            r = run(cond, seed, data, synth, args, pos_weight)
            rows.append(r)
            print(f"{r['condition']:<24} seed {seed}: best epoch {r['best_epoch']:>2} | "
                  f"test F1 {r['test F1']:.4f} P {r['test precision']:.4f} R {r['test recall']:.4f} "
                  f"PR-AUC {r['test PR-AUC']:.4f} | S {r['detected S (%)']:.1f}% "
                  f"V {r['detected V (%)']:.1f}% | {time.time() - t0:.0f}s")
            pd.DataFrame(rows).to_csv(runs_csv, index=False)

    runs = pd.DataFrame(rows)
    order = [c for c in CONDITIONS.values() if c in set(runs.condition)]
    runs["condition"] = pd.Categorical(runs.condition, order, ordered=True)
    runs = runs.sort_values(["condition", "seed"])
    metrics = ["test F1", "test precision", "test recall", "test PR-AUC", "test ROC-AUC",
               "detected S (%)", "detected V (%)", "false alarms N (%)", "val PR-AUC"]
    summary = runs.groupby("condition", observed=True)[metrics].agg(["mean", "std"])
    print("\nSummary on the TEST patients (mean and std over seeds):")
    print(summary.round(4).to_string())
    summary.to_csv(REPORTS / "augmentation_summary.csv")

    fig, axes = plt.subplots(1, 4, figsize=(4.3 * len(order) + 4, 3.8))
    for ax, m in zip(axes, ["test F1", "test PR-AUC", "detected S (%)", "detected V (%)"]):
        means = runs.groupby("condition", observed=True)[m].mean()
        stds = runs.groupby("condition", observed=True)[m].std().fillna(0)
        ax.bar(range(len(means)), means, yerr=stds, capsize=4,
               color=[COLORS[list(CONDITIONS.values()).index(c)] for c in means.index])
        for k, cond in enumerate(means.index):
            vals = runs[runs.condition == cond][m]
            ax.scatter([k] * len(vals), vals, color="black", s=12, zorder=3)
        ax.set_xticks(range(len(means)), [SHORT[c] for c in means.index], fontsize=8)
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "augmentation_results.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results to {REPORTS}")


if __name__ == "__main__":
    main()
