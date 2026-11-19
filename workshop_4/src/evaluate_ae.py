"""
Evaluation of the autoencoders: model selection, reconstruction quality and
anomaly detection, plus the cumulative comparison with Workshops 1-3.

1. Model selection (VALIDATION only): among all runs in runs/workshop4/, the AE
   and the VAE with the highest validation anomaly AUC are copied to
   checkpoints/ae_best.pt and checkpoints/vae_best.pt.
2. Reconstruction on TEST: MSE and Pearson correlation per beat, by AAMI class
   and by patient record; panel of 8 test beats (original vs AE vs VAE) with
   difference curves.
3. Anomaly detection on TEST: scores = AE reconstruction error, VAE
   reconstruction error, VAE reconstruction error + KL. AUC-ROC and PR-AUC;
   the decision threshold is tuned on VALIDATION (max F1) and then applied to
   test (precision, recall, F1, detection per AAMI class).
4. Cumulative table: Workshop 1 Random Forest, Workshop 2 ResNet1D,
   Workshop 3 DQN agent and the Workshop 4 models on the same test beats.

Outputs in reports/. Usage (from workshop_4/):  python src/evaluate_ae.py
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from autoencoder import BEAT_LEN, build_model
from data_ae import REPO, W4_DIR, load_splits, to_loader
from train_ae import score_beats

RUNS = W4_DIR / "runs" / "workshop4"
CKPT = W4_DIR / "checkpoints"
REPORTS = W4_DIR / "reports"
W2_METRICS = REPO / "workshop_2" / "reports" / "test_metrics.csv"
W3_METRICS = REPO / "workshop_3" / "reports" / "test_policy_comparison.csv"
CLASSES = ["N", "S", "V", "F"]


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["model_kwargs"]["kind"], ckpt["model_kwargs"]["z_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def reconstruct(model, X):
    out = []
    for (x,) in to_loader(X, 2048):
        r = model(x, sample=False)["recon"] if model.kind == "vae" else model(x)["recon"]
        out.append(r.squeeze(1).numpy())
    return np.concatenate(out)


def rowwise_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12)


# ---------------------------------------------------------------- selection
def select_models():
    rows = []
    for d in sorted(RUNS.iterdir()):
        f = d / "summary.json"
        if "smoke" in d.name or not f.exists():
            continue
        s = json.load(open(f))
        c = s["config"]
        hist = pd.read_csv(d / "history.csv")
        rows.append({"run": d.name, "model": c["model"].upper(), "z_dim": c["z_dim"],
                     "beta": c["beta"] if c["model"] == "vae" else None,
                     "parameters": s["parameters"], "best_epoch": s["best_epoch"],
                     "val_mse_normal": s["val_mse_at_best"],
                     "val_kl": float(hist.loc[hist.epoch == s["best_epoch"], "val_kl"].iloc[0]),
                     "val_auc": s["val_auc_recon_at_best"]})
    table = pd.DataFrame(rows).sort_values(["model", "z_dim", "beta"])
    chosen = {}
    for kind in ["AE", "VAE"]:
        best = table[table.model == kind].sort_values("val_auc", ascending=False).iloc[0]
        shutil.copy(CKPT / f"{best.run}_best.pt", CKPT / f"{kind.lower()}_best.pt")
        chosen[kind] = best.run
    table["selected"] = table.run.isin(chosen.values())
    table.to_csv(REPORTS / "model_selection.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for d in table.run:
        h = pd.read_csv(RUNS / d / "history.csv")
        style = "-" if d.startswith("ae") else "--"
        axes[0].plot(h.epoch, h.val_mse, style, label=d)
        axes[1].plot(h.epoch, h.val_auc_recon, style, label=d)
    axes[0].set_yscale("log")
    axes[0].set_title("Validation reconstruction MSE (normal beats)")
    axes[1].set_title("Validation anomaly AUC-ROC (reconstruction error)")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curves_all.png", dpi=150)
    plt.close(fig)
    return table, chosen


# ---------------------------------------------------------------- panels
def plot_panel(X, R_ae, R_vae, meta, err_ae, path):
    picks = []
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        idx = np.where(meta.aami_class.to_numpy() == cls)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(err_ae[idx])]
        picks += [order[len(order) // 2], order[int(0.9 * (len(order) - 1))]]
    t = np.arange(BEAT_LEN) / 360 * 1000 - 200          # ms relative to the R-peak
    fig, axes = plt.subplots(len(picks), 2, figsize=(12, 1.9 * len(picks)),
                             gridspec_kw={"width_ratios": [2.2, 1]}, sharex=True)
    for row, i in enumerate(picks):
        a, b = axes[row]
        a.plot(t, X[i], color="black", lw=1.6, label="original")
        a.plot(t, R_ae[i], color="C0", lw=1.1, label="AE")
        a.plot(t, R_vae[i], color="C3", lw=1.1, ls="--", label="VAE")
        a.set_ylabel(f"{meta.aami_class.iloc[i]}  rec {meta.record_id.iloc[i]}", fontsize=8)
        a.set_title(f"MSE AE {np.mean((R_ae[i] - X[i]) ** 2):.3f} | "
                    f"VAE {np.mean((R_vae[i] - X[i]) ** 2):.3f}", fontsize=8, loc="left")
        b.plot(t, R_ae[i] - X[i], color="C0", lw=1)
        b.plot(t, R_vae[i] - X[i], color="C3", lw=1, ls="--")
        b.axhline(0, color="gray", lw=0.6)
        b.set_ylim(-2, 2)
        for ax in (a, b):
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[0, 1].set_title("difference (reconstruction - original)", fontsize=8)
    axes[-1, 0].set_xlabel("ms from R-peak")
    axes[-1, 1].set_xlabel("ms from R-peak")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    REPORTS.mkdir(exist_ok=True)
    selection, chosen = select_models()
    print("Model selection on VALIDATION (anomaly AUC):")
    print(selection.round(4).to_string(index=False))
    print(f"-> ae_best.pt = {chosen['AE']}, vae_best.pt = {chosen['VAE']}\n")

    ae, _ = load_model(CKPT / "ae_best.pt")
    vae, _ = load_model(CKPT / "vae_best.pt")
    data = load_splits()

    # Anomaly scores on validation (threshold tuning) and test
    scores = {}
    for split in ["val", "test"]:
        X, y, _ = data[split]
        mse_ae, _ = score_beats(ae, X)
        mse_vae, kl_vae = score_beats(vae, X)
        scores[split] = {"AE reconstruction error": mse_ae,
                         "VAE reconstruction error": mse_vae,
                         "VAE reconstruction error + KL": mse_vae * BEAT_LEN + kl_vae}

    # ---- 2. Reconstruction quality on test
    X, y, meta = data["test"]
    R_ae, R_vae = reconstruct(ae, X), reconstruct(vae, X)
    rec_rows, by_class = [], []
    for name, R in [("AE", R_ae), ("VAE", R_vae)]:
        err = ((R - X) ** 2).mean(1)
        corr = rowwise_corr(R, X)
        rec_rows.append({"model": name, "MSE all": err.mean(), "MSE normal": err[y == 0].mean(),
                         "MSE abnormal": err[y == 1].mean(),
                         "ratio abnormal/normal": err[y == 1].mean() / err[y == 0].mean(),
                         "corr normal": corr[y == 0].mean(), "corr abnormal": corr[y == 1].mean()})
        for cls in CLASSES:
            m = (meta.aami_class == cls).to_numpy()
            by_class.append({"model": name, "class": cls, "beats": int(m.sum()),
                             "MSE mean": err[m].mean(), "MSE median": np.median(err[m]),
                             "corr mean": corr[m].mean()})
    rec = pd.DataFrame(rec_rows)
    cls_tab = pd.DataFrame(by_class)
    print("Reconstruction quality on TEST:")
    print(rec.round(4).to_string(index=False))
    print(cls_tab.round(4).to_string(index=False))
    rec.to_csv(REPORTS / "test_reconstruction.csv", index=False)
    cls_tab.to_csv(REPORTS / "test_reconstruction_by_class.csv", index=False)

    err_ae = ((R_ae - X) ** 2).mean(1)
    err_vae = ((R_vae - X) ** 2).mean(1)
    per_rec = pd.DataFrame({"record": meta.record_id, "y": y, "ae": err_ae, "vae": err_vae})
    per_rec = per_rec[per_rec.y == 0].groupby("record")[["ae", "vae"]].mean().round(4)
    per_rec.columns = ["AE MSE (normal beats)", "VAE MSE (normal beats)"]
    print("\nNormal-beat reconstruction error per test record (patient variability):")
    print(per_rec.to_string())
    per_rec.to_csv(REPORTS / "test_reconstruction_by_record.csv")
    plot_panel(X, R_ae, R_vae, meta, err_ae, REPORTS / "reconstruction_panel.png")

    # ---- 3. Anomaly detection
    y_val = data["val"][1]
    det_rows, det_class = [], []
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.8))
    for k, (name, s_test) in enumerate(scores["test"].items()):
        s_val = scores["val"][name]
        p, r, t = precision_recall_curve(y_val, s_val)
        f1 = 2 * p * r / (p + r + 1e-12)
        thr = float(t[np.argmax(f1[:-1])])
        pred = (s_test >= thr).astype(int)
        det_rows.append({"method": f"W4 {name}", "val AUC-ROC": roc_auc_score(y_val, s_val),
                         "test AUC-ROC": roc_auc_score(y, s_test),
                         "test PR-AUC": average_precision_score(y, s_test),
                         "threshold (val)": thr, "precision": precision_score(y, pred, zero_division=0),
                         "recall": recall_score(y, pred), "F1": f1_score(y, pred)})
        for cls in CLASSES:
            m = (meta.aami_class == cls).to_numpy()
            det_class.append({"method": name, "class": cls, "flagged (%)": 100 * pred[m].mean()})
        fpr, tpr, _ = roc_curve(y, s_test)
        axes[0].plot(fpr, tpr, label=f"{name} ({det_rows[-1]['test AUC-ROC']:.3f})")
        ax = axes[k + 1]
        bins = np.logspace(np.log10(max(s_test.min(), 1e-4)), np.log10(s_test.max()), 60)
        ax.hist(s_test[y == 0], bins=bins, alpha=0.6, density=True, label="normal")
        ax.hist(s_test[y == 1], bins=bins, alpha=0.6, density=True, label="abnormal")
        ax.axvline(thr, color="black", ls="--", lw=1, label="threshold (val)")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].plot([0, 1], [0, 1], color="gray", lw=0.8)
    axes[0].set_title("Test ROC curves")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "anomaly_detection.png", dpi=150)
    plt.close(fig)

    det = pd.DataFrame(det_rows)
    print("\nAnomaly detection (trained on normal beats only):")
    print(det.round(4).to_string(index=False))
    det.to_csv(REPORTS / "test_anomaly_detection.csv", index=False)
    dc = pd.DataFrame(det_class).pivot(index="method", columns="class", values="flagged (%)")
    print("\nShare of test beats flagged as anomalous, per AAMI class (%):")
    print(dc[CLASSES].round(1).to_string())
    dc.to_csv(REPORTS / "test_anomaly_by_class.csv")

    # ---- 4. Cumulative comparison
    cum = []
    if W2_METRICS.exists():
        w2 = pd.read_csv(W2_METRICS, index_col=0)
        for idx, label in [("Random Forest (W1)", "W1 Random Forest (supervised)"),
                           ("ResNet1D", "W2 ResNet1D (supervised)")]:
            if idx in w2.index:
                r = w2.loc[idx]
                cum.append({"method": label, "test AUC-ROC": r["ROC-AUC"], "test PR-AUC": r["PR-AUC"],
                            "precision": r["Precision"], "recall": r["Recall"], "F1": r["F1-score"]})
    if W3_METRICS.exists():
        w3 = pd.read_csv(W3_METRICS, index_col=0)
        if "DQN agent" in w3.index:
            r = w3.loc["DQN agent"]
            cum.append({"method": "W3 DQN adaptive threshold (RL on top of W2)",
                        "test AUC-ROC": np.nan, "test PR-AUC": np.nan,
                        "precision": r["precision"], "recall": r["recall"], "F1": r["f1"]})
    for r in det_rows:
        cum.append({"method": r["method"] + " (unsupervised)", "test AUC-ROC": r["test AUC-ROC"],
                    "test PR-AUC": r["test PR-AUC"], "precision": r["precision"],
                    "recall": r["recall"], "F1": r["F1"]})
    cum = pd.DataFrame(cum)
    print("\nCumulative comparison on the same 13,869 test beats:")
    print(cum.round(4).to_string(index=False))
    cum.to_csv(REPORTS / "cumulative_comparison.csv", index=False)

    with open(REPORTS / "evaluation_summary.json", "w") as f:
        json.dump({"selected": chosen,
                   "reconstruction": rec.to_dict(orient="records"),
                   "anomaly_detection": det.to_dict(orient="records")}, f, indent=4, default=float)
    print(f"\nSaved tables and figures to {REPORTS}")


if __name__ == "__main__":
    main()
