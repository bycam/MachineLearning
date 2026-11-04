"""
Train the 1D ResNet on MIT-BIH heartbeats.

- Loss: BCEWithLogitsLoss with pos_weight (class imbalance)
- Optimizer: AdamW (weight decay) + cosine annealing learning rate
- Model selection / early stopping: validation PR-AUC
- Decision threshold: tuned on VALIDATION to maximize F1
- The test set is never used here (see evaluate.py)

Logs go to runs/<run_name>/ (TensorBoard + history.csv + curves PNG).
The best model is saved to checkpoints/model_best.pt.

Usage (from workshop_2/):
    python src/train.py --smoke      # 1 quick epoch to check that everything runs
    python src/train.py              # full training
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.tensorboard import SummaryWriter

from data import make_dataloaders
from model import ECGResNet1D, count_parameters

W2_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = W2_DIR / "checkpoints"
RUNS_DIR = W2_DIR / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--pos-weight-mode", choices=["full", "sqrt"], default="full")
    p.add_argument("--patience", type=int, default=7,
                   help="epochs without val PR-AUC improvement before stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="resnet1d")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch on a few batches, just to test the pipeline")
    return p.parse_args()


def set_seed(seed):
    """Fix every source of randomness so the run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)          # data augmentation uses np.random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs),
        "roc_auc": roc_auc_score(y_true, probs),
    }
    return {k: float(v) for k, v in metrics.items()}


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over a loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, all_probs, all_y = 0.0, [], []
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.float().to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_y.append(y.cpu().numpy())

    y_true = np.concatenate(all_y).astype(int)
    probs = np.concatenate(all_probs)
    return total_loss / len(y_true), y_true, probs


def tune_threshold(y_true, probs):
    """Threshold that maximizes F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = np.argmax(f1[:-1])          # last point has no threshold
    return float(thresholds[best]), float(f1[best])


def plot_curves(history, path):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ["loss", "f1", "pr_auc"],
                                 ["Loss", "F1 (threshold 0.5)", "PR-AUC"]):
        ax.plot(h["epoch"], h[f"train_{metric}"], label="train")
        ax.plot(h["epoch"], h[f"val_{metric}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, data = make_dataloaders(batch_size=args.batch_size, seed=args.seed)

    # Class imbalance: an abnormal beat costs (normal / abnormal) times more
    y_train = data["train"][1]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    if args.pos_weight_mode == "sqrt":
        pos_weight = np.sqrt(pos_weight)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    model = ECGResNet1D(base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    max_batches = 20 if args.smoke else None

    run_dir = RUNS_DIR / (args.run_name + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"Device: {device} | parameters: {count_parameters(model):,} | "
          f"pos_weight: {pos_weight:.2f}")
    print(f"{'ep':>3} {'lr':>8} {'tr_loss':>8} {'va_loss':>8} {'va_P':>6} "
          f"{'va_R':>6} {'va_F1':>6} {'va_PRAUC':>8} {'time':>6}")

    history, best_pr_auc, best_epoch, bad_epochs = [], -1.0, 0, 0
    best_path = CHECKPOINT_DIR / ("model_smoke.pt" if args.smoke else "model_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_y, tr_p = run_epoch(model, loaders["train"], criterion,
                                        device, optimizer, max_batches)
        va_loss, va_y, va_p = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        tr_m = compute_metrics(tr_y, tr_p)
        va_m = compute_metrics(va_y, va_p)

        row = {"epoch": epoch, "lr": lr, "train_loss": tr_loss, "val_loss": va_loss}
        row.update({f"train_{k}": v for k, v in tr_m.items()})
        row.update({f"val_{k}": v for k, v in va_m.items()})
        history.append(row)

        writer.add_scalar("lr", lr, epoch)
        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        for k in tr_m:
            writer.add_scalars(k, {"train": tr_m[k], "val": va_m[k]}, epoch)

        print(f"{epoch:>3} {lr:>8.1e} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{va_m['precision']:>6.3f} {va_m['recall']:>6.3f} {va_m['f1']:>6.3f} "
              f"{va_m['pr_auc']:>8.4f} {time.time() - t0:>5.0f}s")

        # Keep the epoch with the best validation PR-AUC (threshold-independent)
        if va_m["pr_auc"] > best_pr_auc:
            best_pr_auc, best_epoch, bad_epochs = va_m["pr_auc"], epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    writer.close()
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    plot_curves(history, run_dir / "learning_curves.png")

    # Reload the best weights and tune the decision threshold on VALIDATION
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, va_y, va_p = run_epoch(model, loaders["val"], criterion, device)
    threshold, val_f1 = tune_threshold(va_y, va_p)
    val_metrics = compute_metrics(va_y, va_p, threshold)

    config = vars(args) | {"pos_weight": float(pos_weight),
                           "parameters": int(count_parameters(model))}
    torch.save({
        "model_state": model.state_dict(),
        "model_kwargs": {"base_channels": args.base_channels, "dropout": args.dropout},
        "threshold": threshold,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "config": config,
    }, best_path)

    with open(run_dir / "summary.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "threshold": threshold,
                   "val_metrics_at_threshold": val_metrics, "config": config},
                  f, indent=4)

    print(f"\nBest epoch: {best_epoch} | val PR-AUC: {best_pr_auc:.4f}")
    print(f"Tuned threshold (val): {threshold:.3f} -> val F1 {val_f1:.4f}, "
          f"P {val_metrics['precision']:.4f}, R {val_metrics['recall']:.4f}")
    print(f"Saved: {best_path}")
    print(f"Curves: {run_dir / 'learning_curves.png'}")


if __name__ == "__main__":
    main()