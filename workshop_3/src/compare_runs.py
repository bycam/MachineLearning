"""
Compare the hyperparameter configurations (validation only, no test data).

Reads runs/workshop3/<run>/validation.csv and summary.json and produces:
  - reports/hyperparameter_search.csv : one row per configuration
  - reports/hyperparameter_search.png : validation return vs environment steps
    (sample efficiency) and exploration rate
The final agent is the configuration with the best SMOOTHED validation return;
its checkpoint is copied to checkpoints/agent_best.pt.

Usage (from workshop_3/):  python src/compare_runs.py
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from env import W3_DIR

RUNS = W3_DIR / "runs" / "workshop3"
CKPT = W3_DIR / "checkpoints"
REPORTS = W3_DIR / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    rows, curves = [], {}
    for run_dir in sorted(RUNS.iterdir()):
        val_path, summ_path = run_dir / "validation.csv", run_dir / "summary.json"
        if "smoke" in run_dir.name or not (val_path.exists() and summ_path.exists()):
            continue
        val = pd.read_csv(val_path)
        with open(summ_path) as f:
            s = json.load(f)
        c, keep = s["config"], s["keep_val_return"]
        beats_keep = val[val.val_return > keep]
        rows.append({
            "run": run_dir.name,
            "train_probs": c.get("train_probs", "p_w2"),
            "lr": c["lr"], "gamma": c["gamma"], "eps_decay": c["eps_decay"],
            "weight_decay": c["weight_decay"],
            "best_val_return": round(s["best_val_return"], 3),
            "best_episode": s["best_episode"],
            "best_smoothed_val_return": round(val.val_return_smoothed.max(), 3),
            "mean_val_return_2nd_half": round(val.val_return[len(val) // 2:].mean(), 3),
            "min_val_return_2nd_half": round(val.val_return[len(val) // 2:].min(), 3),
            "val_f1_at_best": round(s["val_metrics_at_best"]["f1"], 4),
            "steps_to_beat_fixed": int(beats_keep.env_steps.iloc[0]) if len(beats_keep) else None,
            "fixed_threshold_val_return": round(keep, 3),
        })
        curves[run_dir.name] = (val, keep)

    table = pd.DataFrame(rows).sort_values("best_smoothed_val_return", ascending=False)
    table.to_csv(REPORTS / "hyperparameter_search.csv", index=False)
    print(table.to_string(index=False))

    final = table.iloc[0]["run"]
    shutil.copy(CKPT / f"{final}_best.pt", CKPT / "agent_best.pt")
    print(f"\nFinal agent (best smoothed validation return): {final}")
    print(f"Copied {final}_best.pt -> checkpoints/agent_best.pt")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, (val, keep) in curves.items():
        axes[0].plot(val.env_steps, val.val_return_smoothed, label=name,
                     lw=2.5 if name == final else 1.2)
    axes[0].axhline(keep, color="black", ls="--", label="fixed threshold (W2)")
    axes[0].set_xlabel("environment steps (beats seen)")
    axes[0].set_ylabel("smoothed validation return")
    axes[0].set_title("Sample efficiency: validation return vs. experience")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for name in curves:
        ep = pd.read_csv(RUNS / name / "episodes.csv")
        axes[1].plot(ep.episode, ep.epsilon, label=name)
    axes[1].set_xlabel("episode")
    axes[1].set_title("Exploration rate (epsilon)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "hyperparameter_search.png", dpi=150)
    print(f"Saved {REPORTS / 'hyperparameter_search.png'}")


if __name__ == "__main__":
    main()
