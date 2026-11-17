"""
Diagnose why an agent trained on the TRAIN patients may not transfer to
VALIDATION: compare how informative the ResNet1D probability is in each split.

For each split it reports:
  1. Share of abnormal beats in each probability bin (calibration)
  2. Best FIXED threshold for the environment reward, and the reward at 0.784
  3. Best threshold per patient record (an oracle upper bound for adaptation)

Usage (from workshop_3/):  python src/diagnose_shift.py
"""
import numpy as np
import pandas as pd

from env import DEFAULT_REWARDS, TAU_0, load_beats

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
BINS = [0, 0.2, 0.4, 0.6, TAU_0, 0.9, 1.0001]


def reward_per_beat(p, y, tau):
    alarm = p >= tau
    tp = np.sum(alarm & (y == 1))
    fn = np.sum(~alarm & (y == 1))
    fp = np.sum(alarm & (y == 0))
    r = DEFAULT_REWARDS
    return (r["tp"] * tp + r["fn"] * fn + r["fp"] * fp) / len(y)


def best_threshold(p, y):
    scores = [reward_per_beat(p, y, t) for t in GRID]
    i = int(np.argmax(scores))
    return GRID[i], scores[i]


def main():
    beats = load_beats()
    print("1) Share of abnormal beats per ResNet1D probability bin")
    beats["p_bin"] = pd.cut(beats["p"], BINS, right=False)
    table = beats.pivot_table(index="p_bin", columns="split", values="label",
                              aggfunc=["mean", "size"], observed=False)
    print((table["mean"][["train", "val", "test"]] * 100).round(1).astype(str) + "%")
    print("   beats per bin:")
    print(table["size"][["train", "val", "test"]])

    print("\n2) Reward per 1000 beats (TP +1, FN -1, FP -1)")
    for split in ["train", "val", "test"]:
        b = beats[beats.split == split]
        p, y = b.p.to_numpy(), b.label.to_numpy()
        t_best, s_best = best_threshold(p, y)
        print(f"   {split:<5} threshold 0.784 -> {1000 * reward_per_beat(p, y, TAU_0):6.1f} | "
              f"best fixed threshold {t_best:.2f} -> {1000 * s_best:6.1f}")

    print("\n3) Best threshold per record (oracle that knows each patient)")
    rows = []
    for split in ["val", "test"]:
        for rec, b in beats[beats.split == split].groupby("record_id"):
            p, y = b.p.to_numpy(), b.label.to_numpy()
            t_best, s_best = best_threshold(p, y)
            rows.append({"split": split, "record": rec, "beats": len(b),
                         "abnormal": int(y.sum()), "best_tau": t_best,
                         "reward_0.784": round(1000 * reward_per_beat(p, y, TAU_0), 1),
                         "reward_best": round(1000 * s_best, 1)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
