"""
Final evaluation on the TEST patients (run once, after choosing the agent).

- The DQN agent (checkpoints/agent_best.pt, greedy policy) and the baselines
  (random, fixed Workshop 2 threshold, rule-based heuristic) play every test
  episode, so all of them classify the same 13,869 beats.
- Workshop 1 Random Forest and Workshop 2 ResNet1D test metrics are added
  from workshop_2/reports/test_metrics.csv.
- "Oracle per patient": best fixed threshold for each test record chosen with
  the test labels. It is NOT a valid method, only an upper bound of what
  adapting the threshold per patient could achieve.
- Trajectories: 4 test episodes (largest gains and losses vs. the fixed
  threshold) with states, actions and rewards, plus a figure.

Usage (from workshop_3/):
    python src/evaluate.py                # test
    python src/evaluate.py --split val    # same report on validation
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import DQNAgent
from baselines import BASELINES
from env import DEFAULT_REWARDS, OBS_NAMES, TAU_0, W3_DIR, ECGThresholdEnv, load_beats
from rollout import BEATS_PER_HOUR, evaluate_policy

REPORTS = W3_DIR / "reports"
W2_METRICS = W3_DIR.parent / "workshop_2" / "reports" / "test_metrics.csv"
COLUMNS = ["mean_return", "precision", "recall", "f1", "false_alarms_per_hour",
           "alarm_rate_S", "alarm_rate_V", "mean_threshold", "action_keep"]


def oracle_per_record(beats):
    """Best fixed threshold of each record, chosen WITH the labels (upper bound)."""
    r = DEFAULT_REWARDS
    tp = fp = fn = 0
    total_reward, grid = 0.0, np.round(np.arange(0.05, 0.951, 0.01), 2)
    for _, b in beats.groupby("record_id"):
        p, y = b.p.to_numpy(), b.label.to_numpy()
        best = None
        for t in grid:
            a = p >= t
            counts = (np.sum(a & (y == 1)), np.sum(a & (y == 0)), np.sum(~a & (y == 1)))
            reward = r["tp"] * counts[0] + r["fp"] * counts[1] + r["fn"] * counts[2]
            if best is None or reward > best[0]:
                best = (reward, counts)
        total_reward += best[0]
        tp, fp, fn = tp + best[1][0], fp + best[1][1], fn + best[1][2]
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return {"total_return": total_reward, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
            "false_alarms_per_hour": fp / len(beats) * BEATS_PER_HOUR}


def plot_trajectories(trajs, keep_returns, path):
    fig, axes = plt.subplots(len(trajs), 1, figsize=(12, 2.8 * len(trajs)), squeeze=False)
    for ax, (k, t) in zip(axes[:, 0], trajs):
        s = pd.DataFrame(t["steps"])
        x = np.arange(len(s))
        normal, abnormal = s.label == 0, s.label == 1
        ax.scatter(x[normal], s.p[normal], s=8, color="gray", label="normal beat (p)")
        ax.scatter(x[abnormal], s.p[abnormal], s=18, color="C3", label="abnormal beat (p)")
        ax.step(x, s.tau, where="post", color="C0", lw=1.8, label="agent threshold")
        ax.axhline(TAU_0, color="black", ls="--", lw=1, label="fixed threshold 0.784")
        fp, fn = s.outcome == "fp", s.outcome == "fn"
        ax.scatter(x[fp], s.p[fp], marker="x", s=40, color="C1", label="false alarm")
        ax.scatter(x[fn], s.p[fn], marker="v", s=40, color="purple", label="missed")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("probability")
        ax.set_title(f"Record {t['record']}, beats {t['start']}-{t['start'] + len(s)}: "
                     f"agent return {t['return']:+.1f} vs fixed {keep_returns[k]:+.1f}",
                     fontsize=10)
    axes[0, 0].legend(fontsize=7, ncol=4, loc="lower left")
    axes[-1, 0].set_xlabel("beat within the episode")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(W3_DIR / "checkpoints" / "agent_best.pt"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    REPORTS.mkdir(exist_ok=True)
    tag = args.split

    beats = load_beats(args.split)
    env = ECGThresholdEnv(beats, mode="sequential")
    agent, ckpt = DQNAgent.load(args.checkpoint)
    print(f"Agent: {ckpt['run_name']} (episode {ckpt['episode']}, "
          f"validation return {ckpt['val_return']:.3f})")
    print(f"{tag.upper()}: {env.n_segments} episodes, {len(beats):,} beats\n")

    results, trajectories = {}, {}
    for name, make in BASELINES.items():
        policy = make(args.seed) if name == "Random policy" else make()
        results[name], trajectories[name] = evaluate_policy(env, policy, collect_trajectories=True)
    results["DQN agent"], trajectories["DQN agent"] = evaluate_policy(
        env, lambda o: agent.act(o), collect_trajectories=True)

    table = pd.DataFrame(results).T.reindex(columns=COLUMNS).astype(float)
    oracle = oracle_per_record(beats)
    table.loc["Oracle per patient (upper bound)", ["mean_return", "precision", "recall", "f1",
                                                  "false_alarms_per_hour"]] = [
        oracle["total_return"] / env.n_segments, oracle["precision"], oracle["recall"],
        oracle["f1"], oracle["false_alarms_per_hour"]]
    if tag == "test" and W2_METRICS.exists():
        w2 = pd.read_csv(W2_METRICS, index_col=0)
        for model, label in [("Random Forest (W1)", "Random Forest (Workshop 1)")]:
            if model in w2.index:
                table.loc[label, ["precision", "recall", "f1"]] = \
                    w2.loc[model, ["Precision", "Recall", "F1-score"]].to_numpy()
    print(table.astype(float).round(4).to_string())
    table.to_csv(REPORTS / f"{tag}_policy_comparison.csv")

    # Per record: agent vs fixed threshold
    rows = []
    agent_eps, keep_eps = trajectories["DQN agent"], trajectories["Fixed threshold 0.784 (Workshop 2)"]
    for ta, tk in zip(agent_eps, keep_eps):
        sa, sk = pd.DataFrame(ta["steps"]), pd.DataFrame(tk["steps"])
        rows.append({"record": ta["record"], "agent_return": ta["return"],
                     "fixed_return": tk["return"],
                     "agent_fp": (sa.outcome == "fp").sum(), "fixed_fp": (sk.outcome == "fp").sum(),
                     "agent_fn": (sa.outcome == "fn").sum(), "fixed_fn": (sk.outcome == "fn").sum(),
                     "agent_mean_tau": sa.tau.mean()})
    per_rec = pd.DataFrame(rows).groupby("record").agg(
        agent_return=("agent_return", "sum"), fixed_return=("fixed_return", "sum"),
        agent_fp=("agent_fp", "sum"), fixed_fp=("fixed_fp", "sum"),
        agent_fn=("agent_fn", "sum"), fixed_fn=("fixed_fn", "sum"),
        agent_mean_tau=("agent_mean_tau", "mean")).round(2)
    print("\nPer record (agent vs fixed threshold):")
    print(per_rec.to_string())
    per_rec.to_csv(REPORTS / f"{tag}_per_record.csv")

    # Trajectories: 2 largest gains, 1 largest loss, 1 typical episode
    gains = np.array([a["return"] - k["return"] for a, k in zip(agent_eps, keep_eps)])
    order = np.argsort(gains)
    chosen = list(dict.fromkeys([order[-1], order[-2], order[0], order[len(order) // 2]]))
    chosen_trajs = [(k, agent_eps[k]) for k in chosen]
    plot_trajectories(chosen_trajs, [k["return"] for k in keep_eps],
                      REPORTS / f"{tag}_trajectories.png")
    export = []
    for k, t in chosen_trajs:
        steps = pd.DataFrame(t["steps"])
        steps.insert(0, "episode_id", int(k))
        steps.insert(1, "record", t["record"])
        export.append(steps)
    pd.concat(export).to_csv(REPORTS / f"{tag}_trajectories.csv", index=False)
    print(f"\nTrajectories (episodes {', '.join(map(str, chosen))}): "
          f"{REPORTS / f'{tag}_trajectories.png'}")

    # Example of the Q-values the agent sees at the start of the best episode
    env.reset(options={"segment": int(order[-1])})
    obs = env._obs()
    q = agent.q_values(obs)
    print("\nQ-values at the start of the episode with the largest gain:")
    print({n: round(float(v), 3) for n, v in zip(OBS_NAMES, obs)})
    print({a: round(float(v), 3) for a, v in zip(
        ["lower 0.20", "lower 0.05", "keep", "raise 0.05", "raise 0.20"], q)})

    with open(REPORTS / f"{tag}_summary.json", "w") as f:
        json.dump({"agent_run": ckpt["run_name"], "agent_episode": ckpt["episode"],
                   "results": {k: {m: float(v) for m, v in r.items()} for k, r in results.items()},
                   "oracle_per_patient": {k: float(v) for k, v in oracle.items()}}, f, indent=4)
    print(f"Saved reports to {REPORTS}")


if __name__ == "__main__":
    main()
