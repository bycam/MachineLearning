"""
Validate the environment before training (required by the workshop guide).

1. Gymnasium's API checker (spaces, reset/step signatures, dtypes, seeding).
2. 100 random-policy episodes on the training environment:
   mean return, episode-length distribution and action frequencies.
3. Reference: the same 100 episodes with "always keep" (= Workshop 2
   classifier with a fixed 0.784 threshold). A sensible reward must give it
   a much higher return than the random policy.
4. Episode lengths of the sequential validation environment.

Outputs: reports/random_policy_diagnostics.json and .png

Usage (from workshop_3/):  python src/check_env.py
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.env_checker import check_env

from env import ACTION_NAMES, OBS_NAMES, W3_DIR, ECGThresholdEnv, load_beats

REPORTS = W3_DIR / "reports"
N_EPISODES = 100
SEED = 42


def run_episodes(env, policy, n, seed):
    returns, lengths, actions, outcomes = [], [], Counter(), Counter()
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            a = policy(obs)
            obs, r, done, _, info = env.step(a)
            total += r
            steps += 1
            actions[a] += 1
            outcomes[info["outcome"]] += 1
        returns.append(total)
        lengths.append(steps)
    return np.array(returns), np.array(lengths), actions, outcomes


def main():
    REPORTS.mkdir(exist_ok=True)
    train_env = ECGThresholdEnv(load_beats("train"), mode="random")

    print("1) Gymnasium API check...")
    check_env(ECGThresholdEnv(load_beats("train"), mode="random"), skip_render_check=True)
    print("   OK\n")

    print("Example of the first 5 steps (render):")
    obs, info = train_env.reset(seed=0)
    print(" ", train_env.render())
    for _ in range(5):
        train_env.step(train_env.action_space.sample())
        print(" ", train_env.render())
    print("  first observation:", {k: round(float(v), 3) for k, v in zip(OBS_NAMES, obs)}, "\n")

    train_env.action_space.seed(SEED)
    rand = run_episodes(train_env, lambda o: train_env.action_space.sample(), N_EPISODES, SEED)
    keep = run_episodes(train_env, lambda o: 2, N_EPISODES, SEED)

    print(f"2) {N_EPISODES} random-policy episodes (train environment)")
    print(f"   mean return: {rand[0].mean():.2f} +/- {rand[0].std():.2f} "
          f"(min {rand[0].min():.1f}, max {rand[0].max():.1f})")
    print(f"   episode length: min {rand[1].min()}, mean {rand[1].mean():.1f}, max {rand[1].max()}")
    total = sum(rand[2].values())
    print("   action frequency:")
    for a, name in enumerate(ACTION_NAMES):
        print(f"     {a} {name:<11} {rand[2][a]:>6}  ({100 * rand[2][a] / total:.1f}%)")
    print(f"   outcomes: {dict(rand[3])}\n")

    print("3) Same episodes with 'always keep' (Workshop 2 fixed threshold 0.784)")
    print(f"   mean return: {keep[0].mean():.2f} +/- {keep[0].std():.2f}")
    print(f"   outcomes: {dict(keep[3])}\n")

    val_env = ECGThresholdEnv(load_beats("val"), mode="sequential")
    val_lengths = [int(end - start) for _, start, end in val_env.segments]
    print(f"4) Validation environment: {val_env.n_segments} episodes cover "
          f"{sum(val_lengths):,} beats (lengths {min(val_lengths)}-{max(val_lengths)})")

    diagnostics = {
        "n_episodes": N_EPISODES,
        "random_mean_return": float(rand[0].mean()),
        "random_std_return": float(rand[0].std()),
        "random_episode_length_mean": float(rand[1].mean()),
        "random_action_frequency": {ACTION_NAMES[a]: int(c) for a, c in sorted(rand[2].items())},
        "random_outcomes": {k: int(v) for k, v in rand[3].items()},
        "keep_mean_return": float(keep[0].mean()),
        "keep_std_return": float(keep[0].std()),
        "val_episodes": val_env.n_segments,
        "val_episode_lengths": {"min": min(val_lengths), "max": max(val_lengths)},
    }
    with open(REPORTS / "random_policy_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(rand[0], bins=20, alpha=0.7, label="random")
    axes[0].hist(keep[0], bins=20, alpha=0.7, label="always keep (W2)")
    axes[0].set_title("Episode return (100 train episodes)")
    axes[0].legend()
    axes[1].hist(val_lengths, bins=20, color="gray")
    axes[1].set_title("Episode length (validation env)")
    axes[2].bar(range(5), [rand[2][a] for a in range(5)])
    axes[2].set_xticks(range(5), ACTION_NAMES, rotation=20)
    axes[2].set_title("Action frequency (random policy)")
    fig.tight_layout()
    fig.savefig(REPORTS / "random_policy_diagnostics.png", dpi=150)
    print(f"\nSaved diagnostics to {REPORTS}")


if __name__ == "__main__":
    main()
