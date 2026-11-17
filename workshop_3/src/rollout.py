"""
Run a policy over every episode of a sequential environment and compute metrics.

Used by train.py (periodic validation) and evaluate.py (test set), so the RL
agent and all baselines are scored with exactly the same code.
"""
from collections import Counter, defaultdict

import numpy as np

from env import ACTION_NAMES

BEATS_PER_HOUR = 4500        # ~75 beats per minute, to express false alarms per hour


def evaluate_policy(env, policy, collect_trajectories=False):
    """
    policy(obs) -> action. Each of the env.n_segments episodes is played once,
    so every beat of the split is classified exactly once.
    """
    returns, outcomes, actions = [], Counter(), Counter()
    per_class = defaultdict(lambda: [0, 0])          # class -> [beats, alarms]
    taus, trajectories = [], []

    for seg in range(env.n_segments):
        obs, info = env.reset(options={"segment": seg})
        done, ret, steps = False, 0.0, []
        while not done:
            action = policy(obs)
            obs, reward, done, _, step = env.step(action)
            ret += reward
            outcomes[step["outcome"]] += 1
            actions[step["action"]] += 1
            per_class[step["aami"]][0] += 1
            per_class[step["aami"]][1] += step["alarm"]
            taus.append(step["tau"])
            if collect_trajectories:
                steps.append(step)
        returns.append(ret)
        if collect_trajectories:
            trajectories.append({"record": info["record"], "start": info["start"],
                                 "return": ret, "steps": steps})

    tp, fp, fn = outcomes["tp"], outcomes["fp"], outcomes["fn"]
    n_beats = sum(outcomes.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_actions = sum(actions.values())

    metrics = {
        "mean_return": float(np.mean(returns)),
        "total_return": float(np.sum(returns)),
        "episodes": len(returns),
        "beats": n_beats,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": outcomes["tn"],
        "false_alarms_per_hour": fp / n_beats * BEATS_PER_HOUR,
        "mean_threshold": float(np.mean(taus)),
        "std_threshold": float(np.std(taus)),
    }
    for cls in ["N", "S", "V", "F"]:
        if per_class[cls][0]:
            metrics[f"alarm_rate_{cls}"] = per_class[cls][1] / per_class[cls][0]
    for a, name in enumerate(ACTION_NAMES):
        metrics[f"action_{name.replace(' ', '_')}"] = actions[a] / total_actions
    return metrics, trajectories
