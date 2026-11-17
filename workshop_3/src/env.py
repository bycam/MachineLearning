"""
Gymnasium environment: adaptive alarm threshold for ECG beat monitoring.

The agent watches one patient's recording beat by beat. At every beat it
sees the Workshop 2 ResNet1D probability, the rhythm (RR intervals) and its
own recent behaviour, and moves the alarm threshold. The current beat raises
an alarm if p >= new threshold, and the outcome is rewarded.

See docs/mdp_spec.md for the full MDP specification.
"""
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

W3_DIR = Path(__file__).resolve().parents[1]
BEATS_PATH = W3_DIR / "data" / "beats.csv.gz"

ACTION_DELTAS = np.array([-0.20, -0.05, 0.0, +0.05, +0.20], dtype=np.float32)
ACTION_NAMES = ["lower 0.20", "lower 0.05", "keep", "raise 0.05", "raise 0.20"]
TAU_MIN, TAU_MAX = 0.05, 0.95
TAU_0 = 0.784                       # Workshop 2 threshold (tuned on validation)

DEFAULT_REWARDS = {"tp": 1.0, "fn": -1.0, "fp": -1.0, "tn": 0.0, "change": -0.01}

OBS_NAMES = ["tau", "p", "rr_pre_ratio", "rr_post_ratio",
             "p_mean_10", "p_std_10", "alarm_rate_20", "progress"]


def load_beats(split=None):
    """Per-beat table built by src/build_features.py, optionally filtered by split."""
    beats = pd.read_csv(BEATS_PATH, dtype={"record_id": str, "subject_group": str})
    if split is not None:
        beats = beats[beats["split"] == split]
    return beats.reset_index(drop=True)


def _past_window_stats(p, window=10):
    """Mean and std of p over the `window` beats BEFORE each beat (current excluded)."""
    n = len(p)
    mean, std = np.empty(n, np.float32), np.empty(n, np.float32)
    csum = np.concatenate([[0.0], np.cumsum(p)])
    csum2 = np.concatenate([[0.0], np.cumsum(p ** 2)])
    for i in range(n):
        lo, hi = max(0, i - window), i
        if hi == lo:                       # first beat of the record: no history
            mean[i], std[i] = p[i], 0.0
            continue
        k = hi - lo
        m = (csum[hi] - csum[lo]) / k
        var = (csum2[hi] - csum2[lo]) / k - m ** 2
        mean[i], std[i] = m, np.sqrt(max(var, 0.0))
    return mean, std


class ECGThresholdEnv(gym.Env):
    """
    mode="random":     each reset() samples a random segment of `episode_length`
                       consecutive beats from a random record (training).
    mode="sequential": the records are cut into consecutive segments that cover
                       every beat exactly once; each reset() returns the next one
                       (validation / test). `n_segments` episodes = whole split.

    prob_source (which ResNet1D probabilities the agent sees):
      "p"    : column p (validation/test: Workshop 2 model; train: out-of-fold)
      "p_w2" : Workshop 2 model on the train patients (in-sample, over-confident)
      "mix"  : each episode randomly uses "p" or "p_w2" (domain randomization)
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, beats, mode="random", episode_length=200,
                 rewards=None, render_mode=None, prob_source="p"):
        super().__init__()
        assert mode in ("random", "sequential")
        self.mode = mode
        self.L = episode_length
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.render_mode = render_mode
        self.sources = ["p", "p_w2"] if prob_source == "mix" else [prob_source]
        for src in self.sources:
            assert src in beats.columns, f"column {src} not found (run oof_probs.py?)"

        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        low = np.zeros(8, dtype=np.float32)
        high = np.array([1, 1, 3, 3, 1, 0.5, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Per-record arrays (beats are already sorted by record and time)
        self.records = {}
        for rec, g in beats.groupby("record_id", sort=True):
            g = g.sort_values("sample")
            probs = {}
            for src in self.sources:
                p = g[src].to_numpy(np.float32)
                p_mean, p_std = _past_window_stats(p)
                probs[src] = {"p": p, "p_mean": p_mean, "p_std": np.clip(p_std, 0, 0.5)}
            self.records[rec] = {
                "probs": probs,
                "rr_pre": np.clip(g["rr_pre"] / g["rr_avg"], 0, 3).to_numpy(np.float32),
                "rr_post": np.clip(g["rr_post"] / g["rr_avg"], 0, 3).to_numpy(np.float32),
                "label": g["label"].to_numpy(np.int64),
                "aami": g["aami_class"].to_numpy(),
                "sample": g["sample"].to_numpy(np.int64),
            }

        self.record_ids = list(self.records)
        lengths = np.array([len(self.records[r]["label"]) for r in self.record_ids])
        # Random mode: pick records proportionally to the number of valid start positions
        starts = np.maximum(lengths - self.L + 1, 1)
        self._record_probs = starts / starts.sum()
        # Sequential mode: fixed list of segments covering all beats
        self.segments = [(rec, s, min(s + self.L, n))
                         for rec, n in zip(self.record_ids, lengths)
                         for s in range(0, n, self.L)]
        self._next_segment = 0

    @property
    def n_segments(self):
        return len(self.segments)

    # ------------------------------------------------------------------ #
    def _obs(self):
        r, i = self.rec, min(self.i, self.end - 1)
        alarm_rate = np.mean(self.alarm_hist) if self.alarm_hist else 0.0
        return np.array([
            self.tau,
            r["p"][i],
            r["rr_pre"][i],
            r["rr_post"][i],
            r["p_mean"][i],
            r["p_std"][i],
            alarm_rate,
            (i - self.start) / self.L,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "segment" in options:            # evaluate one given segment
            rec_id, start, end = self.segments[options["segment"]]
        elif self.mode == "sequential":
            rec_id, start, end = self.segments[self._next_segment]
            self._next_segment = (self._next_segment + 1) % self.n_segments
        else:
            k = self.np_random.choice(len(self.record_ids), p=self._record_probs)
            rec_id = self.record_ids[k]
            n = len(self.records[rec_id]["label"])
            start = int(self.np_random.integers(0, max(n - self.L, 0) + 1))
            end = min(start + self.L, n)

        rec = self.records[rec_id]
        src = (self.sources[int(self.np_random.integers(len(self.sources)))]
               if len(self.sources) > 1 else self.sources[0])
        self.rec_id, self.source = rec_id, src
        self.rec = {**rec, **rec["probs"][src]}          # p, p_mean, p_std of this source
        self.start, self.end, self.i = start, end, start
        self.tau = TAU_0
        self.alarm_hist = deque(maxlen=20)
        self.counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.last = None
        return self._obs(), {"record": rec_id, "start": start, "length": end - start,
                             "prob_source": src}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action}"
        r, i = self.rec, self.i

        self.tau = float(np.clip(self.tau + ACTION_DELTAS[action], TAU_MIN, TAU_MAX))
        alarm = int(r["p"][i] >= self.tau)
        label = int(r["label"][i])
        outcome = ("tp" if alarm else "fn") if label else ("fp" if alarm else "tn")

        reward = self.rewards[outcome]
        if ACTION_DELTAS[action] != 0:
            reward += self.rewards["change"]

        self.counts[outcome] += 1
        self.alarm_hist.append(alarm)
        self.last = {"beat": i, "sample": int(r["sample"][i]), "p": float(r["p"][i]),
                     "aami": r["aami"][i], "label": label, "action": int(action),
                     "tau": self.tau, "alarm": alarm, "outcome": outcome,
                     "reward": float(reward)}

        self.i += 1
        terminated = self.i >= self.end
        info = dict(self.last)
        if terminated:
            info["episode_counts"] = dict(self.counts)
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        if self.last is None:
            return f"[record {self.rec_id}] start of episode, tau={self.tau:.2f}"
        s = self.last
        return (f"[record {self.rec_id} beat {s['beat'] - self.start:3d}] "
                f"class={s['aami']} p={s['p']:.2f} action={ACTION_NAMES[s['action']]:<10} "
                f"tau={s['tau']:.2f} alarm={s['alarm']} -> {s['outcome'].upper()} "
                f"reward={s['reward']:+.2f}")
