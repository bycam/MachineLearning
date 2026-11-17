"""
Baseline policies for the adaptive-threshold environment.

All of them map an observation to one of the 5 actions, so they are scored
with exactly the same environment and code as the DQN agent:

  - random     : uniform random action (lower bound)
  - fixed      : always "keep" -> fixed threshold 0.784 = Workshop 2 classifier
  - heuristic  : hand-written rule using the same state as the agent
                 (premature beat -> lower the threshold; many recent alarms
                 or noisy probabilities -> raise it)

Observation layout (see env.OBS_NAMES):
  0 tau | 1 p | 2 rr_pre_ratio | 3 rr_post_ratio | 4 p_mean_10 | 5 p_std_10 |
  6 alarm_rate_20 | 7 progress
"""
import numpy as np

from env import TAU_0

KEEP = 2


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(5))


def fixed_threshold_policy():
    return lambda obs: KEEP


def action_towards(tau, target):
    """Action that moves the threshold closest to `target`."""
    diff = target - tau
    if diff <= -0.125:
        return 0            # lower 0.20
    if diff <= -0.025:
        return 1            # lower 0.05
    if diff >= 0.125:
        return 4            # raise 0.20
    if diff >= 0.025:
        return 3            # raise 0.05
    return KEEP


def heuristic_policy(premature=0.85, pause=1.10, low_tau=0.50,
                     busy_alarm_rate=0.30, noisy_std=0.30, high_tau=0.90):
    """
    Clinical rule of thumb, written by hand (no learning):
      - premature beat followed by a pause (classic ectopic beat): alarm more easily
      - too many recent alarms or very unstable probabilities (noise): be stricter
      - otherwise go back to the Workshop 2 threshold
    """
    def policy(obs):
        tau, rr_pre, rr_post = obs[0], obs[2], obs[3]
        p_std, alarm_rate = obs[5], obs[6]
        if alarm_rate > busy_alarm_rate or p_std > noisy_std:
            target = high_tau
        elif rr_pre < premature and rr_post > pause:
            target = low_tau
        else:
            target = TAU_0
        return action_towards(tau, target)
    return policy


BASELINES = {
    "Random policy": random_policy,
    "Fixed threshold 0.784 (Workshop 2)": fixed_threshold_policy,
    "Rule-based heuristic": heuristic_policy,
}
