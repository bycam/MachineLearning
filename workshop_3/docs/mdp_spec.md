# MDP Specification — Adaptive Alarm Threshold for ECG Monitoring

Team 6 — Workshop 3 (Reinforcement Learning)

## 1. Problem statement

In Workshop 2, a 1D ResNet outputs, for every heartbeat, the probability
`p` that the beat is abnormal. A beat raises an alarm when `p >= tau`, with a
**fixed** threshold `tau = 0.784` tuned on the validation set. A single fixed
threshold behaves very differently from patient to patient:

- Record 108 (noisy signal) produced 173 of the 200 test false alarms.
- Records 201/202 and 124 (many supraventricular beats) produced most missed
  abnormal beats, because their probabilities fall just below 0.784.

**Task of the agent:** monitor one patient beat by beat and **adapt the alarm
threshold** online, raising it when the signal produces spurious alarms and
lowering it when abnormal activity is likely, so that more abnormal beats are
detected with fewer false alarms than with the fixed threshold.

**Why RL instead of supervised learning.** The threshold is a *persistent
control variable*: the value chosen now stays in force for the next beats,
so each decision changes the situation the agent faces later. There is no
label telling the correct threshold at each moment; only the consequences of
the threshold (correct alarms, false alarms, missed beats) can be observed
and scored. RL learns a policy from these consequences. RL does not replace
the supervised model: it is a decision layer on top of it, reusing the
Workshop 2 probabilities as part of the state.

## 2. Data and environments

Each beat `t` of a record provides precomputed features:

- `p_t`: abnormal probability from the Workshop 2 ResNet1D (`model_best.pt`)
- `RR_pre_t`, `RR_post_t`: time from the previous R-peak and to the next one
- `RR_avg_t`: mean of the previous 10 RR intervals (the patient's local rhythm)
- `y_t`: true label (0 Normal, 1 Abnormal), used **only** to compute rewards

Episodes are built from the Workshop 1 patient splits (`splits.json`):

| Environment | Patients | Episodes |
|---|---|---|
| Train | 34 train records | random segments of 200 consecutive beats |
| Validation | 7 validation records | all beats, consecutive segments of 200 (fixed) |
| Test | 7 test records | all beats, consecutive segments of 200 (fixed) |

Validation and test episodes cover every beat exactly once, so their metrics
are computed on the same 13,606 / 13,869 beats as in Workshops 1 and 2.

## 3. State space (dimension 8, continuous)

| # | Feature | Range | Meaning |
|---|---|---|---|
| 1 | `tau_t` | [0.05, 0.95] | current alarm threshold |
| 2 | `p_t` | [0, 1] | ResNet1D probability of the current beat |
| 3 | `RR_pre_t / RR_avg_t` | [0, 3] | < 1: beat arrives early (premature) |
| 4 | `RR_post_t / RR_avg_t` | [0, 3] | > 1: compensatory pause after the beat |
| 5 | mean of `p` over the last 10 beats | [0, 1] | recent level of abnormality |
| 6 | std of `p` over the last 10 beats | [0, 0.5] | recent instability / noise |
| 7 | alarm rate over the last 20 beats | [0, 1] | consequence of the agent's own past thresholds |
| 8 | `t / L` | [0, 1] | progress within the episode |

Features 3–4 add the rhythm information that the single-beat ResNet1D does
not see (the main cause of missed supraventricular beats in Workshop 2).
Feature 7 depends on past actions, which makes the problem sequential.
`RR_post_t` requires the next R-peak, i.e. the decision for a beat is taken
about one beat (~0.8 s) later, which is acceptable for monitoring.

## 4. Action set (5 discrete actions)

| Action | Effect on the threshold |
|---|---|
| 0 | `tau -= 0.20` (strongly lower: alarm more easily) |
| 1 | `tau -= 0.05` (slightly lower) |
| 2 | keep `tau` |
| 3 | `tau += 0.05` (slightly raise) |
| 4 | `tau += 0.20` (strongly raise: alarm less easily) |

The threshold is clipped to [0.05, 0.95]. Each episode starts at
`tau_0 = 0.784`, the Workshop 2 threshold, so "always keep" reproduces the
Workshop 2 classifier exactly.

## 5. Transition dynamics

At step `t`, the agent observes `s_t` (which already contains the current
beat's `p_t` and RR features) and chooses `a_t`:

1. `tau_{t+1} = clip(tau_t + delta(a_t), 0.05, 0.95)` — **deterministic**
2. The current beat is classified with the new threshold:
   `alarm_t = 1 if p_t >= tau_{t+1} else 0`
3. A reward is computed from `alarm_t` and `y_t`.
4. The environment moves to the next beat of the recording and builds
   `s_{t+1}`.

The threshold and the alarm history are deterministic functions of the
actions. The sequence of beats is **exogenous**: the agent cannot influence
which beat comes next. Stochasticity comes from which patient segment is
sampled at reset (training) and is unknown to the agent in advance.

## 6. Reward function (dense, every step)

| Outcome | Reward |
|---|---|
| True positive (abnormal beat, alarm) | +1.0 |
| False negative (abnormal beat missed) | −1.0 |
| False positive (normal beat, alarm) | −1.0 |
| True negative (normal beat, no alarm) | 0.0 |
| Any threshold change (actions 0, 1, 3, 4) | −0.01 extra |

- Rewarding true negatives with 0 avoids the trivial policy "never alarm",
  which would otherwise collect +reward on the ~90% normal beats.
- With these values, alarming is worth it when P(abnormal) > 1/3, a
  balance between missed beats and false alarms.
- The small cost on changes discourages the threshold from oscillating.
- The reward is dense, so there is no sparse-reward problem. The relevant
  horizon is long-term: a threshold set now affects the next beats, which is
  captured by the discount factor gamma (searched in {0.90, 0.95, 0.99}).

## 7. Episode termination

An episode ends (`terminated = True`) after the last beat of the segment:
200 beats in training (about 3 minutes of ECG), or the end of the segment
or record in validation/test. There is no failure state: the task is
continuous monitoring, and success is measured by the cumulative reward and
by detection metrics (precision, recall, F1, false alarms per hour).

## 8. Baselines and evaluation

On the same validation/test episodes:

1. Random policy
2. Fixed threshold 0.784 (Workshop 2 classifier, "always keep")
3. Best fixed threshold for the reward, chosen on validation
4. Rule-based heuristic: lower the threshold after premature beats
   (`RR_pre/RR_avg < 0.85`) and raise it when the recent alarm rate is high

Metrics: mean episode return, precision, recall, F1, false alarms per hour
and detection rate per AAMI class, compared with the Workshop 1 Random
Forest and the Workshop 2 ResNet1D on the same test beats. (PR-AUC is not
reported for the agent: a policy produces decisions, not a ranking score.)
