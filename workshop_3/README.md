# Team 6 — ECG Anomaly Detection

## Workshop 3: Reinforcement Learning for Adaptive Alarm Thresholds

Report: [group6_workshop3.pdf](group6_workshop3.pdf) · MDP specification: [docs/mdp_spec.md](docs/mdp_spec.md)

### 1. Overview

The Workshop 2 ResNet1D gives, for each heartbeat, the probability `p` that it is
abnormal, and raises an alarm when `p >= 0.784` (a fixed threshold). A single fixed
threshold behaves very differently across patients (one noisy record produced 173 of
the 200 test false alarms).

Here a **Deep Q-Network** monitors a patient beat by beat and **adapts the alarm
threshold online**:

| MDP | Design |
|---|---|
| State (8) | threshold, ResNet1D probability, RR-interval ratios (rhythm), mean/std of recent probabilities, recent alarm rate, progress |
| Actions (5) | lower 0.20, lower 0.05, keep, raise 0.05, raise 0.20 |
| Reward | TP +1, missed beat −1, false alarm −1, TN 0, −0.01 per threshold change |
| Episode | 200 consecutive beats (train); validation/test episodes cover every beat once |

"Always keep" reproduces the Workshop 2 classifier exactly, so the main baseline is
inside the agent's action space.

### 2. Results on the test set (6 unseen patients, 13,869 beats)

| Policy | Return / episode | Precision | Recall | F1 | False alarms / h |
|---|---:|---:|---:|---:|---:|
| Random policy | −38.28 | 0.215 | 0.826 | 0.342 | 1,112 |
| Random Forest (Workshop 1) | — | 0.707 | 0.692 | 0.700 | — |
| Fixed threshold 0.784 (Workshop 2) | 5.63 | 0.814 | 0.768 | 0.790 | 64.9 |
| Rule-based heuristic | 5.71 | 0.827 | 0.768 | 0.797 | 59.4 |
| **DQN agent** | **7.60** | **0.948** | **0.798** | **0.867** | **16.2** |
| Oracle per patient (upper bound, uses labels) | 8.48 | 0.936 | 0.799 | 0.862 | 20.1 |

Most of the gain comes from the noisy record 108 (false alarms 173 → 24); on the
validation patients the agent is roughly on par with the fixed threshold.

### 3. Key finding: the training environment must match evaluation

The ResNet1D is over-confident on the patients it was trained on, so an agent trained
with those probabilities learned to lower the threshold and failed on new patients.
Fixes: **out-of-fold probabilities** for the training patients (`src/oof_probs.py`) and
**domain randomization** (each training episode uses in-sample or out-of-fold
probabilities at random).

| Training environment | Best validation return | Fixed threshold |
|---|---:|---:|
| Workshop 2 probabilities (`dqn_base`) | 7.14 | 9.25 |
| Out-of-fold probabilities (`dqn_oof`) | 8.59 | 9.25 |
| Random mix (`dqn_mix`) | 9.95 | 9.25 |

### 4. Hyperparameter search (validation)

| Config | lr | γ | ε decay | weight decay | Best val. return | Smoothed max |
|---|---|---|---|---|---:|---:|
| A `dqn_mix` | 1e-3 | 0.95 | 0.995 | 0 | 9.950 | 8.508 |
| **B `dqn_lr1e-4` (final)** | 1e-4 | 0.95 | 0.995 | 0 | 9.550 | **8.834** |
| C `dqn_lr1e-4_g099` | 1e-4 | 0.99 | 0.995 | 0 | 9.434 | 8.612 |
| D `dqn_lr1e-4_eps099_wd` | 1e-4 | 0.95 | 0.990 | 1e-4 | 9.262 | 8.703 |

The final agent is chosen by the best smoothed validation return.

### 5. Project structure

```text
workshop_3/
├── docs/mdp_spec.md        # MDP specification (written before the code)
├── src/
│   ├── build_features.py   # per-beat ResNet1D probability + RR intervals -> data/beats.csv.gz
│   ├── oof_probs.py        # out-of-fold probabilities for the training patients
│   ├── diagnose_shift.py   # calibration per split, best fixed and per-patient thresholds
│   ├── env.py              # gymnasium environment ECGThresholdEnv
│   ├── check_env.py        # API check + 100 random-policy episodes
│   ├── agent.py            # DQN (replay buffer, target network, Double-DQN target)
│   ├── rollout.py          # runs a policy over a split and computes metrics
│   ├── train.py            # training loop, validation, TensorBoard logging
│   ├── baselines.py        # random, fixed threshold, rule-based heuristic
│   ├── compare_runs.py     # hyperparameter table, sample efficiency, picks agent_best.pt
│   └── evaluate.py         # test evaluation, per-record table, trajectories
├── data/                   # beats.csv.gz and out-of-fold folds
├── checkpoints/            # agent_best.pt + best checkpoint of every run
├── runs/workshop3/         # TensorBoard logs, episodes.csv, validation.csv, curves
└── reports/                # diagnostics, comparison tables and figures
```

### 6. How to reproduce

Requires the Workshop 1 data (`workshop_1/data/processed`, `workshop_1/data/raw/mitdb`)
and the Workshop 2 checkpoint. From `workshop_3` (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python src\build_features.py      # ~2 min
python src\oof_probs.py           # ~30 min on CPU (resumable)
python src\diagnose_shift.py
python src\check_env.py

python src\train.py --run-name dqn_base --train-probs p_w2
python src\train.py --run-name dqn_oof --train-probs p
python src\train.py --run-name dqn_mix --train-probs mix
python src\train.py --run-name dqn_lr1e-4 --lr 1e-4
python src\train.py --run-name dqn_lr1e-4_g099 --lr 1e-4 --gamma 0.99
python src\train.py --run-name dqn_lr1e-4_eps099_wd --lr 1e-4 --eps-decay 0.99 --weight-decay 1e-4

python src\compare_runs.py        # selects checkpoints\agent_best.pt
python src\evaluate.py --split val
python src\evaluate.py            # test set
tensorboard --logdir runs\workshop3
```

Each training run takes about 10–12 minutes on CPU. Seeds are fixed (42), and
re-running a configuration reproduces its validation values exactly.

This is an academic project and is not intended for clinical use.
