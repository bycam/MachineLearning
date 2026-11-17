"""
Train a DQN agent on the adaptive-threshold environment.

- Training episodes: random 200-beat segments of the TRAIN patients
- Every `--val-every` episodes: greedy policy (no exploration) on ALL the
  validation beats; the checkpoint with the highest validation return is kept
- Logs: TensorBoard + CSV in runs/workshop3/<run_name>/
- Checkpoints: checkpoints/<run_name>_best.pt, and checkpoints/agent_best.pt
  is updated whenever a run beats the best validation return stored there

Usage (from workshop_3/):
    python src/train.py --smoke                       # quick pipeline test
    python src/train.py --run-name dqn_base           # one configuration
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
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent, ReplayBuffer
from env import W3_DIR, ECGThresholdEnv, load_beats
from rollout import evaluate_policy

RUNS_DIR = W3_DIR / "runs" / "workshop3"
CKPT_DIR = W3_DIR / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="dqn_base")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-min", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.995, help="per episode")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=2_000, help="steps before learning")
    p.add_argument("--train-freq", type=int, default=2, help="env steps per update")
    p.add_argument("--target-update", type=int, default=1_000, help="steps")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--no-double", action="store_true", help="plain DQN target")
    p.add_argument("--val-every", type=int, default=25, help="episodes")
    p.add_argument("--train-probs", choices=["p", "p_w2", "mix"], default="mix",
                   help="ResNet1D probabilities in the TRAIN environment: out-of-fold (p), "
                        "Workshop 2 in-sample (p_w2) or a random mix per episode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_run(hist, val, path, keep_return):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    ax.plot(hist["episode"], hist["return"], alpha=0.25, color="gray", label="episode")
    ax.plot(hist["episode"], hist["return"].rolling(50, min_periods=1).mean(),
            color="C0", label="moving avg (50)")
    ax.set_title("Training return (with exploration)")
    ax.set_xlabel("episode")
    ax.legend()
    ax = axes[1]
    ax.plot(val["env_steps"], val["val_return"], "o-", alpha=0.4, label="validation return")
    ax.plot(val["env_steps"], val["val_return_smoothed"], color="C1", label="smoothed")
    ax.axhline(keep_return, color="C3", ls="--", label="fixed threshold (W2)")
    ax.set_title("Validation return (greedy policy)")
    ax.set_xlabel("environment steps")
    ax.legend()
    ax = axes[2]
    ax.plot(hist["episode"], hist["epsilon"], color="C2")
    ax.set_title("Exploration rate (epsilon)")
    ax.set_xlabel("episode")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.episodes, args.val_every, args.warmup = 6, 3, 200
        args.run_name += "_smoke"
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_env = ECGThresholdEnv(load_beats("train"), mode="random",
                                prob_source=args.train_probs)
    val_env = ECGThresholdEnv(load_beats("val"), mode="sequential")
    obs_dim, n_actions = int(train_env.observation_space.shape[0]), int(train_env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, hidden=args.hidden, lr=args.lr,
                     gamma=args.gamma, weight_decay=args.weight_decay,
                     grad_clip=args.grad_clip, double=not args.no_double)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, seed=args.seed)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    run_ckpt = CKPT_DIR / f"{args.run_name}_best.pt"

    # Reference: fixed Workshop 2 threshold ("always keep") on validation
    keep_metrics, _ = evaluate_policy(val_env, lambda o: 2)
    keep_return = keep_metrics["mean_return"]
    print(f"Validation reference, fixed threshold 0.784: return {keep_return:.3f}, "
          f"F1 {keep_metrics['f1']:.4f}")
    print(f"{'ep':>5} {'steps':>7} {'eps':>5} {'tr_ret':>7} {'loss':>7} {'val_ret':>8} "
          f"{'val_F1':>6} {'val_P':>6} {'val_R':>6} {'keep%':>6} {'time':>5}")

    epsilon, env_steps = args.eps_start, 0
    best_val, smoothed = -np.inf, None
    history, val_history = [], []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = train_env.reset(seed=args.seed + ep)
        done, ep_return, ep_len, losses, qs = False, 0.0, 0, [], []
        while not done:
            action = agent.act(obs, epsilon, rng)
            next_obs, reward, done, _, _ = train_env.step(action)
            buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_return += reward
            ep_len += 1
            env_steps += 1

            if env_steps >= args.warmup and env_steps % args.train_freq == 0:
                loss, q_mean, _ = agent.update(buffer.sample(args.batch_size))
                losses.append(loss)
                qs.append(q_mean)
            if env_steps % args.target_update == 0:
                agent.sync_target()

        row = {"episode": ep, "env_steps": env_steps, "return": ep_return,
               "length": ep_len, "epsilon": epsilon,
               "loss": float(np.mean(losses)) if losses else np.nan,
               "q_mean": float(np.mean(qs)) if qs else np.nan}
        history.append(row)
        writer.add_scalar("train/episode_return", ep_return, ep)
        writer.add_scalar("train/episode_length", ep_len, ep)
        writer.add_scalar("train/epsilon", epsilon, ep)
        if losses:
            writer.add_scalar("train/loss", row["loss"], ep)
            writer.add_scalar("train/q_mean", row["q_mean"], ep)

        epsilon = max(args.eps_min, epsilon * args.eps_decay)

        if ep % args.val_every == 0 or ep == args.episodes:
            m, _ = evaluate_policy(val_env, lambda o: agent.act(o))
            smoothed = m["mean_return"] if smoothed is None else 0.7 * smoothed + 0.3 * m["mean_return"]
            val_history.append({"episode": ep, "env_steps": env_steps,
                                "val_return": m["mean_return"],
                                "val_return_smoothed": smoothed, **{
                                    k: v for k, v in m.items() if k != "mean_return"}})
            for k in ["mean_return", "f1", "precision", "recall",
                      "false_alarms_per_hour", "mean_threshold", "action_keep"]:
                writer.add_scalar(f"val/{k}", m[k], env_steps)
            writer.add_scalar("val/return_smoothed", smoothed, env_steps)

            mark = ""
            if m["mean_return"] > best_val:
                best_val = m["mean_return"]
                agent.save(run_ckpt, run_name=args.run_name, episode=ep,
                           env_steps=env_steps, val_return=float(best_val),
                           val_metrics={k: float(v) for k, v in m.items()},
                           config=vars(args))
                mark = " *"
            print(f"{ep:>5} {env_steps:>7} {epsilon:>5.2f} {ep_return:>7.1f} "
                  f"{row['loss']:>7.4f} {m['mean_return']:>8.3f} {m['f1']:>6.3f} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{100 * m['action_keep']:>5.1f}% {time.time() - t0:>4.0f}s{mark}")

    writer.close()
    hist, val = pd.DataFrame(history), pd.DataFrame(val_history)
    hist.to_csv(run_dir / "episodes.csv", index=False)
    val.to_csv(run_dir / "validation.csv", index=False)
    plot_run(hist, val, run_dir / "training_curves.png", keep_return)

    best = DQNAgent.load(run_ckpt)[1]
    summary = {"run_name": args.run_name, "best_val_return": best_val,
               "best_episode": best["episode"], "best_env_steps": best["env_steps"],
               "val_metrics_at_best": best["val_metrics"],
               "keep_val_return": keep_return, "keep_val_f1": keep_metrics["f1"],
               "config": vars(args), "train_seconds": round(time.time() - t0, 1)}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Keep the overall best agent across all runs as checkpoints/agent_best.pt
    global_path = CKPT_DIR / "agent_best.pt"
    if not args.smoke:
        current = DQNAgent.load(global_path)[1]["val_return"] if global_path.exists() else -np.inf
        if best_val > current:
            DQNAgent.load(run_ckpt)[0].save(global_path, **{k: v for k, v in best.items()
                                                        if k not in ("q_state", "agent_kwargs")})
            print(f"New overall best -> {global_path}")

    print(f"\nBest validation return {best_val:.3f} at episode {best['episode']} "
          f"(fixed threshold: {keep_return:.3f})")
    vm = best["val_metrics"]
    print(f"Validation at best: F1 {vm['f1']:.4f} | P {vm['precision']:.4f} | "
          f"R {vm['recall']:.4f} | false alarms/h {vm['false_alarms_per_hour']:.1f}")
    print(f"Logs: {run_dir}")


if __name__ == "__main__":
    main()
