"""
Deep Q-Network agent (Mnih et al., 2015) with the Double-DQN target
(van Hasselt et al., 2016).

Components required by the workshop guide:
  - Q-network (MLP) that maps the 8-dim state to one Q-value per action
  - experience replay buffer
  - target network, synchronized every `target_update` steps
  - epsilon-greedy exploration (decay handled in train.py)
  - Huber loss, Adam optimizer (optional L2 weight decay), gradient clipping
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """state (8) -> 128 -> 128 -> Q-value for each of the 5 actions."""

    def __init__(self, obs_dim=8, n_actions=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-size circular memory of transitions (s, a, r, s', done).
    Sampling random mini-batches breaks the strong correlation between
    consecutive beats, which would otherwise destabilize learning.
    """

    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]),
                torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.dones[idx]))

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, obs_dim=8, n_actions=5, hidden=128, lr=1e-3, gamma=0.95,
                 weight_decay=0.0, grad_clip=10.0, double=True):
        self.kwargs = dict(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden,
                           lr=lr, gamma=gamma, weight_decay=weight_decay,
                           grad_clip=grad_clip, double=double)
        self.n_actions = n_actions
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.double = double

        self.q = QNetwork(obs_dim, n_actions, hidden)
        self.q_target = copy.deepcopy(self.q)
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr,
                                          weight_decay=weight_decay)

    def act(self, obs, epsilon=0.0, rng=None):
        """Epsilon-greedy: random action with probability epsilon, else argmax Q."""
        if epsilon > 0 and rng is not None and rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def q_values(self, obs):
        with torch.no_grad():
            return self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def update(self, batch):
        """One gradient step on a mini-batch. Returns loss, mean Q and gradient norm."""
        obs, actions, rewards, next_obs, dones = batch
        q = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # Double DQN: the online network chooses the next action,
                # the target network evaluates it (reduces over-estimation)
                next_a = self.q(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.q_target(next_obs).gather(1, next_a).squeeze(1)
            else:
                next_q = self.q_target(next_obs).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q, target)          # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item()), float(q.mean().item()), float(grad_norm)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path, **extra):
        torch.save({"q_state": self.q.state_dict(), "agent_kwargs": self.kwargs,
                    **extra}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(**ckpt["agent_kwargs"])
        agent.q.load_state_dict(ckpt["q_state"])
        agent.sync_target()
        agent.q.eval()
        return agent, ckpt
