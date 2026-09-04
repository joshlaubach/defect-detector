"""
dqn.py

Core DQN building blocks: the replay buffer and the Q-network.

These are kept separate from the training loop so that the DQN agent
can be imported and used in other scripts (evaluation, the dashboard)
without pulling in the full training setup.
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.

    When the buffer is full, new transitions overwrite the oldest ones.
    Transitions are sampled uniformly at random, which is the standard
    approach for DQN and works because DQN is off-policy (old transitions
    remain valid training data regardless of the current policy).

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random minibatch of transitions.

        Returns five tensors (states, actions, rewards, next_states, dones),
        each with leading dimension batch_size, ready to pass to the Q-network.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=device),
            torch.tensor(actions,               dtype=torch.long,    device=device),
            torch.tensor(rewards,               dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones,                 dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Multi-layer perceptron that maps a state vector to Q-values.

    Each output neuron corresponds to one patch action (0 to n_actions-1).
    The agent picks the action with the highest Q-value.

    Architecture:
        state_dim -> 512 -> 256 -> n_actions

    Args:
        state_dim:  Dimension of the input state vector.
        n_actions:  Number of discrete actions (GRID_SIZE^2 = 64).
    """

    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State batch of shape (N, state_dim).

        Returns:
            Q-value tensor of shape (N, n_actions).
        """
        return self.net(x)
