"""
train_dqn.py

Trains the DQN agent to adaptively select which image patches to inspect.

The frozen EfficientNet-B4 (RDD2022 checkpoint) acts as the environment's
reward oracle: it classifies each patch the agent selects and returns a
positive reward when it finds a defect.

Training runs for DQN_TRAIN_STEPS environment steps. Every TARGET_UPDATE_FREQ
steps, the online network's weights are copied to the target network (hard
update). Epsilon decays linearly from 1.0 to 0.05 over the first
DQN_EPSILON_DECAY steps.

Usage:
    python scripts/train/train_dqn.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    set_seeds, SEED,
    GRID_SIZE,
    DQN_LR,
    DQN_GAMMA,
    DQN_BATCH_SIZE,
    DQN_REPLAY_BUFFER_SIZE,
    DQN_TARGET_UPDATE_FREQ,
    DQN_EPSILON_START,
    DQN_EPSILON_END,
    DQN_EPSILON_DECAY,
    DQN_TRAIN_STEPS,
    RDD2022_NUM_CLASSES,
    RDD2022_DIR,
    RDD2022_CHECKPOINT,
    DQN_CHECKPOINT,
    CHECKPOINT_DIR,
)
from src.datasets.rdd2022 import COUNTRIES
from src.models.efficientnet import load_checkpoint
from src.models.dqn import QNetwork, ReplayBuffer
from src.models.dqn_env import PatchInspectionEnv


def collect_image_paths() -> list[Path]:
    """Find all training images across the four RDD2022 country splits."""
    paths: list[Path] = []
    for country in COUNTRIES:
        img_dir = RDD2022_DIR / country / "train" / "images"
        if img_dir.exists():
            paths.extend(sorted(img_dir.glob("*.jpg")))
        else:
            print(f"Warning: {img_dir} not found, skipping.")
    if not paths:
        raise RuntimeError("No RDD2022 images found. Run download_rdd2022.py first.")
    return paths


def get_epsilon(step: int) -> float:
    """Linear epsilon decay from EPSILON_START to EPSILON_END."""
    ratio = min(step / DQN_EPSILON_DECAY, 1.0)
    return DQN_EPSILON_START + ratio * (DQN_EPSILON_END - DQN_EPSILON_START)


class DQNAgent:
    """
    DQN agent with epsilon-greedy exploration and a hard-updated target network.

    Args:
        state_dim:  Dimension of the state vector from PatchInspectionEnv.
        n_actions:  Number of patch actions (GRID_SIZE^2).
        device:     Device to run networks on.
    """

    def __init__(self, state_dim: int, n_actions: int, device: torch.device) -> None:
        self.device = device
        self.n_actions = n_actions

        self.online_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=DQN_LR)
        self.buffer = ReplayBuffer(DQN_REPLAY_BUFFER_SIZE)

    def select_action(self, state, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(self.n_actions, (1,)).item())
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def learn(self) -> float | None:
        """
        Sample a minibatch and update the online network with one gradient step.

        Returns the loss value, or None if the buffer is not yet large enough.
        """
        if len(self.buffer) < DQN_BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            DQN_BATCH_SIZE, self.device
        )

        # Current Q-values for the actions that were actually taken.
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * max_a Q_target(s') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + DQN_GAMMA * next_q * (1.0 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self) -> None:
        """Hard copy: replace target network weights with online network weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())


def main() -> None:
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading frozen EfficientNet-B4 ...")
    efficientnet = load_checkpoint(
        num_classes=RDD2022_NUM_CLASSES,
        checkpoint_path=str(RDD2022_CHECKPOINT),
        device=device,
    )
    for param in efficientnet.parameters():
        param.requires_grad = False

    image_paths = collect_image_paths()
    print(f"Training images available: {len(image_paths):,}")

    env = PatchInspectionEnv(
        image_paths=image_paths,
        model=efficientnet,
        device=device,
        grid_size=GRID_SIZE,
    )

    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    print(f"State dim: {state_dim}  |  Actions: {n_actions}")

    agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, device=device)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    episode_count = 0
    # The agent's real job is "high recall for few patches inspected", so the
    # checkpoint is chosen on a rolling mean of that trade-off, not on the
    # noisy single-episode return.
    recent_recall: list[float] = []
    recent_patches: list[int] = []
    best_gate_score = float("-inf")

    def gate_score(recalls: list[float], patches: list[int]) -> float:
        n = len(recalls)
        return sum(recalls) / n - 0.02 * (sum(patches) / n)

    print(f"Training for {DQN_TRAIN_STEPS:,} steps ...")
    for step in tqdm(range(1, DQN_TRAIN_STEPS + 1)):
        epsilon = get_epsilon(step)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        agent.learn()

        if step % DQN_TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if done:
            episode_count += 1
            recent_recall.append(info["recall"])
            recent_patches.append(info["patches_inspected"])
            recent_recall = recent_recall[-100:]
            recent_patches = recent_patches[-100:]

            if len(recent_recall) == 100:
                score = gate_score(recent_recall, recent_patches)
                if score > best_gate_score:
                    best_gate_score = score
                    torch.save(agent.online_net.state_dict(), DQN_CHECKPOINT)

            if episode_count % 100 == 0:
                print(
                    f"  Step {step:>7,}  Episodes {episode_count:>5,}  "
                    f"Recall(100): {sum(recent_recall) / len(recent_recall):.3f}  "
                    f"Patches(100): {sum(recent_patches) / len(recent_patches):.1f}  "
                    f"Epsilon: {epsilon:.3f}"
                )

            state, _ = env.reset()
            episode_reward = 0.0

    if not DQN_CHECKPOINT.exists():
        # Short run that never completed 100 episodes: still leave a checkpoint.
        torch.save(agent.online_net.state_dict(), DQN_CHECKPOINT)

    print(f"\nTraining complete. Best gate score: {best_gate_score:.3f}")
    print(f"Checkpoint saved to {DQN_CHECKPOINT}")


if __name__ == "__main__":
    main()
