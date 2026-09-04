"""
evaluate_dqn.py

Compares the trained DQN agent against a random patch selection baseline.

For each evaluation episode, both agents inspect all N patches on the same
image. We record how many defective patches each agent has found after
every k patches inspected. Averaging this across episodes gives a recall
curve: x = patches inspected, y = fraction of defects found.

A well-trained DQN should find defects earlier (higher recall at lower k)
compared to the random baseline.

Results are printed to the terminal and the recall curve is saved as
dqn_recall_curve.png in the project root.

Usage:
    python scripts/evaluate_dqn.py
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import (
    set_seeds, SEED,
    GRID_SIZE,
    RDD2022_NUM_CLASSES,
    RDD2022_DIR,
    RDD2022_CHECKPOINT,
    DQN_CHECKPOINT,
)
from src.models.efficientnet import load_checkpoint
from src.models.dqn import QNetwork
from src.models.dqn_env import PatchInspectionEnv

COUNTRIES = ["Japan", "India", "Czech", "Norway"]
N_EVAL_EPISODES = 200


def collect_test_paths() -> list[Path]:
    """Collect test image paths from all four country splits."""
    paths: list[Path] = []
    for country in COUNTRIES:
        img_dir = RDD2022_DIR / country / "test" / "images"
        if img_dir.exists():
            paths.extend(sorted(img_dir.glob("*.jpg")))
        else:
            print(f"Warning: {img_dir} not found, skipping.")
    if not paths:
        raise RuntimeError("No test images found. Run download_rdd2022.py first.")
    return paths


def run_episode_greedy(
    env: PatchInspectionEnv,
    q_net: QNetwork,
    device: torch.device,
) -> list[int]:
    """
    Run one greedy episode (epsilon=0) and return cumulative defects found
    at each step.
    """
    state, _ = env.reset()
    cumulative_defects = []
    defects_so_far = 0
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(q_net(state_t).argmax(dim=1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # A positive reward above the step penalty means a defect was found.
        if reward > 0:
            defects_so_far += 1
        cumulative_defects.append(defects_so_far)

    return cumulative_defects


def run_episode_random(env: PatchInspectionEnv) -> list[int]:
    """
    Run one episode with random action selection and return cumulative
    defects found at each step.
    """
    state, _ = env.reset()
    n_actions = env.action_space.n
    unvisited = list(range(n_actions))
    random.shuffle(unvisited)

    cumulative_defects = []
    defects_so_far = 0
    done = False

    while not done and unvisited:
        action = unvisited.pop(0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward > 0:
            defects_so_far += 1
        cumulative_defects.append(defects_so_far)

    return cumulative_defects


def normalize_to_recall(cumulative: list[int]) -> list[float]:
    """Convert cumulative defect counts to recall fractions."""
    total = cumulative[-1] if cumulative else 0
    if total == 0:
        return [0.0] * len(cumulative)
    return [c / total for c in cumulative]


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

    test_paths = collect_test_paths()
    print(f"Test images: {len(test_paths):,}")

    env = PatchInspectionEnv(
        image_paths=test_paths,
        model=efficientnet,
        device=device,
        grid_size=GRID_SIZE,
    )

    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    print("Loading DQN checkpoint ...")
    q_net = QNetwork(state_dim=state_dim, n_actions=n_actions).to(device)
    q_net.load_state_dict(torch.load(DQN_CHECKPOINT, map_location=device))
    q_net.eval()

    # Run evaluation episodes.
    dqn_recalls: list[list[float]] = []
    random_recalls: list[list[float]] = []

    for _ in tqdm(range(N_EVAL_EPISODES), desc="Evaluating"):
        dqn_curve = run_episode_greedy(env, q_net, device)
        dqn_recalls.append(normalize_to_recall(dqn_curve))

        random_curve = run_episode_random(env)
        random_recalls.append(normalize_to_recall(random_curve))

    # Pad shorter episodes to n_actions length so we can average them.
    def pad(curves: list[list[float]], length: int) -> np.ndarray:
        padded = [c + [c[-1]] * (length - len(c)) for c in curves]
        return np.array(padded)

    dqn_array = pad(dqn_recalls, n_actions)
    random_array = pad(random_recalls, n_actions)

    dqn_mean = dqn_array.mean(axis=0)
    random_mean = random_array.mean(axis=0)

    # Patches inspected to reach 80% recall.
    def steps_to_recall(curve: np.ndarray, threshold: float = 0.8) -> int:
        indices = np.where(curve >= threshold)[0]
        return int(indices[0]) + 1 if len(indices) > 0 else n_actions

    dqn_80 = steps_to_recall(dqn_mean)
    random_80 = steps_to_recall(random_mean)

    print(f"\nPatches to reach 80% defect recall:")
    print(f"  DQN:    {dqn_80} / {n_actions}")
    print(f"  Random: {random_80} / {n_actions}")
    print(f"  Improvement: {random_80 - dqn_80} fewer patches inspected")

    # Plot and save.
    x = list(range(1, n_actions + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(x, dqn_mean,    label="DQN agent",    linewidth=2)
    plt.plot(x, random_mean, label="Random",        linewidth=2, linestyle="--")
    plt.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="80% recall")
    plt.xlabel("Patches inspected")
    plt.ylabel("Defect recall")
    plt.title("DQN vs. Random: defect recall per patches inspected")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dqn_recall_curve.png", dpi=120)
    plt.show()
    print("Saved dqn_recall_curve.png")


if __name__ == "__main__":
    main()
