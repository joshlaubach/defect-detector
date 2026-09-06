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
    python scripts/eval/evaluate_dqn.py
    python scripts/eval/evaluate_dqn.py --episodes 1
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

from src.datasets.rdd2022 import COUNTRIES

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
) -> tuple[list[int], int]:
    """
    Greedy episode (epsilon=0). Returns (cumulative defects found after each
    inspected patch, true defect total for the image). The agent may pick the
    "stop" action, which ends the episode early.
    """
    state, info = env.reset()
    cumulative: list[int] = []
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(q_net(state_t).argmax(dim=1).item())
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if action != env.stop_action:
            cumulative.append(info["defects_found"])

    return cumulative, info["defects_total"]


def run_episode_random(env: PatchInspectionEnv) -> tuple[list[int], int]:
    """Random full sweep over real patches (never picks 'stop')."""
    _, info = env.reset()
    order = list(range(env.n_patches))
    random.shuffle(order)

    cumulative: list[int] = []
    for action in order:
        _, _, terminated, truncated, info = env.step(action)
        cumulative.append(info["defects_found"])
        if terminated or truncated:
            break

    return cumulative, info["defects_total"]


def normalize_to_recall(cumulative: list[int], defect_total: int) -> list[float]:
    """Cumulative defect counts -> recall of the image's true defect total."""
    if defect_total == 0:
        return [1.0] * len(cumulative)
    return [c / defect_total for c in cumulative]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DQN patch inspection agent")
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EVAL_EPISODES})",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

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
    n_patches = env.n_patches

    print("Loading DQN checkpoint ...")
    q_net = QNetwork(state_dim=state_dim, n_actions=n_actions).to(device)
    q_net.load_state_dict(torch.load(DQN_CHECKPOINT, map_location=device))
    q_net.eval()

    # Run evaluation episodes. Skip images with no defects: recall is undefined.
    dqn_recalls: list[list[float]] = []
    random_recalls: list[list[float]] = []
    dqn_stop_steps: list[int] = []

    pbar = tqdm(total=args.episodes, desc="Evaluating")
    while len(dqn_recalls) < args.episodes:
        dqn_curve, total = run_episode_greedy(env, q_net, device)
        random_curve, _ = run_episode_random(env)
        if total == 0 or not dqn_curve or not random_curve:
            continue
        dqn_recalls.append(normalize_to_recall(dqn_curve, total))
        random_recalls.append(normalize_to_recall(random_curve, total))
        dqn_stop_steps.append(len(dqn_curve))
        pbar.update(1)
    pbar.close()

    # Pad each curve forward to a full sweep so episodes can be averaged.
    def pad(curves: list[list[float]], length: int) -> np.ndarray:
        padded = [c + [c[-1]] * (length - len(c)) for c in curves]
        return np.array(padded)

    dqn_array = pad(dqn_recalls, n_patches)
    random_array = pad(random_recalls, n_patches)

    dqn_mean = dqn_array.mean(axis=0)
    random_mean = random_array.mean(axis=0)

    # Patches inspected to reach 80% recall.
    def steps_to_recall(curve: np.ndarray, threshold: float = 0.8) -> int:
        indices = np.where(curve >= threshold)[0]
        return int(indices[0]) + 1 if len(indices) > 0 else n_patches

    dqn_80 = steps_to_recall(dqn_mean)
    random_80 = steps_to_recall(random_mean)
    mean_stop = sum(dqn_stop_steps) / len(dqn_stop_steps)

    print(f"\nEpisodes evaluated (with >=1 defect): {len(dqn_recalls)}")
    print(f"DQN mean patches inspected before stop: {mean_stop:.1f} / {n_patches}")
    print(f"Patches to reach 80% defect recall:")
    print(f"  DQN:    {dqn_80} / {n_patches}")
    print(f"  Random: {random_80} / {n_patches}")
    print(f"  Improvement: {random_80 - dqn_80} fewer patches inspected")
    print(f"Final mean recall  DQN: {dqn_mean[-1]:.3f}   Random: {random_mean[-1]:.3f}")

    # Plot and save.
    x = list(range(1, n_patches + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(x, dqn_mean,    label="DQN agent",    linewidth=2)
    plt.plot(x, random_mean, label="Random",        linewidth=2, linestyle="--")
    plt.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="80% recall")
    plt.xlabel("Patches inspected")
    plt.ylabel("Defect recall")
    plt.title("DQN vs. Random: defect recall per patches inspected")
    plt.legend()
    plt.tight_layout()
    out_path = Path(__file__).resolve().parents[2] / "assets" / "dqn_recall_curve.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
