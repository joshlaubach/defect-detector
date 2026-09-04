"""
config.py

Single source of truth for every hyperparameter and path in the project.
Nothing in any training script or notebook should hardcode a number that
lives here.
"""

import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

GRID_SIZE = 8       # divide each image into an 8x8 grid (64 patches total)
PATCH_SIZE = 380    # resize each patch to 380x380 before passing to EfficientNet


# ---------------------------------------------------------------------------
# EfficientNet-B4
# ---------------------------------------------------------------------------

EFFICIENTNET_MODEL = "efficientnet_b4"
EFFICIENTNET_IMG_SIZE = 380
EFFICIENTNET_LR = 1e-4
EFFICIENTNET_WEIGHT_DECAY = 1e-2
EFFICIENTNET_BATCH_SIZE = 32
EFFICIENTNET_EPOCHS = 20
EFFICIENTNET_NUM_WORKERS = 4

RDD2022_NUM_CLASSES = 5     # background + D00 + D10 + D20 + D40
MVTEC_NUM_CLASSES = 2       # normal + anomalous


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

DQN_LR = 1e-4
DQN_GAMMA = 0.99                # discount factor for future rewards
DQN_BATCH_SIZE = 64
DQN_REPLAY_BUFFER_SIZE = 50000
DQN_TARGET_UPDATE_FREQ = 500    # copy weights to target network every N steps
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY = 10000       # steps over which epsilon anneals
DQN_TRAIN_STEPS = 200000
DQN_STEP_PENALTY = -0.1         # reward subtracted per step to encourage efficiency
DQN_DEFECT_REWARD = 1.0         # reward for finding a defective patch


# ---------------------------------------------------------------------------
# LLaMA 3.2 Vision (via Ollama)
# ---------------------------------------------------------------------------

LLAMA_MODEL = "llama3.2-vision"
OLLAMA_HOST = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RDD2022_DIR = RAW_DIR / "rdd2022"
MVTEC_DIR = RAW_DIR / "mvtec"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RDD2022_CHECKPOINT = CHECKPOINT_DIR / "efficientnet_rdd2022.pth"
MVTEC_CHECKPOINT = CHECKPOINT_DIR / "efficientnet_mvtec.pth"
DQN_CHECKPOINT = CHECKPOINT_DIR / "dqn.pth"
