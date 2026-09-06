"""
dqn_env.py

Gymnasium environment for adaptive patch inspection.

The agent receives a feature representation of the full image and decides
which patch to inspect next. Inspecting a defective patch gives a positive
reward; every step incurs a small penalty to encourage efficiency.

State:
    A 1-D float32 vector containing:
    - The EfficientNet-B4 feature vector of the full image (fixed per episode)
    - A binary mask of length GRID_SIZE^2 marking which patches are inspected

Action:
    An integer in [0, GRID_SIZE^2] . Values [0, GRID_SIZE^2) select the next
    patch to inspect; the final value GRID_SIZE^2 is a "stop" action that
    ends the episode early.

Reward:
    +DQN_DEFECT_REWARD  if the selected patch has any defect above threshold
    +DQN_STEP_PENALTY   applied on every inspect step (negative)
    Re-inspecting a patch gives only the step penalty (no double reward).
    On "stop": DQN_STOP_BONUS * recall - DQN_MISS_PENALTY * missed_defects,
    where recall is defects_found / defects_total for the current image.

Episode end:
    The episode ends when the agent picks "stop" or when every patch has
    been inspected.
"""

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from config import (
    GRID_SIZE,
    PATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    RDD2022_DEFECT_THRESHOLD,
    DQN_DEFECT_REWARD,
    DQN_STEP_PENALTY,
    DQN_STOP_BONUS,
    DQN_MISS_PENALTY,
)
from src.utils.patches import extract_patches, resize_patch


# Standard ImageNet normalization, matching the classifier's training transform.
_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Full-image transform used to compute the state feature vector.
_FULL_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_feature_dim(model: nn.Module, device: torch.device) -> int:
    """Run a dummy forward pass to find the feature vector dimension."""
    dummy = torch.zeros(1, 3, EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE, device=device)
    with torch.no_grad():
        out = model.forward_features(dummy)
        if out.dim() == 4:
            out = out.mean(dim=[2, 3])
    return out.shape[1]


class PatchInspectionEnv(gymnasium.Env):
    """
    Gymnasium environment for adaptive patch inspection.

    Args:
        image_paths: List of paths to images the agent can be trained on.
        model:       Frozen EfficientNet-B4 used both to build the state
                     feature vector and to score individual patches.
        device:      Device the model lives on.
        grid_size:   Number of grid cells per axis (default from config).
        defect_threshold: Sigmoid cutoff above which a patch is considered
                     to contain a defect. Patches with no class above this
                     threshold give no positive reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_paths: list[Path],
        model: nn.Module,
        device: torch.device,
        grid_size: int = GRID_SIZE,
        defect_threshold: float = RDD2022_DEFECT_THRESHOLD,
    ) -> None:
        super().__init__()

        self.image_paths = image_paths
        self.model = model
        self.model.eval()
        self.device = device
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.defect_threshold = defect_threshold

        # Action n_patches is "stop"; patch actions are [0, n_patches).
        self.stop_action = self.n_patches

        feature_dim = _get_feature_dim(model, device)
        obs_dim = feature_dim + self.n_patches

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.n_patches + 1)

        # Episode state, populated in reset().
        self._image_features: np.ndarray = np.zeros(feature_dim, dtype=np.float32)
        self._inspected_mask: np.ndarray = np.zeros(self.n_patches, dtype=np.float32)

        # Pre-compute features and patch defect labels for all images so that
        # DQN steps are array lookups rather than EfficientNet forward passes.
        print("Pre-computing image features and patch labels (runs once) ...")
        self._cached_features: list[np.ndarray] = []
        self._cached_patch_labels: list[np.ndarray] = []
        with torch.no_grad():
            for img_path in self.image_paths:
                pil_img = Image.open(img_path).convert("RGB")
                np_img = np.array(pil_img)

                # Full-image feature vector.
                img_t = _FULL_IMAGE_TRANSFORM(pil_img).unsqueeze(0).to(device)
                feats = model.forward_features(img_t)
                if feats.dim() == 4:
                    feats = feats.mean(dim=[2, 3])
                self._cached_features.append(feats.squeeze(0).cpu().numpy())

                # Per-patch defect labels.
                patches = extract_patches(np_img, grid_size)
                labels = np.zeros(self.n_patches, dtype=np.float32)
                for i, patch in enumerate(patches):
                    resized = resize_patch(patch, PATCH_SIZE)
                    t = _NORMALIZE(Image.fromarray(resized)).unsqueeze(0).to(device)
                    probs = torch.sigmoid(model(t))
                    labels[i] = float((probs > defect_threshold).any().item())
                self._cached_patch_labels.append(labels)
        print(f"  Cached {len(self.image_paths)} images.")
        self._current_idx: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick a random image for this episode using pre-cached data.
        self._current_idx = int(self.np_random.integers(0, len(self.image_paths)))
        self._image_features = self._cached_features[self._current_idx]
        self._inspected_mask = np.zeros(self.n_patches, dtype=np.float32)
        self._defects_found = 0
        self._defects_total = int(self._cached_patch_labels[self._current_idx].sum())

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        truncated = False

        # Stop action: end the episode and settle up on recall.
        if action == self.stop_action:
            reward = self._stop_reward()
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._info(stopped=True)

        reward = DQN_STEP_PENALTY
        if self._inspected_mask[action] == 0.0:
            self._inspected_mask[action] = 1.0
            if self._patch_has_defect(action):
                reward += DQN_DEFECT_REWARD
                self._defects_found += 1

        # A full sweep also ends the episode (and pays the same terminal bonus,
        # so "inspect everything" and "stop once done" are scored consistently).
        if self._inspected_mask.sum() == self.n_patches:
            reward += self._stop_reward()
            terminated = True
        else:
            terminated = False

        return self._get_obs(), reward, terminated, truncated, self._info(stopped=terminated)

    def _stop_reward(self) -> float:
        """Terminal reward: reward high recall, punish leaving defects behind."""
        if self._defects_total == 0:
            # Clean image: stopping promptly is the right call.
            return DQN_STOP_BONUS
        recall = self._defects_found / self._defects_total
        missed = self._defects_total - self._defects_found
        return DQN_STOP_BONUS * recall - DQN_MISS_PENALTY * missed

    def _info(self, stopped: bool) -> dict[str, Any]:
        return {
            "patches_inspected": int(self._inspected_mask.sum()),
            "defects_found": self._defects_found,
            "defects_total": self._defects_total,
            "recall": (
                self._defects_found / self._defects_total
                if self._defects_total else 1.0
            ),
            "stopped": stopped,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._image_features, self._inspected_mask])

    def _patch_has_defect(self, patch_idx: int) -> bool:
        """Look up the pre-cached patch defect label (no EfficientNet call at step time)."""
        return bool(self._cached_patch_labels[self._current_idx][patch_idx])
