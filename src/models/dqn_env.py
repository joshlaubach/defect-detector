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
    An integer in [0, GRID_SIZE^2) selecting the next patch to inspect.

Reward:
    +DQN_DEFECT_REWARD  if the selected patch is classified as defective
    +DQN_STEP_PENALTY   applied every step (negative, to penalize wasted steps)
    Re-inspecting a patch gives only the step penalty (no double reward).

Episode end:
    The episode ends when every patch has been inspected.
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
    DQN_DEFECT_REWARD,
    DQN_STEP_PENALTY,
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
        background_class: Class index that means "no defect". Patches
                     predicted as this class give no positive reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_paths: list[Path],
        model: nn.Module,
        device: torch.device,
        grid_size: int = GRID_SIZE,
        background_class: int = 0,
    ) -> None:
        super().__init__()

        self.image_paths = image_paths
        self.model = model
        self.model.eval()
        self.device = device
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.background_class = background_class

        feature_dim = _get_feature_dim(model, device)
        obs_dim = feature_dim + self.n_patches

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.n_patches)

        # Episode state, populated in reset().
        self._image_features: np.ndarray = np.zeros(feature_dim, dtype=np.float32)
        self._inspected_mask: np.ndarray = np.zeros(self.n_patches, dtype=np.float32)
        self._patches: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick a random image for this episode.
        idx = self.np_random.integers(0, len(self.image_paths))
        img_path = self.image_paths[idx]

        pil_image = Image.open(img_path).convert("RGB")
        np_image = np.array(pil_image)

        # Extract all patches up front so step() does not do file I/O.
        self._patches = extract_patches(np_image, self.grid_size)
        self._inspected_mask = np.zeros(self.n_patches, dtype=np.float32)

        # Compute the full-image feature vector (fixed for this episode).
        image_tensor = _FULL_IMAGE_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(image_tensor)
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
        self._image_features = features.squeeze(0).cpu().numpy()

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = DQN_STEP_PENALTY

        if self._inspected_mask[action] == 0.0:
            self._inspected_mask[action] = 1.0
            patch_class = self._classify_patch(action)
            if patch_class != self.background_class:
                reward += DQN_DEFECT_REWARD

        # Episode ends when every patch has been visited.
        terminated = bool(self._inspected_mask.sum() == self.n_patches)
        truncated = False

        info: dict[str, Any] = {
            "patches_inspected": int(self._inspected_mask.sum()),
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._image_features, self._inspected_mask])

    def _classify_patch(self, patch_idx: int) -> int:
        """Run EfficientNet on one patch and return the predicted class."""
        patch = self._patches[patch_idx]
        resized = resize_patch(patch, PATCH_SIZE)
        pil_patch = Image.fromarray(resized)
        tensor = _NORMALIZE(pil_patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return int(logits.argmax(dim=1).item())
