"""
efficientnet.py

Helpers for building and loading EfficientNet-B4 models.

Keeping model construction in one place means the training scripts,
the notebook, and the DQN environment all get the same model with
the same settings.
"""

from pathlib import Path

import torch
import timm
from torch import nn

from config import EFFICIENTNET_MODEL


def build_efficientnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load EfficientNet-B4 and replace its classifier head.

    Args:
        num_classes: Number of output classes.
        pretrained:  If True, start from ImageNet weights. Set False for testing.

    Returns:
        An nn.Module ready to move to a device and train.
    """
    model = timm.create_model(
        EFFICIENTNET_MODEL,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def load_checkpoint(
    num_classes: int,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Build the model and load weights from a saved checkpoint.

    Args:
        num_classes:     Must match the num_classes used during training.
        checkpoint_path: Path to the .pth file saved by the training script.
        device:          Device to map the weights onto.

    Returns:
        The model in eval mode with loaded weights.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run the training script first."
        )
    model = build_efficientnet(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Run images through the backbone and return feature vectors before the
    classifier head.

    The DQN uses these feature vectors as its state representation.

    Args:
        model:  A model returned by build_efficientnet or load_checkpoint.
        images: Batch of images, shape (N, 3, H, W).

    Returns:
        Feature tensor of shape (N, feature_dim).
    """
    with torch.no_grad():
        # timm exposes forward_features() to get the backbone output.
        features = model.forward_features(images)
        # Global average pool if the output is still spatial (H x W features).
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
    return features
