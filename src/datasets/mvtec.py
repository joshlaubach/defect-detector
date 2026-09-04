"""
mvtec.py

PyTorch Dataset class for the MVTec Anomaly Detection dataset.

Each category in MVTec AD has this structure:
    train/good/          - defect-free training images
    test/good/           - defect-free test images
    test/<defect_type>/  - images with a specific type of anomaly

We use binary labels: 0 = normal, 1 = anomalous.
"""

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from config import EFFICIENTNET_IMG_SIZE


class MVTecDataset(Dataset):
    """
    Loads one MVTec AD category for training or testing.

    Args:
        root_dir:   Path to the MVTec AD root, e.g. data/raw/mvtec.
        category:   One of the 15 MVTec categories, e.g. "bottle".
        split:      "train" loads only normal images.
                    "test" loads both normal and anomalous images.
        transform:  torchvision transforms applied to the image.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.transform = transform or self._default_transform()
        category_dir = Path(root_dir) / category

        if split == "train":
            self.samples: list[tuple[Path, int]] = [
                (p, 0)
                for p in sorted((category_dir / "train" / "good").glob("*.png"))
            ]
        else:
            self.samples = []
            test_dir = category_dir / "test"
            for class_dir in sorted(test_dir.iterdir()):
                label = 0 if class_dir.name == "good" else 1
                for img_path in sorted(class_dir.glob("*.png")):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def build_combined_mvtec(
    root_dir: str,
    categories: list[str],
    split: str = "train",
) -> Dataset:
    """
    Combine multiple MVTec categories into one dataset.

    Args:
        root_dir:   Path to the MVTec AD root.
        categories: List of category names to include.
        split:      "train" or "test".

    Returns:
        A ConcatDataset covering all requested categories.
    """
    datasets = [MVTecDataset(root_dir, cat, split) for cat in categories]
    return ConcatDataset(datasets)
