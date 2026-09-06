"""
rdd2022.py

PyTorch Dataset class for the Road Damage Dataset 2022 (RDD2022).

The dataset uses Pascal VOC XML annotations. Each image can have zero or
more bounding boxes, each tagged with one of four defect classes:
    D00 - pothole
    D10 - longitudinal crack
    D20 - transverse crack
    D40 - alligator (grid) crack

We treat this as multi-label image classification: an image can legitimately
contain more than one defect type (e.g. a pothole next to a crack), so each
class gets its own independent 0/1 label rather than forcing a single
whole-image class. "Background" (no defect) is implicit -- it's whatever a
zero vector means, not a fifth class.
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import transforms

from config import EFFICIENTNET_IMG_SIZE, RDD2022_DIR, SEED

COUNTRIES = ["Japan", "India", "Czech", "Norway"]
VAL_FRACTION = 0.15


LABEL_MAP: dict[str, int] = {
    "D00": 1,
    "D10": 2,
    "D20": 3,
    "D40": 4,
}

NUM_DEFECT_CLASSES = len(LABEL_MAP)


def _parse_coordinate(value: str) -> int:
    """Parse an XML coordinate, which may be integer or decimal text."""
    return int(round(float(value)))


def parse_annotation(xml_path: str) -> list[dict]:
    """
    Parse a Pascal VOC XML file and return a list of bounding box records.

    Each record is a dict with keys:
        label (int): class index from LABEL_MAP
        xmin, ymin, xmax, ymax (int): pixel coordinates

    Args:
        xml_path: Path to the annotation XML file.

    Returns:
        List of bounding box dicts. Empty list if the image has no defects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: list[dict] = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in LABEL_MAP:
            continue
        bndbox = obj.find("bndbox")
        boxes.append({
            "label": LABEL_MAP[name],
            "xmin": _parse_coordinate(bndbox.find("xmin").text),
            "ymin": _parse_coordinate(bndbox.find("ymin").text),
            "xmax": _parse_coordinate(bndbox.find("xmax").text),
            "ymax": _parse_coordinate(bndbox.find("ymax").text),
        })
    return boxes


def multi_hot_label(boxes: list[dict]) -> Tensor:
    """Return a (NUM_DEFECT_CLASSES,) float tensor: 1.0 where that defect
    type appears anywhere among boxes, 0.0 otherwise."""
    label = torch.zeros(NUM_DEFECT_CLASSES, dtype=torch.float32)
    for box in boxes:
        label[box["label"] - 1] = 1.0
    return label


def count_images_per_class(ann_dir: Path) -> tuple[Counter, int]:
    """
    Count how many images in ann_dir contain at least one box of each
    defect class, without loading any images.

    Args:
        ann_dir: Directory of Pascal VOC XML annotation files.

    Returns:
        (counts, total) where counts maps LABEL_MAP class index -> number of
        images containing that class at least once, and total is the number
        of annotation files scanned.
    """
    counts: Counter = Counter()
    total = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        total += 1
        boxes = parse_annotation(str(xml_path))
        for label in {b["label"] for b in boxes}:
            counts[label] += 1
    return counts, total


class RDD2022Dataset(Dataset):
    """
    Loads RDD2022 images and returns them with their multi-label defect vector.

    Args:
        root_dir:  Path to one country's split, e.g. data/raw/rdd2022/Japan.
        split:     "train" or "test".
        transform: torchvision transforms applied to the PIL image.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.img_dir = Path(root_dir) / split / "images"
        self.ann_dir = Path(root_dir) / split / "annotations" / "xmls"
        self.transform = transform or self._default_transform()

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            ann_path = self.ann_dir / (img_path.stem + ".xml")
            if ann_path.exists():
                self.samples.append((img_path, ann_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.get_label(idx)
        return image, label

    def get_label(self, idx: int) -> Tensor:
        """Return just the multi-hot label for idx, without loading the image."""
        _, ann_path = self.samples[idx]
        boxes = parse_annotation(str(ann_path))
        return multi_hot_label(boxes)

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


def train_val_split(
    train_transform: transforms.Compose | None = None,
    val_transform: transforms.Compose | None = None,
    countries: list[str] = COUNTRIES,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    rdd2022_dir: Path = RDD2022_DIR,
) -> tuple[Subset, Subset]:
    """
    Reconstruct the exact train/val split used to fine-tune EfficientNet-B4.

    The notebook builds two datasets over the *same* RDD2022 "train" images
    (one with augmentation, one without), then holds out a deterministic
    ``val_fraction`` slice by permuting indices with a seeded generator. This
    helper reproduces that split so ``evaluate_rdd2022.py`` scores the model
    on the identical validation images the notebook reported.

    Returns:
        (train_subset, val_subset) as torch.utils.data.Subset objects.

    Raises:
        FileNotFoundError: if any requested country directory is missing.
        RuntimeError:      if a country directory has no labeled train samples.
    """
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    missing: list[str] = []

    for country in countries:
        country_dir = rdd2022_dir / country
        if not country_dir.is_dir():
            missing.append(country)
            continue
        train_ds = RDD2022Dataset(str(country_dir), split="train", transform=train_transform)
        if len(train_ds) == 0:
            raise RuntimeError(
                f"RDD2022 {country} exists but has no labeled train samples: {country_dir}"
            )
        train_datasets.append(train_ds)
        val_datasets.append(
            RDD2022Dataset(str(country_dir), split="train", transform=val_transform)
        )

    if missing:
        raise FileNotFoundError(
            f"Missing RDD2022 country directories: {missing}. "
            f'Run "python data/download_rdd2022.py --countries {" ".join(missing)}".'
        )

    train_full = ConcatDataset(train_datasets)
    val_full = ConcatDataset(val_datasets)
    assert len(train_full) == len(val_full), "train/val enumeration mismatch"

    val_size = int(val_fraction * len(train_full))
    order = torch.randperm(
        len(train_full), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_idx = order[: len(train_full) - val_size]
    val_idx = order[len(train_full) - val_size:]
    return Subset(train_full, train_idx), Subset(val_full, val_idx)
