"""
rdd2022.py

PyTorch Dataset class for the Road Damage Dataset 2022 (RDD2022).

The dataset uses Pascal VOC XML annotations. Each image can have zero or
more bounding boxes, each tagged with one of four defect classes:
    D00 - pothole
    D10 - longitudinal crack
    D20 - transverse crack
    D40 - alligator (grid) crack

We treat this as image-level classification: each image gets the label of
whichever defect class appears most often in its bounding boxes, or 0
(background) if there are no annotations.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from config import EFFICIENTNET_IMG_SIZE, RDD2022_DIR


LABEL_MAP: dict[str, int] = {
    "D00": 1,
    "D10": 2,
    "D20": 3,
    "D40": 4,
}


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


class RDD2022Dataset(Dataset):
    """
    Loads RDD2022 images and returns them with their image-level defect label.

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

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        boxes = parse_annotation(str(ann_path))
        label = self._image_level_label(boxes)
        return image, label

    def _image_level_label(self, boxes: list[dict]) -> int:
        """Return the most frequent defect class, or 0 if there are none."""
        if not boxes:
            return 0
        labels = [b["label"] for b in boxes]
        return max(set(labels), key=labels.count)

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
