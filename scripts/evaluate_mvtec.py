"""
evaluate_mvtec.py

Evaluates the trained EfficientNet-B4 on the MVTec AD test split.

Reports AUROC and F1 for each of the 15 categories individually, then
prints macro-averaged scores across all categories. AUROC is the standard
evaluation metric for MVTec AD.

Usage:
    python scripts/evaluate_mvtec.py
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from config import (
    set_seeds, SEED,
    EFFICIENTNET_BATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    EFFICIENTNET_NUM_WORKERS,
    MVTEC_NUM_CLASSES,
    MVTEC_DIR,
    MVTEC_CHECKPOINT,
)
from src.datasets.mvtec import MVTecDataset
from src.models.efficientnet import load_checkpoint

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def evaluate_category(
    model: torch.nn.Module,
    category: str,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run inference on one MVTec category and return (AUROC, F1).

    Returns (-1.0, -1.0) if the category directory does not exist.
    """
    category_dir = str(MVTEC_DIR / category)

    transform = transforms.Compose([
        transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = MVTecDataset(str(MVTEC_DIR), category, split="test", transform=transform)
    except FileNotFoundError:
        print(f"  Warning: {category_dir} not found, skipping.")
        return -1.0, -1.0

    loader = DataLoader(
        dataset,
        batch_size=EFFICIENTNET_BATCH_SIZE,
        shuffle=False,
        num_workers=EFFICIENTNET_NUM_WORKERS,
        pin_memory=True,
    )

    all_probs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    if len(set(all_labels)) < 2:
        # Cannot compute AUROC with only one class present.
        return -1.0, f1_score(all_labels, all_preds, zero_division=0)

    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return auroc, f1


def main() -> None:
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = load_checkpoint(
        num_classes=MVTEC_NUM_CLASSES,
        checkpoint_path=str(MVTEC_CHECKPOINT),
        device=device,
    )

    print(f"{'Category':<16} {'AUROC':>8} {'F1':>8}")
    print("-" * 36)

    aurocs: list[float] = []
    f1s: list[float] = []

    for category in MVTEC_CATEGORIES:
        auroc, f1 = evaluate_category(model, category, device)
        if auroc < 0:
            print(f"{category:<16} {'N/A':>8} {'N/A':>8}")
            continue
        aurocs.append(auroc)
        f1s.append(f1)
        print(f"{category:<16} {auroc:>8.4f} {f1:>8.4f}")

    if aurocs:
        print("-" * 36)
        print(f"{'Macro average':<16} {np.mean(aurocs):>8.4f} {np.mean(f1s):>8.4f}")


if __name__ == "__main__":
    main()
