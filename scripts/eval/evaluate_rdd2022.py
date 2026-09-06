"""
evaluate_rdd2022.py

Evaluates the trained EfficientNet-B4 on the held-out RDD2022 validation split.

The official RDD2022 "test" split ships images with no Pascal VOC annotations,
so classification metrics cannot be computed against it. Instead this script
reconstructs the exact deterministic 15% validation slice the training
notebook held out (src.datasets.rdd2022.train_val_split) and reports on that.

RDD2022 is treated as multi-label (an image can contain more than one defect
type), so this prints a per-class precision/recall/F1 report plus per-class
AUROC computed from the model's independent sigmoid scores.

Usage:
    python scripts/eval/evaluate_rdd2022.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    set_seeds, SEED,
    EFFICIENTNET_BATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    EFFICIENTNET_NUM_WORKERS,
    RDD2022_NUM_CLASSES,
    RDD2022_DEFECT_THRESHOLD,
    RDD2022_CHECKPOINT,
)
from src.datasets.rdd2022 import train_val_split
from src.models.efficientnet import load_checkpoint

CLASS_NAMES = ["Pothole", "Longitudinal crack", "Transverse crack", "Alligator crack"]


def build_val_loader() -> DataLoader:
    """Deterministic 15% validation slice, matching the training notebook."""
    eval_transform = transforms.Compose([
        transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _, val_dataset = train_val_split(
        train_transform=eval_transform, val_transform=eval_transform
    )
    return DataLoader(
        val_dataset,
        batch_size=EFFICIENTNET_BATCH_SIZE,
        shuffle=False,
        num_workers=EFFICIENTNET_NUM_WORKERS,
        pin_memory=True,
    )


def main() -> None:
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_checkpoint(
        num_classes=RDD2022_NUM_CLASSES,
        checkpoint_path=str(RDD2022_CHECKPOINT),
        device=device,
    )

    loader = build_val_loader()
    print(f"Validation samples: {len(loader.dataset):,}")

    all_preds: list[list[float]] = []
    all_labels: list[list[float]] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > RDD2022_DEFECT_THRESHOLD).float()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    probs_array = np.array(all_probs)
    preds_array = np.array(all_preds)
    labels_array = np.array(all_labels)

    print("\n--- Classification Report ---")
    print(classification_report(
        labels_array, preds_array, target_names=CLASS_NAMES, zero_division=0
    ))

    no_defect_true = labels_array.sum(axis=1) == 0
    no_defect_pred = preds_array.sum(axis=1) == 0
    bg_agreement = (no_defect_true == no_defect_pred).mean()
    print(f"No-defect agreement: {bg_agreement:.3f}\n")

    # Per-class AUROC. Each class is already an independent binary target,
    # so this is a plain binary AUROC per column, not one-vs-rest.
    print("--- Per-Class AUROC ---")
    for i, name in enumerate(CLASS_NAMES):
        binary_labels = labels_array[:, i]
        if binary_labels.sum() == 0:
            print(f"  {name}: no positive samples in test set, skipping.")
            continue
        auroc = roc_auc_score(binary_labels, probs_array[:, i])
        print(f"  {name}: {auroc:.4f}")


if __name__ == "__main__":
    main()
