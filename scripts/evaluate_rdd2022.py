"""
evaluate_rdd2022.py

Evaluates the trained EfficientNet-B4 on the RDD2022 test split.

Prints per-class precision, recall, and F1 from sklearn's classification
report, plus per-class AUROC computed from the model's softmax scores.

Usage:
    python scripts/evaluate_rdd2022.py
"""

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
from pathlib import Path

from config import (
    set_seeds, SEED,
    EFFICIENTNET_BATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    EFFICIENTNET_NUM_WORKERS,
    RDD2022_NUM_CLASSES,
    RDD2022_DIR,
    RDD2022_CHECKPOINT,
)
from src.datasets.rdd2022 import RDD2022Dataset
from src.models.efficientnet import load_checkpoint

CLASS_NAMES = ["Background", "Pothole", "Longitudinal crack", "Transverse crack", "Alligator crack"]
COUNTRIES = ["Japan", "India", "Czech", "Norway"]


def build_test_loader() -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = []
    for country in COUNTRIES:
        country_dir = str(RDD2022_DIR / country)
        if Path(country_dir).exists():
            datasets.append(RDD2022Dataset(country_dir, split="test", transform=transform))
        else:
            print(f"Warning: {country_dir} not found, skipping.")

    if not datasets:
        raise RuntimeError("No RDD2022 country directories found. Run download_rdd2022.py first.")

    return DataLoader(
        ConcatDataset(datasets),
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

    loader = build_test_loader()
    print(f"Test samples: {len(loader.dataset):,}")

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Per-class AUROC using one-vs-rest.
    probs_array = np.array(all_probs)
    labels_array = np.array(all_labels)
    print("--- Per-Class AUROC (one-vs-rest) ---")
    for i, name in enumerate(CLASS_NAMES):
        binary_labels = (labels_array == i).astype(int)
        if binary_labels.sum() == 0:
            print(f"  {name}: no positive samples in test set, skipping.")
            continue
        auroc = roc_auc_score(binary_labels, probs_array[:, i])
        print(f"  {name}: {auroc:.4f}")


if __name__ == "__main__":
    main()
