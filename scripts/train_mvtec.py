"""
train_mvtec.py

Fine-tunes EfficientNet-B4 on MVTec AD for binary anomaly detection.

Each image is labeled 0 (normal) or 1 (anomalous). The model is trained
across all 15 MVTec categories combined, then evaluated on the test split.
The best checkpoint is saved to checkpoints/efficientnet_mvtec.pth.

Usage:
    python scripts/train_mvtec.py
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from config import (
    set_seeds, SEED,
    EFFICIENTNET_LR,
    EFFICIENTNET_WEIGHT_DECAY,
    EFFICIENTNET_IMG_SIZE,
    MVTEC_BATCH_SIZE,
    MVTEC_EPOCHS,
    MVTEC_NUM_WORKERS,
    MVTEC_NUM_CLASSES,
    MVTEC_DIR,
    MVTEC_CHECKPOINT,
    CHECKPOINT_DIR,
)
from src.datasets.mvtec import build_combined_mvtec
from src.models.efficientnet import build_efficientnet

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """One pass over the training set. Returns (avg loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a data loader. Returns (avg loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main() -> None:
    set_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mvtec_root = str(MVTEC_DIR)
    train_dataset = build_combined_mvtec(mvtec_root, MVTEC_CATEGORIES, split="train")
    test_dataset = build_combined_mvtec(mvtec_root, MVTEC_CATEGORIES, split="test")

    # Apply transforms by wrapping in a simple adapter since ConcatDataset
    # does not accept a transform argument directly.
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int):
            img, label = self.dataset[idx]
            # img is already a tensor from MVTecDataset's default transform;
            # skip if transforms were applied at construction time.
            return img, label

    train_loader = DataLoader(
        train_dataset,
        batch_size=MVTEC_BATCH_SIZE,
        shuffle=True,
        num_workers=MVTEC_NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=MVTEC_BATCH_SIZE,
        shuffle=False,
        num_workers=MVTEC_NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples:     {len(test_dataset):,}")

    model = build_efficientnet(num_classes=MVTEC_NUM_CLASSES, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=EFFICIENTNET_LR,
        weight_decay=EFFICIENTNET_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, MVTEC_EPOCHS // 2),
        gamma=0.1,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, MVTEC_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{MVTEC_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MVTEC_CHECKPOINT)
            print(f"  Saved best checkpoint (val_acc={best_val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")

    # Final evaluation with AUROC.
    model.load_state_dict(torch.load(MVTEC_CHECKPOINT, map_location=device))
    model.eval()

    all_probs: list[float] = []
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final eval"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    auroc = roc_auc_score(all_labels, all_probs)
    print(f"\nAUROC: {auroc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomalous"]))


if __name__ == "__main__":
    main()
