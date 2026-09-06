"""
waymo_test.py

Qualitative out-of-distribution generalization test on Waymo Open Dataset frames.

The Waymo dataset has no defect labels, so this is not a quantitative evaluation.
The goal is to show that the EfficientNet-B4 model (trained on RDD2022) flags
reasonable regions in real AV camera footage without any domain adaptation.

For each image in the input directory:
    1. Split the image into an 8x8 grid of patches.
    2. Run EfficientNet-B4 on every patch.
    3. Highlight patches predicted as defective and save the annotated image.
    4. Print a per-image summary.

Results are written to --out_dir (default: waymo_results/).

Usage:
    python scripts/waymo_test.py --img_dir /path/to/waymo/frames
    python scripts/waymo_test.py --img_dir /path/to/waymo/frames --out_dir results/ --limit 20
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm

from config import (
    set_seeds, SEED,
    GRID_SIZE,
    PATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    RDD2022_NUM_CLASSES,
    RDD2022_DEFECT_THRESHOLD,
    RDD2022_CHECKPOINT,
)
from src.models.efficientnet import load_checkpoint
from src.utils.patches import extract_patches, resize_patch, highlight_patches
from src.models.report import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collect_images(img_dir: Path, limit: int | None) -> list[Path]:
    """Find image files in img_dir (non-recursive)."""
    paths = [
        p for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise RuntimeError(f"No images found in {img_dir}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def inspect_image(
    img_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Run patch-level inference on one image.

    Returns:
        annotated_image: The image with defective patches highlighted.
        defect_indices:  Flat patch indices that were flagged as defective.
        class_counts:    Dict mapping class index to count of flagged patches.
    """
    pil_image = Image.open(img_path).convert("RGB")
    np_image = np.array(pil_image)
    patches = extract_patches(np_image, GRID_SIZE)

    defect_indices: list[int] = []
    class_counts: dict[int, int] = defaultdict(int)

    tensors = []
    for patch in patches:
        resized = Image.fromarray(resize_patch(patch, PATCH_SIZE))
        tensors.append(_PATCH_TRANSFORM(resized))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = model(batch)
        triggered = (torch.sigmoid(logits) > RDD2022_DEFECT_THRESHOLD).cpu()

    # A patch can trigger more than one defect class at once (e.g. a
    # pothole next to a crack); count each one.
    for patch_idx in range(triggered.shape[0]):
        classes = torch.nonzero(triggered[patch_idx]).squeeze(1).tolist()
        if classes:
            defect_indices.append(patch_idx)
            for cls in classes:
                class_counts[cls + 1] += 1

    annotated = highlight_patches(np_image, defect_indices, GRID_SIZE)
    return annotated, defect_indices, dict(class_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Waymo generalization test")
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Directory containing Waymo camera frames (jpg/png).",
    )
    parser.add_argument(
        "--out_dir",
        default="waymo_results",
        help="Directory to save annotated output images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all).",
    )
    args = parser.parse_args()

    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading EfficientNet-B4 ...")
    model = load_checkpoint(
        num_classes=RDD2022_NUM_CLASSES,
        checkpoint_path=str(RDD2022_CHECKPOINT),
        device=device,
    )

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(img_dir, args.limit)
    print(f"Processing {len(image_paths)} images from {img_dir}")
    print(f"Output directory: {out_dir}\n")

    total_defect_patches = 0
    total_patches = GRID_SIZE * GRID_SIZE * len(image_paths)
    global_class_counts: dict[int, int] = defaultdict(int)
    flagged_frames = 0

    for img_path in tqdm(image_paths, desc="Inspecting"):
        annotated, defect_indices, class_counts = inspect_image(img_path, model, device)

        if defect_indices:
            flagged_frames += 1
            total_defect_patches += len(defect_indices)
            for cls, count in class_counts.items():
                global_class_counts[cls] += count

        out_path = out_dir / img_path.name
        Image.fromarray(annotated).save(out_path)

    # Summary.
    print("\n--- Waymo Generalization Test Results ---")
    print(f"Frames processed:     {len(image_paths)}")
    print(f"Frames with defects:  {flagged_frames} ({100 * flagged_frames / len(image_paths):.1f}%)")
    print(f"Patches inspected:    {total_patches:,}")
    print(f"Patches flagged:      {total_defect_patches:,} ({100 * total_defect_patches / total_patches:.1f}%)")

    if global_class_counts:
        print("\nFlagged patch breakdown:")
        for cls in sorted(global_class_counts):
            name = CLASS_NAMES.get(cls, f"Class {cls}")
            print(f"  {name}: {global_class_counts[cls]}")

    print(f"\nAnnotated images saved to {out_dir}/")


if __name__ == "__main__":
    main()
