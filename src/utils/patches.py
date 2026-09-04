"""
patches.py

Functions for splitting an image into a grid of patches and drawing
highlights back onto the original frame.

The DQN agent works on a flat list of patch indices (0 to GRID_SIZE^2 - 1).
These helpers translate between that index space and actual pixel crops.
"""

import numpy as np
from PIL import Image, ImageDraw

from config import GRID_SIZE, PATCH_SIZE


def extract_patches(image: np.ndarray, grid_size: int = GRID_SIZE) -> list[np.ndarray]:
    """
    Split an image into a grid_size x grid_size grid of patches.

    Patches are returned in row-major order: patch 0 is top-left,
    patch 1 is one cell to the right, and so on.

    Args:
        image:     H x W x C numpy array (uint8 RGB).
        grid_size: Number of cells along each axis.

    Returns:
        List of (H/grid_size) x (W/grid_size) x C patches as numpy arrays.
    """
    h, w = image.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    patches: list[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * cell_h
            y1 = y0 + cell_h
            x0 = col * cell_w
            x1 = x0 + cell_w
            patches.append(image[y0:y1, x0:x1])
    return patches


def patch_index_to_coords(
    idx: int, image_shape: tuple[int, int], grid_size: int = GRID_SIZE
) -> tuple[int, int, int, int]:
    """
    Convert a flat patch index into pixel bounding box coordinates.

    Args:
        idx:         Flat patch index in [0, grid_size^2).
        image_shape: (H, W) of the full image.
        grid_size:   Number of grid cells per axis.

    Returns:
        (x0, y0, x1, y1) pixel coordinates of the patch.
    """
    h, w = image_shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    row = idx // grid_size
    col = idx % grid_size
    y0 = row * cell_h
    x0 = col * cell_w
    return x0, y0, x0 + cell_w, y0 + cell_h


def resize_patch(patch: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray:
    """
    Resize a patch to size x size pixels (the classifier's expected input).

    Args:
        patch: H x W x C numpy array.
        size:  Target side length in pixels.

    Returns:
        size x size x C numpy array (uint8).
    """
    pil_img = Image.fromarray(patch)
    pil_img = pil_img.resize((size, size), Image.BILINEAR)
    return np.array(pil_img)


def highlight_patches(
    image: np.ndarray,
    defect_indices: list[int],
    grid_size: int = GRID_SIZE,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
) -> np.ndarray:
    """
    Draw colored bounding boxes on the image for each defective patch.

    Args:
        image:          H x W x C numpy array (uint8 RGB).
        defect_indices: Flat patch indices that contain defects.
        grid_size:      Number of grid cells per axis.
        color:          RGB tuple for the bounding box color.
        line_width:     Thickness of the drawn rectangle border.

    Returns:
        A copy of the image with boxes drawn on it.
    """
    pil_img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(pil_img)
    h, w = image.shape[:2]
    for idx in defect_indices:
        x0, y0, x1, y1 = patch_index_to_coords(idx, (h, w), grid_size)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
    return np.array(pil_img)
