"""
llama.py

Generates plain-English defect reports using LLaMA 3.2 Vision via Ollama.

No fine-tuning is needed. The model receives the defect patch image along
with structured metadata (defect class, confidence, grid location) and
produces a short structured report covering severity, likely cause, and
a recommended action.

Ollama must be running locally before calling generate_report():
    ollama serve
    ollama pull llama3.2-vision
"""

import io
from typing import Any

import numpy as np
import ollama
from PIL import Image

from config import LLAMA_MODEL, OLLAMA_HOST, SEED
from src.models.report import CLASS_NAMES as _REPORT_CLASS_NAMES


# Single source of truth for class names lives in report.py. LLaMA prompts
# read better lowercase, so lowercase a copy here without duplicating the map.
CLASS_NAMES: dict[int, str] = {
    idx: name.lower() for idx, name in _REPORT_CLASS_NAMES.items() if idx != 0
}

_PROMPT_TEMPLATE = """\
You are an automotive road inspection assistant. Your job is to analyze a \
defect image captured by an autonomous vehicle and write a brief inspection report.

Defect type: {defect_class}
Detection confidence: {confidence:.0%}
Image location: row {row}, column {col} (in an 8x8 patch grid)

Write a short report with exactly three lines:
Severity: <low / medium / high>
Likely cause: <one sentence>
Recommended action: <one sentence>

Be direct and specific. Do not add any extra commentary."""


def _image_to_bytes(image: np.ndarray) -> bytes:
    """Convert a numpy RGB array to JPEG bytes."""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def generate_report(
    patch_image: np.ndarray,
    defect_class: int,
    confidence: float,
    patch_row: int,
    patch_col: int,
) -> str:
    """
    Send a defect patch to LLaMA 3.2 Vision and return the inspection report.

    Args:
        patch_image:  H x W x 3 numpy array (uint8 RGB) of the defective patch.
        defect_class: Integer class label (1-5). Must not be 0 (background).
        confidence:   Model confidence for the predicted class (0 to 1).
        patch_row:    Row index of the patch in the inspection grid (0-indexed).
        patch_col:    Column index of the patch in the inspection grid (0-indexed).

    Returns:
        A three-line report string from the model.

    Raises:
        ValueError:   If defect_class is 0 (background patches should not be reported).
        RuntimeError: If Ollama is not reachable or the model is not available.
    """
    if defect_class == 0:
        raise ValueError("generate_report() should not be called for background patches.")

    class_name = CLASS_NAMES.get(defect_class, f"class {defect_class}")
    prompt = _PROMPT_TEMPLATE.format(
        defect_class=class_name,
        confidence=confidence,
        row=patch_row,
        col=patch_col,
    )

    image_bytes = _image_to_bytes(patch_image)

    client = ollama.Client(host=OLLAMA_HOST)
    response: Any = client.chat(
        model=LLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }
        ],
        # Greedy decoding with a fixed seed so the same patch produces the
        # same report across runs. The Ollama runtime is still not bit-exact,
        # but this removes sampling as a source of variation.
        options={"seed": SEED, "temperature": 0.0},
    )

    return response["message"]["content"].strip()


def check_ollama_available() -> bool:
    """
    Return True if Ollama is running and the vision model is available.

    Useful for a quick health check before starting the Streamlit dashboard.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()
        # ollama<0.4 returns dicts with a "name" key; ollama>=0.4 returns
        # objects with a .model attribute. Handle both so a client upgrade
        # doesn't silently disable narrative reports.
        entries = getattr(response, "models", None) or response.get("models", [])
        names = []
        for entry in entries:
            name = (
                getattr(entry, "model", None)
                or getattr(entry, "name", None)
                or (entry.get("model") or entry.get("name") if isinstance(entry, dict) else None)
            )
            if name:
                names.append(name)
        return any(LLAMA_MODEL in name for name in names)
    except Exception:
        return False
