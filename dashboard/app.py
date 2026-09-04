"""
app.py

Streamlit dashboard for the AV defect detection pipeline.

Upload a road image, then click "Run inspection". The pipeline:
    1. Splits the image into a 8x8 grid of patches (64 total).
    2. Uses the DQN agent to rank patches by inspection priority.
    3. Runs EfficientNet-B4 on each patch to detect defects.
    4. Calls LLaMA 3.2 Vision (via Ollama) for each defective patch
       to generate a natural-language inspection report.
    5. Displays the annotated image and all reports.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Allow imports from the project root when running as `streamlit run dashboard/app.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from config import (
    GRID_SIZE,
    PATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    RDD2022_NUM_CLASSES,
    RDD2022_CHECKPOINT,
    DQN_CHECKPOINT,
)
from src.models.efficientnet import load_checkpoint, extract_features
from src.models.dqn import QNetwork
from src.models.llama import generate_report, check_ollama_available
from src.models.report import build_report, format_summary, DefectReport, CLASS_NAMES
from src.utils.patches import extract_patches, resize_patch, highlight_patches


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AV Defect Inspector",
    page_icon=None,
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_FULL_TRANSFORM = transforms.Compose([
    transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_efficientnet():
    """Load EfficientNet-B4 from the RDD2022 checkpoint."""
    if not RDD2022_CHECKPOINT.exists():
        return None
    try:
        model = load_checkpoint(
            num_classes=RDD2022_NUM_CLASSES,
            checkpoint_path=str(RDD2022_CHECKPOINT),
            device=DEVICE,
        )
        return model
    except Exception:
        return None


@st.cache_resource
def load_dqn(state_dim: int) -> QNetwork | None:
    """Load the trained DQN Q-network."""
    if not DQN_CHECKPOINT.exists():
        return None
    try:
        n_actions = GRID_SIZE * GRID_SIZE
        q_net = QNetwork(state_dim=state_dim, n_actions=n_actions).to(DEVICE)
        q_net.load_state_dict(torch.load(DQN_CHECKPOINT, map_location=DEVICE))
        q_net.eval()
        return q_net
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def get_patch_order(
    pil_image: Image.Image,
    efficientnet: torch.nn.Module,
    q_net: QNetwork | None,
) -> list[int]:
    """
    Return patch indices in the order the agent wants to inspect them.

    If the DQN is available, sort patches by their initial Q-values (highest
    first). Otherwise, return patches in raster scan order (0 to 63).
    """
    if q_net is None:
        return list(range(GRID_SIZE * GRID_SIZE))

    image_tensor = _FULL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = efficientnet.forward_features(image_tensor)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
    feature_vec = features.squeeze(0)

    n_patches = GRID_SIZE * GRID_SIZE
    mask = torch.zeros(n_patches, device=DEVICE)
    state = torch.cat([feature_vec, mask]).unsqueeze(0)

    with torch.no_grad():
        q_values = q_net(state).squeeze(0)

    return q_values.argsort(descending=True).cpu().tolist()


def classify_patch(
    patch: np.ndarray,
    efficientnet: torch.nn.Module,
) -> tuple[int, float]:
    """
    Run EfficientNet on one patch and return (class_index, confidence).
    """
    resized = Image.fromarray(resize_patch(patch, PATCH_SIZE))
    tensor = _PATCH_TRANSFORM(resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = efficientnet(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    predicted_class = int(probs.argmax().item())
    confidence = float(probs[predicted_class].item())
    return predicted_class, confidence


def run_inspection(
    pil_image: Image.Image,
    efficientnet: torch.nn.Module,
    q_net: QNetwork | None,
    use_llama: bool,
    progress_bar,
) -> tuple[list[DefectReport], np.ndarray]:
    """
    Run the full inspection pipeline on one image.

    Returns a list of DefectReport objects and the annotated image.
    """
    np_image = np.array(pil_image.convert("RGB"))
    patches = extract_patches(np_image, GRID_SIZE)
    patch_order = get_patch_order(pil_image, efficientnet, q_net)

    reports: list[DefectReport] = []
    defect_indices: list[int] = []
    n_patches = len(patches)

    for step, patch_idx in enumerate(patch_order):
        progress_bar.progress((step + 1) / n_patches, text=f"Inspecting patch {step + 1}/{n_patches}")

        patch = patches[patch_idx]
        row = patch_idx // GRID_SIZE
        col = patch_idx % GRID_SIZE

        defect_class, confidence = classify_patch(patch, efficientnet)

        if defect_class == 0:
            continue

        defect_indices.append(patch_idx)

        llama_text = ""
        if use_llama:
            try:
                llama_text = generate_report(
                    patch_image=resize_patch(patch, PATCH_SIZE),
                    defect_class=defect_class,
                    confidence=confidence,
                    patch_row=row,
                    patch_col=col,
                )
            except Exception as e:
                llama_text = f"Report unavailable: {e}"

        report = build_report(
            patch_image=resize_patch(patch, PATCH_SIZE),
            patch_idx=patch_idx,
            patch_row=row,
            patch_col=col,
            defect_class=defect_class,
            confidence=confidence,
            llama_text=llama_text,
        )
        reports.append(report)

    annotated = highlight_patches(np_image, defect_indices, GRID_SIZE)
    return reports, annotated


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("AV Defect Inspector")
    st.write(
        "Upload a road image to run the full inspection pipeline: "
        "DQN patch selection, EfficientNet defect classification, "
        "and LLaMA 3.2 Vision report generation."
    )

    # Sidebar: model status.
    st.sidebar.header("Model Status")

    efficientnet = load_efficientnet()
    en_ok = efficientnet is not None
    st.sidebar.write("EfficientNet-B4:", "loaded" if en_ok else "not found (train first)")

    if en_ok:
        dummy = torch.zeros(1, 3, EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE, device=DEVICE)
        with torch.no_grad():
            f = efficientnet.forward_features(dummy)
            if f.dim() == 4:
                f = f.mean(dim=[2, 3])
        state_dim = f.shape[1] + GRID_SIZE * GRID_SIZE
        q_net = load_dqn(state_dim)
    else:
        q_net = None

    dqn_ok = q_net is not None
    st.sidebar.write("DQN agent:", "loaded" if dqn_ok else "not found (train first)")
    if not dqn_ok:
        st.sidebar.write("(falling back to raster scan order)")

    ollama_ok = check_ollama_available()
    st.sidebar.write("LLaMA 3.2 Vision:", "available" if ollama_ok else "not available")
    if not ollama_ok:
        st.sidebar.write("(run: ollama serve && ollama pull llama3.2-vision)")

    st.sidebar.write(f"Device: {DEVICE}")

    # Image upload.
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    pil_image = Image.open(uploaded).convert("RGB")
    st.image(pil_image, caption="Input image", use_column_width=True)

    if not en_ok:
        st.error("EfficientNet-B4 checkpoint not found. Train the model first.")
        return

    if st.button("Run inspection"):
        progress_bar = st.progress(0, text="Starting ...")

        with st.spinner("Running pipeline ..."):
            reports, annotated_image = run_inspection(
                pil_image=pil_image,
                efficientnet=efficientnet,
                q_net=q_net,
                use_llama=ollama_ok,
                progress_bar=progress_bar,
            )

        progress_bar.empty()

        # Show annotated image.
        st.subheader("Inspection result")
        st.image(annotated_image, caption="Defective patches highlighted in red", use_column_width=True)

        if not reports:
            st.success("No defects detected.")
            return

        # Summary text.
        st.subheader("Summary")
        st.text(format_summary(reports))

        # Per-defect details.
        st.subheader(f"Defect details ({len(reports)} found)")
        sorted_reports = sorted(reports, key=lambda r: (-r.severity_rank, r.patch_idx))

        for i, report in enumerate(sorted_reports, start=1):
            label = (
                f"[{i}] {report.class_name} at patch ({report.patch_row}, {report.patch_col})  "
                f"confidence={report.confidence:.0%}  severity={report.severity}"
            )
            with st.expander(label):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(report.patch_image, caption="Patch", use_column_width=True)
                with col2:
                    if report.likely_cause:
                        st.write(f"**Likely cause:** {report.likely_cause}")
                    if report.recommended_action:
                        st.write(f"**Recommended action:** {report.recommended_action}")
                    if report.llama_text:
                        st.caption("Raw model output:")
                        st.code(report.llama_text, language=None)


if __name__ == "__main__":
    main()
