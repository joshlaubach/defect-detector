"""
app.py

Streamlit dashboard for the AV defect detection pipeline. This is a thin UI
layer: all pipeline logic lives in src/pipeline.py so the dashboard and the
CLI (scripts/inspect.py) run exactly the same code.

Upload a road image, click "Run inspection", and the pipeline runs
DQN patch selection -> EfficientNet-B4 classification -> LLaMA 3.2 Vision
report generation, then shows the annotated frame and per-defect reports.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import DefectPipeline, InspectionConfig
from src.models.report import format_summary

st.set_page_config(page_title="AV Defect Inspector", layout="wide")


@st.cache_resource
def get_pipeline(use_dqn: bool, use_llama: bool) -> DefectPipeline:
    return DefectPipeline(InspectionConfig(use_dqn=use_dqn, use_llama=use_llama))


def main() -> None:
    st.title("AV Defect Inspector")
    st.write(
        "Upload a road image to run the full inspection pipeline: DQN patch "
        "selection, EfficientNet-B4 defect classification, and LLaMA 3.2 "
        "Vision report generation."
    )

    st.sidebar.header("Options")
    use_dqn = st.sidebar.checkbox("Use DQN patch gate", value=True)
    use_llama = st.sidebar.checkbox("Generate LLaMA reports", value=True)

    pipeline = get_pipeline(use_dqn, use_llama)

    st.sidebar.header("Model status")
    for stage, status in pipeline.model_status().items():
        st.sidebar.write(f"**{stage}:** {status}")

    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    pil_image = Image.open(uploaded).convert("RGB")
    st.image(pil_image, caption="Input image", use_column_width=True)

    if pipeline.efficientnet is None:
        st.error(f"EfficientNet-B4 unavailable: {pipeline.model_status()['efficientnet']}")
        return

    if not st.button("Run inspection"):
        return

    with st.spinner("Running pipeline ..."):
        result = pipeline.inspect(np.array(pil_image))

    st.subheader("Inspection result")
    st.image(result.annotated_image, caption="Defective patches highlighted", use_column_width=True)
    st.caption(
        f"Inspected {result.patches_inspected}/{result.patches_total} patches"
        f"{' (DQN stopped early)' if result.stopped_early else ''} "
        f"in {result.elapsed_s:.2f}s"
    )

    if not result.reports:
        st.success("No defects detected.")
        return

    st.subheader("Summary")
    st.text(format_summary(result.reports))

    st.subheader(f"Defect details ({len(result.reports)} found)")
    for i, report in enumerate(
        sorted(result.reports, key=lambda r: (-r.severity_rank, r.patch_idx)), start=1
    ):
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
