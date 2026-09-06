"""Tests for LLaMA report parsing and summary formatting."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.report import (
    parse_llama_report,
    build_report,
    format_summary,
)


def test_parse_well_formed():
    text = (
        "Severity: high\n"
        "Likely cause: Freeze-thaw cycles opened the surface.\n"
        "Recommended action: Schedule a patch repair within two weeks."
    )
    parsed = parse_llama_report(text)
    assert parsed["severity"] == "high"
    assert parsed["likely_cause"].startswith("Freeze-thaw")
    assert "two weeks" in parsed["recommended_action"]


def test_parse_is_case_insensitive():
    parsed = parse_llama_report("SEVERITY: Medium\nLIKELY CAUSE: wear\nRECOMMENDED ACTION: monitor")
    assert parsed["severity"] == "medium"
    assert parsed["likely_cause"] == "wear"


def test_parse_missing_lines_fall_back():
    parsed = parse_llama_report("Severity: low")
    assert parsed["severity"] == "low"
    assert parsed["likely_cause"] == ""
    assert parsed["recommended_action"] == ""


def test_parse_garbage_gives_unknown():
    parsed = parse_llama_report("the model rambled without any structure at all")
    assert parsed["severity"] == "unknown"
    assert parsed["likely_cause"] == ""


def test_parse_invalid_severity_value_rejected():
    parsed = parse_llama_report("Severity: catastrophic\nLikely cause: x\nRecommended action: y")
    assert parsed["severity"] == "unknown"


def _report(defect_class, severity_text, idx):
    return build_report(
        patch_image=np.zeros((4, 4, 3), dtype=np.uint8),
        patch_idx=idx,
        patch_row=idx // 8,
        patch_col=idx % 8,
        defect_class=defect_class,
        confidence=0.9,
        llama_text=f"Severity: {severity_text}\nLikely cause: c\nRecommended action: a",
    )


def test_format_summary_orders_by_severity():
    reports = [_report(1, "low", 0), _report(2, "high", 1), _report(3, "medium", 2)]
    out = format_summary(reports)
    # class 2 = "Longitudinal crack" carries the "high" severity -> listed first,
    # ahead of the "low"-severity Pothole (class 1).
    assert out.index("Longitudinal crack") < out.index("Pothole")
    assert "3 defect(s) found" in out


def test_format_summary_empty():
    assert format_summary([]) == "No defects detected."
