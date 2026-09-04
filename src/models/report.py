"""
report.py

Data structures and formatting helpers for defect inspection reports.

The pipeline produces one DefectReport per defective patch. parse_llama_report()
turns the raw three-line LLaMA output into structured fields. format_summary()
produces a human-readable text block suitable for display in the dashboard or
written to a log file.
"""

import re
from dataclasses import dataclass, field

import numpy as np


# Severity ranking used when sorting or color-coding reports.
SEVERITY_RANK: dict[str, int] = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "unknown": 0,
}

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Pothole",
    2: "Longitudinal crack",
    3: "Transverse crack",
    4: "Alligator crack",
    5: "Surface anomaly",
}


@dataclass
class DefectReport:
    """
    All information about one defective patch, including the LLaMA report.

    Fields:
        patch_image:        H x W x 3 numpy array (uint8 RGB) of the patch.
        patch_idx:          Flat patch index in [0, GRID_SIZE^2).
        patch_row:          Row in the inspection grid (0-indexed).
        patch_col:          Column in the inspection grid (0-indexed).
        defect_class:       Integer class label from EfficientNet.
        class_name:         Human-readable class name.
        confidence:         EfficientNet softmax confidence for the predicted class.
        llama_text:         Raw three-line text returned by LLaMA.
        severity:           Parsed severity from llama_text (low / medium / high).
        likely_cause:       Parsed cause sentence from llama_text.
        recommended_action: Parsed action sentence from llama_text.
    """

    patch_image: np.ndarray
    patch_idx: int
    patch_row: int
    patch_col: int
    defect_class: int
    class_name: str
    confidence: float
    llama_text: str
    severity: str = field(default="unknown")
    likely_cause: str = field(default="")
    recommended_action: str = field(default="")

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity.lower(), 0)


def parse_llama_report(text: str) -> dict[str, str]:
    """
    Parse the structured three-line LLaMA output into a dict.

    Expected format (any capitalization):
        Severity: low
        Likely cause: <sentence>
        Recommended action: <sentence>

    Returns a dict with keys "severity", "likely_cause", "recommended_action".
    Falls back to empty strings for lines that do not match.
    """
    result = {"severity": "unknown", "likely_cause": "", "recommended_action": ""}

    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()

        if lower.startswith("severity:"):
            raw = re.sub(r"(?i)^severity:\s*", "", line).strip().lower()
            if raw in SEVERITY_RANK:
                result["severity"] = raw

        elif lower.startswith("likely cause:"):
            result["likely_cause"] = re.sub(r"(?i)^likely cause:\s*", "", line).strip()

        elif lower.startswith("recommended action:"):
            result["recommended_action"] = re.sub(
                r"(?i)^recommended action:\s*", "", line
            ).strip()

    return result


def build_report(
    patch_image: np.ndarray,
    patch_idx: int,
    patch_row: int,
    patch_col: int,
    defect_class: int,
    confidence: float,
    llama_text: str,
) -> DefectReport:
    """
    Construct a DefectReport by combining raw fields with parsed LLaMA output.

    Args:
        patch_image:  Raw patch as a numpy array.
        patch_idx:    Flat patch index.
        patch_row:    Row in the grid.
        patch_col:    Column in the grid.
        defect_class: EfficientNet class prediction.
        confidence:   EfficientNet softmax confidence.
        llama_text:   Raw LLaMA output string.

    Returns:
        A fully populated DefectReport.
    """
    parsed = parse_llama_report(llama_text)
    return DefectReport(
        patch_image=patch_image,
        patch_idx=patch_idx,
        patch_row=patch_row,
        patch_col=patch_col,
        defect_class=defect_class,
        class_name=CLASS_NAMES.get(defect_class, f"Class {defect_class}"),
        confidence=confidence,
        llama_text=llama_text,
        severity=parsed["severity"],
        likely_cause=parsed["likely_cause"],
        recommended_action=parsed["recommended_action"],
    )


def format_summary(reports: list[DefectReport]) -> str:
    """
    Format a list of DefectReport objects into a readable text summary.

    Reports are sorted by severity (high first), then by patch index.

    Args:
        reports: List of DefectReport objects from one inspection run.

    Returns:
        A multi-line string suitable for printing or displaying in the dashboard.
    """
    if not reports:
        return "No defects detected."

    sorted_reports = sorted(
        reports,
        key=lambda r: (-r.severity_rank, r.patch_idx),
    )

    severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for r in reports:
        key = r.severity.lower() if r.severity.lower() in severity_counts else "unknown"
        severity_counts[key] += 1

    lines = [
        f"Inspection complete. {len(reports)} defect(s) found.",
        f"  High: {severity_counts['high']}  "
        f"Medium: {severity_counts['medium']}  "
        f"Low: {severity_counts['low']}",
        "",
    ]

    for i, report in enumerate(sorted_reports, start=1):
        lines.append(
            f"[{i}] {report.class_name} at patch ({report.patch_row}, {report.patch_col})  "
            f"confidence={report.confidence:.0%}  severity={report.severity}"
        )
        if report.likely_cause:
            lines.append(f"    Cause:  {report.likely_cause}")
        if report.recommended_action:
            lines.append(f"    Action: {report.recommended_action}")
        lines.append("")

    return "\n".join(lines).rstrip()
