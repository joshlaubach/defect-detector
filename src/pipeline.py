"""
pipeline.py

The single place the four inspection stages are wired together:

    DQN gate  ->  EfficientNet-B4  ->  LLaMA 3.2 Vision  ->  DefectReport list

Both the CLI (scripts/inspect.py) and the Streamlit dashboard import
`DefectPipeline` from here so there is exactly one implementation of the
data flow. Nothing in this module reads argparse or touches Streamlit.

Stage handoff:
    1. extract_patches() splits the frame into a GRID_SIZE x GRID_SIZE grid.
    2. The DQN Q-network does a greedy sequential rollout over patch actions
       plus a "stop" action, producing the list of patch indices to inspect.
       Without a usable checkpoint it falls back to inspecting every patch.
    3. EfficientNet-B4 classifies each selected patch (multi-label sigmoid).
    4. For every (patch, defect class) above threshold, LLaMA 3.2 Vision
       writes a three-line report, parsed into DefectReport fields.
    5. highlight_patches() draws the boxes; InspectionResult bundles it all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    GRID_SIZE,
    PATCH_SIZE,
    EFFICIENTNET_IMG_SIZE,
    RDD2022_NUM_CLASSES,
    RDD2022_DEFECT_THRESHOLD,
    RDD2022_CHECKPOINT,
    DQN_CHECKPOINT,
    DQN_MAX_INSPECT_BUDGET,
)
from src.models.efficientnet import load_checkpoint, decode_multilabel
from src.models.dqn import QNetwork
from src.models.report import DefectReport, build_report

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_FULL_TRANSFORM = transforms.Compose([
    transforms.Resize((EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Patches are already cropped to PATCH_SIZE by the pipeline before this runs,
# so there is no redundant Resize here (unlike the old dashboard transform).
_PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


@dataclass
class InspectionConfig:
    """Knobs for one pipeline run. Defaults come from config.py."""

    grid_size: int = GRID_SIZE
    defect_threshold: float = RDD2022_DEFECT_THRESHOLD
    use_dqn: bool = True
    use_llama: bool = True
    patch_budget: int = DQN_MAX_INSPECT_BUDGET
    device: str | None = None  # None -> auto (cuda if available)


@dataclass
class InspectionResult:
    """Everything one inspected frame produces."""

    reports: list[DefectReport]
    annotated_image: np.ndarray
    patches_inspected: int
    patches_total: int
    stopped_early: bool
    elapsed_s: float
    per_stage_s: dict[str, float] = field(default_factory=dict)


class DefectPipeline:
    """Loads the models once, then inspects frames on demand."""

    def __init__(self, cfg: InspectionConfig | None = None) -> None:
        self.cfg = cfg or InspectionConfig()
        self.device = torch.device(
            self.cfg.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._efficientnet_error: str | None = None
        self._dqn_error: str | None = None

        self.efficientnet = self._load_efficientnet()
        self.q_net = self._load_dqn() if (self.efficientnet and self.cfg.use_dqn) else None

        # Lazily imported so that a machine without the `ollama` package can
        # still run the classification-only path.
        self._llama = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_efficientnet(self) -> torch.nn.Module | None:
        if not RDD2022_CHECKPOINT.exists():
            self._efficientnet_error = f"checkpoint not found: {RDD2022_CHECKPOINT}"
            return None
        try:
            return load_checkpoint(
                num_classes=RDD2022_NUM_CLASSES,
                checkpoint_path=str(RDD2022_CHECKPOINT),
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via status()
            self._efficientnet_error = str(exc)
            return None

    def _feature_dim(self) -> int:
        dummy = torch.zeros(
            1, 3, EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE, device=self.device
        )
        with torch.no_grad():
            feats = self.efficientnet.forward_features(dummy)
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
        return feats.shape[1]

    def _load_dqn(self) -> QNetwork | None:
        if not DQN_CHECKPOINT.exists():
            self._dqn_error = f"checkpoint not found: {DQN_CHECKPOINT}"
            return None
        n_patches = self.cfg.grid_size * self.cfg.grid_size
        try:
            q_net = QNetwork(
                state_dim=self._feature_dim() + n_patches,
                n_actions=n_patches + 1,  # patches + stop
            ).to(self.device)
            q_net.load_state_dict(torch.load(DQN_CHECKPOINT, map_location=self.device))
            q_net.eval()
            return q_net
        except Exception as exc:  # noqa: BLE001
            # A checkpoint trained before the "stop" action has 64 outputs and
            # will not load here; that is expected until the agent is retrained.
            self._dqn_error = f"{exc} (retrain with scripts/train/train_dqn.py)"
            return None

    def model_status(self) -> dict[str, str]:
        """Human-readable availability of each stage, for the CLI and dashboard."""
        return {
            "efficientnet": "loaded" if self.efficientnet else f"unavailable ({self._efficientnet_error})",
            "dqn": (
                "loaded" if self.q_net
                else f"disabled ({self._dqn_error or 'use_dqn=False'}); inspecting all patches"
            ),
            "llama": "enabled" if self.cfg.use_llama else "disabled (classification only)",
            "device": str(self.device),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _image_features(self, pil_image: Image.Image) -> torch.Tensor:
        tensor = _FULL_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.efficientnet.forward_features(tensor)
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
        return feats.squeeze(0)

    def _select_patches(self, pil_image: Image.Image) -> tuple[list[int], bool]:
        """
        Return (patch_indices_to_inspect, stopped_early).

        With a trained Q-network this is a real greedy rollout: at each step we
        mask out already-inspected patches, take the argmax action, and halt
        when the agent chooses "stop" or the budget is hit. Without one, every
        patch is inspected in raster order.
        """
        n_patches = self.cfg.grid_size * self.cfg.grid_size
        if self.q_net is None:
            return list(range(n_patches)), False

        feature_vec = self._image_features(pil_image)
        mask = torch.zeros(n_patches, device=self.device)
        stop_action = n_patches
        selected: list[int] = []

        for _ in range(min(self.cfg.patch_budget, n_patches)):
            state = torch.cat([feature_vec, mask]).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state).squeeze(0)
            # Never re-pick an inspected patch.
            for idx in selected:
                q_values[idx] = float("-inf")
            action = int(q_values.argmax().item())
            if action == stop_action:
                return selected, True
            selected.append(action)
            mask[action] = 1.0

        return selected, len(selected) < n_patches

    def _classify(self, patch: np.ndarray) -> list[tuple[int, float]]:
        """(class_index, confidence) for every defect class over threshold."""
        tensor = _PATCH_TRANSFORM(Image.fromarray(patch)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.efficientnet(tensor).squeeze(0)
            probs = torch.sigmoid(logits)
        triggered = decode_multilabel(logits, threshold=self.cfg.defect_threshold)
        pairs = [(cls, float(probs[cls - 1].item())) for cls in triggered]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    def _narrative(self, patch: np.ndarray, defect_class: int, confidence: float,
                   row: int, col: int) -> str:
        if not self.cfg.use_llama:
            return ""
        if self._llama is None:
            from src.models import llama as _llama_mod
            self._llama = _llama_mod
        try:
            return self._llama.generate_report(
                patch_image=patch,
                defect_class=defect_class,
                confidence=confidence,
                patch_row=row,
                patch_col=col,
            )
        except Exception as exc:  # noqa: BLE001
            return f"Report unavailable: {exc}"

    def inspect(self, image: np.ndarray) -> InspectionResult:
        """Run the full pipeline on one RGB uint8 frame."""
        if self.efficientnet is None:
            raise RuntimeError(
                f"EfficientNet-B4 is not loaded: {self._efficientnet_error}"
            )

        from src.utils.patches import extract_patches, resize_patch, highlight_patches

        start = time.perf_counter()
        stage_s: dict[str, float] = {}

        np_image = np.ascontiguousarray(image)
        pil_image = Image.fromarray(np_image)
        patches = extract_patches(np_image, self.cfg.grid_size)

        t0 = time.perf_counter()
        selected, stopped_early = self._select_patches(pil_image)
        stage_s["dqn_gate"] = time.perf_counter() - t0

        reports: list[DefectReport] = []
        defect_indices: list[int] = []
        classify_s = 0.0
        llama_s = 0.0

        for patch_idx in selected:
            patch = resize_patch(patches[patch_idx], PATCH_SIZE)
            row = patch_idx // self.cfg.grid_size
            col = patch_idx % self.cfg.grid_size

            t0 = time.perf_counter()
            detections = self._classify(patch)
            classify_s += time.perf_counter() - t0
            if not detections:
                continue

            defect_indices.append(patch_idx)
            for defect_class, confidence in detections:
                t0 = time.perf_counter()
                text = self._narrative(patch, defect_class, confidence, row, col)
                llama_s += time.perf_counter() - t0
                reports.append(build_report(
                    patch_image=patch,
                    patch_idx=patch_idx,
                    patch_row=row,
                    patch_col=col,
                    defect_class=defect_class,
                    confidence=confidence,
                    llama_text=text,
                ))

        stage_s["efficientnet"] = classify_s
        stage_s["llama"] = llama_s

        annotated = highlight_patches(np_image, defect_indices, self.cfg.grid_size)

        return InspectionResult(
            reports=reports,
            annotated_image=annotated,
            patches_inspected=len(selected),
            patches_total=self.cfg.grid_size * self.cfg.grid_size,
            stopped_early=stopped_early,
            elapsed_s=time.perf_counter() - start,
            per_stage_s=stage_s,
        )
