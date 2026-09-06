# Pipeline review

Answers to a reviewer's ten questions about running `defect-detector` end to end.
Each item states what was true before this pass and what changed.

Verdict on the original state: the ML components were individually trained and sound,
but there was no way to run them together outside the single-image Streamlit app, the
results table was empty, and the DQN — the headline RL component — had no effect at
inference. Those were fair reasons to hesitate. The changes below address each one.

---

## 1. Single entry point

**Before:** none. The four stages were only ever chained inside
`dashboard/app.py:run_inspection()`. `scripts/waymo_test.py` was a second, partial
path (EfficientNet only). Running the pipeline headless was not possible.

**After:** `python scripts/run_inspection.py <image|dir>` runs all four stages and
writes an annotated PNG + a JSON report per frame. `src/pipeline.py:DefectPipeline`
holds the chain; the CLI and the dashboard both import it, so there is one
implementation. Named `run_inspection.py` (not `inspect.py`) so it does not shadow the
stdlib `inspect` module.

## 2. Data flow between stages

```
frame ─ extract_patches ─► 64 patch crops
      │
DQN gate (src/pipeline.py:_select_patches)
      │   greedy rollout over [0..63] + stop action; state = [img_features, inspected_mask]
      │   halts on "stop" or DQN_MAX_INSPECT_BUDGET; no checkpoint ⇒ all 64 in raster order
      ▼
selected patch indices
      │
EfficientNet-B4 (_classify)  ─►  decode_multilabel(logits) ─► [(class_idx 1..4, confidence), ...]
      │                            multi-label sigmoid > RDD2022_DEFECT_THRESHOLD (0.5)
      ▼
per (patch, class) detection
      │
LLaMA 3.2 Vision (_narrative, via Ollama)  ─►  3-line text
      │   "Severity: .. / Likely cause: .. / Recommended action: .."
      │   parse_llama_report() → structured fields; unparsed ⇒ severity="unknown", raw text kept
      ▼
DefectReport  ─►  InspectionResult{reports, annotated_image, patches_inspected, per_stage_s, ...}
```

- **DQN output format:** a Python `list[int]` of patch indices plus a `stopped_early`
  flag. Before this pass the agent returned *all 64* indices merely re-sorted, computed
  once from the empty-mask state — so it never actually selected a subset and never made
  a sequential decision. It now does a real step-by-step rollout and can stop early.
- **EfficientNet output:** a list of `(class_index, confidence)` tuples per patch, not a
  bare tensor. Multi-label, so a patch can carry more than one class.
- **LLaMA output:** free text that is *expected* to be three labelled lines. It is not
  enforced JSON; `parse_llama_report()` is a tolerant regex parser with fallbacks.
- **Streamlit input:** single image upload (`jpg/jpeg/png`). No video. The CLI adds
  folder-of-frames mode. Video decoding was deliberately left out.

## 3. Model loading

- **EfficientNet-B4 + DQN:** loaded once when `DefectPipeline` is constructed (the
  dashboard wraps that in `@st.cache_resource`). Checkpoint paths are in `config.py`
  (`RDD2022_CHECKPOINT`, `DQN_CHECKPOINT`).
- **LLaMA:** nothing loads in-process; it is a remote call to Ollama on
  `OLLAMA_HOST` (`http://localhost:11434`).
- **Missing checkpoint:** EfficientNet missing ⇒ `DefectPipeline.model_status()` reports
  it and `inspect()` raises a clear `RuntimeError`; the dashboard blocks the run. DQN
  missing *or incompatible* ⇒ logged as "disabled … inspecting all patches" and the
  pipeline runs without the gate. Ollama down ⇒ `check_ollama_available()` is false and
  each report is `"Report unavailable: <error>"`, classification still completes.
- The committed `dqn.pth` was trained before the stop action (64 outputs) and will not
  load into the new 65-output network — this is expected and handled by the fallback
  until the gate is retrained.

## 4. Configuration

Genuinely centralized in `config.py`: seeds, grid size, patch size, all model
hyperparameters, thresholds, and every path. Nothing downstream hardcodes a value that
lives there. New constants added this pass: `DQN_STOP_BONUS`, `DQN_MISS_PENALTY`,
`DQN_MAX_INSPECT_BUDGET`, plus cuDNN determinism in `set_seeds()` and a
`seeded_generator()` helper. The duplicated `COUNTRIES = [...]` lists in the scripts now
import from `src/datasets/rdd2022.py`.

## 5. Inference time

Measured on an RTX 5080, one 640x640 RDD2022 frame, `--no-llama --no-dqn`:

| Stage | Time |
|-------|------|
| DQN gate | ~0 (disabled / trivial rollout) |
| EfficientNet-B4, 64 patches | ~1.9 s |
| LLaMA 3.2 Vision | not measured here; ~2–5 s **per defect patch** on Ollama |
| **Total, classification only** | **~2 s / frame** |

With LLaMA enabled and a trigger-happy classifier (31 of 64 patches flagged on the
sample frame), a frame can take 1–2 minutes end to end. This is a **batch / post-drive**
tool, not real-time — which matches the framing. The DQN gate exists partly to cut the
number of LLaMA calls. EfficientNet currently runs one patch per forward pass; batching
the 64 patches is an easy future speed-up.

Reproduce timings: `python scripts/run_inspection.py <frame> --benchmark`.

## 6. Example output

`python scripts/run_inspection.py data/raw/rdd2022/Czech/test/images/Czech_000016.jpg --no-llama`

```
=== Czech_000016.jpg ===
  inspected 64/64 patches  |  31 defect(s)  |  1.95s
Inspection complete. 31 defect(s) found.
  High: 0  Medium: 0  Low: 0      (severity is "unknown" without LLaMA)
```

`Czech_000016.json` (one report, trimmed):

```json
{
  "patch_idx": 3, "patch_row": 0, "patch_col": 3,
  "defect_class": 4, "class_name": "Alligator crack",
  "confidence": 0.731,
  "llama_text": "", "severity": "unknown",
  "likely_cause": "", "recommended_action": ""
}
```

With Ollama running, `llama_text` holds the three-line report and `severity` /
`likely_cause` / `recommended_action` are filled. The dashboard shows the annotated
frame, a severity-sorted summary, and one expandable card per defect with the patch
crop and parsed fields.

Note the 31/64 hit rate: the classifier fires on roughly half the patches at
threshold 0.5. That is a model-calibration issue (see item 10), not a pipeline bug.

## 7. Failure modes

| Situation | Behavior |
|-----------|----------|
| DQN selects no patches | Cannot happen: the rollout always inspects ≥1 patch before "stop" is available; with no checkpoint it inspects all 64. |
| Classifier flags nothing | `reports == []`, CLI prints "No defects detected", dashboard shows a success message. |
| LLaMA output unparseable | `parse_llama_report()` returns `severity="unknown"` and empty cause/action; the raw text is still attached to the `DefectReport`. |
| Ollama not running | `check_ollama_available()` false; `_narrative()` returns `"Report unavailable: <error>"`; classification and annotation still succeed. |
| Ollama client upgraded | `check_ollama_available()` now handles both the old dict (`m["name"]`) and new object (`.model`) shapes. |
| EfficientNet checkpoint missing/corrupt | `model_status()` reports it; `inspect()` raises `RuntimeError` with the path. |

## 8. Dataset setup friction

- **RDD2022 resume:** yes, real. `download_rdd2022.py:download_file()` issues a HEAD
  request, checks `Accept-Ranges: bytes`, and resumes with a `Range` header; if the
  server won't do ranges it restarts and says so. Size is verified against
  `Content-Length` afterward.
- **Kaggle setup:** documented step by step in the README ("Kaggle setup for MVTec") and
  in the script docstring — account, API token, `~/.kaggle/kaggle.json`, accept terms.
- **Disk:** RDD2022 ~13.3 GB download / ~15 GB extracted, MVTec ~5 GB, checkpoints
  ~80 MB. The downloader now deletes the archive and any stray `RDD2022/` staging tree
  from older runs (`--keep-archive` to keep the ZIP); a full checkout was sitting at
  ~37 GB because of that leftover.
- **Skip datasets for inference:** yes. `scripts/run_inspection.py` and the dashboard
  need only the two checkpoints. README "Track A" documents this path.

## 9. Training vs. evaluation vs. production

Scripts are now split by purpose:

| Purpose | Location |
|---------|----------|
| One-time training | `notebooks/train_efficientnet.ipynb`, `scripts/train/train_mvtec.py`, `scripts/train/train_dqn.py` |
| Evaluation | `scripts/eval/evaluate_rdd2022.py`, `evaluate_mvtec.py`, `evaluate_dqn.py` |
| Production | `scripts/run_inspection.py`, `src/pipeline.py`, `dashboard/app.py` |
| Diagnostic | `scripts/waymo_test.py` (OOD spot-check, EfficientNet only) |

Every script also got a `sys.path` bootstrap — previously `python scripts/train_dqn.py`
from the repo root failed at `from config import ...` because the repo root was not on
the path.

## 10. Reproducibility

- `SEED = 42` in `config.py`; `set_seeds()` covers Python / NumPy / torch / CUDA and now
  also sets `cudnn.deterministic = True`, `benchmark = False`.
- The RDD2022 train/val split is no longer notebook-only: `train_val_split()` in
  `src/datasets/rdd2022.py` reproduces the exact deterministic 15% hold-out
  (`torch.randperm` with a seeded generator), and `evaluate_rdd2022.py` uses it — so the
  eval script scores the same images the notebook trained against.
- LLaMA calls now pass `options={"seed": SEED, "temperature": 0.0}`. The Ollama runtime
  is still not bit-exact, but sampling is removed as a source of variation.
- Classification outputs (classes, confidences, annotated pixels) are deterministic
  run to run. Narrative prose is near-deterministic.

---

## Model quality caveat

The RDD2022 classifier reached **val macro-F1 ≈ 0.73** (per-class: Pothole 0.69,
Longitudinal 0.65, Transverse 0.84, Alligator 0.73) with a large train/val gap
(train F1 ≈ 0.92). It over-predicts: ~half the patches on a typical frame clear the 0.5
threshold. Options for a follow-up: stronger regularization / more augmentation, a
higher decision threshold tuned on the val set, or per-class thresholds. The
`evaluate_rdd2022.py` AUROC numbers (run to fill the table) will show whether the
ranking is better than the thresholded F1 suggests.

The DQN gate's reward oracle is the same EfficientNet it feeds, so the agent learns to
predict where that classifier fires rather than where ground-truth defects are. With a
noisy, over-firing classifier that signal is weak. The gate is still useful as a
cost-control mechanism (fewer LLaMA calls), and `evaluate_dqn.py` quantifies the
recall/latency trade-off against a random-order baseline once the agent is retrained
with the new stop action.
