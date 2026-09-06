# Defect Detector

An offline post-drive inspection system for autonomous vehicles. After a drive completes,
the pipeline processes camera frames to find road surface defects and generates a
plain-English report describing each one.

The project combines three ML components: a fine-tuned CNN classifier, a reinforcement
learning agent that decides how many image regions to inspect, and a vision-language model
that writes the final report.

---

## Run it end to end

One command runs all four stages on an image or a folder of frames:

```bash
python scripts/run_inspection.py path/to/frame.jpg
python scripts/run_inspection.py path/to/frames/ --out out/ --limit 20
```

What happens, stage by stage:

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | **DQN gate** (`src/models/dqn.py`, `dqn_env.py`) | EfficientNet feature vector of the full frame + a mask of already-inspected patches | A sequence of patch indices to inspect. The agent runs a greedy rollout over 64 patch actions plus a **stop** action and halts when it chooses to stop or hits `DQN_MAX_INSPECT_BUDGET`. With no trained checkpoint it falls back to inspecting all 64 patches. |
| 2 | **EfficientNet-B4** (`src/models/efficientnet.py`) | One 380x380 patch crop | Multi-label sigmoid over 4 RDD2022 classes; `decode_multilabel()` returns `[(class_index, confidence), ...]` for every class above `RDD2022_DEFECT_THRESHOLD` (0.5). |
| 3 | **LLaMA 3.2 Vision** (`src/models/llama.py`, via Ollama) | The defect patch JPEG + defect class, confidence, grid location | A three-line text report (`Severity` / `Likely cause` / `Recommended action`). `report.py:parse_llama_report()` parses it into structured fields; unparseable lines fall back to `severity="unknown"` and the raw text is kept. |
| 4 | **Dashboard** (`dashboard/app.py`) or **CLI** (`scripts/run_inspection.py`) | The `DefectReport` list + annotated frame from `src/pipeline.py` | Streamlit: annotated image, severity-sorted report cards. CLI: `<name>_annotated.png` + `<name>.json` per frame. |

`src/pipeline.py` (`DefectPipeline`) owns this chain; the CLI and the dashboard both import it,
so there is exactly one implementation of the data flow.

### Flags

```bash
--no-dqn        inspect all 64 patches (skip the gate)
--no-llama      classification only, no narrative text
--benchmark     print per-stage timing
--limit N       cap frames in folder mode
```

---

## Architecture

```
Post-drive frames
       |
[1] DQN agent          adaptive patch selection: how many patches to inspect, and when to stop
       |
[2] EfficientNet-B4    patch defect classifier (defect type + confidence)
       |
[3] LLaMA 3.2 Vision   natural language defect report (severity, cause, action)
       |
[4] Streamlit / CLI    inspection output
```

---

## Setup

Pick the track that matches what you want to do.

### Track A: inference only (no datasets)

You need the two checkpoints in `checkpoints/` and, for narrative text, a running Ollama.
No dataset download.

```bash
pip install -r requirements.txt

# place efficientnet_rdd2022.pth and dqn.pth in checkpoints/

# for LLaMA reports:
# install Ollama from https://ollama.com, then:
ollama serve
ollama pull llama3.2-vision

python scripts/run_inspection.py your_frame.jpg
```

### Track B: full retrain

```bash
pip install -r requirements.txt

# 1. datasets
python data/download_rdd2022.py            # ~13.3 GB download
pip install kaggle && python data/download_mvtec.py   # ~5 GB, see Kaggle setup below

# 2. EfficientNet-B4 on RDD2022 (notebook, top to bottom)
#    -> checkpoints/efficientnet_rdd2022.pth
jupyter notebook notebooks/train_efficientnet.ipynb

# 3. EfficientNet-B4 on MVTec AD
python scripts/train/train_mvtec.py         # -> checkpoints/efficientnet_mvtec.pth

# 4. DQN gate (uses the frozen RDD2022 EfficientNet as its reward oracle)
python scripts/train/train_dqn.py           # -> checkpoints/dqn.pth
```

### Track C: dashboard

```bash
streamlit run dashboard/app.py
```

### Kaggle setup for MVTec

1. Create a free account at https://www.kaggle.com
2. Account -> Settings -> API -> Create New Token (downloads `kaggle.json`)
3. Place `kaggle.json` at `~/.kaggle/kaggle.json` (`C:\Users\<you>\.kaggle\kaggle.json` on Windows); `chmod 600` on Linux/Mac
4. Accept the dataset terms at https://www.kaggle.com/datasets/ipythonx/mvtec-ad

The RDD2022 downloader resumes on interrupt (HTTP Range requests) and skips countries that
are already extracted. Pass `--keep-archive` to retain the ZIP.

### Disk space

| Item | Size |
|------|------|
| RDD2022 archive (deleted after extract) | ~13.3 GB |
| RDD2022 extracted | ~15 GB |
| MVTec AD | ~5 GB |
| Checkpoints | ~80 MB |
| Inference only | ~80 MB (checkpoints only) |

---

## Datasets

| Dataset | Role | Size |
|---------|------|------|
| RDD2022 | Supervised training: road defects | 4 country splits, 4 defect classes (D00/D10/D20/D40) |
| MVTec AD | Supervised training: surface anomalies | 15 categories, binary normal/anomalous |
| Waymo Open Dataset | Unlabeled OOD generalization test | applied for separately; see `scripts/waymo_test.py` |

The official RDD2022 **test** split ships images with no Pascal VOC annotations, so
classification metrics are reported on the deterministic 15% validation slice held out
during training (`src/datasets/rdd2022.py:train_val_split`).

---

## Results

Numbers below come from the committed `checkpoints/efficientnet_rdd2022.pth`
(EfficientNet-B4, 39 epochs, early-stopped). Reproduce with the eval scripts.

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| EfficientNet-B4 | RDD2022 (val, 15% held-out) | macro-F1 | 0.73 |
| EfficientNet-B4 | RDD2022 (val) | per-class F1 | Pothole 0.69 / Longitudinal 0.65 / Transverse 0.84 / Alligator 0.73 |
| EfficientNet-B4 | RDD2022 (val) | no-defect agreement | 0.86 |
| EfficientNet-B4 | RDD2022 (val) | macro-AUROC | run `scripts/eval/evaluate_rdd2022.py` |
| EfficientNet-B4 | MVTec AD (test) | macro-AUROC | run `scripts/train/train_mvtec.py` then `scripts/eval/evaluate_mvtec.py` |
| DQN gate vs. random | RDD2022 (val) | patches to 80% recall | run `scripts/eval/evaluate_dqn.py` after retraining the gate |

There is a visible train/val gap (train F1 ~0.92 vs val ~0.73): the classifier overfits and
fires on a large fraction of patches at threshold 0.5. See `REVIEW.md` for the full read.

```bash
python scripts/eval/evaluate_rdd2022.py    # per-class P/R/F1 + AUROC on the val slice
python scripts/eval/evaluate_mvtec.py      # AUROC and F1 per category
python scripts/eval/evaluate_dqn.py        # DQN gate vs random: recall curve -> assets/
```

---

## Project structure

```
defect-detector/
    config.py                     hyperparameters, paths, seeds, set_seeds()
    requirements.txt
    data/
        download_rdd2022.py       resumable Figshare download
        download_mvtec.py         Kaggle download
    notebooks/
        train_efficientnet.ipynb  EfficientNet-B4 fine-tuning on RDD2022
    src/
        pipeline.py               DefectPipeline: the single stage chain
        datasets/
            rdd2022.py            RDD2022Dataset, train_val_split
            mvtec.py             MVTecDataset, build_combined_mvtec
        models/
            efficientnet.py       build/load, decode_multilabel, extract_features
            dqn.py                QNetwork, ReplayBuffer
            dqn_env.py            PatchInspectionEnv (stop action + recall reward)
            llama.py              generate_report, check_ollama_available
            report.py             DefectReport, build_report, parse_llama_report, format_summary
        utils/
            patches.py            extract_patches, resize_patch, highlight_patches
    scripts/
        run_inspection.py         PRODUCTION: full pipeline on image / folder
        waymo_test.py             diagnostic: EfficientNet-only OOD test
        train/
            train_mvtec.py
            train_dqn.py
        eval/
            evaluate_rdd2022.py
            evaluate_mvtec.py
            evaluate_dqn.py
    dashboard/
        app.py                    Streamlit UI over DefectPipeline
    tests/
        test_report.py
        test_dqn_gate.py
    checkpoints/                  git-ignored
```

---

## Tests

```bash
python -m pytest
```

Covers `parse_llama_report` (well-formed, partial, garbage input) and the DQN gate's
stop / budget / reward logic. The heavier stages are exercised by
`scripts/run_inspection.py` end to end.

---

## Key design decisions

**Why EfficientNet-B4?** Compound scaling (depth + width + resolution) gives better accuracy
than ResNet-50 at a similar parameter count, and the 380x380 native input suits fine-grained
defect patterns like hairline cracks.

**Why a DQN gate?** The action space is small and discrete (64 patches + stop), which is
where DQN works well, and being off-policy it can reuse replay-buffer experience. The agent's
job is framed as "reach high recall while inspecting as few patches as possible" via a
terminal reward proportional to recall minus a penalty per missed defect.

**Why LLaMA 3.2 Vision?** A vision-language model sees the actual defect crop rather than
just a class label, so its reports are grounded in what the defect looks like. It runs
locally via Ollama with greedy decoding and a fixed seed for reproducibility; there are no
API dependencies.
