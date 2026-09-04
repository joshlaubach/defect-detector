# Defect Detector

An offline post-drive inspection system for autonomous vehicles. After a drive completes,
the pipeline processes camera footage to find road surface defects and generates a
plain-English report describing each one.

The project combines three ML components: a fine-tuned CNN classifier, a reinforcement
learning agent that decides which image regions to inspect, and a vision-language model
that writes the final report.

---

## Demo

![Dashboard demo](assets/demo.gif)

---

## Architecture

```
Post-drive footage
       |
[1] DQN agent          adaptive patch selection (which patches to inspect?)
       |
[2] EfficientNet-B4    patch defect classifier (defect type + confidence)
       |
[3] LLaMA 3.2 Vision   natural language defect report (severity, cause, action)
       |
[4] Streamlit          visual inspection dashboard
```

| Stage | Component | Role |
|-------|-----------|------|
| 1 | DQN agent | Learns which patches are most likely to contain defects |
| 2 | EfficientNet-B4 | Classifies each patch as one of five defect classes |
| 3 | LLaMA 3.2 Vision | Generates a structured report for each defective patch |
| 4 | Streamlit dashboard | Displays annotated frames and all reports |

---

## Datasets

| Dataset | Role | Size |
|---------|------|------|
| RDD2022 | Supervised training: road defects | 47k images, 4 defect classes |
| MVTec AD | Supervised training: surface anomalies | ~5k images per category, 15 categories |
| Waymo Open Dataset | Unlabeled OOD generalization test | Applied for access separately |

---

## Results

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| EfficientNet-B4 | RDD2022 (val) | Accuracy | TBD after training |
| EfficientNet-B4 | MVTec AD (test) | AUROC (macro) | TBD after training |
| DQN vs. random | RDD2022 (test) | Patches to 80% recall | TBD after training |

*Fill in these numbers after running the evaluation scripts.*

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

```bash
python data/download_rdd2022.py
python data/download_mvtec.py
```

RDD2022 downloads four country splits (Japan, India, Czech, Norway) from GitHub.
MVTec AD is about 5 GB and downloads from mvtec.com.

### 3. Train EfficientNet-B4 on RDD2022

Open and run `notebooks/train_efficientnet.ipynb` from top to bottom. The best
checkpoint is saved to `checkpoints/efficientnet_rdd2022.pth`.

### 4. Train EfficientNet-B4 on MVTec AD

```bash
python scripts/train_mvtec.py
```

Checkpoint saved to `checkpoints/efficientnet_mvtec.pth`.

### 5. Train the DQN agent

The DQN uses the frozen RDD2022 EfficientNet as its reward oracle. Train
EfficientNet-B4 first.

```bash
python scripts/train_dqn.py
```

Checkpoint saved to `checkpoints/dqn.pth`.

### 6. Set up LLaMA 3.2 Vision

Install Ollama from https://ollama.com, then:

```bash
ollama serve
ollama pull llama3.2-vision
```

### 7. Run the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Evaluation

```bash
# RDD2022: per-class precision/recall/F1 and AUROC
python scripts/evaluate_rdd2022.py

# MVTec AD: AUROC and F1 per category
python scripts/evaluate_mvtec.py

# DQN vs. random baseline: defect recall curve
python scripts/evaluate_dqn.py
```

---

## Waymo Generalization Test

Run the trained EfficientNet on unlabeled Waymo frames to check whether the
model generalizes to real AV footage it was never trained on.

```bash
python scripts/waymo_test.py --img_dir /path/to/waymo/frames --limit 50
```

Annotated images are saved to `waymo_results/` by default.

---

## Project Structure

```
defect_detector/
    config.py                    global hyperparameters and random seeds
    requirements.txt             pinned dependencies
    data/
        download_rdd2022.py
        download_mvtec.py
        raw/                     downloaded datasets (git-ignored)
    notebooks/
        train_efficientnet.ipynb EfficientNet-B4 fine-tuning demo
    src/
        datasets/
            rdd2022.py           RDD2022Dataset
            mvtec.py             MVTecDataset, build_combined_mvtec
        models/
            efficientnet.py      build_efficientnet, load_checkpoint, extract_features
            dqn.py               ReplayBuffer, QNetwork
            dqn_env.py           PatchInspectionEnv (Gymnasium)
            llama.py             generate_report, check_ollama_available
            report.py            DefectReport, build_report, format_summary
        utils/
            patches.py           extract_patches, highlight_patches
    scripts/
        train_mvtec.py           EfficientNet-B4 training on MVTec AD
        train_dqn.py             DQN training loop
        evaluate_rdd2022.py      RDD2022 evaluation
        evaluate_mvtec.py        MVTec AD evaluation
        evaluate_dqn.py          DQN vs. random baseline
        waymo_test.py            OOD generalization test
    dashboard/
        app.py                   Streamlit dashboard
    checkpoints/                 saved model weights (git-ignored)
```

---

## Key Design Decisions

**Why EfficientNet-B4?** Its compound scaling (depth + width + resolution together)
gives better accuracy than ResNet-50 at a similar parameter count. The 380x380 native
input resolution is well-suited to fine-grained defect patterns like hairline cracks.

**Why DQN?** The patch selection action space is discrete and small (64 actions), which
is exactly where DQN works well. Being off-policy, DQN can reuse old experience in a
replay buffer, which makes training more sample-efficient than an on-policy method like PPO.

**Why LLaMA 3.2 Vision?** A vision-language model sees the actual defect image rather
than just a text label, so its reports are grounded in what the defect looks like. The
model runs locally via Ollama on an RTX 5080 at 4-bit quantization, so there are no
API dependencies.
