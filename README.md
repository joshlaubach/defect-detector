# Defect Detector

An offline post-drive inspection system for autonomous vehicles. After a drive completes,
the pipeline processes camera footage to find road surface defects and generates a
plain-English report describing each one.

The project combines three ML components: a fine-tuned CNN classifier, a reinforcement
learning agent that decides which image regions to inspect, and a vision-language model
that writes the final report.

---

## Architecture

| Stage | Component | Role |
|-------|-----------|------|
| 1 | DQN agent | Picks which image patches to inspect |
| 2 | EfficientNet-B4 | Classifies each patch as defect or normal |
| 3 | LLaMA 3.2 Vision | Writes a natural-language report for each defect |
| 4 | Streamlit dashboard | Displays annotated frames and the report |

---

## Datasets

| Dataset | Use |
|---------|-----|
| RDD2022 | Supervised training for road defect classification |
| MVTec AD | Supervised training for surface anomaly detection |
| Waymo Open Dataset | Unlabeled out-of-distribution generalization test |

---

## Setup

```bash
pip install -r requirements.txt
```

Download the datasets:

```bash
python data/download_rdd2022.py
python data/download_mvtec.py
```

Train EfficientNet-B4 by running `notebooks/train_efficientnet.ipynb` from top to bottom.

---

## Project Structure

```
defect_detector/
    config.py                   global hyperparameters and seeds
    requirements.txt
    data/
        download_rdd2022.py
        download_mvtec.py
        raw/                    downloaded datasets (git-ignored)
        processed/              preprocessed files (git-ignored)
    notebooks/
        train_efficientnet.ipynb
    src/
        datasets/
            rdd2022.py
            mvtec.py
        models/
            efficientnet.py
            dqn.py
        utils/
            patches.py
    scripts/
        train_dqn.py
        evaluate.py
        waymo_test.py
    dashboard/
        app.py
    checkpoints/                saved model weights (git-ignored)
```

---

## Demo

*(GIF will be added after the Streamlit dashboard is complete.)*
