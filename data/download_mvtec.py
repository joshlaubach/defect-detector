"""
download_mvtec.py

Downloads the MVTec Anomaly Detection dataset from Kaggle.

The MVTec official download URL is no longer available. This script uses the
Kaggle CLI to download the dataset from the well-maintained mirror at
https://www.kaggle.com/datasets/ipythonx/mvtec-ad

Prerequisites:
    1. Create a free account at https://www.kaggle.com
    2. Go to Account -> API -> Create New Token. This downloads kaggle.json.
    3. Place kaggle.json at C:/Users/<you>/.kaggle/kaggle.json (Windows) or
       ~/.kaggle/kaggle.json (Linux/Mac). Set permissions to 600 on Linux/Mac.
    4. pip install kaggle

Usage:
    python data/download_mvtec.py
    python data/download_mvtec.py --out_dir data/raw/mvtec
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


KAGGLE_DATASET = "ipythonx/mvtec-ad"
ARCHIVE_NAME = "mvtec-ad.zip"


def _check_kaggle() -> None:
    """Raise RuntimeError if the kaggle package or credentials are missing."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The kaggle CLI is not installed. Run: pip install kaggle\n"
            "Then place your kaggle.json API token at ~/.kaggle/kaggle.json."
        )

    cred_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]
    if not any(p.exists() for p in cred_paths):
        raise RuntimeError(
            "Kaggle credentials not found. Download kaggle.json from "
            "kaggle.com -> Account -> API -> Create New Token, then place it "
            "at ~/.kaggle/kaggle.json."
        )


def download_mvtec(out_dir: Path) -> None:
    """Download and extract the MVTec AD dataset into out_dir."""
    _check_kaggle()

    archive_path = out_dir / ARCHIVE_NAME

    if not archive_path.exists():
        print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI ...")
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(out_dir),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"kaggle download failed with exit code {result.returncode}. "
                f"Check that your API token is valid and you have accepted the "
                f"dataset terms at https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
            )
    else:
        print(f"Archive already present at {archive_path}, skipping download.")

    print(f"Extracting {archive_path.name} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(
                f"Corrupt ZIP: first bad file is {bad!r}. "
                f"Delete {archive_path} and re-run."
            )
        zf.extractall(out_dir)

    archive_path.unlink()
    print(f"Done: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--out_dir",
        default="data/raw/mvtec",
        help="Directory to save the extracted data",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Skipping download: {out_dir} already exists and is not empty.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_mvtec(out_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
