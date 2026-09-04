"""
download_mvtec.py

Downloads the MVTec Anomaly Detection dataset from the official MVTec website.

The full archive is about 5 GB. After extraction it expands to around 4.9 GB
across 15 object categories. The archive is deleted after extraction.

Usage:
    python data/download_mvtec.py
    python data/download_mvtec.py --out_dir data/raw/mvtec
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


MVTEC_URL = "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection.tar.gz"
ARCHIVE_NAME = "mvtec_anomaly_detection.tar.gz"


class DownloadProgressBar(tqdm):
    """tqdm wrapper that plugs into urllib's reporthook interface."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total: int = -1) -> None:
        if total >= 0:
            self.total = total
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest, showing a progress bar."""
    print(f"Downloading {url}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
        urllib.request.urlretrieve(url, filename=dest, reporthook=bar.update_to)


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
    archive_path = out_dir / ARCHIVE_NAME

    download_file(MVTEC_URL, archive_path)

    print(f"Extracting {archive_path.name} ...")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(out_dir)

    archive_path.unlink()
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
