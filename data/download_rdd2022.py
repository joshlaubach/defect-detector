"""
download_rdd2022.py

Downloads the Road Damage Dataset 2022 (RDD2022) from the official GitHub release.

Each country is packaged as a separate zip file. This script downloads the
requested countries, extracts them, and removes the archives to save disk space.

Usage:
    python data/download_rdd2022.py
    python data/download_rdd2022.py --countries Japan India --out_dir data/raw/rdd2022
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


RELEASE_BASE = (
    "https://github.com/sekilab/RoadDamageDetector/releases/download/"
    "dataset3.0/{country}.zip"
)

ALL_COUNTRIES = ["Japan", "India", "Czech", "Norway"]


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


def download_country(country: str, out_dir: Path) -> None:
    """Download and extract one country split."""
    zip_path = out_dir / f"{country}.zip"
    url = RELEASE_BASE.format(country=country)

    download_file(url, zip_path)

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    zip_path.unlink()
    print(f"Done: {out_dir / country}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RDD2022 dataset")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=ALL_COUNTRIES,
        choices=ALL_COUNTRIES,
        help="Which country splits to download (default: all four)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw/rdd2022",
        help="Directory to save the extracted data",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for country in args.countries:
        country_dir = out_dir / country
        if country_dir.exists():
            print(f"Skipping {country}: already exists at {country_dir}")
            continue
        download_country(country, out_dir)

    print("All requested countries downloaded.")


if __name__ == "__main__":
    main()
