"""
download_rdd2022.py

Downloads the Road Damage Dataset 2022 (RDD2022) from the Figshare mirror.

The full archive (~13.3 GB) contains all four country splits in one ZIP:
    RDD2022/Japan/...
    RDD2022/India/...
    RDD2022/Czech/...
    RDD2022/Norway/...

This script downloads the archive, extracts only the requested countries,
moves them to out_dir/<country>/..., and deletes the archive afterward.

Downloads resume automatically if the archive is partially complete.
Countries that already exist in out_dir are skipped.

Usage:
    python data/download_rdd2022.py
    python data/download_rdd2022.py --countries Japan India --out_dir data/raw/rdd2022
"""

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASET_URL = "https://ndownloader.figshare.com/files/38030910"
ARCHIVE_NAME = "RDD2022_released_through_CRDDC2022.zip"
ZIP_ROOT = "RDD2022"

ALL_COUNTRIES = ["Japan", "India", "Czech", "Norway"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _get_remote_size(url: str) -> int:
    """Return Content-Length from a HEAD request, or -1 if unavailable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            val = resp.headers.get("Content-Length")
            return int(val) if val else -1
    except Exception:
        return -1


def _accepts_range(url: str) -> bool:
    """Return True if the server advertises Accept-Ranges: bytes."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        return False


def download_file(url: str, dest: Path) -> None:
    """
    Download url to dest with resume support and a progress bar.

    If dest already exists and its size matches Content-Length, the download
    is skipped entirely. If dest is a partial file and the server supports
    Range requests, the download resumes from where it left off. Otherwise
    the file is downloaded from scratch.

    Raises RuntimeError on HTTP errors or if the final file size does not
    match Content-Length.
    """
    remote_size = _get_remote_size(url)
    existing_size = dest.stat().st_size if dest.exists() else 0

    # Already complete.
    if remote_size > 0 and existing_size == remote_size:
        gb = remote_size / 1e9
        print(f"{dest.name} already downloaded ({gb:.1f} GB), skipping.")
        return

    # Partial file: attempt resume.
    resume = (
        existing_size > 0
        and remote_size > 0
        and existing_size < remote_size
        and _accepts_range(url)
    )

    if existing_size > 0 and not resume:
        print("Partial archive found but resume is not supported; restarting.")
        dest.unlink()
        existing_size = 0

    if resume:
        gb_done = existing_size / 1e9
        gb_total = remote_size / 1e9
        print(f"Resuming download ({gb_done:.1f} / {gb_total:.1f} GB) ...")
        req = urllib.request.Request(url, headers={"Range": f"bytes={existing_size}-"})
        open_mode = "ab"
    else:
        print(f"Downloading {url}")
        req = urllib.request.Request(url)
        open_mode = "wb"

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP {status} from {url}")

            bar_total = remote_size if remote_size > 0 else None
            with tqdm(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc=dest.name,
                total=bar_total,
                initial=existing_size,
            ) as bar:
                with open(dest, open_mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MB
                        if not chunk:
                            break
                        fh.write(chunk)
                        bar.update(len(chunk))

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} downloading {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error downloading {url}: {exc.reason}"
        ) from exc

    # Verify completeness.
    if remote_size > 0:
        actual = dest.stat().st_size
        if actual != remote_size:
            dest.unlink()
            raise RuntimeError(
                f"Download incomplete: expected {remote_size} bytes, got {actual}. "
                f"Partial file removed. Run again to retry."
            )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _validate_zip(archive_path: Path) -> None:
    """Raise RuntimeError if the ZIP file cannot be opened or is corrupt."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(
                    f"Corrupt ZIP: first bad file is {bad!r}. "
                    f"Delete {archive_path} and re-run to download again."
                )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{archive_path} is not a valid ZIP file. "
            f"Delete it and re-run to download again."
        ) from exc


def extract_country(zf: zipfile.ZipFile, country: str, out_dir: Path) -> None:
    """
    Extract one country's files from the archive into out_dir/<country>/.

    The archive stores files under RDD2022/<country>/...; this function
    strips the RDD2022/ prefix so the final layout is out_dir/<country>/...

    Uses a temporary staging directory so a failed extraction does not leave
    a partial country folder that would be mistaken for a complete one.
    """
    prefix = f"{ZIP_ROOT}/{country}/"
    members = [info for info in zf.infolist() if info.filename.startswith(prefix)]
    if not members:
        raise RuntimeError(
            f"Country {country!r} not found in the archive. "
            f"Expected entries starting with {prefix!r}."
        )

    staging = out_dir / f"_staging_{country}"
    if staging.exists():
        shutil.rmtree(staging)

    try:
        print(f"Extracting {country} ({len(members)} files) ...")
        zf.extractall(staging, members=members)
        extracted_src = staging / ZIP_ROOT / country
        if not extracted_src.exists():
            raise RuntimeError(
                f"Extraction succeeded but {extracted_src} is missing. "
                f"The archive layout may differ from expectations."
            )
        extracted_src.rename(out_dir / country)
        print(f"Done: {out_dir / country}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    needed = []
    for country in args.countries:
        if (out_dir / country).exists():
            print(f"Skipping {country}: already exists at {out_dir / country}")
        else:
            needed.append(country)

    if not needed:
        print("All requested countries are already present.")
        return

    archive_path = out_dir / ARCHIVE_NAME

    try:
        download_file(DATASET_URL, archive_path)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    try:
        _validate_zip(archive_path)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print(f"\nOpening archive: {archive_path.name}")
    extraction_ok = True
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for country in needed:
                try:
                    extract_country(zf, country, out_dir)
                except RuntimeError as exc:
                    print(f"Error extracting {country}: {exc}")
                    extraction_ok = False
    except zipfile.BadZipFile as exc:
        print(f"Error: Could not read archive after download: {exc}")
        return

    if extraction_ok:
        archive_path.unlink()
        print(f"\nArchive removed. All requested countries downloaded to {out_dir}/")
    else:
        print(
            f"\nSome countries failed to extract. "
            f"Archive kept at {archive_path} for retry."
        )


if __name__ == "__main__":
    main()
