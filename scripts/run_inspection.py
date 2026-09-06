"""
run_inspection.py

Single entry point for the defect inspection pipeline. Point it at one image
or a folder of frames; it runs DQN patch selection -> EfficientNet-B4 ->
LLaMA 3.2 Vision and writes an annotated image plus a JSON report per frame.

This is the production / demo path. It needs the two checkpoints in
checkpoints/ and (for narrative text) a running Ollama; it does NOT need any
training dataset downloaded.

(Named run_inspection.py rather than inspect.py so it does not shadow the
Python standard-library `inspect` module on `sys.path`.)

Usage:
    python scripts/run_inspection.py path/to/frame.jpg
    python scripts/run_inspection.py path/to/frames/ --out out/ --limit 20
    python scripts/run_inspection.py frame.jpg --no-llama     # classification only
    python scripts/run_inspection.py frame.jpg --no-dqn       # inspect all 64 patches
    python scripts/run_inspection.py frame.jpg --benchmark    # print stage timings
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import set_seeds, SEED
from src.pipeline import DefectPipeline, InspectionConfig, InspectionResult
from src.models.report import format_summary

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(target: Path, limit: int | None) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        paths = sorted(
            p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"No images found in {target}")
        return paths[:limit] if limit else paths
    raise RuntimeError(f"Not a file or directory: {target}")


def result_to_dict(path: Path, result: InspectionResult) -> dict:
    reports = []
    for r in result.reports:
        d = asdict(r)
        d.pop("patch_image", None)  # not JSON-serializable, saved as a crop instead
        reports.append(d)
    return {
        "frame": path.name,
        "patches_inspected": result.patches_inspected,
        "patches_total": result.patches_total,
        "stopped_early": result.stopped_early,
        "elapsed_s": round(result.elapsed_s, 3),
        "per_stage_s": {k: round(v, 3) for k, v in result.per_stage_s.items()},
        "defect_count": len(result.reports),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the defect inspection pipeline")
    parser.add_argument("target", help="Image file or directory of frames")
    parser.add_argument("--out", default="inspection_out", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max frames (dir mode)")
    parser.add_argument("--no-dqn", action="store_true", help="Inspect every patch")
    parser.add_argument("--no-llama", action="store_true", help="Skip narrative reports")
    parser.add_argument("--benchmark", action="store_true", help="Print per-stage timing")
    args = parser.parse_args()

    set_seeds(SEED)

    cfg = InspectionConfig(use_dqn=not args.no_dqn, use_llama=not args.no_llama)
    pipeline = DefectPipeline(cfg)

    print("Model status:")
    for stage, status in pipeline.model_status().items():
        print(f"  {stage:13s} {status}")
    print()

    image_paths = collect_images(Path(args.target), args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict] = []
    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))
        result = pipeline.inspect(image)

        Image.fromarray(result.annotated_image).save(out_dir / f"{path.stem}_annotated.png")
        record = result_to_dict(path, result)
        (out_dir / f"{path.stem}.json").write_text(json.dumps(record, indent=2))
        combined.append(record)

        print(f"=== {path.name} ===")
        print(
            f"  inspected {result.patches_inspected}/{result.patches_total} patches"
            f"{' (stopped early)' if result.stopped_early else ''}"
            f"  |  {len(result.reports)} defect(s)  |  {result.elapsed_s:.2f}s"
        )
        if args.benchmark:
            for stage, secs in result.per_stage_s.items():
                print(f"    {stage:13s} {secs:.3f}s")
        if result.reports:
            print(format_summary(result.reports))
        print()

    if len(combined) > 1:
        (out_dir / "summary.json").write_text(json.dumps(combined, indent=2))
    print(f"Wrote {len(combined)} result(s) to {out_dir}/")


if __name__ == "__main__":
    main()
