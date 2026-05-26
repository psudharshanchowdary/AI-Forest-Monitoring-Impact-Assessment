from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forest_monitor.data.pipeline import create_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1-2 dataset creation pipeline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--target-patches", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_dataset(output_dir=args.output_dir, patch_size_px=args.patch_size, target_patches=args.target_patches)
    print("Dataset creation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"data.yaml: {ROOT / 'configs' / 'data.yaml'}")


if __name__ == "__main__":
    main()
