from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 YOLOv8 segmentation training")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "outputs")
    parser.add_argument("--name", type=str, default="yolov8_seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO("yolov8s-seg.pt")
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        device=0,
        cache=True,
        plots=True,
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    val = YOLO(str(best_path)).val(data=str(args.data), split="test", imgsz=args.imgsz, plots=True)
    metrics = {
        "model": "YOLOv8-seg",
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "map50": float(val.seg.map50),
        "map50_95": float(val.seg.map),
        "precision": float(val.seg.mp),
        "recall": float(val.seg.mr),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
