from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forest_monitor.models.evaluation import benchmark_torch_model
from forest_monitor.models.mask_rcnn_dataset import YoloSegmentationInstanceDataset, build_mask_rcnn_model, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 4 model comparison")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--yolo-metrics", type=Path, default=ROOT / "outputs" / "evaluation" / "yolo_metrics.json")
    parser.add_argument("--mask-metrics", type=Path, default=ROOT / "outputs" / "evaluation" / "maskrcnn_metrics.json")
    parser.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--maskrcnn-checkpoint", type=Path, default=ROOT / "outputs" / "mask_rcnn" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "comparison")
    return parser.parse_args()


def benchmark_yolo(model_path: Path, image_paths: list[Path], imgsz: int = 640) -> float:
    model = YOLO(str(model_path))
    timings = []
    for image_path in image_paths[:30]:
        start = time.perf_counter()
        _ = model.predict(source=str(image_path), imgsz=imgsz, conf=0.25, verbose=False)
        timings.append(time.perf_counter() - start)
    return float(1.0 / max(sum(timings) / max(len(timings), 1), 1e-6))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_metrics = json.loads(args.yolo_metrics.read_text(encoding="utf-8"))
    mask_metrics = json.loads(args.mask_metrics.read_text(encoding="utf-8"))

    image_paths = sorted(list((args.dataset_root / "val" / "images").glob("*.png")) + list((args.dataset_root / "val" / "images").glob("*.jpg")))
    yolo_fps = benchmark_yolo(args.yolo_checkpoint, image_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloSegmentationInstanceDataset(args.dataset_root, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = build_mask_rcnn_model()
    checkpoint = torch.load(args.maskrcnn_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    fps_images = [dataset[idx][0] for idx in range(min(len(dataset), 20))]
    mask_fps = benchmark_torch_model(model, fps_images, device=device)

    rows = [
        {
            "Model": "YOLOv8-seg",
            "mAP": yolo_metrics["map50_95"],
            "IoU": yolo_metrics["iou"],
            "Precision": yolo_metrics["precision"],
            "Recall": yolo_metrics["recall"],
            "FPS": yolo_fps,
            "Model Size (MB)": args.yolo_checkpoint.stat().st_size / (1024 * 1024),
        },
        {
            "Model": "Mask R-CNN",
            "mAP": mask_metrics["map50_95"],
            "IoU": mask_metrics["iou"],
            "Precision": mask_metrics["precision"],
            "Recall": mask_metrics["recall"],
            "FPS": mask_fps,
            "Model Size (MB)": args.maskrcnn_checkpoint.stat().st_size / (1024 * 1024),
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    plt.figure(figsize=(8, 4))
    plt.bar(table["Model"], table["mAP"], color=["#2a9d8f", "#e76f51"])
    plt.ylabel("mAP@50-95")
    plt.title("Accuracy vs Model")
    plt.tight_layout()
    plt.savefig(args.output_dir / "accuracy_vs_model.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(table["Model"], table["FPS"], color=["#264653", "#f4a261"])
    plt.ylabel("FPS")
    plt.title("Speed vs Model")
    plt.tight_layout()
    plt.savefig(args.output_dir / "speed_vs_model.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
