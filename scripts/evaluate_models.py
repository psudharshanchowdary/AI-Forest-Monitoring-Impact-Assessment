from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forest_monitor.data.pipeline import CLASS_NAMES
from forest_monitor.models.evaluation import (
    accumulate_confusion,
    confusion_to_metrics,
    mask_rcnn_output_to_label_map,
    save_confusion_matrix,
    write_metrics,
    yolo_result_to_label_map,
)
from forest_monitor.models.mask_rcnn_dataset import (
    YoloSegmentationInstanceDataset,
    build_mask_rcnn_model,
    collate_fn,
    masks_to_boxes,
    parse_yolo_segmentation_label,
    polygons_to_instance_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 and Mask R-CNN on the test split")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--maskrcnn-checkpoint", type=Path, default=ROOT / "outputs" / "mask_rcnn" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def target_from_label(label_path: Path, image_shape: tuple[int, int]) -> tuple[dict, np.ndarray]:
    polygons, labels = parse_yolo_segmentation_label(label_path, image_shape=image_shape)
    masks = polygons_to_instance_masks(polygons, image_shape=image_shape)
    boxes = masks_to_boxes(masks)
    target = {
        "masks": torch.as_tensor(masks, dtype=torch.uint8),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
    }
    target_map = np.full(image_shape, fill_value=-1, dtype=np.int16)
    for mask, label in zip(masks, labels):
        target_map[mask > 0] = int(label) - 1
    return target, target_map


def yolo_predictions_to_metric(result, image_shape: tuple[int, int]) -> dict:
    height, width = image_shape
    if result.masks is None or result.boxes is None:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, height, width), dtype=torch.uint8),
        }

    masks = []
    for mask in result.masks.data.detach().cpu().numpy():
        resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(resized)
    return {
        "boxes": result.boxes.xyxy.detach().cpu().float(),
        "scores": result.boxes.conf.detach().cpu().float(),
        "labels": result.boxes.cls.detach().cpu().long() + 1,
        "masks": torch.as_tensor(np.asarray(masks), dtype=torch.uint8),
    }


def evaluate_yolo(args: argparse.Namespace) -> dict[str, float]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.yolo_checkpoint))
    test_images = sorted(list((args.dataset_root / "test" / "images").glob("*.png")) + list((args.dataset_root / "test" / "images").glob("*.jpg")))
    map_metric = MeanAveragePrecision(iou_type="segm")
    confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)

    for image_path in test_images:
        label_path = args.dataset_root / "test" / "labels" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        target, target_map = target_from_label(label_path, image_shape=(height, width))
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
        pred = yolo_predictions_to_metric(result, image_shape=(height, width))
        map_metric.update([pred], [target])
        pred_map = yolo_result_to_label_map(result, image_shape=(height, width), score_threshold=0.25)
        accumulate_confusion(confusion, pred_map, target_map)

    overall = model.val(data=str(args.data), split="test", imgsz=args.imgsz, plots=True)
    pixel_metrics = confusion_to_metrics(confusion)
    metrics = {
        "model": "YOLOv8-seg",
        "map50": float(overall.seg.map50),
        "map50_95": float(overall.seg.map),
        "precision": pixel_metrics["precision"],
        "recall": pixel_metrics["recall"],
        "iou": pixel_metrics["iou"],
    }
    write_metrics(metrics, args.output_dir / "yolo_metrics.json")
    save_confusion_matrix(confusion, args.output_dir / "yolo_confusion_matrix.png", title="YOLOv8 Confusion Matrix")
    return metrics


def evaluate_mask_rcnn_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloSegmentationInstanceDataset(args.dataset_root, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = build_mask_rcnn_model()
    checkpoint = torch.load(args.maskrcnn_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    from forest_monitor.models.evaluation import evaluate_mask_rcnn

    metrics, confusion = evaluate_mask_rcnn(model, loader, device=device)
    metrics["model"] = "Mask R-CNN"
    write_metrics(metrics, args.output_dir / "maskrcnn_metrics.json")
    save_confusion_matrix(confusion, args.output_dir / "maskrcnn_confusion_matrix.png", title="Mask R-CNN Confusion Matrix")
    return metrics


def main() -> None:
    args = parse_args()
    yolo_metrics = evaluate_yolo(args)
    mask_metrics = evaluate_mask_rcnn_checkpoint(args)
    print(json.dumps({"yolo": yolo_metrics, "mask_rcnn": mask_metrics}, indent=2))


if __name__ == "__main__":
    main()
