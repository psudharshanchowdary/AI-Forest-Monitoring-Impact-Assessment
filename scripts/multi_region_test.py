from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely.geometry import box

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forest_monitor.data.pipeline import (
    CLASS_NAMES,
    RegionSpec,
    combine_label_mask,
    fetch_osm_layers,
    fetch_sentinel_patch,
    fetch_worldcover_patch,
    normalize_image,
    rasterize_osm_layers,
    sample_patch_polygons,
)
from forest_monitor.models.evaluation import (
    accumulate_confusion,
    confusion_to_metrics,
    mask_rcnn_output_to_label_map,
    yolo_result_to_label_map,
)
from forest_monitor.models.mask_rcnn_dataset import build_mask_rcnn_model, masks_to_boxes


TEST_REGIONS = [
    RegionSpec("india", box(77.6, 12.6, 79.1, 18.2), "2025-01-01", "2025-12-31", 20),
    RegionSpec("amazon", box(-64.5, -11.2, -60.2, -7.8), "2025-01-01", "2025-12-31", 20),
    RegionSpec("slovakia", box(18.2, 47.7, 22.6, 49.6), "2025-01-01", "2025-12-31", 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 5 multi-region inference and comparison")
    parser.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--maskrcnn-checkpoint", type=Path, default=ROOT / "outputs" / "mask_rcnn" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "regions")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def label_mask_to_metric_target(label_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    masks = []
    labels = []
    for class_index in range(len(CLASS_NAMES)):
        class_mask = (label_mask == class_index).astype(np.uint8)
        num_labels, cc = cv2.connectedComponents(class_mask)
        for component_idx in range(1, num_labels):
            component = (cc == component_idx).astype(np.uint8)
            if component.sum() < 40:
                continue
            masks.append(component)
            labels.append(class_index + 1)
    if not masks:
        height, width = label_mask.shape
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, height, width), dtype=torch.uint8),
        }, np.full((height, width), -1, dtype=np.int16)
    masks_np = np.stack(masks, axis=0)
    boxes_np = masks_to_boxes(masks_np)
    target = {
        "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
    }
    target_map = np.full(label_mask.shape, -1, dtype=np.int16)
    for mask, label in zip(masks_np, labels):
        target_map[mask > 0] = label - 1
    return target, target_map


def yolo_pred_to_metric(result, image_shape: tuple[int, int]) -> dict:
    height, width = image_shape
    if result.masks is None or result.boxes is None:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64), "masks": torch.zeros((0, height, width), dtype=torch.uint8)}
    masks = []
    for mask in result.masks.data.detach().cpu().numpy():
        masks.append(cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST))
    return {
        "boxes": result.boxes.xyxy.detach().cpu().float(),
        "scores": result.boxes.conf.detach().cpu().float(),
        "labels": result.boxes.cls.detach().cpu().long() + 1,
        "masks": torch.as_tensor(np.asarray(masks), dtype=torch.uint8),
    }


def overlay_prediction(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    colors = {
        0: np.array([34, 197, 94], dtype=np.uint8),
        1: np.array([250, 204, 21], dtype=np.uint8),
        2: np.array([59, 130, 246], dtype=np.uint8),
        3: np.array([249, 115, 22], dtype=np.uint8),
        4: np.array([239, 68, 68], dtype=np.uint8),
    }
    output = image.copy().astype(np.float32)
    for class_index, color in colors.items():
        mask = label_map == class_index
        if np.any(mask):
            output[mask] = (0.62 * output[mask]) + (0.38 * color)
    return np.clip(output, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(args.yolo_checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model = build_mask_rcnn_model()
    checkpoint = torch.load(args.maskrcnn_checkpoint, map_location=device)
    mask_model.load_state_dict(checkpoint["model_state_dict"])
    mask_model.to(device)
    mask_model.eval()

    records = []
    for region_idx, region in enumerate(TEST_REGIONS):
        region_dir = args.output_dir / region.name
        (region_dir / "yolo").mkdir(parents=True, exist_ok=True)
        (region_dir / "mask_rcnn").mkdir(parents=True, exist_ok=True)
        yolo_map = MeanAveragePrecision(iou_type="segm")
        mask_map = MeanAveragePrecision(iou_type="segm")
        yolo_confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)
        mask_confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)

        patches = sample_patch_polygons(region, patch_edge_m=args.imgsz * 10, seed=100 + region_idx)
        for patch_idx, patch_polygon in enumerate(patches[: region.patches]):
            sentinel_patch = fetch_sentinel_patch(patch_polygon, start_date=region.start_date, end_date=region.end_date, patch_size_px=args.imgsz)
            worldcover_patch = fetch_worldcover_patch(patch_polygon, patch_size_px=args.imgsz)
            if sentinel_patch is None or worldcover_patch is None:
                continue
            try:
                roads, buildings = fetch_osm_layers(patch_polygon)
            except Exception:
                continue
            road_mask, building_mask = rasterize_osm_layers(patch_polygon, image_shape=worldcover_patch.shape, roads=roads, buildings=buildings)
            gt_mask = combine_label_mask(worldcover_patch, road_mask=road_mask, building_mask=building_mask)
            rgb = normalize_image(sentinel_patch[..., :3])
            target, target_map = label_mask_to_metric_target(gt_mask)

            yolo_result = yolo_model.predict(source=rgb, imgsz=args.imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
            yolo_pred = yolo_pred_to_metric(yolo_result, image_shape=gt_mask.shape)
            yolo_map.update([yolo_pred], [target])
            yolo_label_map = yolo_result_to_label_map(yolo_result, image_shape=gt_mask.shape, score_threshold=0.25)
            accumulate_confusion(yolo_confusion, yolo_label_map, target_map)
            cv2.imwrite(str(region_dir / "yolo" / f"patch_{patch_idx:03d}.png"), cv2.cvtColor(overlay_prediction(rgb, yolo_label_map), cv2.COLOR_RGB2BGR))

            image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().to(device) / 255.0
            with torch.no_grad():
                mask_output = mask_model([image_tensor])[0]
            mask_output_cpu = {key: value.detach().cpu() for key, value in mask_output.items()}
            mask_map.update([mask_output_cpu], [target])
            mask_label_map = mask_rcnn_output_to_label_map(mask_output_cpu, image_shape=gt_mask.shape, score_threshold=0.5)
            accumulate_confusion(mask_confusion, mask_label_map, target_map)
            cv2.imwrite(str(region_dir / "mask_rcnn" / f"patch_{patch_idx:03d}.png"), cv2.cvtColor(overlay_prediction(rgb, mask_label_map), cv2.COLOR_RGB2BGR))

        yolo_scores = yolo_map.compute()
        mask_scores = mask_map.compute()
        yolo_pixel = confusion_to_metrics(yolo_confusion)
        mask_pixel = confusion_to_metrics(mask_confusion)
        records.append({"Region": region.name, "Model": "YOLOv8-seg", "mAP": float(yolo_scores["map"].item()), "IoU": yolo_pixel["iou"]})
        records.append({"Region": region.name, "Model": "Mask R-CNN", "mAP": float(mask_scores["map"].item()), "IoU": mask_pixel["iou"]})

    df = pd.DataFrame(records)
    df.to_csv(args.output_dir / "region_metrics.csv", index=False)
    print(df.to_string(index=False))

    plt.figure(figsize=(9, 4))
    for idx, metric in enumerate(["mAP", "IoU"], start=1):
        plt.subplot(1, 2, idx)
        pivot = df.pivot(index="Region", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=plt.gca(), rot=20)
        plt.title(f"Region-wise {metric}")
        plt.tight_layout()
    plt.savefig(args.output_dir / "region_comparison.png", dpi=160)
    plt.close()

    (args.output_dir / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
