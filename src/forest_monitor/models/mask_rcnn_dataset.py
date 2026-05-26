"""Torch dataset and model helpers for Mask R-CNN."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from forest_monitor.constants import CLASS_NAMES


def parse_yolo_segmentation_label(label_path: Path, image_shape: tuple[int, int]) -> tuple[list[np.ndarray], list[int]]:
    height, width = image_shape
    polygons: list[np.ndarray] = []
    labels: list[int] = []
    if not label_path.exists() or label_path.read_text(encoding="utf-8").strip() == "":
        return polygons, labels

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(parts[0])
        coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        polygons.append(coords)
        labels.append(class_id + 1)
    return polygons, labels


def polygons_to_instance_masks(polygons: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    masks = []
    for polygon in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        masks.append(mask)
    if not masks:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack(masks, axis=0)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return np.asarray(boxes, dtype=np.float32)


class YoloSegmentationInstanceDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.image_paths = sorted(list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.jpeg")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        polygons, labels = parse_yolo_segmentation_label(label_path, image_shape=(height, width))
        masks_np = polygons_to_instance_masks(polygons, image_shape=(height, width))
        boxes_np = masks_to_boxes(masks_np)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor([(mask > 0).sum() for mask in masks_np], dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_mask_rcnn_model(num_classes: int = len(CLASS_NAMES) + 1):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
