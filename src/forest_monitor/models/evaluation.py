"""Shared evaluation helpers for YOLOv8, Mask R-CNN, and fusion experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from forest_monitor.constants import CLASS_NAMES
from forest_monitor.models.mask_rcnn_dataset import masks_to_boxes, parse_yolo_segmentation_label, polygons_to_instance_masks

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional plotting dependency at runtime
    sns = None


def empty_detection_prediction(image_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    height, width = image_shape
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, height, width), dtype=torch.uint8),
    }


def target_to_label_map(target: dict, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    label_map = np.full((height, width), fill_value=-1, dtype=np.int16)
    masks = target["masks"].cpu().numpy() if torch.is_tensor(target["masks"]) else target["masks"]
    labels = target["labels"].cpu().numpy() if torch.is_tensor(target["labels"]) else target["labels"]
    for mask, label in zip(masks, labels):
        label_map[mask > 0] = int(label) - 1
    return label_map


def build_target_from_label_file(label_path: Path, image_shape: tuple[int, int]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    polygons, labels = parse_yolo_segmentation_label(label_path, image_shape=image_shape)
    masks = polygons_to_instance_masks(polygons, image_shape=image_shape)
    if masks.shape[0] == 0:
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8),
        }
        return target, np.full(image_shape, -1, dtype=np.int16)
    boxes = masks_to_boxes(masks)
    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(masks, dtype=torch.uint8),
    }
    target_map = np.full(image_shape, fill_value=-1, dtype=np.int16)
    for mask, label in zip(masks, labels):
        target_map[mask > 0] = int(label) - 1
    return target, target_map


def masks_from_yolo_file(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    polygons, labels = parse_yolo_segmentation_label(label_path, image_shape=image_shape)
    masks = polygons_to_instance_masks(polygons, image_shape=image_shape)
    return masks, np.asarray(labels, dtype=np.int64)


def yolo_result_to_label_map(result, image_shape: tuple[int, int], score_threshold: float = 0.25) -> np.ndarray:
    height, width = image_shape
    label_map = np.full((height, width), fill_value=-1, dtype=np.int16)
    if result.masks is None or result.boxes is None:
        return label_map

    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(scores)
    for idx in order:
        if scores[idx] < score_threshold:
            continue
        mask = cv2.resize(masks[idx].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        label_map[mask > 0] = labels[idx]
    return label_map


def mask_rcnn_output_to_label_map(output: dict, image_shape: tuple[int, int], score_threshold: float = 0.5) -> np.ndarray:
    height, width = image_shape
    label_map = np.full((height, width), fill_value=-1, dtype=np.int16)
    if len(output["scores"]) == 0:
        return label_map

    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy().astype(np.int64)
    masks = output["masks"].detach().cpu().numpy()
    order = np.argsort(scores)
    for idx in order:
        if scores[idx] < score_threshold:
            continue
        mask = (masks[idx, 0] > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        label_map[mask > 0] = labels[idx] - 1
    return label_map


def label_map_to_detection_prediction(
    label_map: np.ndarray,
    score_stack: np.ndarray | None = None,
    min_component_pixels: int = 40,
) -> dict[str, torch.Tensor]:
    masks: list[np.ndarray] = []
    labels: list[int] = []
    scores: list[float] = []

    for class_index in range(len(CLASS_NAMES)):
        class_mask = (label_map == class_index).astype(np.uint8)
        component_count, component_map = cv2.connectedComponents(class_mask)
        for component_idx in range(1, component_count):
            component = (component_map == component_idx).astype(np.uint8)
            if int(component.sum()) < min_component_pixels:
                continue
            masks.append(component)
            labels.append(class_index + 1)
            if score_stack is None:
                scores.append(1.0)
            else:
                component_scores = score_stack[..., class_index][component > 0]
                scores.append(float(np.clip(component_scores.mean() if component_scores.size else 0.5, 0.0, 1.0)))

    if not masks:
        return empty_detection_prediction(label_map.shape)

    masks_np = np.stack(masks, axis=0)
    boxes_np = masks_to_boxes(masks_np)
    return {
        "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
        "scores": torch.as_tensor(scores, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks": torch.as_tensor(masks_np, dtype=torch.uint8),
    }


def accumulate_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    valid = target >= 0
    pred_valid = np.where(prediction >= 0, prediction, len(CLASS_NAMES))
    tgt_valid = np.where(target >= 0, target, len(CLASS_NAMES))
    for tgt, pred in zip(tgt_valid[valid].ravel(), pred_valid[valid].ravel()):
        confusion[int(tgt), int(pred)] += 1


def compute_f1_per_class(confusion: np.ndarray) -> dict[str, float]:
    f1_scores: dict[str, float] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_scores[class_name] = float((2.0 * precision * recall) / (precision + recall + 1e-9))
    return f1_scores


def confusion_to_metrics(confusion: np.ndarray) -> dict[str, float]:
    tp = float(np.trace(confusion[: len(CLASS_NAMES), : len(CLASS_NAMES)]))
    fp = float(confusion[:, : len(CLASS_NAMES)].sum() - tp)
    fn = float(confusion[: len(CLASS_NAMES), :].sum() - tp)
    iou = tp / (tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-9)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_confusion_matrix(confusion: np.ndarray, output_path: Path, title: str) -> None:
    labels = CLASS_NAMES + ["background"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    if sns is not None:
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)
    else:  # pragma: no cover - fallback when seaborn is unavailable
        plt.imshow(confusion, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_training_plot(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_map = [item["val_map_50_95"] for item in history]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, color="#2a9d8f")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_map, color="#e76f51")
    plt.title("Validation mAP@50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def benchmark_torch_model(model, images: Iterable[torch.Tensor], device: torch.device, repeats: int = 30) -> float:
    images = list(images)
    if not images:
        return 0.0
    model.eval()
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            image = images[_ % len(images)].to(device)
            start = time.perf_counter()
            _ = model([image])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
    return float(1.0 / max(np.mean(timings), 1e-6))


def evaluate_mask_rcnn(model, dataloader, device: torch.device, score_threshold: float = 0.5) -> tuple[dict[str, float], np.ndarray]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    map_metric = MeanAveragePrecision(iou_type="segm")
    confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            outputs_cpu = [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
            targets_cpu = [{key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
            map_metric.update(outputs_cpu, targets_cpu)

            for image_tensor, output, target in zip(images, outputs_cpu, targets_cpu):
                _, height, width = image_tensor.shape
                pred_map = mask_rcnn_output_to_label_map(output, image_shape=(height, width), score_threshold=score_threshold)
                target_map = target_to_label_map(target, image_shape=(height, width))
                accumulate_confusion(confusion, prediction=pred_map, target=target_map)

    scores = map_metric.compute()
    pixel_metrics = confusion_to_metrics(confusion)
    metrics = {
        "map50": float(scores["map_50"].item()),
        "map50_95": float(scores["map"].item()),
        "precision": pixel_metrics["precision"],
        "recall": pixel_metrics["recall"],
        "iou": pixel_metrics["iou"],
        "f1": pixel_metrics["f1"],
    }
    return metrics, confusion
