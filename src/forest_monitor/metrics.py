"""Segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass(slots=True)
class PerformanceMetrics:
    map_50_95: float
    iou: float
    precision: float
    recall: float


def _extract_instance_masks(instance_map: np.ndarray, min_pixels: int = 18) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for label_id in np.unique(instance_map):
        if label_id == 0:
            continue
        mask = instance_map == label_id
        if int(mask.sum()) >= min_pixels:
            masks.append(mask)
    return masks


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def _average_precision(
    pred_masks: list[np.ndarray],
    pred_scores: list[float],
    gt_masks: list[np.ndarray],
    iou_threshold: float,
) -> float:
    if not pred_masks and not gt_masks:
        return float("nan")
    if not pred_masks or not gt_masks:
        return 0.0

    order = np.argsort(np.asarray(pred_scores, dtype=np.float32))[::-1]
    matched_gt: set[int] = set()
    true_pos = np.zeros(len(order), dtype=np.float32)
    false_pos = np.zeros(len(order), dtype=np.float32)

    for rank, pred_idx in enumerate(order):
        pred_mask = pred_masks[pred_idx]
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_mask in enumerate(gt_masks):
            iou = _mask_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_iou >= iou_threshold and best_gt not in matched_gt:
            true_pos[rank] = 1.0
            matched_gt.add(best_gt)
        else:
            false_pos[rank] = 1.0

    cum_tp = np.cumsum(true_pos)
    cum_fp = np.cumsum(false_pos)
    recalls = cum_tp / max(float(len(gt_masks)), EPS)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, EPS)

    recall_curve = np.concatenate(([0.0], recalls, [1.0]))
    precision_curve = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precision_curve) - 2, -1, -1):
        precision_curve[idx] = max(precision_curve[idx], precision_curve[idx + 1])

    change_points = np.where(recall_curve[1:] != recall_curve[:-1])[0]
    return float(
        np.sum((recall_curve[change_points + 1] - recall_curve[change_points]) * precision_curve[change_points + 1])
    )


def evaluate_segmentation(
    pred_binary: np.ndarray,
    pred_instances: np.ndarray,
    pred_scores: list[float],
    reference_binary: np.ndarray,
    reference_instances: np.ndarray,
) -> PerformanceMetrics:
    pred_binary = pred_binary.astype(bool)
    reference_binary = reference_binary.astype(bool)

    if not np.any(pred_binary) and not np.any(reference_binary):
        unavailable = float("nan")
        return PerformanceMetrics(map_50_95=unavailable, iou=unavailable, precision=unavailable, recall=unavailable)

    tp = float(np.logical_and(pred_binary, reference_binary).sum())
    fp = float(np.logical_and(pred_binary, ~reference_binary).sum())
    fn = float(np.logical_and(~pred_binary, reference_binary).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    pred_masks = _extract_instance_masks(pred_instances)
    gt_masks = _extract_instance_masks(reference_instances)
    if len(pred_scores) != len(pred_masks):
        pred_scores = [1.0 for _ in pred_masks]

    thresholds = np.arange(0.5, 1.0, 0.05)
    ap_values = [
        _average_precision(pred_masks=pred_masks, pred_scores=pred_scores, gt_masks=gt_masks, iou_threshold=float(th))
        for th in thresholds
    ]
    finite_ap = np.asarray([value for value in ap_values if np.isfinite(value)], dtype=np.float32)
    map_50_95 = float(np.mean(finite_ap)) if finite_ap.size else float("nan")

    return PerformanceMetrics(
        map_50_95=float(np.clip(map_50_95, 0.0, 1.0)) if np.isfinite(map_50_95) else float("nan"),
        iou=float(np.clip(iou, 0.0, 1.0)),
        precision=float(np.clip(precision, 0.0, 1.0)),
        recall=float(np.clip(recall, 0.0, 1.0)),
    )
