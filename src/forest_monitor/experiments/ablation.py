"""Week 6 ablation study utilities for the forest monitoring system."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import mapping

from forest_monitor.constants import CLASS_NAMES
from forest_monitor.environment import generate_air_quality_history, generate_wildfire_events
from forest_monitor.geometry import geometry_centroid
from forest_monitor.models.evaluation import (
    accumulate_confusion,
    build_target_from_label_file,
    compute_f1_per_class,
    confusion_to_metrics,
    label_map_to_detection_prediction,
    save_confusion_matrix,
    write_metrics,
)
from forest_monitor.pipeline.ndvi import compute_ndvi

SAMPLE_ID_RE = re.compile(r"^(?P<region>[A-Za-z0-9-]+)_(?P<patch_idx>\d{5})_(?P<aug>[A-Za-z0-9_]+)$")
AUGMENTATION_GAIN = {
    "base": (1.0, 0.0),
    "bright": (1.08, 12.0 / 255.0),
    "contrast": (0.92, -8.0 / 255.0),
}
SEVERITY_TO_SCORE = {"Low": 0.25, "Moderate": 0.55, "High": 0.85}


@dataclass(slots=True)
class AblationVariant:
    name: str
    display_name: str
    use_segmentation: bool
    use_ndvi: bool
    use_nir: bool
    use_environment: bool
    segmentation_weight: float
    spectral_weight: float


@dataclass(slots=True)
class SampleToken:
    sample_id: str
    region_name: str
    patch_index: int
    augmentation: str


@dataclass(slots=True)
class EnvironmentContext:
    aqi_score: float
    wildfire_score: float
    stress_score: float
    wildfire_count: int


@dataclass(slots=True)
class ReconstructedSample:
    sample_id: str
    rgb: np.ndarray
    multispectral: np.ndarray
    ndvi: np.ndarray
    geometry: dict[str, Any]
    target: dict[str, torch.Tensor]
    target_map: np.ndarray


def _default_regions() -> list[Any]:
    from forest_monitor.data.pipeline import DEFAULT_REGIONS

    return list(DEFAULT_REGIONS)


def _sample_patch_polygons(region: Any, patch_size_px: int, seed: int) -> list[Any]:
    from forest_monitor.data.pipeline import sample_patch_polygons

    return sample_patch_polygons(region, patch_edge_m=patch_size_px * 10, seed=seed)


def _fetch_sentinel_patch(polygon: Any, start_date: str, end_date: str, patch_size_px: int) -> np.ndarray | None:
    from forest_monitor.data.pipeline import fetch_sentinel_patch

    return fetch_sentinel_patch(polygon, start_date=start_date, end_date=end_date, patch_size_px=patch_size_px)


class MultispectralSampleBuilder:
    def __init__(
        self,
        dataset_root: Path,
        split: str = "test",
        patch_size_px: int = 640,
        base_seed: int = 17,
        cache_dir: Path | None = None,
        regions: Iterable[Any] | None = None,
    ) -> None:
        region_list = list(regions) if regions is not None else _default_regions()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.patch_size_px = patch_size_px
        self.base_seed = base_seed
        self.cache_dir = cache_dir or (self.dataset_root / ".cache_multispectral" / split)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_paths = sorted(list((self.dataset_root / split / "images").glob("*.png")) + list((self.dataset_root / split / "images").glob("*.jpg")))
        self.regions = {region.name: region for region in region_list}
        self.region_index = {region.name: idx for idx, region in enumerate(region_list)}
        self.patch_cache: dict[tuple[str, int], list[Any]] = {}

    def __iter__(self):
        for image_path in self.image_paths:
            yield self.build_sample(image_path)

    def _parse_sample_id(self, sample_id: str) -> SampleToken:
        match = SAMPLE_ID_RE.match(sample_id)
        if match is None:
            raise ValueError(f"Unsupported sample_id format: {sample_id}")
        return SampleToken(
            sample_id=sample_id,
            region_name=match.group("region"),
            patch_index=int(match.group("patch_idx")),
            augmentation=match.group("aug"),
        )

    def _region_polygon(self, token: SampleToken):
        key = (token.region_name, self.patch_size_px)
        if key not in self.patch_cache:
            region = self.regions[token.region_name]
            seed = self.base_seed + self.region_index[token.region_name]
            self.patch_cache[key] = _sample_patch_polygons(region, patch_size_px=self.patch_size_px, seed=seed)
        return self.patch_cache[key][token.patch_index]

    def _apply_multispectral_augmentation(self, stack: np.ndarray, augmentation: str) -> np.ndarray:
        if augmentation == "flip_h":
            return np.flip(stack, axis=1).copy()
        if augmentation == "flip_v":
            return np.flip(stack, axis=0).copy()
        if augmentation == "rot90":
            return np.rot90(stack, k=3).copy()
        gain, bias = AUGMENTATION_GAIN.get(augmentation, AUGMENTATION_GAIN["base"])
        return np.clip((stack * gain) + bias, 0.0, 1.0).astype(np.float32)

    def _load_multispectral(self, token: SampleToken) -> tuple[np.ndarray, dict[str, Any]]:
        cache_path = self.cache_dir / f"{token.sample_id}.npz"
        polygon = self._region_polygon(token)
        geometry = mapping(polygon)
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached["stack"].astype(np.float32), geometry

        region = self.regions[token.region_name]
        stack = _fetch_sentinel_patch(polygon, start_date=region.start_date, end_date=region.end_date, patch_size_px=self.patch_size_px)
        if stack is None:
            raise RuntimeError(f"Unable to fetch Sentinel-2 patch for {token.sample_id}")
        stack = self._apply_multispectral_augmentation(stack.astype(np.float32), token.augmentation)
        np.savez_compressed(cache_path, stack=stack)
        return stack, geometry

    def build_sample(self, image_path: Path) -> ReconstructedSample:
        token = self._parse_sample_id(image_path.stem)
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        label_path = self.dataset_root / self.split / "labels" / f"{image_path.stem}.txt"
        target, target_map = build_target_from_label_file(label_path, image_shape=(height, width))
        multispectral, geometry = self._load_multispectral(token)
        if multispectral.shape[:2] != rgb.shape[:2]:
            multispectral = cv2.resize(multispectral, (width, height), interpolation=cv2.INTER_LINEAR)
        ndvi = compute_ndvi(multispectral[..., 0], multispectral[..., 3])
        return ReconstructedSample(
            sample_id=image_path.stem,
            rgb=rgb,
            multispectral=multispectral,
            ndvi=ndvi.astype(np.float32),
            geometry=geometry,
            target=target,
            target_map=target_map,
        )


def default_ablation_variants() -> list[AblationVariant]:
    return [
        AblationVariant("full_model", "Full model", True, True, True, True, 0.64, 0.36),
        AblationVariant("no_ndvi", "Without NDVI", True, False, True, True, 0.70, 0.30),
        AblationVariant("no_nir", "Without NIR band", True, False, False, True, 0.76, 0.24),
        AblationVariant("ndvi_only", "Without segmentation (NDVI only)", False, True, True, True, 0.0, 1.0),
        AblationVariant("no_environment", "Without environmental context", True, True, True, False, 0.64, 0.36),
    ]


def _yolo_score_stack(result, image_shape: tuple[int, int], score_threshold: float = 0.25) -> np.ndarray:
    height, width = image_shape
    scores = np.zeros((height, width, len(CLASS_NAMES)), dtype=np.float32)
    if result.masks is None or result.boxes is None:
        return scores
    masks = result.masks.data.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(confidences)
    for idx in order:
        if confidences[idx] < score_threshold:
            continue
        label = int(labels[idx])
        mask = cv2.resize(masks[idx].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
        scores[..., label][mask] = np.maximum(scores[..., label][mask], float(confidences[idx]))
    return scores


def _mask_rcnn_score_stack(output: dict[str, torch.Tensor], image_shape: tuple[int, int], score_threshold: float = 0.5) -> np.ndarray:
    height, width = image_shape
    scores = np.zeros((height, width, len(CLASS_NAMES)), dtype=np.float32)
    if len(output["scores"]) == 0:
        return scores
    confidences = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy().astype(np.int64)
    masks = output["masks"].detach().cpu().numpy()
    order = np.argsort(confidences)
    for idx in order:
        if confidences[idx] < score_threshold:
            continue
        label = int(labels[idx]) - 1
        mask = cv2.resize((masks[idx, 0] > 0.5).astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) >= 0.5
        scores[..., label][mask] = np.maximum(scores[..., label][mask], float(confidences[idx]))
    return scores


def _normalize_stack(stack: np.ndarray) -> np.ndarray:
    total = stack.sum(axis=-1, keepdims=True)
    return (stack / np.clip(total, 1e-6, None)).astype(np.float32)


def _spectral_score_stack(multispectral: np.ndarray, ndvi: np.ndarray, variant: AblationVariant) -> np.ndarray:
    red = multispectral[..., 0]
    green = multispectral[..., 1]
    blue = multispectral[..., 2]
    nir = multispectral[..., 3]
    brightness = np.clip((red + green + blue) / 3.0, 0.0, 1.0)
    visible_veg = np.clip((2.0 * green) - red - blue, 0.0, 1.0)
    color_variance = np.clip(np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue), 0.0, 1.5)
    grayness = np.clip(1.0 - (color_variance / 1.5), 0.0, 1.0)

    if variant.use_nir:
        ndwi = (green - nir) / (green + nir + 1e-6)
        nir_score = np.clip((nir - 0.28) / 0.32, 0.0, 1.0)
    else:
        ndwi = np.clip((blue - red) / (blue + red + 1e-6), -1.0, 1.0)
        nir_score = np.clip(visible_veg, 0.0, 1.0)

    ndvi_signal = np.clip((ndvi - 0.20) / 0.45, 0.0, 1.0) if variant.use_ndvi else np.zeros_like(red)
    forest = np.clip((0.58 * ndvi_signal) + (0.28 * nir_score) + (0.14 * visible_veg), 0.0, 1.0)
    if not variant.use_ndvi and variant.use_nir:
        forest = np.clip((0.72 * nir_score) + (0.28 * visible_veg), 0.0, 1.0)
    if not variant.use_nir:
        forest = np.clip((0.78 * visible_veg) + (0.22 * np.clip((green - red) * 1.8, 0.0, 1.0)), 0.0, 1.0)

    if variant.use_ndvi:
        field = np.clip(1.0 - (np.abs(ndvi - 0.23) / 0.20), 0.0, 1.0)
    else:
        field = np.clip(1.0 - (np.abs(visible_veg - 0.22) / 0.24), 0.0, 1.0)
    field *= (1.0 - np.clip((nir_score - 0.55) / 0.35, 0.0, 0.75))

    lake = np.clip((ndwi - 0.05) / 0.45, 0.0, 1.0) * np.clip(1.0 - brightness, 0.35, 1.0)
    road = np.clip((brightness - 0.24) / 0.38, 0.0, 1.0) * grayness * (1.0 - forest)
    building = np.clip((brightness - 0.35) / 0.45, 0.0, 1.0) * np.clip(1.0 - visible_veg, 0.0, 1.0)

    stack = np.stack([forest, field, lake, road, building], axis=-1) + 1e-4
    return _normalize_stack(stack)


def _environment_context(geometry: dict[str, Any]) -> EnvironmentContext:
    latitude, longitude = geometry_centroid(geometry)
    wildfire_events = generate_wildfire_events(geometry, latitude=latitude, longitude=longitude)
    aqi_history = generate_air_quality_history(geometry, latitude=latitude, longitude=longitude)
    latest_aqi = aqi_history[-1].us_aqi if aqi_history else 60.0
    aqi_score = float(np.clip(latest_aqi / 150.0, 0.0, 1.0))
    wildfire_values = []
    for event in wildfire_events:
        severity_score = SEVERITY_TO_SCORE.get(event.severity, 0.45)
        distance_factor = 1.0 - min(event.distance_km / 350.0, 0.9)
        wildfire_values.append(severity_score * max(distance_factor, 0.1))
    wildfire_score = float(np.clip(np.mean(wildfire_values) if wildfire_values else 0.0, 0.0, 1.0))
    stress_score = float(np.clip((0.52 * aqi_score) + (0.48 * wildfire_score), 0.0, 1.0))
    return EnvironmentContext(aqi_score=aqi_score, wildfire_score=wildfire_score, stress_score=stress_score, wildfire_count=len(wildfire_events))


def _apply_environment_adjustment(score_stack: np.ndarray, segmentation_stack: np.ndarray, spectral_stack: np.ndarray, context: EnvironmentContext) -> np.ndarray:
    adjusted = score_stack.copy()
    if context.stress_score <= 0.0:
        return adjusted
    disagreement = np.abs(segmentation_stack[..., 0] - spectral_stack[..., 0])
    degradation = context.stress_score * (0.35 + (0.65 * disagreement)) * (1.0 - spectral_stack[..., 0])
    adjusted[..., 0] *= (1.0 - (0.16 * degradation))
    adjusted[..., 1] += 0.07 * degradation
    adjusted[..., 3] += 0.03 * degradation
    adjusted[..., 4] += 0.06 * degradation
    return _normalize_stack(adjusted + 1e-4)


def _fuse_prediction(sample: ReconstructedSample, variant: AblationVariant, yolo_model: Any, mask_model: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    height, width = sample.rgb.shape[:2]
    segmentation_stack = np.zeros((height, width, len(CLASS_NAMES)), dtype=np.float32)

    if variant.use_segmentation:
        yolo_result = yolo_model.predict(source=sample.rgb, imgsz=max(height, width), conf=0.25, retina_masks=True, verbose=False)[0]
        yolo_stack = _yolo_score_stack(yolo_result, image_shape=(height, width), score_threshold=0.25)
        image_tensor = torch.from_numpy(sample.rgb.transpose(2, 0, 1)).float().to(device) / 255.0
        with torch.no_grad():
            mask_output = mask_model([image_tensor])[0]
        mask_output = {key: value.detach().cpu() for key, value in mask_output.items()}
        mask_stack = _mask_rcnn_score_stack(mask_output, image_shape=(height, width), score_threshold=0.5)
        segmentation_stack = _normalize_stack((0.55 * yolo_stack) + (0.45 * mask_stack) + 1e-4)

    spectral_stack = _spectral_score_stack(sample.multispectral, sample.ndvi, variant=variant)
    fused = (variant.segmentation_weight * segmentation_stack) + (variant.spectral_weight * spectral_stack) if variant.use_segmentation else spectral_stack.copy()

    if variant.use_environment:
        fused = _apply_environment_adjustment(
            score_stack=_normalize_stack(fused + 1e-4),
            segmentation_stack=segmentation_stack if variant.use_segmentation else spectral_stack,
            spectral_stack=spectral_stack,
            context=_environment_context(sample.geometry),
        )
    else:
        fused = _normalize_stack(fused + 1e-4)

    label_map = np.argmax(fused, axis=-1).astype(np.int16)
    return label_map, fused


def _evaluate_variant(variant: AblationVariant, dataset: MultispectralSampleBuilder, yolo_model: Any, mask_model: Any, device: torch.device, output_dir: Path) -> dict[str, float]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    map_metric = MeanAveragePrecision(iou_type="segm")
    confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)

    for sample in dataset:
        label_map, score_stack = _fuse_prediction(sample, variant, yolo_model=yolo_model, mask_model=mask_model, device=device)
        prediction = label_map_to_detection_prediction(label_map, score_stack=score_stack)
        map_metric.update([prediction], [sample.target])
        accumulate_confusion(confusion, prediction=label_map, target=sample.target_map)

    map_scores = map_metric.compute()
    pixel_metrics = confusion_to_metrics(confusion)
    f1_scores = compute_f1_per_class(confusion)
    metrics = {
        "variant": variant.display_name,
        "map50": float(map_scores["map_50"].item()),
        "map50_95": float(map_scores["map"].item()),
        "iou": pixel_metrics["iou"],
        "precision": pixel_metrics["precision"],
        "recall": pixel_metrics["recall"],
        "f1": pixel_metrics["f1"],
        **{f"f1_{class_name}": score for class_name, score in f1_scores.items()},
    }
    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, variant_dir / "metrics.json")
    save_confusion_matrix(confusion, variant_dir / "confusion_matrix.png", title=f"{variant.display_name} Confusion Matrix")
    return metrics


def run_ablation_study(
    dataset_root: Path,
    yolo_checkpoint: Path,
    maskrcnn_checkpoint: Path,
    output_dir: Path,
    variants: Iterable[AblationVariant] | None = None,
    patch_size_px: int = 640,
    base_seed: int = 17,
) -> tuple[pd.DataFrame, list[str]]:
    from ultralytics import YOLO
    from forest_monitor.models.mask_rcnn_dataset import build_mask_rcnn_model

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_variants = list(variants or default_ablation_variants())

    yolo_model = YOLO(str(yolo_checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model = build_mask_rcnn_model()
    checkpoint = torch.load(maskrcnn_checkpoint, map_location=device)
    mask_model.load_state_dict(checkpoint["model_state_dict"])
    mask_model.to(device)
    mask_model.eval()

    rows = []
    for variant in active_variants:
        dataset = MultispectralSampleBuilder(
            dataset_root=dataset_root,
            split="test",
            patch_size_px=patch_size_px,
            base_seed=base_seed,
            cache_dir=output_dir / "cache" / variant.name,
        )
        rows.append(_evaluate_variant(variant, dataset, yolo_model=yolo_model, mask_model=mask_model, device=device, output_dir=output_dir))

    table = pd.DataFrame(rows).sort_values("map50_95", ascending=False).reset_index(drop=True)
    table.to_csv(output_dir / "ablation_results.csv", index=False)
    table.to_json(output_dir / "ablation_results.json", orient="records", indent=2)

    baseline = table.set_index("variant").loc["Full model"]
    conclusions: list[str] = []
    for _, row in table.iterrows():
        if row["variant"] == "Full model":
            continue
        accuracy_delta = ((baseline["map50_95"] - row["map50_95"]) / max(baseline["map50_95"], 1e-6)) * 100.0
        iou_delta = ((baseline["iou"] - row["iou"]) / max(baseline["iou"], 1e-6)) * 100.0
        direction = "reduced" if accuracy_delta >= 0.0 else "improved"
        conclusions.append(
            f"{row['variant']} {direction} mAP@50-95 by {abs(accuracy_delta):.2f}% and IoU by {abs(iou_delta):.2f}% relative to the full model."
        )
    (output_dir / "conclusions.txt").write_text("\n".join(conclusions), encoding="utf-8")
    return table, conclusions
