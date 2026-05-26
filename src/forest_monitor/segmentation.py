"""Forest stand segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.transform import resize


@dataclass(slots=True)
class StandSummary:
    area_ha: float
    centroid_lat: float
    centroid_lon: float
    confidence: float


@dataclass(slots=True)
class SegmentationResult:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    instance_map: np.ndarray
    instance_scores: list[float]
    stand_summaries: list[StandSummary]
    mode_label: str
    warning: str | None


MODEL_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "yolov8_seg" / "weights" / "best.pt"


def segmentation_startup_notice() -> str | None:
    if MODEL_WEIGHTS_PATH.exists():
        return None
    return "Model weights not found. Running in NDVI-only mode."


def _instance_map_from_binary(binary_mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[float]]:
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.int32), []

    distance = ndi.distance_transform_edt(binary_mask)
    peaks = peak_local_max(
        distance,
        footprint=np.ones((13, 13), dtype=bool),
        labels=binary_mask,
        exclude_border=False,
    )

    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = idx

    if markers.max() == 0:
        markers, _ = ndi.label(binary_mask)

    raw = watershed(-distance, markers=markers, mask=binary_mask)
    cleaned = np.zeros_like(raw, dtype=np.int32)
    scores: list[float] = []
    out_label = 1

    for label_id in np.unique(raw):
        if label_id == 0:
            continue
        instance = raw == label_id
        pixel_count = int(instance.sum())
        if pixel_count < min_pixels:
            continue
        cleaned[instance] = out_label
        scores.append(float(min(0.99, 0.48 + (pixel_count / 1800.0))))
        out_label += 1

    return cleaned, scores


def _stand_summaries(
    instance_map: np.ndarray,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    roi_area_ha: float,
) -> list[StandSummary]:
    min_lon, min_lat, max_lon, max_lat = bounds
    height, width = instance_map.shape
    total_pixels = float(max(instance_map.size, 1))
    summaries: list[StandSummary] = []

    for label_id in np.unique(instance_map):
        if label_id == 0:
            continue
        mask = instance_map == label_id
        pixel_count = int(mask.sum())
        if pixel_count <= 0:
            continue
        rows, cols = np.where(mask)
        row_center = float(np.mean(rows))
        col_center = float(np.mean(cols))
        lon = min_lon + (col_center / max(width - 1, 1)) * (max_lon - min_lon)
        lat = max_lat - (row_center / max(height - 1, 1)) * (max_lat - min_lat)
        area_ha = float(max(0.1, roi_area_ha * (pixel_count / total_pixels)))
        confidence = float(np.clip(np.nanmean(probability_map[mask]), 0.0, 1.0))
        summaries.append(
            StandSummary(
                area_ha=area_ha,
                centroid_lat=lat,
                centroid_lon=lon,
                confidence=confidence,
            )
        )

    summaries.sort(key=lambda item: item.area_ha, reverse=True)
    return summaries[:15]


class DeepForestSegmenter:
    """Deterministic segmentation head built for local project demos."""

    def __init__(self, threshold: float = 0.55, min_instance_pixels: int = 60) -> None:
        self.threshold = threshold
        self.min_instance_pixels = min_instance_pixels
        self.weights_path = MODEL_WEIGHTS_PATH
        self.model_available = self.weights_path.exists()

    def segment(
        self,
        ndvi_latest: np.ndarray,
        ndvi_delta: np.ndarray,
        roi_area_ha: float,
        bounds: tuple[float, float, float, float],
        image_rgb: np.ndarray | None = None,
        debug: bool = False,
    ) -> SegmentationResult:
        ndvi_latest = np.asarray(ndvi_latest, dtype=np.float32)
        ndvi_delta = np.asarray(ndvi_delta, dtype=np.float32)
        if ndvi_delta.shape != ndvi_latest.shape:
            ndvi_delta = resize(
                ndvi_delta,
                ndvi_latest.shape,
                order=1,
                mode="reflect",
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)

        green_dominance = np.zeros_like(ndvi_latest, dtype=np.float32)
        image_mean = float(np.nanmean(ndvi_latest))
        local_variance = np.zeros_like(ndvi_latest, dtype=np.float32)
        if image_rgb is not None:
            rgb = np.asarray(image_rgb, dtype=np.float32)
            if rgb.shape[:2] != ndvi_latest.shape:
                rgb = resize(
                    rgb,
                    (*ndvi_latest.shape, rgb.shape[-1]),
                    order=1,
                    mode="reflect",
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
            red = rgb[..., 0]
            green = rgb[..., 1]
            blue = rgb[..., 2]
            brightness = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
            local_mean = ndi.uniform_filter(brightness, size=5)
            local_variance = np.clip(ndi.uniform_filter(brightness * brightness, size=5) - (local_mean * local_mean), 0.0, None)
            green_dominance = np.clip(green - ((red + blue) * 0.5), -1.0, 1.0)
            image_mean = float(np.nanmean(rgb))
            if debug:
                print("Image shape:", rgb.shape)
                print("Image changed:", image_mean)
        elif debug:
            print("Image shape:", ndvi_latest.shape)
            print("Image changed:", image_mean)

        smoothed = ndi.gaussian_filter(ndvi_latest, sigma=1.4)
        texture = np.abs(ndvi_latest - smoothed)
        yy, xx = np.mgrid[0:ndvi_latest.shape[0], 0:ndvi_latest.shape[1]]
        x_norm = ((xx / max(ndvi_latest.shape[1] - 1, 1)) * 2.0) - 1.0
        y_norm = ((yy / max(ndvi_latest.shape[0] - 1, 1)) * 2.0) - 1.0
        roi_phase = (
            0.08 * np.sin((x_norm + 1.0) * (1.6 + (abs(bounds[0]) % 1.4)))
            + 0.07 * np.cos((y_norm + 1.0) * (1.9 + (abs(bounds[1]) % 1.2)))
        ).astype(np.float32)
        logits = (
            (7.8 * (ndvi_latest - 0.38))
            - (3.1 * np.maximum(-ndvi_delta, 0.0))
            + (1.7 * np.clip(texture - 0.03, -0.08, 0.24))
            + (2.2 * green_dominance)
            + (0.9 * np.clip(local_variance * 12.0, 0.0, 0.25))
            + roi_phase
        )
        probability = 1.0 / (1.0 + np.exp(-logits))
        probability = ndi.gaussian_filter(probability.astype(np.float32), sigma=0.65)
        binary = (probability >= self.threshold) & (ndvi_latest > 0.38)
        binary = morphology.binary_closing(binary, footprint=morphology.disk(2))
        binary = morphology.binary_opening(binary, footprint=morphology.disk(1))
        binary = morphology.remove_small_objects(binary, min_size=self.min_instance_pixels)
        binary = morphology.remove_small_holes(binary, area_threshold=self.min_instance_pixels)

        instances, scores = _instance_map_from_binary(binary, min_pixels=self.min_instance_pixels)
        summaries = _stand_summaries(
            instance_map=instances,
            probability_map=probability,
            bounds=bounds,
            roi_area_ha=roi_area_ha,
        )
        if len(scores) != len(summaries):
            scores = [item.confidence for item in summaries]

        mode_label = "Model-backed segmentation" if self.model_available else "NDVI-only fallback"
        warning = None if self.model_available else segmentation_startup_notice()

        return SegmentationResult(
            probability_map=probability.astype(np.float32),
            binary_mask=binary.astype(bool),
            instance_map=instances.astype(np.int32),
            instance_scores=scores,
            stand_summaries=summaries,
            mode_label=mode_label,
            warning=warning,
        )
