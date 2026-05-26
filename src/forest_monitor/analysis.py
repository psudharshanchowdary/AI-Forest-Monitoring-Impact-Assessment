"""End-to-end monitoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.transform import resize

from .environment import carbon_density_profile
from .geometry import approximate_area_ha, geometry_bounds, geometry_centroid
from .metrics import PerformanceMetrics, evaluate_segmentation
from .pipeline.ndvi import compute_ndvi
from .segmentation import DeepForestSegmenter, SegmentationResult
from .synthetic import SceneData, generate_scene_pair

FOREST_NDVI_THRESHOLD = 0.40
CHANGE_DROP_THRESHOLD = -0.10
NON_FOREST_MEAN_NDVI_THRESHOLD = 0.12
MIN_FOREST_PIXEL_THRESHOLD = 50
HIGH_CLOUD_COVER_THRESHOLD = 30.0


class AnalysisPipelineError(RuntimeError):
    """Base exception for user-facing analysis failures."""


class SatelliteDataFetchError(AnalysisPipelineError):
    """Raised when imagery generation or retrieval fails."""


@dataclass(slots=True)
class RiskAssessment:
    score: float
    level: str
    rationale: list[str]


@dataclass(slots=True)
class MonitoringResult:
    baseline_scene: SceneData
    current_scene: SceneData
    ndvi_change: np.ndarray
    baseline_forest_mask: np.ndarray
    current_forest_mask: np.ndarray
    forest_loss_mask: np.ndarray
    loss_area_ha: float
    carbon_loss_tco2e: float
    forest_loss_percent: float
    mean_ndvi_drop: float
    mean_current_ndvi: float
    region_classification: str
    segmentation: SegmentationResult
    metrics: PerformanceMetrics
    risk: RiskAssessment
    roi_area_ha: float
    baseline_cloud_cover_pct: float
    current_cloud_cover_pct: float
    baseline_forest_pixels: int
    current_forest_pixels: int
    carbon_density_tco2e_per_ha: float
    carbon_density_note: str
    analysis_warnings: list[str]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "baseline_item": self.baseline_scene.item_id,
            "current_item": self.current_scene.item_id,
            "baseline_date": self.baseline_scene.acquisition_date,
            "current_date": self.current_scene.acquisition_date,
            "forest_loss_percent": round(self.forest_loss_percent, 3),
            "loss_area_ha": round(self.loss_area_ha, 3),
            "carbon_loss_tco2e": round(self.carbon_loss_tco2e, 3),
            "mean_ndvi_drop": round(self.mean_ndvi_drop, 4),
            "mean_current_ndvi": round(self.mean_current_ndvi, 4),
            "risk_level": self.risk.level,
            "risk_score": round(self.risk.score, 4),
            "roi_area_ha": round(self.roi_area_ha, 3),
            "baseline_cloud_cover_pct": round(self.baseline_cloud_cover_pct, 2),
            "current_cloud_cover_pct": round(self.current_cloud_cover_pct, 2),
            "baseline_forest_pixels": self.baseline_forest_pixels,
            "current_forest_pixels": self.current_forest_pixels,
            "carbon_density_tco2e_per_ha": round(self.carbon_density_tco2e_per_ha, 2),
            "carbon_density_note": self.carbon_density_note,
            "analysis_warnings": list(self.analysis_warnings),
            "metrics": {
                "mAP_50_95": _safe_metric_value(self.metrics.map_50_95),
                "IoU": _safe_metric_value(self.metrics.iou),
                "precision": _safe_metric_value(self.metrics.precision),
                "recall": _safe_metric_value(self.metrics.recall),
            },
        }


from .pipeline.sentinel_ndvi import SentinelChangeResult, SentinelScenePaths, run_sentinel_ndvi_monitoring


def _safe_metric_value(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return round(float(value), 4)


def _align_to_shape(array: np.ndarray, target_shape: tuple[int, int], *, order: int) -> np.ndarray:
    aligned = np.asarray(array)
    if aligned.shape == target_shape:
        return aligned.astype(np.float32 if order > 0 else aligned.dtype)
    resized = resize(
        aligned,
        target_shape,
        order=order,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=(order > 0),
    )
    if order == 0:
        return resized.astype(aligned.dtype)
    return resized.astype(np.float32)


def _reference_instances(reference_binary: np.ndarray) -> np.ndarray:
    refined = morphology.closing(reference_binary, footprint=morphology.disk(2))
    refined = morphology.remove_small_objects(refined, min_size=40)
    labeled, _ = ndi.label(refined)
    return labeled.astype(np.int32)


def classify_risk(forest_loss_percent: float, mean_ndvi_drop: float, carbon_loss_tco2e: float) -> RiskAssessment:
    loss_factor = min(forest_loss_percent / 45.0, 1.0)
    drop_factor = min(abs(min(mean_ndvi_drop, 0.0)) / 0.30, 1.0)
    carbon_factor = min(carbon_loss_tco2e / 4000.0, 1.0)
    score = (0.48 * loss_factor) + (0.24 * drop_factor) + (0.28 * carbon_factor)
    rationale = [
        f"Forest loss: {forest_loss_percent:.2f}%",
        f"Mean NDVI drop: {mean_ndvi_drop:.3f}",
        f"Estimated carbon loss: {carbon_loss_tco2e:.1f} tCO2e",
    ]
    if score < 0.33:
        return RiskAssessment(score=score, level="Low", rationale=rationale)
    if score < 0.67:
        return RiskAssessment(score=score, level="Medium", rationale=rationale)
    return RiskAssessment(score=score, level="High", rationale=rationale)


def _stat_line(name: str, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        print(f"{name}: no valid pixels")
        return
    print(
        f"{name} min/max/mean/std: "
        f"{float(np.nanmin(finite)):.4f} / "
        f"{float(np.nanmax(finite)):.4f} / "
        f"{float(np.nanmean(finite)):.4f} / "
        f"{float(np.nanstd(finite)):.4f}"
    )


def run_monitoring_pipeline(
    geometry: dict[str, Any],
    baseline_start: date,
    baseline_end: date,
    current_start: date,
    current_end: date,
    progress_callback: Callable[[float, str], None] | None = None,
) -> MonitoringResult:
    def report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)

    report(0.08, "Step 1/4: Fetching satellite data...")
    try:
        baseline, current = generate_scene_pair(
            geometry=geometry,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
        )
    except Exception as exc:
        raise SatelliteDataFetchError(
            "Satellite data fetch failed. Please try again or select a smaller region."
        ) from exc

    report(0.35, "Step 2/4: Computing NDVI...")
    baseline.ndvi = compute_ndvi(baseline.red, baseline.nir).astype(np.float32)
    current.ndvi = compute_ndvi(current.red, current.nir).astype(np.float32)

    if current.ndvi.shape != baseline.ndvi.shape:
        target_shape = baseline.ndvi.shape
        current.ndvi = _align_to_shape(current.ndvi, target_shape, order=1)
        current.red = _align_to_shape(current.red, target_shape, order=1)
        current.green = _align_to_shape(current.green, target_shape, order=1)
        current.blue = _align_to_shape(current.blue, target_shape, order=1)
        current.nir = _align_to_shape(current.nir, target_shape, order=1)
        current.rgb = np.dstack([current.red, current.green, current.blue]).astype(np.float32)
        current.valid_mask = _align_to_shape(current.valid_mask.astype(np.uint8), target_shape, order=0) > 0
        current.land_cover = _align_to_shape(current.land_cover, target_shape, order=0).astype(np.uint8)
        current.forest_reference = _align_to_shape(current.forest_reference.astype(np.uint8), target_shape, order=0) > 0

    baseline.valid_mask = np.asarray(baseline.valid_mask, dtype=bool) & np.isfinite(baseline.ndvi)
    current.valid_mask = np.asarray(current.valid_mask, dtype=bool) & np.isfinite(current.ndvi)
    common_valid = baseline.valid_mask & current.valid_mask

    ndvi_change = (current.ndvi - baseline.ndvi).astype(np.float32)
    ndvi_change[~common_valid] = np.nan

    baseline_valid_ndvi = baseline.ndvi[baseline.valid_mask]
    current_valid_ndvi = current.ndvi[current.valid_mask]
    mean_baseline_ndvi = float(np.nanmean(baseline_valid_ndvi)) if baseline_valid_ndvi.size else 0.0
    mean_current_ndvi = float(np.nanmean(current_valid_ndvi)) if current_valid_ndvi.size else 0.0

    _stat_line("Baseline NDVI", baseline.ndvi[baseline.valid_mask])
    _stat_line("Current NDVI", current.ndvi[current.valid_mask])

    roi_area_ha = approximate_area_ha(geometry)
    bounds = geometry_bounds(geometry)
    centroid_lat, _ = geometry_centroid(geometry)
    carbon_profile = carbon_density_profile(centroid_lat)
    baseline_cloud_cover_pct = float(getattr(baseline, "cloud_cover_pct", 0.0))
    current_cloud_cover_pct = float(getattr(current, "cloud_cover_pct", 0.0))
    baseline_forest_pixels = int(((baseline.ndvi > FOREST_NDVI_THRESHOLD) & baseline.valid_mask).sum())
    current_forest_pixels = int(((current.ndvi > FOREST_NDVI_THRESHOLD) & current.valid_mask).sum())
    analysis_warnings: list[str] = []

    if max(baseline_cloud_cover_pct, current_cloud_cover_pct) > HIGH_CLOUD_COVER_THRESHOLD:
        analysis_warnings.append(
            "Cloud cover exceeds 30% in one or more scenes. Results may be inaccurate."
        )

    no_forest_detected = baseline_forest_pixels < MIN_FOREST_PIXEL_THRESHOLD
    if no_forest_detected:
        analysis_warnings.append("No forest detected in this region. Try a forested area.")

    non_forest_region = bool(
        no_forest_detected
        or (not current_valid_ndvi.size)
        or (mean_current_ndvi < NON_FOREST_MEAN_NDVI_THRESHOLD)
    )

    if non_forest_region:
        baseline_forest = np.zeros_like(baseline.ndvi, dtype=bool)
        current_forest = np.zeros_like(current.ndvi, dtype=bool)
        forest_loss = np.zeros_like(current.ndvi, dtype=bool)
        forest_loss_percent = 0.0
        mean_ndvi_drop = 0.0
        loss_area_ha = 0.0
        carbon_loss_tco2e = 0.0
    else:
        baseline_forest = (baseline.ndvi > FOREST_NDVI_THRESHOLD) & baseline.valid_mask
        current_forest = (current.ndvi > FOREST_NDVI_THRESHOLD) & current.valid_mask
        smoothed_change = ndi.gaussian_filter(np.nan_to_num(ndvi_change, nan=0.0), sigma=0.75).astype(np.float32)
        smoothed_change[~common_valid] = np.nan
        forest_loss = baseline_forest & common_valid & (smoothed_change < CHANGE_DROP_THRESHOLD)
        forest_loss = morphology.binary_closing(forest_loss, footprint=morphology.disk(2))
        forest_loss = morphology.binary_opening(forest_loss, footprint=morphology.disk(1))
        forest_loss = morphology.remove_small_objects(forest_loss, min_size=24)
        forest_loss = morphology.remove_small_holes(forest_loss, area_threshold=24)

        baseline_forest_pixels_for_ratio = max(float(baseline_forest.sum()), 1.0)
        forest_loss_pixels = float(forest_loss.sum())
        forest_loss_percent = (forest_loss_pixels / baseline_forest_pixels_for_ratio) * 100.0
        mean_ndvi_drop = float(np.nanmean(smoothed_change[forest_loss])) if forest_loss_pixels > 0 else 0.0
        loss_area_ha = roi_area_ha * (forest_loss_pixels / float(max(current.ndvi.size, 1)))
        severity = np.clip(abs(min(mean_ndvi_drop, 0.0)) / 0.24, 0.25, 1.6)
        carbon_loss_tco2e = float(loss_area_ha * carbon_profile.tco2e_per_ha * severity)

    dndvi_min = float(np.nanmin(ndvi_change)) if np.isfinite(ndvi_change).any() else 0.0
    dndvi_max = float(np.nanmax(ndvi_change)) if np.isfinite(ndvi_change).any() else 0.0
    dndvi_mean = float(np.nanmean(ndvi_change)) if np.isfinite(ndvi_change).any() else 0.0
    dndvi_std = float(np.nanstd(ndvi_change)) if np.isfinite(ndvi_change).any() else 0.0
    forest_pixel_count = int(((current.ndvi > FOREST_NDVI_THRESHOLD) & current.valid_mask).sum())
    loss_pixel_count = int(forest_loss.sum()) if not non_forest_region else 0
    print("dNDVI min:", dndvi_min)
    print("dNDVI max:", dndvi_max)
    print("dNDVI mean:", dndvi_mean)
    print("dNDVI std:", dndvi_std)
    print("Forest pixels:", forest_pixel_count)
    print("Loss pixels:", loss_pixel_count)
    print("Baseline cloud cover %:", baseline_cloud_cover_pct)
    print("Current cloud cover %:", current_cloud_cover_pct)

    report(0.62, "Step 3/4: Running segmentation...")
    segmenter = DeepForestSegmenter()
    segmentation = segmenter.segment(
        ndvi_latest=np.where(current.valid_mask, current.ndvi, -1.0),
        ndvi_delta=np.where(common_valid, np.nan_to_num(ndvi_change, nan=0.0), 0.0),
        roi_area_ha=roi_area_ha,
        bounds=bounds,
        image_rgb=current.rgb,
        debug=True,
    )
    if non_forest_region:
        segmentation = SegmentationResult(
            probability_map=segmentation.probability_map,
            binary_mask=np.zeros_like(segmentation.binary_mask, dtype=bool),
            instance_map=np.zeros_like(segmentation.instance_map, dtype=np.int32),
            instance_scores=[],
            stand_summaries=[],
            mode_label=segmentation.mode_label,
            warning=segmentation.warning,
        )

    report(0.82, "Step 4/4: Calculating metrics...")
    reference_binary = morphology.remove_small_objects(current.forest_reference & current.valid_mask, min_size=40)
    if non_forest_region:
        reference_binary = np.zeros_like(reference_binary, dtype=bool)
    reference_instances = _reference_instances(reference_binary)
    metrics = evaluate_segmentation(
        pred_binary=segmentation.binary_mask,
        pred_instances=segmentation.instance_map,
        pred_scores=segmentation.instance_scores,
        reference_binary=reference_binary,
        reference_instances=reference_instances,
    )

    report(1.0, "Analysis complete")
    risk = classify_risk(
        forest_loss_percent=float(np.clip(forest_loss_percent, 0.0, 100.0)),
        mean_ndvi_drop=mean_ndvi_drop,
        carbon_loss_tco2e=carbon_loss_tco2e,
    )
    if non_forest_region:
        risk.rationale.append(
            f"Mean NDVI {mean_current_ndvi:.3f} is below {NON_FOREST_MEAN_NDVI_THRESHOLD:.2f}; classified as non-forest."
        )
    return MonitoringResult(
        baseline_scene=baseline,
        current_scene=current,
        ndvi_change=ndvi_change,
        baseline_forest_mask=baseline_forest,
        current_forest_mask=current_forest,
        forest_loss_mask=forest_loss,
        loss_area_ha=float(loss_area_ha),
        carbon_loss_tco2e=float(carbon_loss_tco2e),
        forest_loss_percent=float(np.clip(forest_loss_percent, 0.0, 100.0)),
        mean_ndvi_drop=float(mean_ndvi_drop),
        mean_current_ndvi=float(mean_current_ndvi),
        region_classification=("Non-forest region" if non_forest_region else "Forest region"),
        segmentation=segmentation,
        metrics=metrics,
        risk=risk,
        roi_area_ha=roi_area_ha,
        baseline_cloud_cover_pct=baseline_cloud_cover_pct,
        current_cloud_cover_pct=current_cloud_cover_pct,
        baseline_forest_pixels=baseline_forest_pixels,
        current_forest_pixels=current_forest_pixels,
        carbon_density_tco2e_per_ha=float(carbon_profile.tco2e_per_ha),
        carbon_density_note=carbon_profile.disclaimer,
        analysis_warnings=analysis_warnings,
    )


def run_monitoring_pipeline_from_sentinel_rasters(
    *,
    baseline_scene: SentinelScenePaths,
    current_scene: SentinelScenePaths,
    roi_geometry: dict[str, Any],
    segmentation: Any = None,
    smoothing_sigma: float = 1.0,
    debug: bool = True,
) -> SentinelChangeResult:
    try:
        return run_sentinel_ndvi_monitoring(
            baseline_scene=baseline_scene,
            current_scene=current_scene,
            roi_geometry=roi_geometry,
            segmentation=segmentation,
            smoothing_sigma=smoothing_sigma,
            debug=debug,
        )
    except Exception as exc:
        raise SatelliteDataFetchError(
            "Satellite data fetch failed. Please try again or select a smaller region."
        ) from exc
