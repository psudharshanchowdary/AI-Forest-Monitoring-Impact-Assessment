"""Scientifically correct Sentinel-2 NDVI monitoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_geom
from scipy import ndimage as ndi
from shapely.geometry.base import BaseGeometry

from forest_monitor.geometry import approximate_area_ha, to_geometry_dict

OPAQUE_CLOUD_BIT = 1 << 10
CIRRUS_CLOUD_BIT = 1 << 11
FOREST_NDVI_THRESHOLD = 0.40
NON_FOREST_NDVI_THRESHOLD = 0.10
SIGNIFICANT_LOSS_THRESHOLD = -0.15
DEFAULT_CARBON_DENSITY_TCO2E_PER_HA = 215.0
DEFAULT_SCL_INVALID = {0, 1, 3, 6, 8, 9, 10, 11}
DEFAULT_SCL_WATER = {6}

GeometryLike = dict[str, Any] | BaseGeometry


@dataclass(slots=True)
class SentinelScenePaths:
    red: Path
    nir: Path
    qa60: Path | None = None
    scl: Path | None = None
    green: Path | None = None
    blue: Path | None = None
    label: str = "scene"


@dataclass(slots=True)
class RasterGrid:
    crs: CRS
    transform: Affine
    width: int
    height: int


@dataclass(slots=True)
class SentinelSceneData:
    red: np.ndarray
    nir: np.ndarray
    qa60: np.ndarray | None
    scl: np.ndarray | None
    ndvi: np.ndarray
    valid_mask: np.ndarray
    cloud_mask: np.ndarray
    water_mask: np.ndarray
    grid: RasterGrid
    label: str
    ndvi_min: float
    ndvi_max: float
    ndvi_mean: float


@dataclass(slots=True)
class SentinelChangeResult:
    baseline: SentinelSceneData
    current: SentinelSceneData
    ndvi_change: np.ndarray
    common_valid_mask: np.ndarray
    baseline_forest_mask: np.ndarray
    current_forest_mask: np.ndarray
    forest_loss_mask: np.ndarray
    aligned_segmentation: np.ndarray | None
    mean_baseline_ndvi: float
    mean_current_ndvi: float
    dndvi_min: float
    dndvi_max: float
    forest_pixels: int
    forest_percent: float
    forest_loss_pixels: int
    forest_loss_percent: float
    loss_area_ha: float
    carbon_loss_tco2e: float
    classification: str
    debug_lines: list[str]


@dataclass(slots=True)
class SegmentationAlignment:
    data: np.ndarray
    transform: Affine | None = None
    crs: CRS | str | None = None


def _roi_geometry(geometry: GeometryLike) -> dict[str, Any]:
    return to_geometry_dict(geometry)


def _transform_geometry(geometry: dict[str, Any], dataset_crs: CRS) -> dict[str, Any]:
    return transform_geom("EPSG:4326", dataset_crs, geometry, precision=10)


def _scale_reflectance(data: np.ndarray) -> np.ndarray:
    scaled = data.astype(np.float32)
    finite_max = float(np.nanmax(scaled)) if np.isfinite(scaled).any() else 0.0
    if finite_max > 1.5:
        scaled = scaled / 10000.0
    return np.clip(scaled, 0.0, 1.0)


def _masked_gaussian(data: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return data.astype(np.float32)
    weights = valid_mask.astype(np.float32)
    filled = np.where(valid_mask, data, 0.0).astype(np.float32)
    smooth_values = ndi.gaussian_filter(filled, sigma=sigma)
    smooth_weights = ndi.gaussian_filter(weights, sigma=sigma)
    result = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(smooth_values, smooth_weights, out=result, where=smooth_weights > 1e-6)
    result[~valid_mask] = np.nan
    return result


def _clip_band(
    dataset_path: Path,
    geometry: dict[str, Any],
    reference_grid: RasterGrid | None,
    *,
    resampling: Resampling,
    scale_reflectance: bool,
    dst_dtype: np.dtype,
    dst_fill: float | int,
) -> tuple[np.ndarray, RasterGrid]:
    with rasterio.open(dataset_path) as src:
        roi_in_src_crs = _transform_geometry(geometry, src.crs)
        clipped, clipped_transform = mask(src, [roi_in_src_crs], crop=True, filled=False, indexes=1)
        if clipped.size == 0:
            raise ValueError(f"ROI does not overlap raster: {dataset_path}")

        source_grid = RasterGrid(crs=src.crs, transform=clipped_transform, width=clipped.shape[1], height=clipped.shape[0])
        destination_grid = reference_grid or source_grid

        if np.issubdtype(np.dtype(dst_dtype), np.floating):
            source_fill = np.nan
            source = clipped.data.astype(np.float32)
            source[np.ma.getmaskarray(clipped)] = np.nan
            if scale_reflectance:
                source = _scale_reflectance(source)
        else:
            source_fill = 0
            source = clipped.filled(0).astype(dst_dtype)

        destination = np.full((destination_grid.height, destination_grid.width), dst_fill, dtype=dst_dtype)
        reproject(
            source=source,
            destination=destination,
            src_transform=source_grid.transform,
            src_crs=source_grid.crs,
            src_nodata=source_fill,
            dst_transform=destination_grid.transform,
            dst_crs=destination_grid.crs,
            dst_nodata=dst_fill,
            resampling=resampling,
        )
        return destination, destination_grid


def compute_ndvi(red: np.ndarray, nir: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    red = _scale_reflectance(red)
    nir = _scale_reflectance(nir)

    denominator = nir + red + 1e-6
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    safe = np.isfinite(red) & np.isfinite(nir)
    if valid_mask is not None:
        safe &= valid_mask
    np.divide(nir - red, denominator, out=ndvi, where=safe)
    return np.clip(ndvi, -1.0, 1.0)


def _derive_cloud_and_water_masks(
    red: np.ndarray,
    nir: np.ndarray,
    qa60: np.ndarray | None,
    scl: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    cloud_mask = np.zeros(red.shape, dtype=bool)
    water_mask = np.zeros(red.shape, dtype=bool)

    if qa60 is not None:
        qa60_int = qa60.astype(np.uint16)
        cloud_mask |= ((qa60_int & OPAQUE_CLOUD_BIT) != 0) | ((qa60_int & CIRRUS_CLOUD_BIT) != 0)

    if scl is not None:
        scl_int = scl.astype(np.uint8)
        cloud_mask |= np.isin(scl_int, list(DEFAULT_SCL_INVALID - DEFAULT_SCL_WATER))
        water_mask |= np.isin(scl_int, list(DEFAULT_SCL_WATER))
    else:
        provisional_ndvi = compute_ndvi(red, nir)
        water_mask |= np.isfinite(provisional_ndvi) & (provisional_ndvi < 0.0) & (nir < 0.12)

    return cloud_mask, water_mask


def load_sentinel_scene(
    scene_paths: SentinelScenePaths,
    roi_geometry: GeometryLike,
    reference_grid: RasterGrid | None = None,
    *,
    smoothing_sigma: float = 1.0,
) -> SentinelSceneData:
    geometry = _roi_geometry(roi_geometry)
    red, grid = _clip_band(
        scene_paths.red,
        geometry,
        reference_grid,
        resampling=Resampling.bilinear,
        scale_reflectance=True,
        dst_dtype=np.float32,
        dst_fill=np.nan,
    )
    nir, _ = _clip_band(
        scene_paths.nir,
        geometry,
        grid,
        resampling=Resampling.bilinear,
        scale_reflectance=True,
        dst_dtype=np.float32,
        dst_fill=np.nan,
    )
    qa60 = None
    if scene_paths.qa60 is not None:
        qa60, _ = _clip_band(
            scene_paths.qa60,
            geometry,
            grid,
            resampling=Resampling.nearest,
            scale_reflectance=False,
            dst_dtype=np.uint16,
            dst_fill=0,
        )
    scl = None
    if scene_paths.scl is not None:
        scl, _ = _clip_band(
            scene_paths.scl,
            geometry,
            grid,
            resampling=Resampling.nearest,
            scale_reflectance=False,
            dst_dtype=np.uint8,
            dst_fill=0,
        )

    cloud_mask, water_mask = _derive_cloud_and_water_masks(red, nir, qa60, scl)
    base_valid = np.isfinite(red) & np.isfinite(nir) & ~cloud_mask & ~water_mask
    ndvi = compute_ndvi(red, nir, valid_mask=base_valid)
    ndvi = _masked_gaussian(ndvi, base_valid, sigma=smoothing_sigma)
    valid_mask = np.isfinite(ndvi) & base_valid
    valid_ndvi = ndvi[valid_mask]
    ndvi_min = float(np.nanmin(valid_ndvi)) if valid_ndvi.size else 0.0
    ndvi_max = float(np.nanmax(valid_ndvi)) if valid_ndvi.size else 0.0
    ndvi_mean = float(np.nanmean(valid_ndvi)) if valid_ndvi.size else 0.0

    return SentinelSceneData(
        red=red,
        nir=nir,
        qa60=qa60,
        scl=scl,
        ndvi=ndvi,
        valid_mask=valid_mask,
        cloud_mask=cloud_mask,
        water_mask=water_mask,
        grid=grid,
        label=scene_paths.label,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
        ndvi_mean=ndvi_mean,
    )


def align_segmentation_to_grid(
    segmentation: Path | np.ndarray | SegmentationAlignment | None,
    reference_grid: RasterGrid,
    roi_geometry: GeometryLike | None = None,
) -> np.ndarray | None:
    if segmentation is None:
        return None

    if isinstance(segmentation, Path):
        if roi_geometry is None:
            raise ValueError("roi_geometry is required when aligning a segmentation raster path")
        geometry = _roi_geometry(roi_geometry)
        aligned, _ = _clip_band(
            segmentation,
            geometry,
            reference_grid,
            resampling=Resampling.nearest,
            scale_reflectance=False,
            dst_dtype=np.uint8,
            dst_fill=0,
        )
        return aligned.astype(bool)

    if isinstance(segmentation, SegmentationAlignment):
        if segmentation.transform is None or segmentation.crs is None:
            if segmentation.data.shape != (reference_grid.height, reference_grid.width):
                raise ValueError("Segmentation array shape mismatch and no transform/CRS metadata provided")
            return segmentation.data.astype(bool)
        destination = np.zeros((reference_grid.height, reference_grid.width), dtype=np.uint8)
        reproject(
            source=segmentation.data.astype(np.uint8),
            destination=destination,
            src_transform=segmentation.transform,
            src_crs=segmentation.crs,
            src_nodata=0,
            dst_transform=reference_grid.transform,
            dst_crs=reference_grid.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        return destination.astype(bool)

    array = np.asarray(segmentation)
    if array.shape != (reference_grid.height, reference_grid.width):
        raise ValueError("Segmentation array shape mismatch; pass SegmentationAlignment with transform/crs metadata")
    return array.astype(bool)


def _pixel_area_ha(grid: RasterGrid, roi_geometry: GeometryLike) -> float:
    if grid.crs.is_projected:
        pixel_area = abs(grid.transform.a * grid.transform.e - grid.transform.b * grid.transform.d)
        return float(pixel_area / 10000.0)
    roi_area_ha = approximate_area_ha(roi_geometry)
    return float(roi_area_ha / max(grid.width * grid.height, 1))


def run_sentinel_ndvi_monitoring(
    *,
    baseline_scene: SentinelScenePaths,
    current_scene: SentinelScenePaths,
    roi_geometry: GeometryLike,
    segmentation: Path | np.ndarray | SegmentationAlignment | None = None,
    smoothing_sigma: float = 1.0,
    forest_threshold: float = FOREST_NDVI_THRESHOLD,
    non_forest_threshold: float = NON_FOREST_NDVI_THRESHOLD,
    change_threshold: float = SIGNIFICANT_LOSS_THRESHOLD,
    carbon_density_tco2e_per_ha: float = DEFAULT_CARBON_DENSITY_TCO2E_PER_HA,
    min_loss_pixels: int = 16,
    debug: bool = True,
) -> SentinelChangeResult:
    baseline = load_sentinel_scene(baseline_scene, roi_geometry, reference_grid=None, smoothing_sigma=smoothing_sigma)
    current = load_sentinel_scene(current_scene, roi_geometry, reference_grid=baseline.grid, smoothing_sigma=smoothing_sigma)

    common_valid = baseline.valid_mask & current.valid_mask
    dndvi = np.full(baseline.ndvi.shape, np.nan, dtype=np.float32)
    dndvi[common_valid] = current.ndvi[common_valid] - baseline.ndvi[common_valid]

    mean_baseline_ndvi = float(np.nanmean(np.where(common_valid, baseline.ndvi, np.nan))) if np.any(common_valid) else np.nan
    mean_current_ndvi = float(np.nanmean(np.where(common_valid, current.ndvi, np.nan))) if np.any(common_valid) else np.nan
    non_forest_region = (not np.any(common_valid)) or (np.isnan(mean_current_ndvi)) or (mean_current_ndvi < non_forest_threshold)

    if non_forest_region:
        baseline_forest = np.zeros_like(common_valid, dtype=bool)
        current_forest = np.zeros_like(common_valid, dtype=bool)
        forest_loss = np.zeros_like(common_valid, dtype=bool)
        forest_pixels = 0
        forest_percent = 0.0
        forest_loss_pixels = 0
        forest_loss_percent = 0.0
        loss_area_ha = 0.0
        carbon_loss_tco2e = 0.0
        classification = "Non-forest region"
    else:
        baseline_forest = common_valid & (baseline.ndvi > forest_threshold)
        current_forest = common_valid & (current.ndvi > forest_threshold)
        raw_loss = baseline_forest & (dndvi < change_threshold)
        effective_min_loss_pixels = max(1, min(min_loss_pixels, max(int(baseline_forest.sum()) // 4, 1)))
        forest_loss = sieve(raw_loss.astype(np.uint8), size=effective_min_loss_pixels, connectivity=8).astype(bool)
        forest_pixels = int(current_forest.sum())
        forest_percent = float((forest_pixels / max(int(common_valid.sum()), 1)) * 100.0)
        forest_loss_pixels = int(forest_loss.sum())
        forest_loss_percent = float((forest_loss_pixels / max(int(baseline_forest.sum()), 1)) * 100.0)
        loss_area_ha = float(forest_loss_pixels * _pixel_area_ha(baseline.grid, roi_geometry))
        carbon_loss_tco2e = float(loss_area_ha * carbon_density_tco2e_per_ha)
        classification = "Forest region"

    segmentation_aligned = align_segmentation_to_grid(segmentation, baseline.grid, roi_geometry=roi_geometry)
    valid_dndvi = dndvi[common_valid]
    dndvi_min = float(np.nanmin(valid_dndvi)) if valid_dndvi.size else np.nan
    dndvi_max = float(np.nanmax(valid_dndvi)) if valid_dndvi.size else np.nan

    debug_lines = [
        f"Baseline NDVI min/max/mean: {baseline.ndvi_min:.4f} / {baseline.ndvi_max:.4f} / {baseline.ndvi_mean:.4f}",
        f"Current NDVI min/max/mean: {current.ndvi_min:.4f} / {current.ndvi_max:.4f} / {current.ndvi_mean:.4f}",
        f"Mean NDVI (baseline): {mean_baseline_ndvi:.4f}",
        f"Mean NDVI (current): {mean_current_ndvi:.4f}",
        f"dNDVI min/max: {dndvi_min:.4f} / {dndvi_max:.4f}",
        f"Forest pixels: {forest_pixels}",
        f"Forest loss pixels: {forest_loss_pixels}",
        f"Classification: {classification}",
    ]
    if debug:
        for line in debug_lines:
            print(line)

    return SentinelChangeResult(
        baseline=baseline,
        current=current,
        ndvi_change=dndvi,
        common_valid_mask=common_valid,
        baseline_forest_mask=baseline_forest,
        current_forest_mask=current_forest,
        forest_loss_mask=forest_loss,
        aligned_segmentation=segmentation_aligned,
        mean_baseline_ndvi=mean_baseline_ndvi,
        mean_current_ndvi=mean_current_ndvi,
        dndvi_min=dndvi_min,
        dndvi_max=dndvi_max,
        forest_pixels=forest_pixels,
        forest_percent=forest_percent,
        forest_loss_pixels=forest_loss_pixels,
        forest_loss_percent=forest_loss_percent,
        loss_area_ha=loss_area_ha,
        carbon_loss_tco2e=carbon_loss_tco2e,
        classification=classification,
        debug_lines=debug_lines,
    )


def plot_intermediate_outputs(
    result: SentinelChangeResult,
    output_path: Path | None = None,
    *,
    show: bool = False,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ndvi_cmap = plt.get_cmap("RdYlGn").copy()
    ndvi_cmap.set_bad(color="#202020")
    change_cmap = plt.get_cmap("RdBu").copy()
    change_cmap.set_bad(color="#202020")

    panels = [
        (result.baseline.ndvi, "Baseline NDVI", ndvi_cmap, (-1.0, 1.0)),
        (result.current.ndvi, "Current NDVI", ndvi_cmap, (-1.0, 1.0)),
        (result.ndvi_change, "dNDVI", change_cmap, (-0.6, 0.6)),
        (result.common_valid_mask.astype(np.float32), "Valid Pixels", "gray", (0.0, 1.0)),
        (result.baseline_forest_mask.astype(np.float32), "Baseline Forest", "Greens", (0.0, 1.0)),
        (result.current_forest_mask.astype(np.float32), "Current Forest", "Greens", (0.0, 1.0)),
        (result.forest_loss_mask.astype(np.float32), "Forest Loss", "Reds", (0.0, 1.0)),
        (
            result.aligned_segmentation.astype(np.float32) if result.aligned_segmentation is not None else np.zeros_like(result.current_forest_mask, dtype=np.float32),
            "Aligned Segmentation",
            "viridis",
            (0.0, 1.0),
        ),
    ]

    for ax, (data, title, cmap, limits) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
