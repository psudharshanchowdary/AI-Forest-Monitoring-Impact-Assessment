from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from forest_monitor.pipeline.sentinel_ndvi import (
    SegmentationAlignment,
    SentinelScenePaths,
    run_sentinel_ndvi_monitoring,
)

ROI = {
    "type": "Polygon",
    "coordinates": [[[78.30, 17.55], [78.58, 17.55], [78.58, 17.29], [78.30, 17.29], [78.30, 17.55]]],
}


def _write_raster(path: Path, data: np.ndarray, *, transform, crs: str = "EPSG:4326") -> Path:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_sentinel_ndvi_monitoring_detects_real_loss(tmp_path: Path):
    transform = from_bounds(78.30, 17.29, 78.58, 17.55, 6, 6)
    baseline_red = np.full((6, 6), 500, dtype=np.uint16)
    baseline_nir = np.full((6, 6), 250, dtype=np.uint16)
    baseline_red[1:5, 1:5] = 1000
    baseline_nir[1:5, 1:5] = 7000

    current_red = baseline_red.copy()
    current_nir = baseline_nir.copy()
    current_red[2:4, 2:4] = 3200
    current_nir[2:4, 2:4] = 3400

    qa60 = np.zeros((6, 6), dtype=np.uint16)
    scl = np.full((6, 6), 4, dtype=np.uint8)
    scl[0, :] = 6
    scl[:, 0] = 6
    scl[-1, :] = 6
    scl[:, -1] = 6

    baseline = SentinelScenePaths(
        red=_write_raster(tmp_path / "baseline_red.tif", baseline_red, transform=transform),
        nir=_write_raster(tmp_path / "baseline_nir.tif", baseline_nir, transform=transform),
        qa60=_write_raster(tmp_path / "baseline_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "baseline_scl.tif", scl, transform=transform),
        label="baseline",
    )
    current = SentinelScenePaths(
        red=_write_raster(tmp_path / "current_red.tif", current_red, transform=transform),
        nir=_write_raster(tmp_path / "current_nir.tif", current_nir, transform=transform),
        qa60=_write_raster(tmp_path / "current_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "current_scl.tif", scl, transform=transform),
        label="current",
    )

    segmentation = SegmentationAlignment(
        data=np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
        transform=from_bounds(78.30, 17.29, 78.58, 17.55, 3, 3),
        crs="EPSG:4326",
    )

    result = run_sentinel_ndvi_monitoring(
        baseline_scene=baseline,
        current_scene=current,
        roi_geometry=ROI,
        segmentation=segmentation,
        debug=False,
    )

    assert result.classification == "Forest region"
    assert result.baseline.ndvi_min > 0.5
    assert result.baseline.ndvi_max <= 1.0
    assert result.baseline.ndvi_mean > 0.5
    assert result.current.ndvi_mean > 0.3
    assert result.forest_percent > 0.0
    assert result.forest_loss_percent > 0.0
    assert result.forest_loss_mask.any()
    assert result.dndvi_min < -0.15
    assert result.aligned_segmentation is not None
    assert result.aligned_segmentation.shape == result.current.ndvi.shape


def test_water_roi_returns_zero_forest(tmp_path: Path):
    transform = from_bounds(78.30, 17.29, 78.58, 17.55, 5, 5)
    red = np.full((5, 5), 900, dtype=np.uint16)
    nir = np.full((5, 5), 300, dtype=np.uint16)
    qa60 = np.zeros((5, 5), dtype=np.uint16)
    scl = np.full((5, 5), 6, dtype=np.uint8)

    baseline = SentinelScenePaths(
        red=_write_raster(tmp_path / "water_baseline_red.tif", red, transform=transform),
        nir=_write_raster(tmp_path / "water_baseline_nir.tif", nir, transform=transform),
        qa60=_write_raster(tmp_path / "water_baseline_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "water_baseline_scl.tif", scl, transform=transform),
    )
    current = SentinelScenePaths(
        red=_write_raster(tmp_path / "water_current_red.tif", red, transform=transform),
        nir=_write_raster(tmp_path / "water_current_nir.tif", nir, transform=transform),
        qa60=_write_raster(tmp_path / "water_current_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "water_current_scl.tif", scl, transform=transform),
    )

    result = run_sentinel_ndvi_monitoring(
        baseline_scene=baseline,
        current_scene=current,
        roi_geometry=ROI,
        debug=False,
    )

    assert result.classification == "Non-forest region"
    assert result.current.ndvi_mean < 0.1
    assert result.forest_percent == 0.0
    assert result.forest_loss_percent == 0.0
    assert result.carbon_loss_tco2e == 0.0


def test_cloud_pixel_is_masked_from_ndvi(tmp_path: Path):
    transform = from_bounds(78.30, 17.29, 78.58, 17.55, 4, 4)
    red = np.full((4, 4), 1000, dtype=np.uint16)
    nir = np.full((4, 4), 7000, dtype=np.uint16)
    qa60 = np.zeros((4, 4), dtype=np.uint16)
    qa60[1, 1] = 1 << 10
    scl = np.full((4, 4), 4, dtype=np.uint8)

    baseline = SentinelScenePaths(
        red=_write_raster(tmp_path / "cloud_baseline_red.tif", red, transform=transform),
        nir=_write_raster(tmp_path / "cloud_baseline_nir.tif", nir, transform=transform),
        qa60=_write_raster(tmp_path / "cloud_baseline_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "cloud_baseline_scl.tif", scl, transform=transform),
    )
    current = SentinelScenePaths(
        red=_write_raster(tmp_path / "cloud_current_red.tif", red, transform=transform),
        nir=_write_raster(tmp_path / "cloud_current_nir.tif", nir, transform=transform),
        qa60=_write_raster(tmp_path / "cloud_current_qa60.tif", qa60, transform=transform),
        scl=_write_raster(tmp_path / "cloud_current_scl.tif", scl, transform=transform),
    )

    result = run_sentinel_ndvi_monitoring(
        baseline_scene=baseline,
        current_scene=current,
        roi_geometry=ROI,
        debug=False,
    )

    assert result.current.cloud_mask[1, 1]
    assert not result.current.valid_mask[1, 1]
    assert np.isnan(result.current.ndvi[1, 1])
