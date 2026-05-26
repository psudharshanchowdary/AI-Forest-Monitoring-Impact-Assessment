from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from forest_monitor.analysis import run_monitoring_pipeline
from forest_monitor.pipeline.ndvi import compute_ndwi
from forest_monitor.visualization import calculate_percentages, classify_land_cover, dndvi_histogram_figure


def _window() -> tuple[date, date, date, date]:
    current_end = date.today()
    current_start = current_end - timedelta(days=180)
    baseline_start = current_start.replace(year=current_start.year - 2)
    baseline_end = current_end.replace(year=current_end.year - 2)
    return baseline_start, baseline_end, current_start, current_end


def test_dense_forest_roi_has_near_zero_water_when_rgb_is_used() -> None:
    baseline_start, baseline_end, current_start, current_end = _window()
    amazon_roi = {
        "type": "Polygon",
        "coordinates": [[[-63.40, -9.65], [-62.55, -9.65], [-62.55, -10.35], [-63.40, -10.35], [-63.40, -9.65]]],
    }
    result = run_monitoring_pipeline(
        geometry=amazon_roi,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=current_end,
    )

    land_cover = classify_land_cover(
        result.current_scene.ndvi,
        green=result.current_scene.green,
        nir=result.current_scene.nir,
        rgb=result.current_scene.rgb,
    )
    water_percent = float((land_cover == 4).mean() * 100.0)
    forest_percent = float((land_cover == 1).mean() * 100.0)

    assert water_percent < 2.0
    assert forest_percent > 50.0


def test_ocean_roi_still_classifies_as_water() -> None:
    baseline_start, baseline_end, current_start, current_end = _window()
    ocean_roi = {
        "type": "Polygon",
        "coordinates": [[[70.0, 0.5], [70.8, 0.5], [70.8, -0.3], [70.0, -0.3], [70.0, 0.5]]],
    }
    result = run_monitoring_pipeline(
        geometry=ocean_roi,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=current_end,
    )

    land_cover = classify_land_cover(
        result.current_scene.ndvi,
        green=result.current_scene.green,
        nir=result.current_scene.nir,
        rgb=result.current_scene.rgb,
    )
    water_percent = float((land_cover == 4).mean() * 100.0)

    assert water_percent > 95.0


def test_low_ndvi_bright_pixels_become_soil_not_water() -> None:
    ndvi = np.array([[0.04, 0.04], [0.06, 0.08]], dtype=np.float32)
    rgb = np.array(
        [
            [[0.62, 0.58, 0.52], [0.64, 0.60, 0.56]],
            [[0.60, 0.57, 0.54], [0.61, 0.59, 0.55]],
        ],
        dtype=np.float32,
    )
    green = rgb[..., 1]
    nir = np.array([[0.62, 0.61], [0.60, 0.60]], dtype=np.float32)
    land_cover = classify_land_cover(ndvi, green=green, nir=nir, rgb=rgb)

    assert np.all(land_cover == 3)


def test_positive_ndwi_pixels_become_water() -> None:
    ndvi = np.array([[0.02, 0.01], [0.05, 0.04]], dtype=np.float32)
    green = np.array([[0.34, 0.30], [0.28, 0.35]], dtype=np.float32)
    nir = np.array([[0.10, 0.12], [0.11, 0.14]], dtype=np.float32)
    rgb = np.dstack([green * 0.7, green, green * 1.15]).astype(np.float32)
    ndwi = compute_ndwi(green, nir)

    land_cover = classify_land_cover(ndvi, ndwi=ndwi, rgb=rgb)

    assert np.all(land_cover == 4)


def test_calculate_percentages_counts_water_pixels() -> None:
    land_cover = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    percentages = calculate_percentages(land_cover)

    assert percentages["Water"] == 25.0
    assert sum(percentages.values()) == 100.0


def test_dndvi_histogram_figure_has_data() -> None:
    ndvi_change = np.array([[0.02, -0.14, 0.0], [0.08, -0.22, 0.11]], dtype=np.float32)
    figure = dndvi_histogram_figure(ndvi_change)

    assert len(figure.data) == 1
    assert figure.data[0].type == "histogram"
