from datetime import date

from forest_monitor.analysis import run_monitoring_pipeline


FOREST_ROI = {
    "type": "Polygon",
    "coordinates": [[[78.30, 17.55], [78.58, 17.55], [78.58, 17.29], [78.30, 17.29], [78.30, 17.55]]],
}

ALT_ROI = {
    "type": "Polygon",
    "coordinates": [[[-63.35, -9.75], [-63.05, -9.75], [-63.05, -10.00], [-63.35, -10.00], [-63.35, -9.75]]],
}

LOW_VEG_ROI = {
    "type": "Polygon",
    "coordinates": [[[80.0, -20.0], [80.4, -20.0], [80.4, -19.6], [80.0, -19.6], [80.0, -20.0]]],
}


def _run(geometry: dict):
    return run_monitoring_pipeline(
        geometry=geometry,
        baseline_start=date(2022, 9, 1),
        baseline_end=date(2023, 2, 28),
        current_start=date(2024, 9, 1),
        current_end=date(2025, 2, 28),
    )


def test_roi_changes_outputs():
    first = _run(FOREST_ROI)
    second = _run(ALT_ROI)
    assert first.baseline_scene.item_id != second.baseline_scene.item_id
    assert abs(first.forest_loss_percent - second.forest_loss_percent) > 0.01 or abs(first.mean_current_ndvi - second.mean_current_ndvi) > 0.01


def test_low_vegetation_roi_is_non_forest():
    result = _run(LOW_VEG_ROI)
    assert result.mean_current_ndvi < 0.20
    if result.mean_current_ndvi < 0.12:
        assert result.region_classification == "Non-forest region"
        assert result.forest_loss_percent == 0.0
        assert result.loss_area_ha == 0.0
        assert "No forest detected in this region. Try a forested area." in result.analysis_warnings
