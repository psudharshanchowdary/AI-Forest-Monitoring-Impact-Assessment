from datetime import datetime

import numpy as np

from forest_monitor.environment import AirQualitySnapshot, WildfireEvent
from forest_monitor.reporting import PDFReportData, build_pdf_report, build_report_payload


def test_build_report_payload_accepts_forest_impact_assessment() -> None:
    payload = build_report_payload(
        analysis_summary={"forest_loss_percent": 12.5},
        wildfire_events=[
            WildfireEvent(
                title="Moderate wildfire hotspot",
                latitude=12.3,
                longitude=77.4,
                date="2026-03-20",
                source="NASA EONET (simulated)",
                distance_km=14.2,
                severity="Moderate",
            )
        ],
        aqi_latest={"us_aqi": 88.0, "aqi_category": "Moderate"},
        impact_assessment={"area_ha": 4.5, "impact_level": "Moderate"},
        combined_risk_score=0.61,
    )

    assert payload["forest_impact_assessment"]["area_ha"] == 4.5
    assert payload["construction_impact"]["impact_level"] == "Moderate"
    assert payload["combined_environmental_risk_score"] == 0.61
    assert payload["wildfire_events"][0]["severity"] == "Moderate"


def test_build_pdf_report_returns_pdf_bytes() -> None:
    ndvi = np.array([[0.2, 0.4], [0.6, 0.7]], dtype=np.float32)
    rgb = np.dstack([ndvi, np.clip(ndvi + 0.15, 0.0, 1.0), ndvi * 0.5]).astype(np.float32)
    probability = np.array([[0.3, 0.65], [0.78, 0.88]], dtype=np.float32)
    instance_map = np.array([[0, 1], [1, 2]], dtype=np.int32)
    forest_loss_mask = np.array([[False, True], [False, False]])

    report = PDFReportData(
        title="Forest Monitoring & Impact Assessment Report",
        generated_at=datetime(2026, 4, 20, 10, 15, 0),
        roi_name="Western Ghats (India)",
        roi_details={
            "ROI centroid": "12.12345, 76.54321",
            "ROI area (ha)": "123.45",
            "Baseline scene": "BASE-001 (2024-04-01)",
            "Current scene": "CURR-001 (2026-04-01)",
        },
        metrics={
            "Environmental Risk": "Medium",
            "Forest Health": "Healthy",
            "Forest Loss %": "5.20%",
            "Estimated Carbon Loss": "112.00 tCO2e",
            "Wildfire Risk": "Low",
        },
        baseline_ndvi=ndvi,
        current_ndvi=ndvi + 0.05,
        current_ndwi=np.array([[0.32, 0.28], [0.10, -0.05]], dtype=np.float32),
        ndvi_change=np.full_like(ndvi, -0.05),
        current_rgb=rgb,
        segmentation_probability_map=probability,
        segmentation_instance_map=instance_map,
        forest_loss_mask=forest_loss_mask,
        aqi_history=[
            AirQualitySnapshot(
                timestamp="2026-04-20T00:00:00Z",
                us_aqi=82.0,
                pm25=35.0,
                pm10=52.0,
                ozone=61.0,
                aqi_category="Moderate",
            ),
            AirQualitySnapshot(
                timestamp="2026-04-20T01:00:00Z",
                us_aqi=76.0,
                pm25=31.0,
                pm10=46.0,
                ozone=58.0,
                aqi_category="Moderate",
            ),
        ],
        wildfire_events=[
            WildfireEvent(
                title="Moderate wildfire hotspot",
                latitude=12.3,
                longitude=77.4,
                date="2026-04-19",
                source="NASA EONET (simulated)",
                distance_km=14.2,
                severity="Moderate",
            )
        ],
    )

    pdf_bytes = build_pdf_report(report)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000
