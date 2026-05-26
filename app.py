from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

from forest_monitor.analysis import MonitoringResult, SatelliteDataFetchError, run_monitoring_pipeline
from forest_monitor.environment import (
    classify_aqi_level,
    estimate_wildfire_risk,
    generate_air_quality_history,
    generate_wildfire_events,
)
from forest_monitor.geometry import geometry_bounds, geometry_centroid, geometry_signature
from forest_monitor.pipeline.ndvi import compute_ndwi
from forest_monitor.reporting import PDFReportData, build_pdf_report
from forest_monitor.segmentation import segmentation_startup_notice
from forest_monitor.visualization import (
    aqi_trend_figure,
    change_figure,
    classified_map_figure,
    classify_land_cover,
    class_distribution_figure,
    create_annotated_rgb,
    dndvi_histogram_figure,
    image_figure,
    instance_map_figure,
    ndvi_figure,
    ndwi_figure,
    wildfire_timeline_figure,
)

st.set_page_config(page_title="AI-Based Forest Monitoring and Impact Assessment System", layout="wide")


PRESET_REGIONS: dict[str, dict[str, Any]] = {
    "Western Ghats (India)": {
        "description": "Tropical montane forest corridor along the western edge of peninsular India.",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[74.65, 15.55], [76.45, 15.55], [76.45, 11.05], [74.65, 11.05], [74.65, 15.55]]],
        },
    },
    "Sundarbans": {
        "description": "Mangrove-dominated delta ecosystem across India and Bangladesh.",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[88.10, 22.35], [89.35, 22.35], [89.35, 21.35], [88.10, 21.35], [88.10, 22.35]]],
        },
    },
    "Nilgiri Biosphere": {
        "description": "High-biodiversity forest landscape spanning Tamil Nadu, Karnataka, and Kerala.",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[76.10, 11.90], [77.20, 11.90], [77.20, 10.70], [76.10, 10.70], [76.10, 11.90]]],
        },
    },
    "Amazon Rainforest (demo)": {
        "description": "Dense tropical rainforest preset for project demo and benchmarking.",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-63.40, -9.65], [-62.55, -9.65], [-62.55, -10.35], [-63.40, -10.35], [-63.40, -9.65]]],
        },
    },
}

DENSITY_FACTORS = {"Low": 0.8, "Medium": 1.0, "High": 1.25}
CARBON_FACTORS_TCO2E_PER_HA = {"Low": 110.0, "Medium": 175.0, "High": 240.0}
IMPACT_FACTORS = {"Deforestation": 1.0, "Degradation": 0.55}
INDIA_DEFAULT_CENTER = [20.5937, 78.9629]
INDIA_DEFAULT_ZOOM = 5
HECTARE_TO_ACRE = 2.47105


@dataclass(slots=True)
class ComparisonWindow:
    baseline_start: date
    baseline_end: date
    current_start: date
    current_end: date

    @property
    def label(self) -> str:
        return (
            f"Baseline (past): {self.baseline_start.isoformat()} to {self.baseline_end.isoformat()} | "
            f"Current (present): {self.current_start.isoformat()} to {self.current_end.isoformat()}"
        )


@dataclass(slots=True)
class AnalysisContext:
    centroid: tuple[float, float]
    wildfire_events: list[Any]
    aqi_history: list[Any]
    latest_aqi: Any | None
    wildfire_score: float
    wildfire_level: str


SESSION_DEFAULTS = {
    "roi_geometry": None,
    "roi_signature": None,
    "analysis_result": None,
    "analysis_signature": None,
    "active_region_label": "Custom ROI",
    "pending_analysis": False,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
            :root {
                --bg: #07111b;
                --bg-soft: #0b1826;
                --panel: rgba(13, 23, 35, 0.94);
                --panel-2: rgba(18, 31, 47, 0.96);
                --line: rgba(111, 147, 188, 0.18);
                --text: #edf5ff;
                --muted: #9eb1c8;
                --green: #29d17f;
                --yellow: #ffd166;
                --red: #ff5f5f;
                --blue: #61c6ff;
            }
            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
            }
            body {
                overflow-x: hidden;
            }
            h1, h2, h3, h4, .metric-value, .top-nav h1, .section-heading h2 {
                font-family: "Space Grotesk", "Manrope", sans-serif;
            }
            .stApp {
                background:
                    radial-gradient(1100px 520px at 0% -5%, rgba(38, 88, 135, 0.38) 0%, transparent 45%),
                    radial-gradient(900px 420px at 100% 0%, rgba(31, 119, 93, 0.22) 0%, transparent 40%),
                    linear-gradient(180deg, var(--bg) 0%, #050d15 100%);
                color: var(--text);
            }
            header[data-testid="stHeader"] {
                background: transparent;
            }
            .block-container {
                max-width: 1440px;
                padding-top: 1rem;
                padding-bottom: 2.8rem;
            }
            section[data-testid="stSidebar"] {
                display: none !important;
            }
            .top-nav {
                background:
                    radial-gradient(420px 220px at 100% 0%, rgba(56, 130, 196, 0.16), transparent 60%),
                    linear-gradient(135deg, rgba(10, 20, 31, 0.97) 0%, rgba(17, 38, 54, 0.98) 100%);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 26px 28px;
                box-shadow: 0 22px 52px rgba(0, 0, 0, 0.28);
                margin-bottom: 18px;
                animation: fadeSlide 380ms ease;
            }
            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.9fr);
                gap: 18px;
                align-items: stretch;
            }
            .hero-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.3rem 0.72rem;
                border-radius: 999px;
                background: rgba(97, 198, 255, 0.09);
                border: 1px solid rgba(97, 198, 255, 0.18);
                color: #cce9ff;
                font-size: 0.76rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.9rem;
            }
            .top-nav h1 {
                margin: 0;
                font-size: 2.05rem;
                font-weight: 760;
                color: #f7fbff;
                letter-spacing: 0.015em;
            }
            .top-nav p {
                margin: 0.4rem 0 0;
                color: var(--muted);
                font-size: 1rem;
                max-width: 760px;
                line-height: 1.6;
            }
            .hero-meta {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 1rem;
            }
            .hero-stat {
                background: rgba(9, 18, 29, 0.72);
                border: 1px solid rgba(103, 134, 165, 0.14);
                border-radius: 16px;
                padding: 14px 14px 12px;
                backdrop-filter: blur(12px);
            }
            .hero-stat-label {
                color: var(--muted);
                font-size: 0.72rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .hero-stat-value {
                color: #f7fbff;
                font-size: 1rem;
                font-weight: 720;
                margin-top: 0.4rem;
                line-height: 1.35;
            }
            .hero-sidecard {
                background: linear-gradient(180deg, rgba(10, 21, 32, 0.94) 0%, rgba(14, 29, 44, 0.96) 100%);
                border: 1px solid rgba(110, 149, 191, 0.16);
                border-radius: 20px;
                padding: 18px 18px 16px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }
            .hero-sidecard h3 {
                margin: 0;
                font-size: 1.02rem;
                color: #eff7ff;
            }
            .hero-sidecard p {
                margin: 0.38rem 0 0;
                font-size: 0.88rem;
                color: var(--muted);
                line-height: 1.55;
            }
            .feature-list {
                display: grid;
                gap: 10px;
                margin-top: 1rem;
            }
            .feature-item {
                display: grid;
                grid-template-columns: 28px 1fr;
                gap: 10px;
                align-items: start;
            }
            .feature-badge {
                width: 28px;
                height: 28px;
                border-radius: 10px;
                display: grid;
                place-items: center;
                background: rgba(97, 198, 255, 0.12);
                color: #d8efff;
                border: 1px solid rgba(97, 198, 255, 0.18);
                font-size: 0.84rem;
                font-weight: 700;
            }
            .control-bar {
                background: linear-gradient(180deg, rgba(12, 24, 37, 0.96) 0%, rgba(15, 29, 43, 0.96) 100%);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
                margin-bottom: 16px;
                backdrop-filter: blur(12px);
            }
            .control-label {
                color: var(--muted);
                font-size: 0.79rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin: 0 0 0.38rem;
            }
            .control-chip {
                display: inline-flex;
                gap: 0.45rem;
                align-items: center;
                background: rgba(13, 27, 41, 0.95);
                border: 1px solid rgba(106, 140, 177, 0.2);
                border-radius: 999px;
                color: #d9e7f8;
                padding: 0.45rem 0.8rem;
                font-size: 0.88rem;
                margin: 0.25rem 0.35rem 0 0;
            }
            .workspace-shell {
                background: linear-gradient(180deg, rgba(10, 21, 33, 0.96) 0%, rgba(13, 25, 37, 0.98) 100%);
                border: 1px solid rgba(111, 147, 188, 0.18);
                border-radius: 22px;
                padding: 18px 18px 14px;
                box-shadow: 0 20px 44px rgba(0, 0, 0, 0.22);
                margin-bottom: 18px;
            }
            .panel-head {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 0.9rem;
            }
            .panel-kicker {
                color: #8db6dc;
                font-size: 0.72rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }
            .panel-head h2 {
                margin: 0;
                font-size: 1.35rem;
                color: #f2f8ff;
            }
            .panel-head p {
                margin: 0.3rem 0 0;
                color: var(--muted);
                font-size: 0.9rem;
            }
            .legend-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                justify-content: flex-end;
            }
            .legend-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                border-radius: 999px;
                padding: 0.42rem 0.72rem;
                background: rgba(14, 28, 42, 0.92);
                border: 1px solid rgba(107, 143, 180, 0.18);
                color: #d9e7f7;
                font-size: 0.8rem;
            }
            .legend-swatch {
                width: 10px;
                height: 10px;
                border-radius: 999px;
            }
            .section-heading {
                margin: 0.35rem 0 1rem;
            }
            .section-kicker {
                color: #8db6dc;
                font-size: 0.73rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }
            .section-heading h2 {
                margin: 0.32rem 0 0;
                color: #f4f9ff;
                font-size: 1.5rem;
            }
            .section-heading p {
                margin: 0.42rem 0 0;
                color: var(--muted);
                font-size: 0.92rem;
                max-width: 860px;
                line-height: 1.6;
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 12px;
                margin-top: 0.9rem;
            }
            .info-card {
                background: rgba(9, 18, 29, 0.8);
                border: 1px solid rgba(107, 143, 180, 0.16);
                border-radius: 16px;
                padding: 14px;
            }
            .info-card h4 {
                margin: 0;
                color: #eff7ff;
                font-size: 0.98rem;
            }
            .info-card p {
                margin: 0.42rem 0 0;
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.55;
            }
            .summary-band {
                background: linear-gradient(135deg, rgba(10, 21, 32, 0.92) 0%, rgba(17, 36, 55, 0.96) 100%);
                border: 1px solid rgba(111, 147, 188, 0.18);
                border-radius: 18px;
                padding: 14px 16px;
                margin: 0.9rem 0 1.05rem;
                color: #eaf4ff;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
            }
            .summary-band strong {
                color: #f8fbff;
            }
            .metric-card {
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px 16px 15px;
                min-height: 132px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
            }
            .metric-top {
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #d6e4f3;
                font-size: 0.84rem;
                margin-bottom: 0.7rem;
            }
            .metric-icon {
                width: 30px;
                height: 30px;
                border-radius: 10px;
                display: grid;
                place-items: center;
                background: rgba(28, 55, 82, 0.9);
                border: 1px solid rgba(98, 142, 183, 0.25);
                color: var(--blue);
                font-weight: 700;
            }
            .metric-value {
                font-size: 1.9rem;
                font-weight: 780;
                line-height: 1.02;
                color: #f6fbff;
            }
            .metric-sub {
                margin-top: 0.34rem;
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
            }
            .risk-badge {
                display: inline-block;
                margin-top: 0.52rem;
                padding: 0.2rem 0.56rem;
                border-radius: 999px;
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.02em;
            }
            .risk-low { background: rgba(41, 209, 127, 0.16); color: var(--green); border: 1px solid rgba(41, 209, 127, 0.4); }
            .risk-medium { background: rgba(255, 209, 102, 0.16); color: var(--yellow); border: 1px solid rgba(255, 209, 102, 0.48); }
            .risk-high { background: rgba(255, 95, 95, 0.16); color: var(--red); border: 1px solid rgba(255, 95, 95, 0.48); }
            .section-card {
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px 16px 14px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            }
            div[data-testid="stPlotlyChart"] {
                background: linear-gradient(180deg, rgba(10, 20, 31, 0.88) 0%, rgba(14, 28, 42, 0.96) 100%);
                border: 1px solid rgba(111, 147, 188, 0.16);
                border-radius: 18px;
                padding: 8px 8px 2px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            }
            [data-testid="stAlert"] {
                background: linear-gradient(180deg, rgba(11, 26, 39, 0.96) 0%, rgba(14, 28, 42, 0.98) 100%);
                border: 1px solid rgba(111, 147, 188, 0.18);
                border-radius: 16px;
                color: #edf5ff;
            }
            .stTabs [data-baseweb="tab-list"] {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                border-bottom: 1px solid rgba(101, 132, 165, 0.16);
            }
            .stTabs [data-baseweb="tab"] {
                background: rgba(20, 33, 49, 0.8);
                border: 1px solid rgba(89, 119, 152, 0.2);
                border-radius: 12px 12px 0 0;
                color: #bfd1e6;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(27, 54, 80, 0.96);
                color: #eef6ff;
                border-color: rgba(108, 145, 186, 0.28);
            }
            .streamlit-expanderHeader {
                font-weight: 700;
                color: #edf5ff;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 13px;
                font-weight: 740;
                transition: transform 150ms ease, box-shadow 150ms ease;
            }
            .stButton > button {
                background: linear-gradient(180deg, #2ac769 0%, #179e4e 100%);
                border: 1px solid rgba(56, 139, 82, 0.6);
                color: white;
                box-shadow: 0 10px 22px rgba(23, 158, 78, 0.24);
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                transform: translateY(-1px);
            }
            .stDownloadButton > button {
                background: linear-gradient(180deg, #183047 0%, #102435 100%);
                border: 1px solid rgba(111, 147, 188, 0.24);
                color: #eef5ff;
                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
            }
            @media (max-width: 1080px) {
                .hero-grid, .hero-meta, .info-grid {
                    grid-template-columns: 1fr;
                }
                .legend-row {
                    justify-content: flex-start;
                }
            }
            @keyframes fadeSlide {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_class(level: str) -> str:
    normalized = (level or "").strip().lower()
    if normalized in {"low", "healthy"}:
        return "risk-badge risk-low"
    if normalized in {"medium", "moderate"}:
        return "risk-badge risk-medium"
    return "risk-badge risk-high"


def hectares_to_acres(area_ha: float) -> float:
    return float(area_ha) * HECTARE_TO_ACRE


def format_area_with_secondary_unit(area_ha: float, decimals: int = 2) -> str:
    area_acres = hectares_to_acres(area_ha)
    return f"{area_ha:.{decimals}f} ha | {area_acres:.{decimals}f} acres"


def format_carbon_with_units(carbon_tco2e: float, decimals: int = 1) -> str:
    return f"{carbon_tco2e:.{decimals}f} tCO2e"


def render_runtime_notices() -> None:
    notice = segmentation_startup_notice()
    if notice:
        st.info(f"{notice} Spatial outputs will fall back to the NDVI-guided segmentation path for this demo.")


def render_analysis_warnings(result: MonitoringResult) -> None:
    for warning in result.analysis_warnings:
        st.warning(warning)


def metric_card(label: str, value: str, icon: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f'<div class="{risk_class(badge)}">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span>{label}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_cards(items: list[tuple[str, str]]) -> None:
    html = "".join(
        f'<div class="info-card"><h4>{title}</h4><p>{description}</p></div>'
        for title, description in items
    )
    st.markdown(f'<div class="info-grid">{html}</div>', unsafe_allow_html=True)


def initialize_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value



def clear_analysis_state() -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_signature = None



def clear_roi_state() -> None:
    st.session_state.roi_geometry = None
    st.session_state.roi_signature = None
    clear_analysis_state()



def store_analysis_result(result: MonitoringResult, geometry: dict[str, Any]) -> None:
    signature = geometry_signature(geometry)
    st.session_state.analysis_result = result
    st.session_state.roi_geometry = geometry
    st.session_state.roi_signature = signature
    st.session_state.analysis_signature = signature



def has_current_analysis() -> bool:
    return (
        st.session_state.analysis_result is not None
        and st.session_state.analysis_signature is not None
        and st.session_state.analysis_signature == st.session_state.roi_signature
    )



def shift_year_safe(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)



def build_comparison_window() -> ComparisonWindow:
    current_end = date.today()
    current_start = current_end - timedelta(days=180)
    return ComparisonWindow(
        baseline_start=shift_year_safe(current_start, -2),
        baseline_end=shift_year_safe(current_end, -2),
        current_start=current_start,
        current_end=current_end,
    )



def active_region_description() -> str:
    if st.session_state.active_region_label in PRESET_REGIONS:
        return str(PRESET_REGIONS[st.session_state.active_region_label]["description"])
    return "Draw any custom polygon or rectangle to monitor vegetation health and forest change over time."



def apply_preset_region(region_label: str) -> None:
    preset = PRESET_REGIONS[region_label]
    geometry = preset["geometry"]
    signature = geometry_signature(geometry)
    st.session_state.active_region_label = region_label
    st.session_state.roi_geometry = geometry
    st.session_state.roi_signature = signature
    st.session_state.pending_analysis = True
    clear_analysis_state()



def forest_health_status(result: MonitoringResult) -> dict[str, str]:
    mean_ndvi = result.mean_current_ndvi
    ndvi_drop = result.mean_ndvi_drop
    if mean_ndvi > 0.5 and ndvi_drop > -0.08:
        return {
            "status": "Healthy",
            "explanation": "Current NDVI indicates strong canopy vigor with limited recent decline.",
        }
    if mean_ndvi < 0.3 or ndvi_drop <= -0.15 or result.forest_loss_percent >= 20.0:
        return {
            "status": "Degraded",
            "explanation": "Low greenness or a strong NDVI drop suggests stressed or disturbed forest condition.",
        }
    return {
        "status": "Moderate",
        "explanation": "Vegetation remains present, but the canopy signal indicates partial stress or mixed-condition cover.",
    }



def forest_impact_estimate(
    *,
    area_ha: float,
    vegetation_density: str,
    impact_type: str,
    roi_area_ha: float,
) -> dict[str, float | str]:
    density_factor = DENSITY_FACTORS[vegetation_density]
    impact_factor = IMPACT_FACTORS[impact_type]
    carbon_factor = CARBON_FACTORS_TCO2E_PER_HA[vegetation_density]

    effective_area = max(area_ha, 0.0)
    carbon_loss_tco2e = effective_area * carbon_factor * impact_factor
    forest_reduction_pct = min(100.0, (effective_area * impact_factor * 100.0) / max(roi_area_ha, 1.0))
    risk_increase_pct = min(100.0, 8.0 + (forest_reduction_pct * 0.9) + ((density_factor - 0.8) * 32.0))

    if risk_increase_pct < 25.0:
        impact_level = "Low"
    elif risk_increase_pct < 55.0:
        impact_level = "Moderate"
    else:
        impact_level = "High"

    explanation = (
        f"{impact_type} over {effective_area:.2f} ha with {vegetation_density.lower()} vegetation density "
        f"is estimated to remove {forest_reduction_pct:.1f}% of the monitored forest footprint."
    )
    return {
        "area_ha": float(effective_area),
        "vegetation_density": vegetation_density,
        "impact_type": impact_type,
        "carbon_loss_tco2e": float(carbon_loss_tco2e),
        "forest_reduction_pct": float(forest_reduction_pct),
        "risk_increase_pct": float(risk_increase_pct),
        "impact_level": impact_level,
        "explanation": explanation,
    }



def render_hero_banner() -> None:
    st.markdown(
        f"""
        <div class="top-nav">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">AI-Based Monitoring Dashboard</div>
                    <h1>AI-Based Forest Monitoring and Impact Assessment System</h1>
                    <p>This system monitors vegetation health and forest changes over time using satellite imagery. It combines NDVI-based change analysis, forest-stand segmentation, environmental context, and impact assessment into one professional review dashboard.</p>
                    <div class="hero-meta">
                        <div class="hero-stat">
                            <div class="hero-stat-label">Monitoring Scope</div>
                            <div class="hero-stat-value">Baseline (past) vs Current (present) vegetation condition and forest loss tracking</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Analysis Outputs</div>
                            <div class="hero-stat-value">NDVI maps, change intensity, forest stands, forest health, carbon loss, wildfire and AQI context</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Active Region</div>
                            <div class="hero-stat-value">{st.session_state.active_region_label}</div>
                        </div>
                    </div>
                </div>
                <div class="hero-sidecard">
                    <h3>Monitoring workflow</h3>
                    <p>{active_region_description()}</p>
                    <div class="feature-list">
                        <div class="feature-item"><div class="feature-badge">1</div><div><strong>Select a preset forest region or draw a custom ROI</strong><br/>The map supports both guided presets and manual selection.</div></div>
                        <div class="feature-item"><div class="feature-badge">2</div><div><strong>Run the monitoring pipeline</strong><br/>The system recomputes NDVI, change detection, segmentation, and diagnostics for the current ROI.</div></div>
                        <div class="feature-item"><div class="feature-badge">3</div><div><strong>Interpret forest health and impact</strong><br/>Review status, loss, environmental pressure, and projected impact in one place.</div></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_control_bar(window: ComparisonWindow) -> bool:
    options = ["Custom ROI", *PRESET_REGIONS.keys()]
    current_label = st.session_state.active_region_label
    default_index = options.index(current_label) if current_label in options else 0

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    controls = st.columns([1.35, 1.75, 0.9])
    with controls[0]:
        chosen_region = st.selectbox("Forest Region", options=options, index=default_index)
    with controls[1]:
        st.markdown('<div class="control-label">Monitoring Context</div>', unsafe_allow_html=True)
        roi_status_text = "ROI selected" if st.session_state.roi_geometry is not None else "Awaiting ROI"
        result_status_text = "Results ready" if has_current_analysis() else ("Queued" if st.session_state.pending_analysis else "Analysis not started")
        st.markdown(
            f'<span class="control-chip">{roi_status_text}</span>'
            f'<span class="control-chip">{result_status_text}</span>'
            f'<span class="control-chip">{window.label}</span>',
            unsafe_allow_html=True,
        )
    with controls[2]:
        st.markdown('<div class="control-label">Pipeline</div>', unsafe_allow_html=True)
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if chosen_region != st.session_state.active_region_label:
        if chosen_region == "Custom ROI":
            st.session_state.active_region_label = "Custom ROI"
            clear_analysis_state()
        else:
            apply_preset_region(chosen_region)
        st.rerun()

    return run_analysis



def current_roi_risk() -> str | None:
    if has_current_analysis():
        return st.session_state.analysis_result.risk.level
    return None



def map_with_roi(existing_geometry: dict | None, roi_risk: str | None) -> folium.Map:
    fmap = folium.Map(location=INDIA_DEFAULT_CENTER, zoom_start=INDIA_DEFAULT_ZOOM, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=False,
        show=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Labels",
        overlay=True,
        control=False,
        show=True,
    ).add_to(fmap)

    if existing_geometry is not None:
        color = {"Low": "#29D17F", "Medium": "#FFD166", "High": "#FF5F5F"}.get(roi_risk, "#61C6FF")
        folium.GeoJson(
            existing_geometry,
            style_function=lambda _: {
                "color": color,
                "weight": 4,
                "fillColor": color,
                "fillOpacity": 0.08,
            },
            tooltip=(f"ROI Risk: {roi_risk}" if roi_risk else "Selected ROI"),
        ).add_to(fmap)
        min_lon, min_lat, max_lon, max_lat = geometry_bounds(existing_geometry)
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": True},
    ).add_to(fmap)
    Fullscreen(position="topright").add_to(fmap)
    return fmap



def extract_geometry(map_data: dict | None) -> dict | None:
    if not map_data:
        return None

    def normalize(candidate: dict | None) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        if "geometry" in candidate and isinstance(candidate["geometry"], dict):
            return normalize(candidate["geometry"])
        if candidate.get("type") == "FeatureCollection":
            features = candidate.get("features")
            if isinstance(features, list) and features:
                return normalize(features[-1])
            return None
        if "type" in candidate and "coordinates" in candidate:
            return candidate
        return None

    def leaf_id(candidate: dict | None) -> int:
        if not isinstance(candidate, dict):
            return -1
        if candidate.get("type") == "FeatureCollection":
            features = candidate.get("features")
            if isinstance(features, list) and features:
                return max(leaf_id(item) for item in features)
            return -1
        props = candidate.get("properties")
        if isinstance(props, dict):
            value = props.get("_leaflet_id")
            if isinstance(value, (int, float)):
                return int(value)
        return -1

    drawing = map_data.get("last_active_drawing")
    geom = normalize(drawing)
    if geom is not None:
        return geom

    all_drawings = map_data.get("all_drawings")
    if isinstance(all_drawings, list) and all_drawings:
        candidates: list[tuple[int, int, dict]] = []
        for idx, item in enumerate(all_drawings):
            geom = normalize(item)
            if geom is None:
                continue
            candidates.append((idx, leaf_id(item), geom))
        if candidates:
            with_leaf_ids = [item for item in candidates if item[1] >= 0]
            if with_leaf_ids:
                return max(with_leaf_ids, key=lambda item: (item[1], item[0]))[2]
            return candidates[-1][2]
    return None



def render_workspace_intro() -> None:
    section_header(
        "Workspace",
        "Interactive Satellite Monitoring Workspace",
        "Choose a preset forest region or draw a custom ROI over the satellite layer, then run the monitoring pipeline to evaluate vegetation health, forest change, and environmental impact.",
    )
    st.markdown(
        """
        <div class="workspace-shell">
            <div class="panel-head">
                <div>
                    <div class="panel-kicker">Map Canvas</div>
                    <h2>Forest Monitoring ROI Surface</h2>
                    <p>The selected ROI border updates after analysis to reflect the latest environmental risk status. The default view opens over India so the initial map frame matches the main demo geography and the preset forest regions.</p>
                </div>
                <div class="legend-row">
                    <span class="legend-pill"><span class="legend-swatch" style="background:#29D17F"></span>Low Risk</span>
                    <span class="legend-pill"><span class="legend-swatch" style="background:#FFD166"></span>Moderate Risk</span>
                    <span class="legend-pill"><span class="legend-swatch" style="background:#FF5F5F"></span>High Risk</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_map_canvas() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    map_widget = map_with_roi(existing_geometry=st.session_state.roi_geometry, roi_risk=current_roi_risk())
    map_data = st_folium(map_widget, height=620, use_container_width=True, key="roi_map")
    st.caption("Default map view: India (20.5937, 78.9629) at zoom level 5, chosen to give the panel an immediate national context before a preset or custom ROI is selected.")
    return extract_geometry(map_data), map_data



def sync_roi_selection(map_data: dict[str, Any] | None, selected_geometry: dict[str, Any] | None, run_analysis: bool) -> None:
    if selected_geometry is not None:
        signature = geometry_signature(selected_geometry)
        if signature != st.session_state.roi_signature:
            st.session_state.active_region_label = "Custom ROI"
            st.session_state.roi_geometry = selected_geometry
            st.session_state.roi_signature = signature
            st.session_state.pending_analysis = False
            if st.session_state.analysis_signature != signature:
                clear_analysis_state()
            if not run_analysis:
                st.rerun()
        return

    if map_data and map_data.get("all_drawings") == [] and st.session_state.roi_geometry is not None and st.session_state.active_region_label == "Custom ROI":
        clear_roi_state()
        st.rerun()



def render_workspace_guidance(window: ComparisonWindow) -> None:
    info_cards(
        [
            (
                "Preset Forest Regions",
                "Western Ghats, Sundarbans, Nilgiri Biosphere, and Amazon Rainforest presets auto-load the ROI, zoom the map, and queue a fresh analysis run.",
            ),
            (
                "Baseline vs Current",
                f"Baseline (past): {window.baseline_start.isoformat()} to {window.baseline_end.isoformat()}. Current (present): {window.current_start.isoformat()} to {window.current_end.isoformat()}.",
            ),
            (
                "Monitoring Outputs",
                "The dashboard combines NDVI, change detection, forest stands, land cover, wildfire context, AQI trends, forest health, and impact estimation.",
            ),
        ]
    )

    if st.session_state.roi_geometry is None:
        st.info("Select a preset forest region or draw a custom ROI on the satellite map to start monitoring.")



def run_requested_analysis(run_analysis: bool, window: ComparisonWindow) -> None:
    should_run = run_analysis or st.session_state.pending_analysis
    if not should_run or st.session_state.roi_geometry is None:
        return

    geometry_for_run = st.session_state.roi_geometry
    clear_analysis_state()
    progress = st.progress(0, text="Step 1/4: Fetching satellite data...")
    status_line = st.empty()

    def cb(value: float, message: str) -> None:
        progress.progress(int(value * 100), text=message)
        status_line.caption(message)

    try:
        with st.spinner("Running forest monitoring pipeline..."):
            result = run_monitoring_pipeline(
                geometry=geometry_for_run,
                baseline_start=window.baseline_start,
                baseline_end=window.baseline_end,
                current_start=window.current_start,
                current_end=window.current_end,
                progress_callback=cb,
            )
        store_analysis_result(result, geometry_for_run)
    except SatelliteDataFetchError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Analysis failed during processing. Please try again or select a smaller region.")
    finally:
        progress.empty()
        status_line.empty()
        st.session_state.pending_analysis = False



def build_analysis_context(result: MonitoringResult) -> AnalysisContext:
    geometry = st.session_state.roi_geometry
    centroid = geometry_centroid(geometry)
    wildfire_events = generate_wildfire_events(geometry, latitude=centroid[0], longitude=centroid[1])
    aqi_history = generate_air_quality_history(geometry, latitude=centroid[0], longitude=centroid[1])
    latest_aqi = aqi_history[-1] if aqi_history else None
    wildfire_score, wildfire_level = estimate_wildfire_risk(
        forest_loss_percent=result.forest_loss_percent,
        wildfire_count=len(wildfire_events),
        current_aqi=(latest_aqi.us_aqi if latest_aqi else None),
    )
    return AnalysisContext(
        centroid=centroid,
        wildfire_events=wildfire_events,
        aqi_history=aqi_history,
        latest_aqi=latest_aqi,
        wildfire_score=wildfire_score,
        wildfire_level=wildfire_level,
    )



def render_decision_snapshot(result: MonitoringResult, context: AnalysisContext) -> None:
    health = forest_health_status(result)
    section_header(
        "Decision Support",
        "Forest Monitoring Snapshot",
        "Operational monitoring outputs summarize vegetation condition, forest health, change intensity, carbon impact, and environmental pressure for the selected ROI.",
    )
    st.markdown(
        f"""<div class="summary-band"><strong>ROI centroid:</strong> {context.centroid[0]:.4f}, {context.centroid[1]:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Baseline (past):</strong> {result.baseline_scene.item_id} ({result.baseline_scene.acquisition_date}) &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Current (present):</strong> {result.current_scene.item_id} ({result.current_scene.acquisition_date})</div>""",
        unsafe_allow_html=True,
    )
    cards = st.columns(5)
    with cards[0]:
        metric_card("Environmental Risk", result.risk.level, "R", f"Score {result.risk.score:.2f}", badge=result.risk.level)
    with cards[1]:
        metric_card("Forest Health", health["status"], "H", health["explanation"], badge=health["status"])
    with cards[2]:
        metric_card(
            "Forest Loss",
            f"{result.forest_loss_percent:.2f}%",
            "F",
            f"Loss area: {format_area_with_secondary_unit(result.loss_area_ha, decimals=1)}",
        )
    with cards[3]:
        metric_card(
            "Estimated Carbon Loss",
            format_carbon_with_units(result.carbon_loss_tco2e, decimals=0),
            "C",
            f"tCO2e = tonnes of CO2 equivalent | Density basis: {result.carbon_density_tco2e_per_ha:.0f} tCO2e/ha",
        )
    with cards[4]:
        metric_card("Wildfire Risk", context.wildfire_level, "W", f"Nearby events {len(context.wildfire_events)}", badge=context.wildfire_level)
    st.markdown(
        f"""<div class="summary-band"><strong>Forest health interpretation:</strong> {health['explanation']} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Current mean NDVI:</strong> {result.mean_current_ndvi:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Mean NDVI drop:</strong> {result.mean_ndvi_drop:.3f}</div>""",
        unsafe_allow_html=True,
    )
    st.caption(result.carbon_density_note)



def render_spatial_tab(result: MonitoringResult) -> None:
    st.caption("Spatial evidence layers show forest structure, canopy greenness, and change intensity for the selected ROI.")
    current_ndwi = compute_ndwi(result.current_scene.green, result.current_scene.nir)
    land_cover = classify_land_cover(
        result.current_scene.ndvi,
        ndwi=current_ndwi,
        rgb=result.current_scene.rgb,
    )
    row1 = st.columns(2)
    with row1[0]:
        annotated = create_annotated_rgb(
            rgb=result.current_scene.rgb,
            ndvi=result.current_scene.ndvi,
            probability_map=result.segmentation.probability_map,
        )
        fig_rgb = image_figure(annotated.image, "Annotated RGB Forest Detection", height=420)
        fig_rgb.add_annotation(
            x=annotated.label_col,
            y=annotated.label_row,
            text=f"Forest {annotated.confidence * 100:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowwidth=1.3,
            ax=35,
            ay=-35,
            bgcolor="rgba(0,0,0,0.75)",
            bordercolor="rgba(255,255,255,0.25)",
            font={"color": "#f7fff8", "size": 12},
        )
        st.plotly_chart(fig_rgb, use_container_width=True)
        st.caption("Annotated RGB forest detection overlays the current satellite image with the canopy mask. Brighter green regions indicate forest evidence aligned to the NDVI grid.")
    with row1[1]:
        st.plotly_chart(instance_map_figure(result.segmentation.instance_map, "Detected Forest Stands"), use_container_width=True)
        st.caption("Detected forest stands show individual connected forest patches extracted from the current ROI after segmentation cleanup.")

    row2 = st.columns(3)
    with row2[0]:
        st.plotly_chart(ndvi_figure(result.baseline_scene.ndvi, "Baseline NDVI (Past)"), use_container_width=True)
        baseline_valid = result.baseline_scene.ndvi[np.isfinite(result.baseline_scene.ndvi)]
        st.caption(
            f"Past canopy condition. NDVI ranges from red (sparse/non-vegetated) to green (healthy vegetation). min {np.nanmin(baseline_valid):.3f} | max {np.nanmax(baseline_valid):.3f} | mean {np.nanmean(baseline_valid):.3f}"
        )
    with row2[1]:
        st.plotly_chart(ndvi_figure(result.current_scene.ndvi, "Current NDVI (Present)"), use_container_width=True)
        current_valid = result.current_scene.ndvi[np.isfinite(result.current_scene.ndvi)]
        st.caption(
            f"Present canopy condition for the selected ROI. min {np.nanmin(current_valid):.3f} | max {np.nanmax(current_valid):.3f} | mean {np.nanmean(current_valid):.3f}"
        )
    with row2[2]:
        st.plotly_chart(change_figure(result.ndvi_change, "NDVI Change (Present - Past)"), use_container_width=True)
        change_valid = result.ndvi_change[np.isfinite(result.ndvi_change)]
        st.caption(
            f"Red = vegetation loss, green = vegetation gain, white = little or no change. min {np.nanmin(change_valid):.3f} | max {np.nanmax(change_valid):.3f} | mean {np.nanmean(change_valid):.3f}"
        )

    row3 = st.columns(2)
    with row3[0]:
        st.plotly_chart(ndwi_figure(current_ndwi, "NDWI Water Detection"), use_container_width=True)
        ndwi_valid = current_ndwi[np.isfinite(current_ndwi)]
        st.caption(
            f"Positive NDWI values indicate likely surface water. min {np.nanmin(ndwi_valid):.3f} | max {np.nanmax(ndwi_valid):.3f} | mean {np.nanmean(ndwi_valid):.3f}"
        )
    with row3[1]:
        st.plotly_chart(classified_map_figure(land_cover, "Final Land-Cover Classification"), use_container_width=True)
        st.caption("Final classified map uses NDWI for water, NDVI for forest and vegetation, and assigns remaining low-index land to soil.")

    st.plotly_chart(dndvi_histogram_figure(result.ndvi_change), use_container_width=True)
    st.caption("The dNDVI histogram summarizes how much of the ROI moved toward loss or gain. Values below -0.10 indicate likely vegetation decline.")



def render_environment_tab(result: MonitoringResult, context: AnalysisContext) -> None:
    st.caption("Environmental context combines NDVI-derived land cover, air quality pressure, and nearby wildfire activity around the ROI.")
    current_ndwi = compute_ndwi(result.current_scene.green, result.current_scene.nir)
    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(
            class_distribution_figure(
                result.current_scene.ndvi,
                current_ndwi,
                rgb_current=result.current_scene.rgb,
            ),
            use_container_width=True,
        )
        st.caption("Land-cover distribution combines NDVI and NDWI so open water is counted separately from soil and sparse vegetation.")
    with row1[1]:
        st.plotly_chart(aqi_trend_figure(context.aqi_history), use_container_width=True)
        st.caption("Air quality trend shows the recent AQI profile near the ROI. Higher AQI values indicate poorer air quality and added ecosystem stress.")

    row2 = st.columns(2)
    with row2[0]:
        st.plotly_chart(wildfire_timeline_figure(context.wildfire_events), use_container_width=True)
        st.caption("Nearby wildfire chart lists recent fire activity around the ROI by date and severity so the panel can relate vegetation stress to fire pressure.")
    with row2[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Risk Context")
        aqi_level = classify_aqi_level(context.latest_aqi.us_aqi if context.latest_aqi else None)
        st.caption(f"Current AQI category: {aqi_level} | Wildfire risk score: {context.wildfire_score:.2f}")
        for event in context.wildfire_events:
            st.write(f"{event.date} | {event.severity} | {event.distance_km:.1f} km | {event.source}")
        st.markdown('</div>', unsafe_allow_html=True)



def render_results_tabs(result: MonitoringResult, context: AnalysisContext) -> None:
    tabs = st.tabs(["Spatial Monitoring", "Environmental Context"])
    with tabs[0]:
        render_spatial_tab(result)
    with tabs[1]:
        render_environment_tab(result, context)



def render_forest_impact_section(result: MonitoringResult) -> tuple[dict[str, Any], float]:
    section_header(
        "Impact Assessment",
        "Forest Impact Estimator",
        "Estimate how additional disturbance over the monitored forest area could affect carbon loss, forest reduction, and projected risk increase using hectares-based inputs.",
    )
    inputs = st.columns(3)
    with inputs[0]:
        area_affected_ha = st.number_input("Area affected (ha)", min_value=0.1, value=2.0, step=0.1)
    with inputs[1]:
        vegetation_density = st.selectbox("Vegetation density", options=["Low", "Medium", "High"])
    with inputs[2]:
        impact_type = st.selectbox("Impact type", options=["Deforestation", "Degradation"])

    impact = forest_impact_estimate(
        area_ha=area_affected_ha,
        vegetation_density=vegetation_density,
        impact_type=impact_type,
        roi_area_ha=result.roi_area_ha,
    )
    impact_cards = st.columns(4)
    with impact_cards[0]:
        metric_card(
            "Carbon Loss",
            format_carbon_with_units(float(impact["carbon_loss_tco2e"]), decimals=0),
            "C",
            "tCO2e = tonnes of CO2 equivalent",
        )
    with impact_cards[1]:
        metric_card(
            "Forest Reduction",
            f"{impact['forest_reduction_pct']:.1f}%",
            "F",
            f"Affected area: {format_area_with_secondary_unit(float(impact['area_ha']), decimals=2)}",
        )
    with impact_cards[2]:
        metric_card("Risk Increase", f"{impact['risk_increase_pct']:.1f}%", "R", "Projected monitoring risk uplift")
    with impact_cards[3]:
        metric_card("Impact Level", str(impact["impact_level"]), "I", impact["impact_type"], badge=str(impact["impact_level"]))

    projected_combined_score = min(1.0, result.risk.score + (float(impact["risk_increase_pct"]) / 160.0))
    projected_level = "Low" if projected_combined_score < 0.33 else ("Medium" if projected_combined_score < 0.67 else "High")
    st.markdown(
        f"""<div class="summary-band"><strong>Projected combined monitoring risk:</strong> {projected_combined_score:.2f} ({projected_level}) &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Impact estimate:</strong> {impact['explanation']}</div>""",
        unsafe_allow_html=True,
    )
    st.caption("Impact estimator uses hectares as the primary area unit. Acres are shown as a secondary reference for presentation clarity.")
    return impact, projected_combined_score



def render_report_download(result: MonitoringResult, context: AnalysisContext, impact: dict[str, Any], combined_score: float) -> None:
    health = forest_health_status(result)
    region_label = str(st.session_state.active_region_label)
    roi_bounds = geometry_bounds(st.session_state.roi_geometry)
    report = PDFReportData(
        title="Forest Monitoring & Impact Assessment Report",
        generated_at=datetime.now(),
        roi_name=region_label,
        roi_details={
            "ROI centroid": f"{context.centroid[0]:.5f}, {context.centroid[1]:.5f}",
            "ROI area (ha)": f"{result.roi_area_ha:.2f}",
            "ROI bounds": f"{roi_bounds[0]:.4f}, {roi_bounds[1]:.4f} to {roi_bounds[2]:.4f}, {roi_bounds[3]:.4f}",
            "Baseline scene": f"{result.baseline_scene.item_id} ({result.baseline_scene.acquisition_date})",
            "Current scene": f"{result.current_scene.item_id} ({result.current_scene.acquisition_date})",
        },
        metrics={
            "Environmental Risk": result.risk.level,
            "Forest Health": health["status"],
            "Forest Loss %": f"{result.forest_loss_percent:.2f}%",
            "Estimated Carbon Loss": f"{result.carbon_loss_tco2e:.2f} tCO2e",
            "Wildfire Risk": context.wildfire_level,
            "Projected Combined Risk": f"{combined_score:.2f}",
            "Forest Impact Level": str(impact["impact_level"]),
        },
        baseline_ndvi=result.baseline_scene.ndvi,
        current_ndvi=result.current_scene.ndvi,
        current_ndwi=compute_ndwi(result.current_scene.green, result.current_scene.nir),
        ndvi_change=result.ndvi_change,
        current_rgb=result.current_scene.rgb,
        segmentation_probability_map=result.segmentation.probability_map,
        segmentation_instance_map=result.segmentation.instance_map,
        forest_loss_mask=result.forest_loss_mask,
        aqi_history=context.aqi_history,
        wildfire_events=context.wildfire_events,
    )

    try:
        pdf_bytes = build_pdf_report(report)
    except Exception as exc:
        st.error(f"PDF report generation failed: {exc}")
        return

    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name=f"forest_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )



def render_results_dashboard(result: MonitoringResult) -> None:
    context = build_analysis_context(result)
    render_analysis_warnings(result)
    render_decision_snapshot(result, context)
    render_results_tabs(result, context)
    impact, combined_score = render_forest_impact_section(result)
    render_report_download(result, context, impact, combined_score)



def main() -> None:
    inject_styles()
    initialize_session_state()
    render_hero_banner()
    render_runtime_notices()

    comparison_window = build_comparison_window()
    run_analysis = render_control_bar(comparison_window)
    render_workspace_intro()
    selected_geometry, map_data = render_map_canvas()
    sync_roi_selection(map_data, selected_geometry, run_analysis)
    render_workspace_guidance(comparison_window)
    run_requested_analysis(run_analysis, comparison_window)

    result: MonitoringResult | None = st.session_state.analysis_result
    if result is not None and st.session_state.roi_geometry is not None:
        render_results_dashboard(result)


main()
