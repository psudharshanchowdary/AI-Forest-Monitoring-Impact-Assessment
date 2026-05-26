"""Report builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .environment import AirQualitySnapshot, WildfireEvent
from .visualization import classify_land_cover, create_annotated_rgb

A4_WIDTH = 8.27
A4_HEIGHT = 11.69
PDF_DPI = 220
PAGE_FACE = "#f7fafc"
TEXT_COLOR = "#102a43"
MUTED_TEXT = "#486581"
ACCENT = "#163b2d"
GRID = "#d9e2ec"


@dataclass(slots=True)
class PDFReportData:
    title: str
    generated_at: datetime
    roi_name: str
    roi_details: Mapping[str, str]
    metrics: Mapping[str, str]
    baseline_ndvi: np.ndarray
    current_ndvi: np.ndarray
    current_ndwi: np.ndarray
    ndvi_change: np.ndarray
    current_rgb: np.ndarray
    segmentation_probability_map: np.ndarray
    segmentation_instance_map: np.ndarray
    forest_loss_mask: np.ndarray
    aqi_history: Sequence[AirQualitySnapshot]
    wildfire_events: Sequence[WildfireEvent]


def build_report_payload(
    analysis_summary: dict[str, Any],
    wildfire_events: list,
    aqi_latest: dict[str, Any] | None,
    impact_assessment: dict[str, Any] | None = None,
    combined_risk_score: float = 0.0,
    construction_impact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    impact = impact_assessment if impact_assessment is not None else (construction_impact or {})
    return {
        "analysis": analysis_summary,
        "wildfire_events": [asdict(item) for item in wildfire_events],
        "latest_air_quality": aqi_latest,
        "forest_impact_assessment": impact,
        "construction_impact": impact,
        "combined_environmental_risk_score": round(combined_risk_score, 4),
    }


def build_pdf_report(report: PDFReportData) -> bytes:
    """Generate a styled multi-page PDF report for the current dashboard result."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pages = [
            _build_cover_page(report),
            _build_ndvi_page(report),
            _build_change_page(report),
            _build_segmentation_page(report),
            _build_environment_page(report),
        ]
        for figure in pages:
            pdf.savefig(figure, dpi=PDF_DPI, bbox_inches="tight", facecolor=PAGE_FACE)
            plt.close(figure)
        metadata = pdf.infodict()
        metadata["Title"] = report.title
        metadata["Author"] = "AI-Based Forest Monitoring Dashboard"
        metadata["Subject"] = "Forest Monitoring & Impact Assessment Report"
        metadata["Keywords"] = "forest monitoring, ndvi, segmentation, wildfire, air quality"
        metadata["CreationDate"] = report.generated_at
    return buffer.getvalue()


def _new_page() -> tuple[Figure, list]:
    figure = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), facecolor=PAGE_FACE)
    return figure, []


def _apply_page_header(figure: Figure, page_title: str, page_subtitle: str, page_number: int) -> None:
    figure.text(0.06, 0.965, page_title, fontsize=17, fontweight="bold", color=ACCENT, ha="left", va="top")
    figure.text(0.06, 0.942, page_subtitle, fontsize=9.5, color=MUTED_TEXT, ha="left", va="top")
    figure.text(0.94, 0.965, f"Page {page_number}", fontsize=9, color=MUTED_TEXT, ha="right", va="top")
    figure.add_artist(plt.Line2D([0.06, 0.94], [0.928, 0.928], color=GRID, linewidth=1.0))


def _footer_text(figure: Figure, text: str) -> None:
    figure.text(0.06, 0.03, text, fontsize=8.5, color=MUTED_TEXT, ha="left", va="bottom")


def _build_cover_page(report: PDFReportData) -> Figure:
    figure, _ = _new_page()
    _apply_page_header(
        figure,
        report.title,
        "Forest Monitoring & Impact Assessment Report",
        1,
    )

    figure.text(0.06, 0.88, "Current ROI and Report Details", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    roi_lines = [
        f"Generated at: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"ROI name: {report.roi_name}",
    ]
    roi_lines.extend(f"{key}: {value}" for key, value in report.roi_details.items())
    figure.text(0.06, 0.858, "\n".join(roi_lines), fontsize=9.5, color=TEXT_COLOR, va="top", linespacing=1.6)

    figure.text(0.06, 0.64, "Key Monitoring Metrics", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    metric_rows = [["Metric", "Value"], *[[key, value] for key, value in report.metrics.items()]]
    table_ax = figure.add_axes([0.06, 0.35, 0.88, 0.25])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=metric_rows,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.46, 0.42],
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#e7f0ea")
            cell.set_text_props(weight="bold", color=ACCENT)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fbfd")
            cell.set_text_props(color=TEXT_COLOR)

    figure.text(0.06, 0.29, "Report Scope", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    scope = (
        "This report summarizes the latest dashboard analysis for the selected ROI, including vegetation index "
        "monitoring, forest stand evidence, land-cover interpretation, and environmental context."
    )
    figure.text(0.06, 0.262, scope, fontsize=9.5, color=TEXT_COLOR, va="top", wrap=True)
    _footer_text(
        figure,
        "Prepared from the current dashboard state. Values and figures reflect the latest selected ROI and analysis run.",
    )
    return figure


def _build_ndvi_page(report: PDFReportData) -> Figure:
    figure = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), facecolor=PAGE_FACE)
    _apply_page_header(
        figure,
        "NDVI Analysis",
        "Baseline and current canopy condition derived from the selected monitoring window.",
        2,
    )
    gs = figure.add_gridspec(1, 2, left=0.06, right=0.94, top=0.89, bottom=0.16, hspace=0.0, wspace=0.18)

    ax1 = figure.add_subplot(gs[0, 0])
    ax2 = figure.add_subplot(gs[0, 1])

    _heatmap_panel(
        ax1,
        report.baseline_ndvi,
        title="Baseline NDVI (Past)",
        caption="Past vegetation condition. Higher NDVI indicates denser and healthier canopy.",
        cmap="RdYlGn",
        colorbar_label="NDVI",
        vmin=-0.2,
        vmax=0.9,
    )
    _heatmap_panel(
        ax2,
        report.current_ndvi,
        title="Current NDVI (Present)",
        caption="Present vegetation condition for the selected ROI.",
        cmap="RdYlGn",
        colorbar_label="NDVI",
        vmin=-0.2,
        vmax=0.9,
    )
    _footer_text(
        figure,
        "NDVI Analysis: B04/B08 vegetation index layers provide the canopy condition baseline and the latest present-state vegetation map for the current ROI.",
    )
    return figure


def _build_change_page(report: PDFReportData) -> Figure:
    figure = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), facecolor=PAGE_FACE)
    _apply_page_header(
        figure,
        "Change Detection",
        "NDVI change shows vegetation decline or recovery between the baseline and present scenes.",
        3,
    )
    gs = figure.add_gridspec(1, 1, left=0.08, right=0.92, top=0.88, bottom=0.20)
    ax = figure.add_subplot(gs[0, 0])
    _heatmap_panel(
        ax,
        report.ndvi_change,
        title="NDVI Change (Present - Past)",
        caption="Current NDVI minus baseline NDVI. Negative zones indicate vegetation decline and possible forest loss hotspots.",
        cmap="RdBu",
        colorbar_label="dNDVI",
        vmin=-0.5,
        vmax=0.5,
    )
    _footer_text(
        figure,
        "Change Detection: dNDVI is calculated as present NDVI minus past NDVI. Strong negative values mark stressed or declining vegetation.",
    )
    return figure


def _build_segmentation_page(report: PDFReportData) -> Figure:
    figure = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), facecolor=PAGE_FACE)
    _apply_page_header(
        figure,
        "Segmentation and Land-Cover Interpretation",
        "Forest detection evidence generated from RGB, NDVI, and stand-level segmentation outputs.",
        4,
    )
    gs = figure.add_gridspec(2, 2, left=0.06, right=0.94, top=0.89, bottom=0.16, hspace=0.44, wspace=0.18)

    annotated = create_annotated_rgb(
        rgb=report.current_rgb,
        ndvi=report.current_ndvi,
        probability_map=report.segmentation_probability_map,
    )
    ax1 = figure.add_subplot(gs[0, 0])
    ax2 = figure.add_subplot(gs[0, 1])
    ax3 = figure.add_subplot(gs[1, :])

    _image_panel(
        ax1,
        annotated.image,
        title="Annotated RGB Forest Detection",
        caption=f"Forest-highlighted RGB overlay. Estimated dominant forest confidence: {annotated.confidence * 100:.1f}%.",
    )
    _instance_map_panel(
        ax2,
        report.segmentation_instance_map,
        title="Detected Forest Stands",
        caption="Instance-level forest stands detected for the selected ROI.",
    )
    land_cover = classify_land_cover(
        report.current_ndvi,
        ndwi=report.current_ndwi,
        rgb=report.current_rgb,
    )
    _land_cover_bar_chart(
        ax3,
        land_cover,
        title="Land-Cover Distribution",
        caption="Class distribution derived from NDVI plus NDWI so water pixels are separated from soil and vegetation more reliably.",
    )
    _footer_text(
        figure,
        "Segmentation section combines a forest-highlighted RGB overlay, detected stand map, and fused land-cover distribution for interpretation.",
    )
    return figure


def _build_environment_page(report: PDFReportData) -> Figure:
    figure = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), facecolor=PAGE_FACE)
    _apply_page_header(
        figure,
        "Environmental Context",
        "Air quality and nearby wildfire indicators provide additional pressure signals around the monitored ROI.",
        5,
    )
    gs = figure.add_gridspec(2, 1, left=0.08, right=0.94, top=0.89, bottom=0.14, hspace=0.42)
    ax1 = figure.add_subplot(gs[0, 0])
    ax2 = figure.add_subplot(gs[1, 0])

    _aqi_chart(
        ax1,
        report.aqi_history,
        title="Air Quality Trend",
        caption="US AQI and PM2.5 trend across the recent monitoring window.",
    )
    _wildfire_chart(
        ax2,
        report.wildfire_events,
        title="Nearby Wildfire Chart",
        caption="Nearby wildfire counts grouped by date for the current ROI context.",
    )
    _footer_text(
        figure,
        "Environmental context blends AQI history and nearby wildfire activity to support final interpretation of monitoring risk.",
    )
    return figure


def _heatmap_panel(
    axis,
    array: np.ndarray,
    *,
    title: str,
    caption: str,
    cmap: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
) -> None:
    image = axis.imshow(np.nan_to_num(array, nan=vmin), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=MUTED_TEXT)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label(colorbar_label, fontsize=8.5, color=MUTED_TEXT)
    _panel_caption(axis, caption, y=-0.12, width=34)


def _image_panel(axis, image: np.ndarray, *, title: str, caption: str) -> None:
    axis.imshow(image)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    _panel_caption(axis, caption, y=-0.12, width=34)


def _instance_map_panel(axis, instance_map: np.ndarray, *, title: str, caption: str) -> None:
    labels = np.asarray(instance_map, dtype=np.int32)
    max_label = int(labels.max())
    if max_label <= 0:
        palette = ListedColormap(["#08131e"])
        axis.imshow(np.zeros_like(labels), cmap=palette, vmin=0, vmax=1)
        axis.text(0.5, 0.5, "No forest stands detected", transform=axis.transAxes, ha="center", va="center", fontsize=10, color=MUTED_TEXT)
    else:
        cmap = plt.cm.get_cmap("tab20", max_label + 1)
        axis.imshow(labels, cmap=cmap, vmin=0, vmax=max_label)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    _panel_caption(axis, caption, y=-0.12, width=34)


def _land_cover_bar_chart(axis, land_cover: np.ndarray, *, title: str, caption: str) -> None:
    total = float(max(land_cover.size, 1))
    labels = ["Forest", "Vegetation", "Soil", "Water"]
    values = [
        float((land_cover == 1).sum() * 100.0 / total),
        float((land_cover == 2).sum() * 100.0 / total),
        float((land_cover == 3).sum() * 100.0 / total),
        float((land_cover == 4).sum() * 100.0 / total),
    ]
    colors = ["#20d190", "#61c96f", "#d5b26e", "#4ea8ff"]
    axis.bar(labels, values, color=colors, edgecolor="none")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    axis.set_ylabel("Area (%)", color=MUTED_TEXT)
    axis.set_ylim(0.0, max(100.0, max(values, default=0.0) + 15.0))
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="x", labelrotation=0, labelsize=9, colors=TEXT_COLOR)
    axis.tick_params(axis="y", labelsize=8.5, colors=MUTED_TEXT)
    for idx, value in enumerate(values):
        axis.text(idx, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)
    _panel_caption(axis, caption, y=-0.16, width=80)


def _aqi_chart(axis, history: Sequence[AirQualitySnapshot], *, title: str, caption: str) -> None:
    if not history:
        axis.text(0.5, 0.5, "No AQI history available", ha="center", va="center", fontsize=10, color=MUTED_TEXT)
        axis.axis("off")
        return

    x = np.arange(len(history))
    aqi = np.array([item.us_aqi for item in history], dtype=np.float32)
    pm25 = np.array([item.pm25 for item in history], dtype=np.float32)

    axis.plot(x, aqi, color="#ff7f51", linewidth=2.2, label="US AQI")
    axis.fill_between(x, 0, 50, color="#2bd47f", alpha=0.10)
    axis.fill_between(x, 50, 100, color="#ffd166", alpha=0.10)
    axis.fill_between(x, 100, 220, color="#ff5f5f", alpha=0.08)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    axis.set_ylabel("US AQI", color=MUTED_TEXT)
    axis.set_xlabel("Hourly index", color=MUTED_TEXT)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(labelsize=8.5, colors=MUTED_TEXT)

    axis_2 = axis.twinx()
    axis_2.plot(x, pm25, color="#3a86ff", linewidth=1.9, label="PM2.5")
    axis_2.set_ylabel("PM2.5", color=MUTED_TEXT)
    axis_2.tick_params(labelsize=8.5, colors=MUTED_TEXT)
    axis_2.spines["top"].set_visible(False)
    axis_2.spines["left"].set_visible(False)
    axis_2.spines["right"].set_color(GRID)

    handles_1, labels_1 = axis.get_legend_handles_labels()
    handles_2, labels_2 = axis_2.get_legend_handles_labels()
    axis.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, fontsize=8.5, loc="upper left")
    _panel_caption(axis, caption, y=-0.16, width=84)


def _wildfire_chart(axis, events: Sequence[WildfireEvent], *, title: str, caption: str) -> None:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.date] = counts.get(event.date, 0) + 1

    dates = sorted(counts)
    values = [counts[item] for item in dates]
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10)
    if values:
        bars = axis.bar(dates, values, color="#ff5f5f", edgecolor="none")
        axis.bar_label(bars, labels=[str(value) for value in values], padding=3, fontsize=8.5, color=TEXT_COLOR)
    else:
        axis.text(0.5, 0.5, "No nearby wildfire events available", ha="center", va="center", fontsize=10, color=MUTED_TEXT)
    axis.set_ylabel("Event count", color=MUTED_TEXT)
    axis.set_xlabel("Date", color=MUTED_TEXT)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(labelsize=8.5, colors=MUTED_TEXT)
    _panel_caption(axis, caption, y=-0.16, width=84)


def _panel_caption(axis, caption: str, *, y: float, width: int) -> None:
    wrapped = textwrap.fill(caption, width=width)
    axis.text(0.0, y, wrapped, transform=axis.transAxes, fontsize=8.2, color=MUTED_TEXT, va="top")
