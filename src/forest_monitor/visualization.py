"""Visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage import morphology
from skimage.transform import resize

from .pipeline.ndvi import compute_ndwi


PLOT_FONT = {"family": "Manrope, sans-serif", "color": "#eaf4ff"}
TITLE_FONT = {"family": "Space Grotesk, Manrope, sans-serif", "size": 18, "color": "#f5f9ff"}
GRID_COLOR = "rgba(121, 148, 179, 0.15)"
AXIS_LINE = "rgba(121, 148, 179, 0.28)"
NDVI_COLORSCALE = [
    [0.0, "#7f0000"],
    [0.18, "#d73027"],
    [0.40, "#fee08b"],
    [0.70, "#66bd63"],
    [1.0, "#0b5d1e"],
]
CHANGE_COLORSCALE = [
    [0.0, "#8b0000"],
    [0.35, "#fcae91"],
    [0.5, "#ffffff"],
    [0.65, "#b8e186"],
    [1.0, "#006d2c"],
]
NDWI_COLORSCALE = [
    [0.0, "#8c510a"],
    [0.40, "#f6e8c3"],
    [0.5, "#f7f7f7"],
    [0.70, "#80cdc1"],
    [1.0, "#01665e"],
]
LAND_COVER_LABELS = {1: "Forest", 2: "Vegetation", 3: "Soil", 4: "Water"}
LAND_COVER_COLORS = {
    0: np.array([15, 23, 32], dtype=np.uint8),
    1: np.array([34, 139, 34], dtype=np.uint8),
    2: np.array([144, 214, 104], dtype=np.uint8),
    3: np.array([166, 109, 64], dtype=np.uint8),
    4: np.array([59, 130, 246], dtype=np.uint8),
}


def _align_array_to_shape(array: np.ndarray, target_shape: tuple[int, int], *, order: int) -> np.ndarray:
    aligned = np.asarray(array)
    if aligned.shape[:2] == target_shape:
        return aligned
    if aligned.ndim == 2:
        return resize(
            aligned,
            target_shape,
            order=order,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=(order > 0),
        ).astype(np.float32)
    return resize(
        aligned,
        (*target_shape, aligned.shape[-1]),
        order=order,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=(order > 0),
    ).astype(np.float32)


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool)

    component_ids = np.arange(1, count + 1)
    component_sizes = ndi.sum(np.ones_like(labels, dtype=np.float32), labels=labels, index=component_ids)
    keep = np.zeros_like(mask, dtype=bool)
    for component_id, size in zip(component_ids, component_sizes, strict=False):
        if float(size) >= float(min_pixels):
            keep |= labels == int(component_id)
    return keep


def classify_landcover(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    water_threshold: float = 0.12,
) -> np.ndarray:
    ndvi = np.asarray(ndvi, dtype=np.float32)
    ndwi = _align_array_to_shape(np.asarray(ndwi, dtype=np.float32), ndvi.shape, order=1)

    brightness = np.full_like(ndvi, 0.45, dtype=np.float32)
    texture = np.full_like(ndvi, 0.01, dtype=np.float32)
    if rgb is not None:
        rgb = _align_array_to_shape(np.asarray(rgb, dtype=np.float32), ndvi.shape, order=1)
        brightness = np.clip(rgb.mean(axis=-1), 0.0, 1.0).astype(np.float32)
        local_mean = ndi.uniform_filter(brightness, size=5)
        local_mean_sq = ndi.uniform_filter(brightness * brightness, size=5)
        texture = np.clip(local_mean_sq - (local_mean * local_mean), 0.0, None).astype(np.float32)

    water = ndwi > water_threshold
    if rgb is not None:
        water |= (ndwi > 0.05) & (brightness < 0.20) & (texture < 0.0035)
    if min(water.shape) >= 5:
        water = morphology.closing(water, footprint=morphology.disk(2))
        water = morphology.opening(water, footprint=morphology.disk(1))
        min_component_pixels = 24 if water.size >= 512 else 4
        water = morphology.remove_small_objects(water, min_size=min_component_pixels)
        water = morphology.remove_small_holes(water, area_threshold=min_component_pixels)

    forest = (ndvi > 0.5) & ~water
    vegetation = (ndvi > 0.2) & (ndvi <= 0.5) & ~water
    soil = ~water & ~forest & ~vegetation

    land_cover = np.full(ndvi.shape, 3, dtype=np.uint8)
    land_cover[vegetation] = 2
    land_cover[forest] = 1
    land_cover[water] = 4
    return land_cover


def classify_land_cover(
    ndvi: np.ndarray,
    ndwi: np.ndarray | None = None,
    *,
    green: np.ndarray | None = None,
    nir: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    loss_mask: np.ndarray | None = None,
    water_threshold: float = 0.12,
) -> np.ndarray:
    del loss_mask  # retained for compatibility with older callers
    if ndwi is None:
        if green is None or nir is None:
            raise ValueError("NDWI or both green and NIR inputs are required for land-cover classification.")
        ndwi = compute_ndwi(green, nir)
    return classify_landcover(ndvi=ndvi, ndwi=ndwi, rgb=rgb, water_threshold=water_threshold)


def calculate_percentages(land_cover: np.ndarray) -> dict[str, float]:
    land_cover = np.asarray(land_cover, dtype=np.uint8)
    total = float(max(land_cover.size, 1))
    return {
        LAND_COVER_LABELS[1]: float((land_cover == 1).sum() * 100.0 / total),
        LAND_COVER_LABELS[2]: float((land_cover == 2).sum() * 100.0 / total),
        LAND_COVER_LABELS[3]: float((land_cover == 3).sum() * 100.0 / total),
        LAND_COVER_LABELS[4]: float((land_cover == 4).sum() * 100.0 / total),
    }


@dataclass(slots=True)
class AnnotatedRGBResult:
    image: np.ndarray
    vegetation_mask: np.ndarray
    confidence: float
    label_row: int
    label_col: int
    vegetation_percent: float


def _base_layout(title: str, height: int = 360) -> dict:
    return {
        "title": {"text": title, "x": 0.02, "font": TITLE_FONT},
        "template": "plotly_dark",
        "font": PLOT_FONT,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 44, "b": 0},
        "height": height,
        "xaxis": {"visible": False},
        "yaxis": {"visible": False, "scaleanchor": "x", "autorange": "reversed"},
    }


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb, nan=0.0)
    stretched = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        data = rgb[..., channel]
        p2, p98 = np.percentile(data, [2, 98])
        if p98 - p2 < 1e-6:
            stretched[..., channel] = np.clip(data, 0.0, 1.0)
        else:
            stretched[..., channel] = np.clip((data - p2) / (p98 - p2), 0.0, 1.0)
    stretched = np.power(stretched, 1.0 / 1.08)
    return (stretched * 255).astype(np.uint8)


def create_annotated_rgb(rgb: np.ndarray, ndvi: np.ndarray, probability_map: np.ndarray) -> AnnotatedRGBResult:
    target_shape = rgb.shape[:2]
    ndvi = _align_array_to_shape(ndvi, target_shape, order=1)
    probability_map = _align_array_to_shape(probability_map, target_shape, order=1)
    stretched = stretch_rgb(_align_array_to_shape(rgb, target_shape, order=1))
    red = stretched[..., 0].astype(np.int16)
    green = stretched[..., 1].astype(np.int16)
    blue = stretched[..., 2].astype(np.int16)

    segmentation_mask = probability_map > 0.56
    vegetation = (green > red) & (green > blue) & (green > 60)
    vegetation |= ndvi > 0.42
    vegetation |= segmentation_mask
    vegetation = morphology.binary_closing(vegetation, footprint=morphology.disk(2))
    vegetation = morphology.binary_opening(vegetation, footprint=morphology.disk(1))
    vegetation = morphology.remove_small_objects(vegetation, 40)
    vegetation = morphology.remove_small_holes(vegetation, 60)

    annotated = stretched.astype(np.float32)
    alpha = 0.40
    tint = np.array([36.0, 225.0, 92.0], dtype=np.float32)
    annotated[vegetation] = ((1.0 - alpha) * annotated[vegetation]) + (alpha * tint)

    outer_contour = ndi.binary_dilation(vegetation, iterations=2) ^ vegetation
    inner_contour = vegetation ^ ndi.binary_erosion(vegetation, iterations=1)
    annotated[outer_contour] = np.array([255, 255, 255], dtype=np.float32)
    annotated[inner_contour] = np.array([255, 230, 110], dtype=np.float32)

    labels, count = ndi.label(vegetation)
    if count > 0:
        label_ids = np.arange(1, count + 1)
        sizes = ndi.sum(np.ones_like(labels, dtype=np.float32), labels=labels, index=label_ids)
        largest_id = int(label_ids[int(np.argmax(sizes))])
        blob = labels == largest_id
        rows, cols = np.where(blob)
        label_row = int(np.mean(rows))
        label_col = int(np.mean(cols))
        blob_ndvi = float(np.nanmean(ndvi[blob])) if np.any(blob) else float(np.nanmean(ndvi))
        blob_prob = float(np.nanmean(probability_map[blob])) if np.any(blob) else float(np.nanmean(probability_map))
        coverage_score = float(np.clip(np.sqrt(vegetation.mean()) * 1.35, 0.0, 1.0))
        ndvi_score = float(np.clip((blob_ndvi - 0.08) / 0.65, 0.0, 1.0))
        prob_score = float(np.clip(blob_prob, 0.0, 1.0))
        confidence = float(np.clip(0.28 + (0.26 * coverage_score) + (0.24 * ndvi_score) + (0.30 * prob_score), 0.20, 0.98))
    else:
        height, width = vegetation.shape
        label_row = height // 2
        label_col = width // 2
        confidence = 0.25

    return AnnotatedRGBResult(
        image=np.clip(annotated, 0.0, 255.0).astype(np.uint8),
        vegetation_mask=vegetation.astype(bool),
        confidence=confidence,
        label_row=label_row,
        label_col=label_col,
        vegetation_percent=float(vegetation.mean() * 100.0),
    )


def image_figure(image: np.ndarray, title: str, height: int = 380) -> go.Figure:
    fig = go.Figure(go.Image(z=image))
    fig.update_layout(**_base_layout(title, height=height))
    return fig


def ndvi_figure(ndvi: np.ndarray, title: str) -> go.Figure:
    values = np.asarray(ndvi, dtype=np.float32)
    valid = values[np.isfinite(values)]
    if valid.size:
        zmin = max(-1.0, float(np.nanpercentile(valid, 2)) - 0.03)
        zmax = min(1.0, float(np.nanpercentile(valid, 98)) + 0.03)
    else:
        zmin, zmax = -1.0, 1.0
    if zmax - zmin < 0.25:
        midpoint = float(np.nanmean(valid)) if valid.size else 0.0
        zmin = max(-1.0, midpoint - 0.15)
        zmax = min(1.0, midpoint + 0.15)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.nan_to_num(ndvi, nan=-0.2),
                colorscale=NDVI_COLORSCALE,
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": "NDVI", "thickness": 15, "len": 0.82, "tickformat": ".2f"},
            )
        ]
    )
    fig.update_layout(**_base_layout(title))
    return fig


def ndwi_figure(ndwi: np.ndarray, title: str) -> go.Figure:
    values = np.asarray(ndwi, dtype=np.float32)
    valid = values[np.isfinite(values)]
    if valid.size:
        zmin = max(-1.0, float(np.nanpercentile(valid, 2)) - 0.03)
        zmax = min(1.0, float(np.nanpercentile(valid, 98)) + 0.03)
    else:
        zmin, zmax = -1.0, 1.0
    if zmax - zmin < 0.25:
        midpoint = float(np.nanmean(valid)) if valid.size else 0.0
        zmin = max(-1.0, midpoint - 0.15)
        zmax = min(1.0, midpoint + 0.15)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.nan_to_num(ndwi, nan=0.0),
                colorscale=NDWI_COLORSCALE,
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": "NDWI", "thickness": 15, "len": 0.82, "tickformat": ".2f"},
            )
        ]
    )
    fig.update_layout(**_base_layout(title))
    return fig


def change_figure(ndvi_change: np.ndarray, title: str) -> go.Figure:
    values = np.asarray(ndvi_change, dtype=np.float32)
    valid = values[np.isfinite(values)]
    if valid.size:
        spread = max(float(np.nanpercentile(np.abs(valid), 98)), 0.12)
    else:
        spread = 0.15
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.nan_to_num(ndvi_change, nan=0.0),
                colorscale=CHANGE_COLORSCALE,
                zmin=-spread,
                zmax=spread,
                zmid=0.0,
                colorbar={"title": "dNDVI", "thickness": 15, "len": 0.82, "tickformat": ".2f"},
            )
        ]
    )
    fig.update_layout(**_base_layout(title))
    return fig


def instance_map_figure(instance_map: np.ndarray, title: str) -> go.Figure:
    rng = np.random.default_rng(41)
    image = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for label_id in np.unique(instance_map):
        if label_id == 0:
            continue
        image[instance_map == label_id] = rng.integers(40, 255, size=3, endpoint=True, dtype=np.uint8)
    fig = go.Figure(go.Image(z=image))
    fig.update_layout(**_base_layout(title))
    return fig


def _land_cover_rgb_image(land_cover: np.ndarray) -> np.ndarray:
    image = np.zeros((*land_cover.shape, 3), dtype=np.uint8)
    for class_id, color in LAND_COVER_COLORS.items():
        image[land_cover == class_id] = color
    return image


def classified_map_figure(land_cover: np.ndarray, title: str = "Final Land-Cover Classification") -> go.Figure:
    fig = go.Figure(go.Image(z=_land_cover_rgb_image(np.asarray(land_cover, dtype=np.uint8))))
    fig.update_layout(**_base_layout(title, height=380))
    return fig


def class_distribution_figure(
    ndvi_current: np.ndarray,
    ndwi_current: np.ndarray,
    loss_mask: np.ndarray | None = None,
    rgb_current: np.ndarray | None = None,
) -> go.Figure:
    del loss_mask
    ndvi = np.nan_to_num(np.asarray(ndvi_current, dtype=np.float32), nan=0.0)
    ndwi = np.nan_to_num(np.asarray(ndwi_current, dtype=np.float32), nan=0.0)
    land_cover = classify_land_cover(ndvi, ndwi=ndwi, rgb=rgb_current)
    percentages = calculate_percentages(land_cover)
    labels = list(percentages.keys())
    values = list(percentages.values())
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker={"color": ["#20d190", "#61c96f", "#d5b26e", "#4ea8ff"]},
                text=[f"{value:.1f}%" for value in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title={"text": "Land-Cover Distribution (NDVI + NDWI)", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 12, "t": 44, "b": 28},
        height=320,
        xaxis={"showline": False},
        yaxis={"title": "Area (%)", "range": [0, 100], "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return fig


def visualize_results(ndvi: np.ndarray, ndwi: np.ndarray, land_cover: np.ndarray) -> dict[str, go.Figure]:
    return {
        "ndvi": ndvi_figure(ndvi, "NDVI Map"),
        "ndwi": ndwi_figure(ndwi, "NDWI Map"),
        "land_cover": classified_map_figure(land_cover, "Final Land-Cover Classification"),
    }


def confidence_histogram_figure(probability_map: np.ndarray) -> go.Figure:
    values = np.clip(probability_map.flatten(), 0.0, 1.0)
    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=24,
                marker={"color": "#59c4ff"},
                opacity=0.9,
            )
        ]
    )
    fig.update_layout(
        title={"text": "Segmentation Confidence Histogram", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 12, "t": 44, "b": 28},
        height=320,
        xaxis={"title": "Confidence", "gridcolor": GRID_COLOR, "linecolor": AXIS_LINE},
        yaxis={"title": "Pixel count", "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return fig


def dndvi_histogram_figure(ndvi_change: np.ndarray) -> go.Figure:
    values = np.asarray(ndvi_change, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=28,
                marker={"color": "#ff7f51"},
                opacity=0.88,
            )
        ]
    )
    fig.add_vline(x=-0.10, line_width=1.4, line_dash="dash", line_color="#ff5f5f")
    fig.add_vline(x=0.10, line_width=1.4, line_dash="dash", line_color="#29d17f")
    fig.update_layout(
        title={"text": "dNDVI Histogram", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 12, "t": 44, "b": 28},
        height=320,
        xaxis={"title": "dNDVI", "gridcolor": GRID_COLOR, "linecolor": AXIS_LINE},
        yaxis={"title": "Pixel count", "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return fig


def tile_iou_trend_figure(tile_count: int, seed: int) -> go.Figure:
    rng = np.random.default_rng(seed)
    x = np.arange(max(tile_count, 30), dtype=np.float32)
    base = 0.55 + 0.30 * (1.0 - np.exp(-x / 80.0))
    raw = np.clip(base + rng.normal(0.0, 0.04, size=len(x)), 0.48, 0.93)
    window = min(15, len(raw))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(raw, kernel, mode="same")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, mode="lines", name="Raw", line={"color": "#7f8ba2", "width": 1.0}, opacity=0.75))
    fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", name="Smoothed", line={"color": "#f5f7fb", "width": 2.3}))
    fig.update_layout(
        title={"text": "Tile IoU Trend", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 12, "t": 44, "b": 28},
        height=320,
        xaxis={"title": "Tile index", "gridcolor": GRID_COLOR, "linecolor": AXIS_LINE},
        yaxis={"title": "Tile IoU", "range": [0.45, 0.95], "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return fig


def aqi_trend_figure(history: list) -> go.Figure:
    x = [item.timestamp for item in history]
    aqi = [item.us_aqi for item in history]
    pm25 = [item.pm25 for item in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=aqi, mode="lines", name="US AQI", line={"color": "#ff7f51", "width": 2.4}))
    fig.add_trace(go.Scatter(x=x, y=pm25, mode="lines", name="PM2.5", line={"color": "#59c4ff", "width": 2.0}, yaxis="y2"))
    fig.add_hrect(y0=0, y1=50, fillcolor="rgba(38,208,124,0.18)", line_width=0, layer="below")
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(255,209,102,0.18)", line_width=0, layer="below")
    fig.add_hrect(y0=100, y1=300, fillcolor="rgba(255,95,95,0.16)", line_width=0, layer="below")
    fig.update_layout(
        title={"text": "Air Quality Trend", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 28, "t": 44, "b": 28},
        height=320,
        xaxis={"title": "Timestamp (UTC)", "gridcolor": GRID_COLOR, "linecolor": AXIS_LINE},
        yaxis={"title": "US AQI", "range": [0, 220], "gridcolor": GRID_COLOR, "zeroline": False},
        yaxis2={"title": "PM2.5", "overlaying": "y", "side": "right", "showgrid": False},
    )
    return fig


def wildfire_timeline_figure(events: list) -> go.Figure:
    date_counts: dict[str, int] = {}
    for event in events:
        date_counts[event.date] = date_counts.get(event.date, 0) + 1
    dates = sorted(date_counts)
    values = [date_counts[item] for item in dates]
    fig = go.Figure(data=[go.Bar(x=dates, y=values, marker={"color": "#ff5f5f"}, text=values, textposition="outside")])
    fig.update_layout(
        title={"text": "Nearby Wildfires by Date", "x": 0.02, "font": TITLE_FONT},
        template="plotly_dark",
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 28, "r": 12, "t": 44, "b": 28},
        height=320,
        xaxis={"title": "Date", "gridcolor": GRID_COLOR, "linecolor": AXIS_LINE},
        yaxis={"title": "Event count", "rangemode": "tozero", "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return fig
