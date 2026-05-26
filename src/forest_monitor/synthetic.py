"""Synthetic Sentinel-like scene generation driven by ROI geometry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Point, shape

try:
    from shapely import contains_xy
except ImportError:  # pragma: no cover - fallback for older Shapely builds
    contains_xy = None

from .geometry import geometry_bounds, geometry_centroid, seed_from_geometry
from .pipeline.ndvi import compute_ndvi

LAND_CODE_FOREST = 1
LAND_CODE_VEGETATION = 2
LAND_CODE_SOIL = 3
LAND_CODE_WATER = 4
LAND_CODE_URBAN = 5

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
LAND_POLYGON_PATH = ASSET_ROOT / "land_polygons.geojson"


@dataclass(slots=True)
class SceneData:
    ndvi: np.ndarray
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    nir: np.ndarray
    rgb: np.ndarray
    acquisition_date: str
    item_id: str
    valid_mask: np.ndarray
    land_cover: np.ndarray
    forest_reference: np.ndarray
    cloud_cover_pct: float


@lru_cache(maxsize=1)
def _load_land_geometry():
    if not LAND_POLYGON_PATH.exists():
        return None
    payload = json.loads(LAND_POLYGON_PATH.read_text())
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
        if not features:
            return None
        if len(features) == 1:
            return shape(features[0]["geometry"])
        from shapely.ops import unary_union

        return unary_union([shape(feature["geometry"]) for feature in features])
    if payload.get("type") == "Feature":
        return shape(payload["geometry"])
    return shape(payload)


def _date_midpoint(start: date, end: date) -> date:
    return start + ((end - start) // 2)


def _smooth_noise(height: int, width: int, rng: np.random.Generator, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    field = np.zeros((height, width), dtype=np.float32)
    frequencies = [1.5, 2.6, 4.1, 6.7]
    for idx, freq in enumerate(frequencies, start=1):
        phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = 1.0 / idx
        field += amp * np.sin((xx / max(width - 1, 1)) * freq * np.pi + phase_x)
        field += amp * np.cos((yy / max(height - 1, 1)) * (freq * 0.92) * np.pi + phase_y)
        field += amp * np.sin(((xx + yy) / max(width + height - 2, 1)) * freq * np.pi + (phase_x * 0.55))
    field += rng.normal(0.0, 0.22, size=(height, width)).astype(np.float32)
    field = ndi.gaussian_filter(field, sigma=sigma)
    field -= float(field.min())
    field /= float(field.max() + 1e-6)
    return field.astype(np.float32)


def _point_mask(geometry, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    if geometry is None:
        return np.ones(lon_grid.shape, dtype=bool)
    if contains_xy is not None:
        return contains_xy(geometry, lon_grid, lat_grid)
    flat = np.fromiter(
        (geometry.contains(Point(lon, lat)) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())),
        dtype=bool,
        count=lon_grid.size,
    )
    return flat.reshape(lon_grid.shape)


def _land_mask(bounds: tuple[float, float, float, float], size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = bounds
    yy, xx = np.mgrid[0:size, 0:size]
    x_norm = ((xx / max(size - 1, 1)) * 2.0) - 1.0
    y_norm = ((yy / max(size - 1, 1)) * 2.0) - 1.0
    lon_grid = min_lon + ((xx / max(size - 1, 1)) * (max_lon - min_lon))
    lat_grid = max_lat - ((yy / max(size - 1, 1)) * (max_lat - min_lat))

    raw_land = _point_mask(_load_land_geometry(), lon_grid, lat_grid)
    raw_land = ndi.binary_closing(raw_land, structure=np.ones((3, 3), dtype=bool))
    coastal_transition = ndi.gaussian_filter(raw_land.astype(np.float32), sigma=5.0)
    return raw_land.astype(bool), coastal_transition.astype(np.float32), lon_grid.astype(np.float32), lat_grid.astype(np.float32), x_norm.astype(np.float32), y_norm.astype(np.float32)


def _build_land_cover(
    *,
    raw_land: np.ndarray,
    coastal_transition: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    base_noise: np.ndarray,
    secondary_noise: np.ndarray,
    tertiary_noise: np.ndarray,
) -> dict[str, np.ndarray]:
    lat_abs = np.abs(lat_grid)
    tropical = np.exp(-np.square(lat_abs / 18.0)).astype(np.float32)
    temperate = np.exp(-np.square((lat_abs - 46.0) / 18.0)).astype(np.float32)

    relief = np.clip((0.58 * base_noise) + (0.28 * secondary_noise) + (0.14 * tertiary_noise), 0.0, 1.0)
    humidity = np.clip(
        0.10
        + (0.46 * tropical)
        + (0.24 * temperate)
        + (0.22 * relief)
        + (0.10 * coastal_transition)
        - (0.08 * np.abs(y_norm)),
        0.0,
        1.0,
    )
    dryness = np.clip(
        0.12
        + (0.22 * np.square(np.sin(np.radians((lon_grid * 1.15) - (lat_grid * 0.55)))))
        + (0.18 * (1.0 - relief))
        + (0.12 * np.abs(x_norm)),
        0.0,
        1.0,
    )

    inland_water_signal = np.clip(
        (0.46 * (1.0 - relief))
        + (0.26 * np.clip(np.cos((x_norm * 7.0) + (secondary_noise * 5.0)), 0.0, 1.0))
        + (0.18 * np.clip(np.sin((y_norm * 6.5) - (base_noise * 4.0)), 0.0, 1.0))
        + (0.12 * humidity),
        0.0,
        1.0,
    )
    inland_water = raw_land & (humidity > 0.42) & (dryness < 0.42) & (inland_water_signal > 0.93)
    inland_water = ndi.binary_opening(inland_water, structure=np.ones((3, 3), dtype=bool))
    inland_water = ndi.binary_closing(inland_water, structure=np.ones((5, 5), dtype=bool))

    water = (~raw_land) | inland_water
    urban_pressure = np.clip(
        0.06
        + (0.24 * (1.0 - humidity))
        + (0.20 * secondary_noise)
        + (0.14 * np.abs(x_norm))
        + (0.10 * np.abs(y_norm)),
        0.0,
        1.0,
    ) * raw_land.astype(np.float32)
    forest_potential = np.clip(
        (0.55 * humidity)
        + (0.23 * relief)
        + (0.10 * tropical)
        + (0.06 * temperate)
        - (0.34 * urban_pressure)
        - (0.08 * dryness),
        0.0,
        1.0,
    ) * raw_land.astype(np.float32)
    vegetation_potential = np.clip(
        0.10
        + (0.36 * humidity)
        + (0.24 * relief)
        + (0.14 * secondary_noise)
        - (0.16 * urban_pressure),
        0.0,
        1.0,
    ) * raw_land.astype(np.float32)
    soil_potential = np.clip(
        0.08
        + (0.34 * dryness)
        + (0.18 * (1.0 - humidity))
        + (0.14 * tertiary_noise),
        0.0,
        1.0,
    ) * raw_land.astype(np.float32)

    urban = raw_land & ~water & (urban_pressure > 0.62) & (forest_potential < 0.46)
    forest = raw_land & ~water & ~urban & (forest_potential > 0.56)
    vegetation = raw_land & ~water & ~urban & ~forest & (vegetation_potential >= soil_potential)
    soil = raw_land & ~water & ~urban & ~forest & ~vegetation

    land_cover = np.full(raw_land.shape, LAND_CODE_WATER, dtype=np.uint8)
    land_cover[soil] = LAND_CODE_SOIL
    land_cover[vegetation] = LAND_CODE_VEGETATION
    land_cover[forest] = LAND_CODE_FOREST
    land_cover[urban] = LAND_CODE_URBAN

    return {
        "humidity": humidity.astype(np.float32),
        "relief": relief.astype(np.float32),
        "dryness": dryness.astype(np.float32),
        "forest": forest.astype(bool),
        "vegetation": vegetation.astype(bool),
        "soil": soil.astype(bool),
        "water": water.astype(bool),
        "urban": urban.astype(bool),
        "land_cover": land_cover,
    }


def _reflectance_from_cover(
    *,
    forest: np.ndarray,
    vegetation: np.ndarray,
    soil: np.ndarray,
    water: np.ndarray,
    urban: np.ndarray,
    humidity: np.ndarray,
    relief: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    red = np.zeros(humidity.shape, dtype=np.float32)
    green = np.zeros(humidity.shape, dtype=np.float32)
    blue = np.zeros(humidity.shape, dtype=np.float32)
    nir = np.zeros(humidity.shape, dtype=np.float32)

    forest_vigor = np.clip(0.58 + (0.30 * humidity) + (0.12 * relief) + rng.normal(0.0, 0.028, humidity.shape), 0.0, 1.0)
    red[forest] = 0.026 + (0.022 * (1.0 - forest_vigor[forest]))
    green[forest] = 0.092 + (0.095 * forest_vigor[forest])
    blue[forest] = 0.022 + (0.020 * (1.0 - forest_vigor[forest]))
    nir[forest] = 0.54 + (0.26 * forest_vigor[forest])

    vegetation_vigor = np.clip(0.34 + (0.34 * humidity) + (0.16 * relief) + rng.normal(0.0, 0.032, humidity.shape), 0.0, 1.0)
    red[vegetation] = 0.060 + (0.052 * (1.0 - vegetation_vigor[vegetation]))
    green[vegetation] = 0.115 + (0.105 * vegetation_vigor[vegetation])
    blue[vegetation] = 0.040 + (0.028 * (1.0 - vegetation_vigor[vegetation]))
    nir[vegetation] = 0.34 + (0.19 * vegetation_vigor[vegetation])

    soil_moisture = np.clip(0.18 + (0.30 * humidity) + (0.10 * relief) + rng.normal(0.0, 0.03, humidity.shape), 0.0, 1.0)
    red[soil] = 0.18 + (0.10 * (1.0 - soil_moisture[soil]))
    green[soil] = 0.12 + (0.035 * soil_moisture[soil])
    blue[soil] = 0.080 + (0.026 * soil_moisture[soil])
    nir[soil] = 0.12 + (0.055 * soil_moisture[soil])

    urban_albedo = np.clip(0.28 + (0.26 * relief) + rng.normal(0.0, 0.035, humidity.shape), 0.0, 1.0)
    red[urban] = 0.18 + (0.11 * urban_albedo[urban])
    green[urban] = 0.18 + (0.09 * urban_albedo[urban])
    blue[urban] = 0.17 + (0.09 * urban_albedo[urban])
    nir[urban] = 0.14 + (0.05 * urban_albedo[urban])

    water_depth = np.clip(0.44 + (0.22 * (1.0 - relief)) + (0.16 * humidity) + rng.normal(0.0, 0.02, humidity.shape), 0.0, 1.0)
    red[water] = 0.020 + (0.016 * (1.0 - water_depth[water]))
    green[water] = 0.028 + (0.020 * (1.0 - water_depth[water]))
    blue[water] = 0.060 + (0.050 * water_depth[water])
    nir[water] = 0.008 + (0.015 * (1.0 - water_depth[water]))

    canopy_texture = ndi.gaussian_filter(rng.normal(0.0, 1.0, humidity.shape).astype(np.float32), sigma=0.55)
    mid_texture = ndi.gaussian_filter(rng.normal(0.0, 1.0, humidity.shape).astype(np.float32), sigma=1.1)

    for band in (red, green, blue, nir):
        band += rng.normal(0.0, 0.005, band.shape).astype(np.float32)
        band[forest] += 0.010 * canopy_texture[forest]
        band[vegetation] += 0.007 * mid_texture[vegetation]
        band[soil] += 0.004 * mid_texture[soil]
        np.clip(band, 0.0, 1.0, out=band)
        band[:] = ndi.gaussian_filter(band, sigma=0.42)
        np.clip(band, 0.0, 1.0, out=band)

    return red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32), nir.astype(np.float32)


def generate_scene_pair(
    geometry: dict,
    baseline_start: date,
    baseline_end: date,
    current_start: date,
    current_end: date,
    size: int = 320,
) -> tuple[SceneData, SceneData]:
    bounds = geometry_bounds(geometry)
    lat, lon = geometry_centroid(geometry)
    seed = seed_from_geometry(geometry, baseline_start, baseline_end, current_start, current_end)
    rng = np.random.default_rng(seed)

    raw_land, coastal_transition, lon_grid, lat_grid, x_norm, y_norm = _land_mask(bounds, size)
    base_noise = _smooth_noise(size, size, rng, sigma=2.1)
    secondary_noise = _smooth_noise(size, size, rng, sigma=3.1)
    tertiary_noise = _smooth_noise(size, size, rng, sigma=1.35)

    baseline_cover = _build_land_cover(
        raw_land=raw_land,
        coastal_transition=coastal_transition,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        x_norm=x_norm,
        y_norm=y_norm,
        base_noise=base_noise,
        secondary_noise=secondary_noise,
        tertiary_noise=tertiary_noise,
    )

    years_apart = max((current_end - baseline_end).days / 365.25, 0.5)
    forest_edge = baseline_cover["forest"] & ~ndi.binary_erosion(baseline_cover["forest"], iterations=5)
    deforestation_pressure = np.clip(
        0.12
        + (0.32 * baseline_cover["dryness"])
        + (0.22 * tertiary_noise)
        + (0.18 * np.where(baseline_cover["urban"], 1.0, 0.0))
        + (0.16 * forest_edge.astype(np.float32))
        + (0.12 * np.abs(x_norm)),
        0.0,
        1.0,
    )
    loss_threshold = 0.68 - (0.06 * min(years_apart, 3.0))
    loss_mask = baseline_cover["forest"] & ((deforestation_pressure + (0.22 * secondary_noise)) > loss_threshold)
    loss_mask = ndi.binary_opening(loss_mask, structure=np.ones((3, 3), dtype=bool))
    loss_mask = ndi.binary_closing(loss_mask, structure=np.ones((5, 5), dtype=bool))

    degradation_mask = baseline_cover["forest"] & ~loss_mask & ((deforestation_pressure + (0.10 * base_noise)) > 0.76)
    regrowth_mask = baseline_cover["vegetation"] & (baseline_cover["humidity"] > 0.56) & (base_noise > 0.72) & ~baseline_cover["urban"]

    current_forest = (baseline_cover["forest"] & ~loss_mask) | regrowth_mask
    loss_to_urban = loss_mask & (deforestation_pressure > 0.78)
    loss_to_soil = loss_mask & ~loss_to_urban & (baseline_cover["dryness"] > 0.48)
    loss_to_vegetation = loss_mask & ~loss_to_urban & ~loss_to_soil

    current_urban = baseline_cover["urban"] | loss_to_urban
    current_vegetation = (baseline_cover["vegetation"] & ~regrowth_mask) | loss_to_vegetation
    current_soil = baseline_cover["soil"] | loss_to_soil
    current_water = baseline_cover["water"]

    current_vegetation &= ~current_forest & ~current_urban & ~current_water
    current_soil &= ~current_forest & ~current_urban & ~current_water & ~current_vegetation
    current_urban &= ~current_forest & ~current_water

    baseline_red, baseline_green, baseline_blue, baseline_nir = _reflectance_from_cover(
        forest=baseline_cover["forest"],
        vegetation=baseline_cover["vegetation"],
        soil=baseline_cover["soil"],
        water=baseline_cover["water"],
        urban=baseline_cover["urban"],
        humidity=baseline_cover["humidity"],
        relief=baseline_cover["relief"],
        rng=np.random.default_rng(seed + 17),
    )
    baseline_ndvi = compute_ndvi(baseline_red, baseline_nir).astype(np.float32)

    current_red, current_green, current_blue, current_nir = _reflectance_from_cover(
        forest=current_forest,
        vegetation=current_vegetation,
        soil=current_soil,
        water=current_water,
        urban=current_urban,
        humidity=np.clip(baseline_cover["humidity"] - (0.08 * degradation_mask.astype(np.float32)), 0.0, 1.0),
        relief=np.clip(baseline_cover["relief"] + (0.03 * current_urban.astype(np.float32)), 0.0, 1.0),
        rng=np.random.default_rng(seed + 31),
    )

    ndvi_loss_signal = (baseline_ndvi > 0.50) & raw_land & ((deforestation_pressure + (0.24 * secondary_noise)) > 0.63)
    moderate_loss = (loss_to_vegetation | degradation_mask | ndvi_loss_signal) & ~loss_to_soil & ~loss_to_urban
    severe_loss = loss_to_soil | loss_to_urban | (ndvi_loss_signal & (baseline_cover["dryness"] > 0.52))
    current_red[moderate_loss] = np.clip((current_red[moderate_loss] * 1.12) + 0.018, 0.0, 1.0)
    current_green[moderate_loss] = np.clip(current_green[moderate_loss] * 0.92, 0.0, 1.0)
    current_nir[moderate_loss] = np.clip(current_nir[moderate_loss] * 0.76, 0.0, 1.0)
    current_red[severe_loss] = np.clip((current_red[severe_loss] * 1.24) + 0.032, 0.0, 1.0)
    current_green[severe_loss] = np.clip(current_green[severe_loss] * 0.84, 0.0, 1.0)
    current_blue[severe_loss] = np.clip(current_blue[severe_loss] * 1.04, 0.0, 1.0)
    current_nir[severe_loss] = np.clip(current_nir[severe_loss] * 0.58, 0.0, 1.0)
    regrowth_boost = regrowth_mask & ~loss_mask
    current_green[regrowth_boost] = np.clip(current_green[regrowth_boost] * 1.06, 0.0, 1.0)
    current_nir[regrowth_boost] = np.clip(current_nir[regrowth_boost] * 1.08, 0.0, 1.0)

    current_ndvi = compute_ndvi(current_red, current_nir).astype(np.float32)

    baseline_land_cover = np.full(raw_land.shape, LAND_CODE_WATER, dtype=np.uint8)
    baseline_land_cover[baseline_cover["soil"]] = LAND_CODE_SOIL
    baseline_land_cover[baseline_cover["vegetation"]] = LAND_CODE_VEGETATION
    baseline_land_cover[baseline_cover["forest"]] = LAND_CODE_FOREST
    baseline_land_cover[baseline_cover["urban"]] = LAND_CODE_URBAN

    current_land_cover = np.full(raw_land.shape, LAND_CODE_WATER, dtype=np.uint8)
    current_land_cover[current_soil] = LAND_CODE_SOIL
    current_land_cover[current_vegetation] = LAND_CODE_VEGETATION
    current_land_cover[current_forest] = LAND_CODE_FOREST
    current_land_cover[current_urban] = LAND_CODE_URBAN

    baseline_mid = _date_midpoint(baseline_start, baseline_end)
    current_mid = _date_midpoint(current_start, current_end)
    sig = f"{seed:08x}"
    lat_tag = f"{lat:+06.2f}"
    lon_tag = f"{lon:+07.2f}"

    baseline = SceneData(
        ndvi=baseline_ndvi,
        red=baseline_red,
        green=baseline_green,
        blue=baseline_blue,
        nir=baseline_nir,
        rgb=np.dstack([baseline_red, baseline_green, baseline_blue]).astype(np.float32),
        acquisition_date=baseline_mid.isoformat(),
        item_id=f"S2L2A-BAS-{lat_tag}-{lon_tag}-{sig}",
        valid_mask=(baseline_land_cover != LAND_CODE_WATER),
        land_cover=baseline_land_cover,
        forest_reference=((baseline_ndvi > 0.5) & (baseline_land_cover != LAND_CODE_WATER)),
        cloud_cover_pct=0.0,
    )
    current = SceneData(
        ndvi=current_ndvi,
        red=current_red,
        green=current_green,
        blue=current_blue,
        nir=current_nir,
        rgb=np.dstack([current_red, current_green, current_blue]).astype(np.float32),
        acquisition_date=current_mid.isoformat(),
        item_id=f"S2L2A-CUR-{lat_tag}-{lon_tag}-{sig}",
        valid_mask=(current_land_cover != LAND_CODE_WATER),
        land_cover=current_land_cover,
        forest_reference=((current_ndvi > 0.5) & (current_land_cover != LAND_CODE_WATER)),
        cloud_cover_pct=0.0,
    )
    return baseline, current
