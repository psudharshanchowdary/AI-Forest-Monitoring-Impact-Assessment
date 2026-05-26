"""Dataset creation pipeline for satellite patch collection and YOLO segmentation export."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, box, mapping

from forest_monitor.constants import CLASS_NAMES, CLASS_TO_INDEX

WORLD_COVER_CLASS_MAP = {
    "forest": {10, 95},
    "field": {40},
    "lake": {80},
    "building": {50},
}
ROAD_TAGS = {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"]}
BUILDING_TAGS = {"building": True}


@dataclass(slots=True)
class RegionSpec:
    name: str
    polygon: Polygon
    start_date: str
    end_date: str
    patches: int


@dataclass(slots=True)
class DatasetConfig:
    output_dir: Path
    patch_size_px: int = 640
    pixel_scale_m: int = 10
    base_seed: int = 17
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass(slots=True)
class PatchSample:
    sample_id: str
    region: str
    polygon: Polygon
    image_path: Path
    label_path: Path


DEFAULT_REGIONS = [
    RegionSpec(
        name="india",
        polygon=box(77.6, 12.6, 79.1, 18.2),
        start_date="2024-01-01",
        end_date="2024-12-31",
        patches=180,
    ),
    RegionSpec(
        name="amazon",
        polygon=box(-64.5, -11.2, -60.2, -7.8),
        start_date="2024-01-01",
        end_date="2024-12-31",
        patches=180,
    ),
    RegionSpec(
        name="slovakia",
        polygon=box(18.2, 47.7, 22.6, 49.6),
        start_date="2024-01-01",
        end_date="2024-12-31",
        patches=180,
    ),
]


def initialize_earth_engine() -> None:
    import ee

    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def sentinel_composite(region_geometry, start_date: str, end_date: str):
    import ee

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda image: image.divide(10000.0).copyProperties(image, image.propertyNames()))
    )
    composite = collection.median().select(["B4", "B3", "B2", "B8"], ["red", "green", "blue", "nir"])
    return composite


def worldcover_labels(region_geometry):
    import ee

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return worldcover.clip(region_geometry)


def random_point_in_polygon(polygon: Polygon, rng: random.Random) -> Point:
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if polygon.contains(point):
            return point


def square_patch_polygon(center: Point, patch_edge_m: float) -> Polygon:
    lat = center.y
    half_edge_lat = (patch_edge_m / 2.0) / 111_320.0
    half_edge_lon = (patch_edge_m / 2.0) / max(20_000.0, 111_320.0 * np.cos(np.radians(lat)))
    return box(center.x - half_edge_lon, center.y - half_edge_lat, center.x + half_edge_lon, center.y + half_edge_lat)


def sample_patch_polygons(region: RegionSpec, patch_edge_m: float, seed: int) -> list[Polygon]:
    rng = random.Random(seed)
    patches: list[Polygon] = []
    for _ in range(region.patches):
        center = random_point_in_polygon(region.polygon, rng)
        patches.append(square_patch_polygon(center, patch_edge_m=patch_edge_m))
    return patches


def fetch_sentinel_patch(polygon: Polygon, start_date: str, end_date: str, patch_size_px: int) -> np.ndarray | None:
    import ee
    import geemap

    region_geometry = ee.Geometry(mapping(polygon))
    image = sentinel_composite(region_geometry, start_date=start_date, end_date=end_date)
    array = geemap.ee_to_numpy(image, region=region_geometry, scale=10, bands=["red", "green", "blue", "nir"])
    if array is None:
        return None
    array = np.nan_to_num(array, nan=0.0).astype(np.float32)
    if array.shape[0] == 0 or array.shape[1] == 0:
        return None
    return cv2.resize(array, (patch_size_px, patch_size_px), interpolation=cv2.INTER_LINEAR)


def fetch_worldcover_patch(polygon: Polygon, patch_size_px: int) -> np.ndarray | None:
    import ee
    import geemap

    region_geometry = ee.Geometry(mapping(polygon))
    labels = worldcover_labels(region_geometry)
    array = geemap.ee_to_numpy(labels, region=region_geometry, scale=10, bands=["Map"])
    if array is None:
        return None
    if array.ndim == 3:
        array = array[..., 0]
    array = np.nan_to_num(array, nan=0).astype(np.uint8)
    if array.shape[0] == 0 or array.shape[1] == 0:
        return None
    return cv2.resize(array, (patch_size_px, patch_size_px), interpolation=cv2.INTER_NEAREST)


def fetch_osm_layers(polygon: Polygon) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    patch_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    roads = ox.features_from_polygon(polygon, tags=ROAD_TAGS)
    buildings = ox.features_from_polygon(polygon, tags=BUILDING_TAGS)
    if not roads.empty:
        roads = roads.to_crs(patch_gdf.crs)
    if not buildings.empty:
        buildings = buildings.to_crs(patch_gdf.crs)
    return roads, buildings


def rasterize_osm_layers(
    polygon: Polygon,
    image_shape: tuple[int, int],
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    transform = from_bounds(*polygon.bounds, width=width, height=height)

    road_shapes: list[tuple[object, int]] = []
    if not roads.empty:
        for geom in roads.geometry:
            if geom is None or geom.is_empty:
                continue
            road_shapes.append((geom.buffer(0.00012), 1))

    building_shapes: list[tuple[object, int]] = []
    if not buildings.empty:
        for geom in buildings.geometry:
            if geom is None or geom.is_empty:
                continue
            building_shapes.append((geom, 1))

    road_mask = rasterio.features.rasterize(
        road_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    building_mask = rasterio.features.rasterize(
        building_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return road_mask.astype(bool), building_mask.astype(bool)


def combine_label_mask(worldcover: np.ndarray, road_mask: np.ndarray, building_mask: np.ndarray) -> np.ndarray:
    label_mask = np.full(worldcover.shape, 255, dtype=np.uint8)
    forest_mask = np.isin(worldcover, list(WORLD_COVER_CLASS_MAP["forest"]))
    field_mask = np.isin(worldcover, list(WORLD_COVER_CLASS_MAP["field"]))
    water_mask = np.isin(worldcover, list(WORLD_COVER_CLASS_MAP["lake"]))
    built_mask = np.isin(worldcover, list(WORLD_COVER_CLASS_MAP["building"]))

    label_mask[field_mask] = CLASS_TO_INDEX["field"]
    label_mask[forest_mask] = CLASS_TO_INDEX["forest"]
    label_mask[water_mask] = CLASS_TO_INDEX["lake"]
    label_mask[built_mask] = CLASS_TO_INDEX["building"]
    label_mask[road_mask] = CLASS_TO_INDEX["road"]
    label_mask[building_mask] = CLASS_TO_INDEX["building"]
    return label_mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    normalized = np.clip(image, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)


def ndvi_to_colormap(ndvi: np.ndarray) -> np.ndarray:
    scaled = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def apply_augmentations(image: np.ndarray, mask: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    augmented = [("base", image, mask)]
    augmented.append(("flip_h", cv2.flip(image, 1), cv2.flip(mask, 1)))
    augmented.append(("flip_v", cv2.flip(image, 0), cv2.flip(mask, 0)))
    augmented.append(("rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)))

    brighter = cv2.convertScaleAbs(image, alpha=1.08, beta=12)
    darker = cv2.convertScaleAbs(image, alpha=0.92, beta=-8)
    augmented.append(("bright", brighter, mask.copy()))
    augmented.append(("contrast", darker, mask.copy()))
    return augmented


def mask_to_yolo_segments(mask: np.ndarray, min_area: int = 40) -> list[tuple[int, list[float]]]:
    segments: list[tuple[int, list[float]]] = []
    height, width = mask.shape
    for class_name, class_index in CLASS_TO_INDEX.items():
        class_mask = (mask == class_index).astype(np.uint8)
        if class_mask.sum() < min_area:
            continue
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3 or cv2.contourArea(contour) < min_area:
                continue
            epsilon = 0.0025 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            coordinates: list[float] = []
            for point in polygon[:, 0, :]:
                coordinates.extend([float(point[0]) / width, float(point[1]) / height])
            if len(coordinates) >= 6:
                segments.append((class_index, coordinates))
    return segments


def write_yolo_label(label_path: Path, segments: Iterable[tuple[int, list[float]]]) -> None:
    lines = []
    for class_index, coordinates in segments:
        line = " ".join([str(class_index), *[f"{coord:.6f}" for coord in coordinates]])
        lines.append(line)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml(output_path: Path) -> None:
    output_path.write_text(
        "path: ../dataset\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: forest\n  1: field\n  2: lake\n  3: road\n  4: building\n",
        encoding="utf-8",
    )


def split_samples(samples: list[PatchSample], cfg: DatasetConfig) -> dict[str, list[PatchSample]]:
    rng = random.Random(cfg.base_seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * cfg.train_split)
    val_end = train_end + int(total * cfg.val_split)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def stage_sample(output_dir: Path, sample_id: str, image_rgb: np.ndarray, label_mask: np.ndarray) -> PatchSample | None:
    staging_images = output_dir / "_staging" / "images"
    staging_labels = output_dir / "_staging" / "labels"
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    image_path = staging_images / f"{sample_id}.png"
    label_path = staging_labels / f"{sample_id}.txt"
    segments = mask_to_yolo_segments(label_mask)
    if not segments:
        return None
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    write_yolo_label(label_path, segments)
    return PatchSample(sample_id=sample_id, region=sample_id.split("_")[0], polygon=Polygon(), image_path=image_path, label_path=label_path)


def move_split_samples(split_samples_map: dict[str, list[PatchSample]], output_dir: Path) -> None:
    for split_name, samples in split_samples_map.items():
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            shutil.copy2(sample.image_path, images_dir / sample.image_path.name)
            shutil.copy2(sample.label_path, labels_dir / sample.label_path.name)


def write_manifest(samples: list[PatchSample], output_dir: Path) -> None:
    manifest_path = output_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "region", "image_path", "label_path"])
        for sample in samples:
            writer.writerow([sample.sample_id, sample.region, sample.image_path.as_posix(), sample.label_path.as_posix()])


def create_dataset(
    output_dir: Path,
    patch_size_px: int = 640,
    target_patches: int = 540,
    regions: Iterable[RegionSpec] = DEFAULT_REGIONS,
) -> dict[str, int]:
    initialize_earth_engine()
    output_dir = output_dir.resolve()
    cfg = DatasetConfig(output_dir=output_dir, patch_size_px=patch_size_px)
    patch_edge_m = patch_size_px * cfg.pixel_scale_m
    created: list[PatchSample] = []
    ndvi_dir = output_dir / "ndvi_maps"
    ndvi_dir.mkdir(parents=True, exist_ok=True)

    for region_idx, region in enumerate(regions):
        patch_polygons = sample_patch_polygons(region, patch_edge_m=patch_edge_m, seed=cfg.base_seed + region_idx)
        for patch_idx, patch_polygon in enumerate(patch_polygons):
            sentinel_patch = fetch_sentinel_patch(patch_polygon, start_date=region.start_date, end_date=region.end_date, patch_size_px=patch_size_px)
            worldcover_patch = fetch_worldcover_patch(patch_polygon, patch_size_px=patch_size_px)
            if sentinel_patch is None or worldcover_patch is None:
                continue
            try:
                roads, buildings = fetch_osm_layers(patch_polygon)
            except Exception:
                roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            road_mask, building_mask = rasterize_osm_layers(
                polygon=patch_polygon,
                image_shape=worldcover_patch.shape,
                roads=roads,
                buildings=buildings,
            )
            combined_mask = combine_label_mask(worldcover_patch, road_mask=road_mask, building_mask=building_mask)
            rgb = normalize_image(sentinel_patch[..., :3])
            ndvi = compute_ndvi(red=sentinel_patch[..., 0], nir=sentinel_patch[..., 3])
            ndvi_preview = ndvi_to_colormap(ndvi)
            cv2.imwrite(str(ndvi_dir / f"{region.name}_{patch_idx:05d}.png"), cv2.cvtColor(ndvi_preview, cv2.COLOR_RGB2BGR))

            for aug_name, aug_image, aug_mask in apply_augmentations(rgb, combined_mask):
                sample_id = f"{region.name}_{patch_idx:05d}_{aug_name}"
                sample = stage_sample(output_dir=output_dir, sample_id=sample_id, image_rgb=aug_image, label_mask=aug_mask)
                if sample is not None:
                    created.append(sample)
                if len(created) >= target_patches:
                    break
            if len(created) >= target_patches:
                break
        if len(created) >= target_patches:
            break

    if len(created) < target_patches:
        raise RuntimeError(f"Only created {len(created)} samples. Increase regions or patch count to reach {target_patches}.")

    split_map = split_samples(created, cfg=cfg)
    move_split_samples(split_map, output_dir=output_dir)
    write_manifest(created, output_dir=output_dir)
    write_data_yaml(output_dir.parent / "configs" / "data.yaml")

    summary = {split_name: len(items) for split_name, items in split_map.items()}
    summary["total"] = len(created)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
