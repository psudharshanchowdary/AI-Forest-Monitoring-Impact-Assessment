"""Geometry helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

GeometryLike = dict[str, Any] | BaseGeometry


def to_geometry_dict(geometry: GeometryLike) -> dict[str, Any]:
    if isinstance(geometry, BaseGeometry):
        return mapping(geometry)
    if geometry.get("type") == "Feature":
        return geometry["geometry"]
    return geometry


def geometry_shape(geometry: GeometryLike):
    return shape(to_geometry_dict(geometry))


def geometry_bounds(geometry: GeometryLike) -> tuple[float, float, float, float]:
    geom = geometry_shape(geometry)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def geometry_centroid(geometry: GeometryLike) -> tuple[float, float]:
    geom = geometry_shape(geometry)
    centroid = geom.centroid
    return float(centroid.y), float(centroid.x)


def geometry_signature(geometry: GeometryLike) -> str:
    geom = to_geometry_dict(geometry)
    packed = json.dumps(geom, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(packed.encode("utf-8"), digest_size=16).hexdigest()


def seed_from_geometry(geometry: GeometryLike, *parts: object) -> int:
    payload = "|".join([geometry_signature(geometry), *[str(item) for item in parts]])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def approximate_area_ha(geometry: GeometryLike) -> float:
    geom = geometry_shape(geometry)
    lat, _ = geometry_centroid(geometry)
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = max(20_000.0, 111_320.0 * float(np.cos(np.radians(lat))))
    area_m2 = float(geom.area) * meters_per_degree_lat * meters_per_degree_lon
    return max(area_m2 / 10_000.0, 1.0)
