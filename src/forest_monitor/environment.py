"""Environmental context generators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .geometry import seed_from_geometry


@dataclass(slots=True)
class WildfireEvent:
    title: str
    latitude: float
    longitude: float
    date: str
    source: str
    distance_km: float
    severity: str


@dataclass(slots=True)
class AirQualitySnapshot:
    timestamp: str
    us_aqi: float
    pm25: float
    pm10: float
    ozone: float
    aqi_category: str


@dataclass(slots=True)
class CarbonDensityProfile:
    biome_label: str
    tC_per_ha: float
    tco2e_per_ha: float
    disclaimer: str


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * (np.sin(dlon / 2.0) ** 2)
    return float(radius_km * 2.0 * np.arcsin(np.sqrt(a)))


def classify_aqi_level(us_aqi: float | None) -> str:
    if us_aqi is None:
        return "Moderate"
    if us_aqi < 50:
        return "Good"
    if us_aqi <= 100:
        return "Moderate"
    return "Unhealthy"


def generate_wildfire_events(
    geometry: dict,
    latitude: float,
    longitude: float,
    max_events: int = 4,
) -> list[WildfireEvent]:
    seed = seed_from_geometry(geometry, "wildfires")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, max_events + 1))
    severities = ["Low", "Moderate", "High"]
    today = datetime.now(timezone.utc)
    events: list[WildfireEvent] = []

    for _ in range(count):
        lat_offset = float(rng.uniform(-2.0, 2.0))
        lon_offset = float(rng.uniform(-2.0, 2.0))
        event_lat = float(np.clip(latitude + lat_offset, -80.0, 80.0))
        event_lon = float(((longitude + lon_offset + 180.0) % 360.0) - 180.0)
        event_dt = today - timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
        severity = str(rng.choice(severities))
        events.append(
            WildfireEvent(
                title=f"{severity} wildfire hotspot",
                latitude=event_lat,
                longitude=event_lon,
                date=event_dt.date().isoformat(),
                source="NASA EONET (simulated)",
                distance_km=_haversine_km(latitude, longitude, event_lat, event_lon),
                severity=severity,
            )
        )

    events.sort(key=lambda item: item.distance_km)
    return events


def generate_air_quality_history(geometry: dict, latitude: float, longitude: float) -> list[AirQualitySnapshot]:
    seed = seed_from_geometry(geometry, "aqi")
    rng = np.random.default_rng(seed)
    location_bias = (np.sin(np.radians(latitude * 2.1)) * 8.0) + (np.cos(np.radians(longitude * 1.7)) * 6.0)
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=(7 * 24) - 1)
    history: list[AirQualitySnapshot] = []

    for idx in range(7 * 24):
        timestamp = start_time + timedelta(hours=idx)
        seasonal_wave = 14.0 * np.sin((2.0 * np.pi * idx) / 24.0) + 8.0 * np.sin((2.0 * np.pi * idx) / 72.0)
        pm25 = float(np.clip(35.0 + location_bias + seasonal_wave + rng.normal(0.0, 6.5), 18.0, 95.0))
        us_aqi = float(np.clip((pm25 * 1.55) + rng.normal(0.0, 7.5), 25.0, 180.0))
        history.append(
            AirQualitySnapshot(
                timestamp=timestamp.isoformat().replace("+00:00", "Z"),
                us_aqi=us_aqi,
                pm25=pm25,
                pm10=float(np.clip((pm25 * 1.3) + rng.normal(0.0, 5.0), 22.0, 150.0)),
                ozone=float(np.clip(40.0 + 0.6 * us_aqi + rng.normal(0.0, 4.0), 20.0, 180.0)),
                aqi_category=classify_aqi_level(us_aqi),
            )
        )
    return history


def estimate_wildfire_risk(
    forest_loss_percent: float,
    wildfire_count: int,
    current_aqi: float | None,
) -> tuple[float, str]:
    loss_factor = min(forest_loss_percent / 35.0, 1.0)
    fire_factor = min(wildfire_count / 5.0, 1.0)
    aqi_factor = min((current_aqi or 60.0) / 160.0, 1.0)
    score = (0.45 * loss_factor) + (0.30 * fire_factor) + (0.25 * aqi_factor)
    if score < 0.33:
        return score, "Low"
    if score < 0.67:
        return score, "Medium"
    return score, "High"


def carbon_density_profile(latitude: float) -> CarbonDensityProfile:
    lat_abs = abs(float(latitude))
    if lat_abs <= 23.5:
        tC_per_ha = 150.0
        biome_label = "Tropical forest average"
    elif lat_abs <= 35.0:
        tC_per_ha = 110.0
        biome_label = "Subtropical / monsoon forest average"
    elif lat_abs <= 50.0:
        tC_per_ha = 90.0
        biome_label = "Temperate forest average"
    else:
        tC_per_ha = 70.0
        biome_label = "Boreal / dry forest average"

    tco2e_per_ha = tC_per_ha * 3.667
    disclaimer = (
        f"Carbon estimate based on {biome_label.lower()} of {tC_per_ha:.0f} tC/ha "
        f"(IPCC 2006), approximately {tco2e_per_ha:.0f} tCO2e/ha."
    )
    return CarbonDensityProfile(
        biome_label=biome_label,
        tC_per_ha=tC_per_ha,
        tco2e_per_ha=tco2e_per_ha,
        disclaimer=disclaimer,
    )
