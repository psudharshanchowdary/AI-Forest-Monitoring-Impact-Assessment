"""Database-specific SQLAlchemy types with graceful PostGIS fallback."""

from __future__ import annotations

from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

try:  # pragma: no cover - exercised when GeoAlchemy2 is installed
    from geoalchemy2 import Geometry as Geometry  # type: ignore
    from geoalchemy2.elements import WKTElement as WKTElement  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without GeoAlchemy2
    WKTElement = None

    class Geometry(TypeDecorator):
        """Fallback geometry type that stores WKT as text when GeoAlchemy2 is unavailable."""

        impl = Text
        cache_ok = True

        def __init__(self, geometry_type: str = "GEOMETRY", srid: int = 4326, spatial_index: bool = False, **_: object) -> None:
            super().__init__()
            self.geometry_type = geometry_type
            self.srid = srid
            self.spatial_index = spatial_index
