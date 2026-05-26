"""SQLAlchemy ORM models for PostgreSQL/PostGIS persistence."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .types import Geometry


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ROI(TimestampMixin, Base):
    __tablename__ = "roi"

    roi_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    roi_name: Mapped[str] = mapped_column(String(120), nullable=False)
    region_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    geometry: Mapped[Any] = mapped_column(Geometry("POLYGON", srid=4326, spatial_index=True), nullable=False)
    centroid: Mapped[Any | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True), nullable=True)
    geometry_geojson: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    area_ha: Mapped[float] = mapped_column(Float, nullable=False)

    analysis_runs: Mapped[list[AnalysisRun]] = relationship(back_populates="roi", cascade="all, delete-orphan")


class AnalysisRun(TimestampMixin, Base):
    __tablename__ = "analysis_run"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    roi_id: Mapped[str] = mapped_column(ForeignKey("roi.roi_id", ondelete="CASCADE"), nullable=False)
    baseline_start: Mapped[date] = mapped_column(Date, nullable=False)
    baseline_end: Mapped[date] = mapped_column(Date, nullable=False)
    current_start: Mapped[date] = mapped_column(Date, nullable=False)
    current_end: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    region_classification: Mapped[str | None] = mapped_column(String(120), nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    forest_loss_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss_area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)
    carbon_loss_tco2e: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_current_ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_ndvi_drop: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)

    roi: Mapped[ROI] = relationship(back_populates="analysis_runs")
    scenes: Mapped[list[SatelliteScene]] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    ndvi_result: Mapped[NDVIResult | None] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    segmentation_result: Mapped[SegmentationResult | None] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    wildfire_events: Mapped[list[WildfireEventRecord]] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    aqi_records: Mapped[list[AirQualityRecord]] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    impact_assessment: Mapped[ImpactAssessment | None] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")
    report_artifact: Mapped[ReportArtifact | None] = relationship(back_populates="analysis_run", cascade="all, delete-orphan")


class SatelliteScene(TimestampMixin, Base):
    __tablename__ = "satellite_scene"

    scene_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), nullable=False)
    phase: Mapped[str] = mapped_column(String(24), nullable=False)
    source: Mapped[str] = mapped_column(String(120), default="Sentinel-2 Level-2A", nullable=False)
    item_id: Mapped[str] = mapped_column(String(160), nullable=False)
    acquisition_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    red_band_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    nir_band_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    rgb_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_pixel_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="scenes")


class NDVIResult(TimestampMixin, Base):
    __tablename__ = "ndvi_result"

    ndvi_result_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), unique=True, nullable=False)
    baseline_mean_ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_mean_ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndvi_drop_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_ndvi_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_ndvi_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    ndvi_change_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    forest_mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    loss_mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="ndvi_result")


class SegmentationResult(TimestampMixin, Base):
    __tablename__ = "segmentation_result"

    segmentation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), unique=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    detected_stands: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mean_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_map_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    binary_mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    instance_map_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotated_rgb_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="segmentation_result")
    stands: Mapped[list[StandSummary]] = relationship(back_populates="segmentation_result", cascade="all, delete-orphan")


class StandSummary(TimestampMixin, Base):
    __tablename__ = "stand_summary"

    stand_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    segmentation_id: Mapped[str] = mapped_column(ForeignKey("segmentation_result.segmentation_id", ondelete="CASCADE"), nullable=False)
    area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    centroid: Mapped[Any | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    segmentation_result: Mapped[SegmentationResult] = relationship(back_populates="stands")


class WildfireEventRecord(TimestampMixin, Base):
    __tablename__ = "wildfire_event"

    wildfire_event_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    location: Mapped[Any | None] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True), nullable=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="wildfire_events")


class AirQualityRecord(TimestampMixin, Base):
    __tablename__ = "aqi_record"

    aqi_record_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    us_aqi: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    ozone: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="aqi_records")


class ImpactAssessment(TimestampMixin, Base):
    __tablename__ = "impact_assessment"

    impact_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), unique=True, nullable=False)
    area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    vegetation_density: Mapped[str] = mapped_column(String(24), nullable=False)
    impact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    carbon_loss_tco2e: Mapped[float | None] = mapped_column(Float, nullable=True)
    forest_reduction_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_increase_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    impact_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="impact_assessment")


class ReportArtifact(Base):
    __tablename__ = "report_artifact"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_run.run_id", ondelete="CASCADE"), unique=True, nullable=False)
    report_format: Mapped[str] = mapped_column(String(16), default="json", nullable=False)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="report_artifact")
