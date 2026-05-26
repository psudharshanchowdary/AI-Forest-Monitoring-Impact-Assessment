CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS roi (
    roi_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    roi_name TEXT NOT NULL,
    region_label TEXT,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    centroid GEOMETRY(POINT, 4326),
    geometry_geojson JSONB,
    area_ha DOUBLE PRECISION NOT NULL CHECK (area_ha >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_run (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    roi_id UUID NOT NULL REFERENCES roi(roi_id) ON DELETE CASCADE,
    baseline_start DATE NOT NULL,
    baseline_end DATE NOT NULL,
    current_start DATE NOT NULL,
    current_end DATE NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    region_classification TEXT,
    risk_level TEXT,
    risk_score NUMERIC(6, 4),
    forest_loss_percent NUMERIC(8, 3),
    loss_area_ha DOUBLE PRECISION,
    carbon_loss_tco2e DOUBLE PRECISION,
    mean_current_ndvi REAL,
    mean_ndvi_drop REAL,
    roi_area_ha DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS satellite_scene (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    phase TEXT NOT NULL CHECK (phase IN ('baseline', 'current')),
    source TEXT NOT NULL DEFAULT 'Sentinel-2 Level-2A',
    item_id TEXT NOT NULL,
    acquisition_date DATE,
    red_band_path TEXT,
    nir_band_path TEXT,
    rgb_path TEXT,
    valid_pixel_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ndvi_result (
    ndvi_result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL UNIQUE REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    baseline_mean_ndvi REAL,
    current_mean_ndvi REAL,
    ndvi_drop_mean REAL,
    baseline_ndvi_path TEXT,
    current_ndvi_path TEXT,
    ndvi_change_path TEXT,
    forest_mask_path TEXT,
    loss_mask_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS segmentation_result (
    segmentation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL UNIQUE REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    detected_stands INTEGER NOT NULL DEFAULT 0,
    mean_confidence REAL,
    probability_map_path TEXT,
    binary_mask_path TEXT,
    instance_map_path TEXT,
    annotated_rgb_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stand_summary (
    stand_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segmentation_id UUID NOT NULL REFERENCES segmentation_result(segmentation_id) ON DELETE CASCADE,
    area_ha DOUBLE PRECISION NOT NULL,
    centroid GEOMETRY(POINT, 4326),
    confidence REAL NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wildfire_event (
    wildfire_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    event_date DATE NOT NULL,
    location GEOMETRY(POINT, 4326),
    severity TEXT NOT NULL,
    source TEXT NOT NULL,
    distance_km DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aqi_record (
    aqi_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    observed_at TIMESTAMPTZ NOT NULL,
    us_aqi REAL,
    pm25 REAL,
    pm10 REAL,
    ozone REAL,
    category TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS impact_assessment (
    impact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL UNIQUE REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    area_ha DOUBLE PRECISION NOT NULL,
    vegetation_density TEXT NOT NULL,
    impact_type TEXT NOT NULL,
    carbon_loss_tco2e DOUBLE PRECISION,
    forest_reduction_pct DOUBLE PRECISION,
    risk_increase_pct DOUBLE PRECISION,
    impact_level TEXT,
    explanation TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS report_artifact (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL UNIQUE REFERENCES analysis_run(run_id) ON DELETE CASCADE,
    report_format TEXT NOT NULL DEFAULT 'json',
    report_path TEXT,
    payload JSONB,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roi_geometry ON roi USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_analysis_run_roi_id ON analysis_run (roi_id);
CREATE INDEX IF NOT EXISTS idx_analysis_run_created_at ON analysis_run (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_satellite_scene_run_id ON satellite_scene (run_id);
CREATE INDEX IF NOT EXISTS idx_wildfire_event_run_id ON wildfire_event (run_id);
CREATE INDEX IF NOT EXISTS idx_wildfire_event_location ON wildfire_event USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_aqi_record_run_id ON aqi_record (run_id);
CREATE INDEX IF NOT EXISTS idx_stand_summary_segmentation_id ON stand_summary (segmentation_id);
CREATE INDEX IF NOT EXISTS idx_stand_summary_centroid ON stand_summary USING GIST (centroid);
