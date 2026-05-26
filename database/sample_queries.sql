-- 1. Latest monitoring summary per ROI
SELECT
    r.roi_name,
    ar.run_id,
    ar.created_at,
    ar.risk_level,
    ar.forest_loss_percent,
    ar.carbon_loss_tco2e,
    ar.mean_current_ndvi
FROM roi r
JOIN analysis_run ar ON ar.roi_id = r.roi_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY r.roi_id ORDER BY ar.created_at DESC) = 1;

-- 2. Top ROIs by forest loss
SELECT
    r.roi_name,
    ar.forest_loss_percent,
    ar.loss_area_ha,
    ar.carbon_loss_tco2e,
    ar.risk_level
FROM analysis_run ar
JOIN roi r ON r.roi_id = ar.roi_id
ORDER BY ar.forest_loss_percent DESC NULLS LAST
LIMIT 10;

-- 3. Baseline and current scene metadata for one run
SELECT
    ar.run_id,
    ss.phase,
    ss.item_id,
    ss.acquisition_date,
    ss.red_band_path,
    ss.nir_band_path,
    ss.rgb_path
FROM analysis_run ar
JOIN satellite_scene ss ON ss.run_id = ar.run_id
WHERE ar.run_id = :run_id
ORDER BY ss.phase;

-- 4. Forest stand summaries with centroids
SELECT
    r.roi_name,
    ssum.area_ha,
    ST_AsText(ssum.centroid) AS centroid_wkt,
    ssum.confidence
FROM stand_summary ssum
JOIN segmentation_result sr ON sr.segmentation_id = ssum.segmentation_id
JOIN analysis_run ar ON ar.run_id = sr.run_id
JOIN roi r ON r.roi_id = ar.roi_id
WHERE ar.run_id = :run_id
ORDER BY ssum.area_ha DESC;

-- 5. AQI statistics for the last 24 hours of a run
SELECT
    run_id,
    MIN(us_aqi) AS min_aqi,
    MAX(us_aqi) AS max_aqi,
    AVG(us_aqi) AS avg_aqi,
    AVG(pm25) AS avg_pm25
FROM aqi_record
WHERE run_id = :run_id
  AND observed_at >= NOW() - INTERVAL '24 hours'
GROUP BY run_id;

-- 6. Nearby wildfire events within 100 km of ROI centroid
SELECT
    r.roi_name,
    we.title,
    we.event_date,
    we.severity,
    we.distance_km,
    ST_DistanceSphere(we.location, r.centroid) / 1000.0 AS recomputed_distance_km
FROM wildfire_event we
JOIN analysis_run ar ON ar.run_id = we.run_id
JOIN roi r ON r.roi_id = ar.roi_id
WHERE ST_DWithin(we.location::geography, r.centroid::geography, 100000)
ORDER BY we.event_date DESC;

-- 7. Impact assessment joined with final report reference
SELECT
    r.roi_name,
    ia.area_ha,
    ia.impact_type,
    ia.vegetation_density,
    ia.carbon_loss_tco2e,
    ia.risk_increase_pct,
    ra.report_path
FROM impact_assessment ia
JOIN analysis_run ar ON ar.run_id = ia.run_id
JOIN roi r ON r.roi_id = ar.roi_id
LEFT JOIN report_artifact ra ON ra.run_id = ar.run_id
WHERE ar.run_id = :run_id;
