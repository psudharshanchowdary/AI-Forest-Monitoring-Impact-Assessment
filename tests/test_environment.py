from forest_monitor.environment import carbon_density_profile, classify_aqi_level, estimate_wildfire_risk, generate_air_quality_history, generate_wildfire_events


ROI = {
    "type": "Polygon",
    "coordinates": [[[78.30, 17.55], [78.58, 17.55], [78.58, 17.29], [78.30, 17.29], [78.30, 17.55]]],
}


def test_environment_generators_return_content():
    wildfires = generate_wildfire_events(ROI, latitude=17.42, longitude=78.45)
    aqi = generate_air_quality_history(ROI, latitude=17.42, longitude=78.45)
    assert 2 <= len(wildfires) <= 4
    assert len(aqi) == 7 * 24


def test_risk_levels_are_ordered():
    low_score, low_level = estimate_wildfire_risk(forest_loss_percent=2.0, wildfire_count=0, current_aqi=42.0)
    high_score, high_level = estimate_wildfire_risk(forest_loss_percent=32.0, wildfire_count=4, current_aqi=150.0)
    assert low_score < high_score
    assert low_level == "Low"
    assert high_level in {"Medium", "High"}
    assert classify_aqi_level(45.0) == "Good"
    assert classify_aqi_level(90.0) == "Moderate"
    assert classify_aqi_level(120.0) == "Unhealthy"


def test_carbon_density_profile_varies_by_latitude():
    tropical = carbon_density_profile(10.0)
    temperate = carbon_density_profile(42.0)

    assert tropical.tco2e_per_ha > temperate.tco2e_per_ha
    assert "IPCC 2006" in tropical.disclaimer
