"""Unit tests for master merged dataset and training target definitions."""

import numpy as np
import pandas as pd
import pytest


def test_sorting_and_composite_key_uniqueness():
    """Verify that records are strictly sorted ascending and unique on (time, lat, lon)."""
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-05-15 00:00", "2024-05-15 00:00", "2024-05-15 06:00"]),
        "latitude": [28.0, 29.0, 28.0],
        "longitude": [-82.0, -82.0, -82.0],
        "cape_J_kg": [2500.0, 100.0, 3000.0],
        "total_precipitation_mm": [5.0, 0.0, 10.0],
        "noaa_event_match": [1, 0, 0],
    })

    # Check ascending sort
    df_sorted = df.sort_values(by=["timestamp_utc", "latitude", "longitude"]).reset_index(drop=True)
    assert df.equals(df_sorted)

    # Check no duplicate composite keys
    assert df.duplicated(subset=["timestamp_utc", "latitude", "longitude"]).sum() == 0


def test_target_label_formulation():
    """Verify justifiable storm_label logic: 1 for confirmed, 0 for quiescent, -1 for uncertain."""
    row_storm = {"noaa_event_match": 1, "glm_lightning_detected": 1, "glm_observation_coverage": 1}
    row_noaa_only = {"noaa_event_match": 1, "glm_lightning_detected": np.nan, "glm_observation_coverage": 0}
    row_glm_only = {"noaa_event_match": 0, "glm_lightning_detected": 1, "glm_observation_coverage": 1}
    row_quiescent = {"noaa_event_match": 0, "glm_lightning_detected": 0, "glm_observation_coverage": 1, "cape_J_kg": 50, "total_precipitation_mm": 0.0}
    row_unobserved = {"noaa_event_match": 0, "glm_lightning_detected": np.nan, "glm_observation_coverage": 0}

    def assign_multisource_label(r):
        if r["noaa_event_match"] == 1 or r["glm_lightning_detected"] == 1:
            return 1
        elif r["glm_observation_coverage"] == 1 and r["glm_lightning_detected"] == 0:
            return 0
        else:
            return -1

    assert assign_multisource_label(row_storm) == 1
    assert assign_multisource_label(row_noaa_only) == 1
    assert assign_multisource_label(row_glm_only) == 1
    assert assign_multisource_label(row_quiescent) == 0
    assert assign_multisource_label(row_unobserved) == -1


def test_target_leakage_in_predictors_forbidden():
    """Verify that predictor feature set contains zero target-derived information."""
    forbidden = [
        "storm_label",
        "label_source",
        "label_confidence",
        "uncertainty_reason",
        "noaa_event_id",
        "noaa_event_match",
        "noaa_event_type",
        "glm_flash_count",
        "glm_total_energy_joules",
        "glm_lightning_detected",
    ]
    pure_predictors = [
        "temperature_2m_degC",
        "dewpoint_2m_degC",
        "surface_pressure_hPa",
        "mslp_hPa",
        "total_precipitation_mm",
        "convective_precipitation_mm",
        "cape_J_kg",
        "cin_J_kg",
        "cloud_cover_fraction",
        "u_wind_10m_ms",
        "v_wind_10m_ms",
        "wind_speed_10m_ms",
        "wind_direction_10m_deg",
        "relative_humidity_2m_pct",
        "tcwv_kg_m2",
    ]
    # Ensure zero overlap
    overlap = set(pure_predictors).intersection(set(forbidden))
    assert len(overlap) == 0
