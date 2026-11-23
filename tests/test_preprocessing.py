"""Unit tests for NOAA Preprocessing Module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    NOAAPreprocessor,
    parse_damage_value,
    validate_coordinates,
)


def test_parse_damage_value():
    assert parse_damage_value("10.00K") == 10000.0
    assert parse_damage_value("1.5M") == 1500000.0
    assert parse_damage_value("2.5B") == 2500000000.0
    assert parse_damage_value("500") == 500.0
    assert parse_damage_value("0.00K") == 0.0
    assert parse_damage_value("0") == 0.0
    assert parse_damage_value(None) is None
    assert parse_damage_value("") is None
    assert parse_damage_value("nan") is None
    assert parse_damage_value("UNKNOWN") is None


def test_validate_coordinates():
    # Valid US Florida coordinates
    is_valid, status = validate_coordinates(28.5383, -81.3792)
    assert is_valid is True
    assert status == "VALID"

    # Null Island (0.0, 0.0)
    is_valid, status = validate_coordinates(0.0, 0.0)
    assert is_valid is False
    assert status == "NULL_ISLAND"

    # Out of bounds
    is_valid, status = validate_coordinates(120.0, -80.0)
    assert is_valid is False
    assert status == "OUT_OF_BOUNDS"

    # Missing
    is_valid, status = validate_coordinates(np.nan, -80.0)
    assert is_valid is False
    assert status == "MISSING"

    is_valid, status = validate_coordinates("invalid", -80.0)
    assert is_valid is False
    assert status == "INVALID_NUMERIC"


def test_clean_details_pipeline():
    raw_data = {
        "EVENT_ID": ["1001", "1002", "1001", "1003"],  # Note duplicate 1001
        "STATE": ["Florida ", " TEXAS", "Florida ", "LOUISIANA"],
        "EVENT_TYPE": ["Thunderstorm Wind", "Hail", "Thunderstorm Wind", "Heavy Rain"],
        "BEGIN_DATE_TIME": [
            "30-APR-24 20:33:00",
            "04/01/2024 15:00:00",
            "30-APR-24 20:33:00",
            "invalid_date",
        ],
        "DAMAGE_PROPERTY": ["15.00K", "1.2M", "15.00K", None],
        "DAMAGE_CROPS": ["0.00K", "500K", "0.00K", "200"],
        "BEGIN_LAT": [28.5, 30.1, 28.5, np.nan],
        "BEGIN_LON": [-81.5, -95.2, -81.5, np.nan],
        "MAGNITUDE": ["55.00", "1.75", "55.00", None],
        "MAGNITUDE_TYPE": ["EG", None, "EG", None],
        "INJURIES_DIRECT": ["1", "0", "1", "0"],
        "DEATHS_DIRECT": ["0", "0", "0", "0"],
    }
    df_raw = pd.DataFrame(raw_data)

    preprocessor = NOAAPreprocessor()
    clean_df = preprocessor.clean_details(df_raw)

    # Verify duplicate removal (exact duplicate 1001 removed)
    assert len(clean_df) == 3
    assert preprocessor.quality_stats["duplicate_records"] == 1

    # Verify normalized column names
    assert "event_id" in clean_df.columns
    assert "state" in clean_df.columns

    # Verify numeric damages
    assert clean_df.loc[clean_df["event_id"] == "1001", "damage_property_num"].values[0] == 15000.0
    assert clean_df.loc[clean_df["event_id"] == "1002", "damage_property_num"].values[0] == 1200000.0

    # Verify coordinate validation flags
    assert clean_df.loc[clean_df["event_id"] == "1001", "has_valid_coords"].values[0] == True
    assert clean_df.loc[clean_df["event_id"] == "1003", "has_valid_coords"].values[0] == False
    assert clean_df.loc[clean_df["event_id"] == "1003", "coord_status"].values[0] == "MISSING"

    # Verify state normalization
    assert clean_df["state"].tolist() == ["FLORIDA", "TEXAS", "LOUISIANA"]


def test_clean_details_handles_missing_schema_columns():
    # Test that older files with missing columns do not crash
    raw_data = {
        "EVENT_ID": ["2001"],
        "STATE": ["FLORIDA"],
        "EVENT_TYPE": ["Hail"],
        # Missing EPISODE_NARRATIVE, DAMAGE_CROPS, etc.
    }
    df_raw = pd.DataFrame(raw_data)
    preprocessor = NOAAPreprocessor()
    clean_df = preprocessor.clean_details(df_raw)

    assert "episode_narrative" in clean_df.columns
    assert pd.isna(clean_df["episode_narrative"].values[0])
    assert "episode_narrative" in preprocessor.quality_stats["missing_schema_columns"]
