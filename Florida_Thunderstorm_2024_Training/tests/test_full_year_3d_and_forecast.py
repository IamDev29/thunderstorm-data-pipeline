"""
Validation tests for full-year 3D dataset integrity, separated classification datasets,
and 6-hour-ahead forward forecasting dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

DATA_PROCESSED = Path("data/processed")
REPORTS_DIR = Path("reports")

def get_data_path(filename: str) -> Path:
    p1 = DATA_PROCESSED / filename
    if p1.exists():
        return p1
    p2 = Path("Florida_Thunderstorm_2024_Training") / DATA_PROCESSED / filename
    if p2.exists():
        return p2
    return p1


def test_full_year_3d_master_integrity():
    """Verify master dataset completeness, 3D coverage, and primary key uniqueness."""
    master_path = get_data_path("florida_thunderstorm_2024_master.csv")
    assert master_path.exists(), f"Master dataset not found at {master_path}"

    df = pd.read_csv(master_path, parse_dates=["valid_time_utc"], low_memory=False)

    # 1. Row count: 218 grid cells * 1,464 timestamps = 319,152
    assert len(df) == 319152, f"Expected 319,152 rows, found {len(df)}"

    # 2. Unique timestamps and coordinates
    unique_times = df["valid_time_utc"].nunique()
    assert unique_times == 1464, f"Expected 1,464 timestamps, found {unique_times}"
    assert df["valid_time_utc"].min() == pd.Timestamp("2024-01-01 00:00:00")
    assert df["valid_time_utc"].max() == pd.Timestamp("2024-12-31 18:00:00")

    # 3. Primary key uniqueness
    dupes = df.duplicated(subset=["valid_time_utc", "latitude", "longitude"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate primary keys"

    # 4. 100% complete pressure levels
    assert "pressure_levels_complete" in df.columns
    assert df["pressure_levels_complete"].all(), "Not all rows have complete pressure-level coverage"

    # 5. Pressure level variables have 0 non-finite values
    pressure_vars = [
        "temperature_850hPa_degC", "temperature_500hPa_degC", "temperature_300hPa_degC",
        "relative_humidity_850hPa_pct", "relative_humidity_500hPa_pct", "relative_humidity_300hPa_pct",
        "u_wind_850hPa_ms", "u_wind_500hPa_ms", "u_wind_300hPa_ms",
        "v_wind_850hPa_ms", "v_wind_500hPa_ms", "v_wind_300hPa_ms",
        "geopotential_850hPa_m2_s2", "geopotential_500hPa_m2_s2", "geopotential_300hPa_m2_s2",
        "vertical_velocity_850hPa_Pa_s", "vertical_velocity_500hPa_Pa_s", "vertical_velocity_300hPa_Pa_s",
        "bulk_shear_850_300_ms", "temp_diff_850_500_degC"
    ]
    for v in pressure_vars:
        assert v in df.columns, f"Expected pressure variable {v} missing"
        assert df[v].isna().sum() == 0, f"Variable {v} contains NaNs"


def test_noaa_event_matching_and_positive_labels():
    """Verify NOAA convective event matching and positive label distribution across 2024."""
    master_path = get_data_path("florida_thunderstorm_2024_master.csv")
    df = pd.read_csv(master_path, parse_dates=["valid_time_utc"], low_memory=False)

    assert "target_noaa_event" in df.columns
    pos_rows = df[df["target_noaa_event"] == 1]

    # Total confirmed positive grid-time keys across all 12 months is 369
    assert len(pos_rows) == 369, f"Expected 369 positive rows, found {len(pos_rows)}"

    # Positives must span across all quarters (Q1-Q4)
    pos_months = pos_rows["valid_time_utc"].dt.month.unique()
    assert len(pos_months) >= 10, f"Positive events should span the full year, found {len(pos_months)} months"


def test_separated_classification_datasets():
    """Verify the 3 benchmark datasets: Dataset A (3D), Dataset B (diagnostic), Dataset C (surface)."""
    # Dataset A: NOAA Centered Full-Year 3D
    ds_a_path = get_data_path("florida_thunderstorm_2024_training_noaa_centered.csv")
    assert ds_a_path.exists(), f"Dataset A missing: {ds_a_path}"
    df_a = pd.read_csv(ds_a_path, low_memory=False)
    assert (df_a["target_noaa_event"] == 1).sum() == 369, "Dataset A must contain all 369 positive events"
    assert "bulk_shear_850_300_ms" in df_a.columns

    # Dataset B: Diagnostic
    ds_b_path = get_data_path("florida_thunderstorm_2024_training_diagnostic.csv")
    assert ds_b_path.exists(), f"Dataset B missing: {ds_b_path}"
    df_b = pd.read_csv(ds_b_path, low_memory=False)
    assert (df_b["thunderstorm_event"] == 1).sum() == 369

    # Dataset C: Surface Full-Year
    ds_c_path = get_data_path("florida_thunderstorm_2024_training_surface_full_year.csv")
    assert ds_c_path.exists(), f"Dataset C missing: {ds_c_path}"
    df_c = pd.read_csv(ds_c_path, low_memory=False)
    assert (df_c["target_noaa_event"] == 1).sum() == 369
    assert "temperature_850hPa_degC" not in df_c.columns


def test_six_hour_ahead_forecasting_dataset():
    """Verify 6-hour-ahead forecast target validity, truncation, and persistence baseline."""
    fc_path = get_data_path("florida_thunderstorm_2024_forecasting_6h.csv")
    assert fc_path.exists(), f"Forecast dataset missing: {fc_path}"
    df_fc = pd.read_csv(fc_path, parse_dates=["forecast_init_time_utc"], low_memory=False)

    # 1. 319,152 rows
    assert len(df_fc) == 319152

    # 2. Check target column exists
    assert "future_noaa_event_6h" in df_fc.columns

    # 3. Truncated year-end handling
    year_end_rows = df_fc[df_fc["forecast_init_time_utc"] == pd.Timestamp("2024-12-31 18:00:00")]
    assert len(year_end_rows) == 218
    assert year_end_rows["future_noaa_event_6h"].isna().all(), "Year-end target must be NaN (truncated)"

    # 4. Valid target rows (excluding 2024-12-31 18:00)
    valid_rows = df_fc[df_fc["forecast_init_time_utc"] < pd.Timestamp("2024-12-31 18:00:00")]
    assert valid_rows["future_noaa_event_6h"].isna().sum() == 0, "No NaNs in evaluated forecast target"
    pos_targets = (valid_rows["future_noaa_event_6h"] == 1).sum()
    assert pos_targets == 375, f"Expected 375 positive forecast targets, found {pos_targets}"

    # 5. Persistence predictor column
    assert "prior_event_at_init" in df_fc.columns
    # Check that persistence predictor is strictly binary or 0/1
    assert df_fc["prior_event_at_init"].isin([0, 1]).all()
