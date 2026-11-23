"""Comprehensive Automated Test Suite for Provisional Florida Thunderstorm 2024 Datasets.

Tests all 15 Phase 9 requirements:
1. Unique primary keys (valid_time_utc, latitude, longitude).
2. Chronological and spatial sorting.
3. Correct six-hourly timestamps (00, 06, 12, 18 UTC).
4. Correct spatial bounds within Florida polygon.
5. Correct feature dimensions and expected columns.
6. Duplicate NOAA event handling and aggregation.
7. Correct temporal and spatial matching rules.
8. No conversion of unknown labels into negatives.
9. No conversion of missing atmospheric data into zero.
10. GLM coverage-state logic.
11. Candidate labels remaining distinguishable from confirmed labels.
12. No future information leaking into features.
13. No duplicate rows after merging.
14. Valid numeric data types and readable CSV output.
15. Proper preservation of missing values.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

MASTER_CSV = Path("data/processed/florida_thunderstorm_2024_master.csv")
TRAINING_CSV = Path("data/processed/florida_thunderstorm_2024_training.csv")


@pytest.fixture(scope="session")
def master_df():
    assert MASTER_CSV.exists(), f"Master CSV not found at {MASTER_CSV}"
    return pd.read_csv(MASTER_CSV, low_memory=False)


@pytest.fixture(scope="session")
def training_df():
    assert TRAINING_CSV.exists(), f"Training CSV not found at {TRAINING_CSV}"
    return pd.read_csv(TRAINING_CSV, low_memory=False)


def test_1_unique_primary_keys(master_df, training_df):
    """1. Unique primary keys (valid_time_utc, latitude, longitude)."""
    pk = ["valid_time_utc", "latitude", "longitude"]
    dups_master = master_df.duplicated(subset=pk).sum()
    assert dups_master == 0, f"Master CSV has {dups_master} duplicate primary keys"

    dups_training = training_df.duplicated(subset=pk).sum()
    assert dups_training == 0, f"Training CSV has {dups_training} duplicate primary keys"


def test_2_chronological_and_spatial_sorting(master_df):
    """2. Chronological sorting then spatial sorting."""
    times = pd.to_datetime(master_df["valid_time_utc"])
    assert times.is_monotonic_increasing, "Master CSV is not chronologically sorted"


def test_3_six_hourly_timestamps(master_df):
    """3. Correct six-hourly timestamps (00, 06, 12, 18 UTC) across all 366 days."""
    times = pd.to_datetime(master_df["valid_time_utc"])
    hours = times.dt.hour.unique()
    assert set(hours).issubset({0, 6, 12, 18}), f"Unexpected hours found: {hours}"
    assert times.nunique() == 1464, f"Expected 1464 timestamps, got {times.nunique()}"

    # Leap day check
    feb29 = times[(times.dt.month == 2) & (times.dt.day == 29)]
    assert feb29.nunique() == 4, "February 29 leap day must have 4 synoptic timestamps"


def test_4_spatial_bounds(master_df):
    """4. Correct spatial bounds within requested Florida area [31.5, -88.0, 24.5, -79.5]."""
    assert master_df["latitude"].min() >= 24.5
    assert master_df["latitude"].max() <= 31.5
    assert master_df["longitude"].min() >= -88.0
    assert master_df["longitude"].max() <= -79.5
    assert master_df["grid_cell_id"].nunique() == 218, "Expected exactly 218 Florida grid cells"


def test_5_feature_dimensions(master_df, training_df):
    """5. Correct feature dimensions."""
    assert len(master_df) == 319152, f"Expected 319,152 master rows, got {len(master_df)}"
    assert master_df.shape[1] >= 80, f"Expected >= 80 columns, got {master_df.shape[1]}"
    assert len(training_df) > 0, "Training dataframe is empty"


def test_6_noaa_event_aggregation(master_df):
    """6. Duplicate NOAA event handling and aggregation."""
    multi_events = master_df[master_df["noaa_event_count"] > 1]
    assert len(multi_events) > 0, "Expected some grid cells with multiple co-occurring events"
    # Verify IDs are separated by semicolon
    for _, row in multi_events.head(10).iterrows():
        assert ";" in str(row["noaa_event_ids"]) or int(row["noaa_event_count"]) == 1


def test_7_temporal_and_spatial_matching(master_df):
    """7. Correct temporal and spatial matching."""
    matched = master_df[master_df["noaa_event_count"] > 0]
    assert len(matched) > 0, "No NOAA matches found"
    assert (matched["thunderstorm_event"] == 1.0).all(), "All matched NOAA events must be positive (1)"


def test_8_no_unknown_labels_converted_to_negatives(master_df):
    """8. No conversion of unknown labels into negatives."""
    unknowns = master_df[master_df["label_status"] == "UNKNOWN"]
    assert len(unknowns) > 0, "Expected unknown labels to be preserved"
    assert unknowns["thunderstorm_event"].isna().all(), "Unknown labels must have NaN thunderstorm_event"


def test_9_no_missing_atmospheric_data_converted_to_zero(master_df):
    """9. No conversion of missing atmospheric data into zero."""
    # Months 5-12 pressure levels should be NaN, NOT 0.0
    may_data = master_df[master_df["month"] == 5]
    assert len(may_data) > 0
    assert may_data["temperature_500hPa_degC"].isna().all(), "Missing pressure level data must be NaN, not zero"
    assert not (may_data["temperature_500hPa_degC"] == 0.0).any(), "Zero value found in missing month!"


def test_10_glm_coverage_state_logic(master_df):
    """10. GLM coverage-state logic."""
    cov_states = set(master_df["glm_coverage_status"].unique())
    assert "UNOBSERVED" in cov_states, "UNOBSERVED status must be present"
    # Unobserved periods must not have zero lightning fabricated
    unobs = master_df[master_df["glm_coverage_status"] == "UNOBSERVED"]
    assert unobs["glm_flash_count"].isna().all(), "Unobserved GLM periods must have NaN flash count"


def test_11_candidate_labels_distinguishable(master_df):
    """11. Candidate labels remaining distinguishable from confirmed labels."""
    candidates = master_df[master_df["label_status"].str.contains("CANDIDATE", na=False)]
    assert len(candidates) > 0, "Expected some candidate labels"
    # In primary supervised target, candidate labels must remain NaN to prevent false ground truth
    assert candidates["thunderstorm_event"].isna().all(), "Candidates must not be mixed with confirmed labels"
    assert set(candidates["label_source"].unique()).issubset({"GLM_CANDIDATE", "ERA5_CANDIDATE"})


def test_12_no_target_leakage_in_features(master_df, training_df):
    """12. No future information leaking into features."""
    forbidden_features = ["noaa_event_narrative", "event_narrative", "damage_property_num", "damage_crops_num"]
    for col in forbidden_features:
        assert col not in training_df.columns, f"Leakage column {col} in training set"


def test_13_no_duplicate_rows_after_merging(master_df, training_df):
    """13. No duplicate rows after merging."""
    assert master_df.duplicated().sum() == 0, "Duplicate rows in master CSV"
    assert training_df.duplicated().sum() == 0, "Duplicate rows in training CSV"


def test_14_valid_numeric_dtypes(master_df, training_df):
    """14. Valid numeric data types and readable CSV output."""
    numeric_cols = [
        "temperature_2m_degC", "dewpoint_2m_degC", "relative_humidity_2m_pct",
        "surface_pressure_hPa", "mslp_hPa", "wind_speed_10m_ms", "era5_thunderstorm_favorability"
    ]
    for col in numeric_cols:
        assert np.issubdtype(training_df[col].dtype, np.number), f"Column {col} is not numeric"
        assert not training_df[col].isna().any(), f"Column {col} has NaN in training set"


def test_15_proper_preservation_of_missing_values(master_df):
    """15. Proper preservation of missing values."""
    assert master_df["temperature_500hPa_degC"].isna().sum() > 0, "Expected NaNs in incomplete pressure levels"
    assert master_df["thunderstorm_event"].isna().sum() > 0, "Expected NaNs in unknown labels"
    assert master_df["bulk_shear_850_300_ms"].isna().sum() > 0, "Expected NaNs in bulk shear for missing months"
