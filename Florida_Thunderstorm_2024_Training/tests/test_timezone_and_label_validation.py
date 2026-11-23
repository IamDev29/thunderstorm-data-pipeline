"""
Comprehensive validation tests for NOAA timezone conversions,
event-matching consistency, label provenance, primary key uniqueness,
and feature leakage prevention.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

DATA_PROCESSED = Path("data/processed")
REPORTS_DIR = Path("reports")

def test_eastern_and_central_timezone_conversions():
    """Verify that Eastern and Central Florida events are converted with distinct, correct offsets."""
    noaa_file = DATA_PROCESSED / "noaa_florida_2024.csv"
    assert noaa_file.exists(), f"Processed NOAA file missing: {noaa_file}"
    df = pd.read_csv(noaa_file)

    # 1. Distinguish Eastern (EST-5) and Central (CST-6) records
    assert "cz_timezone" in df.columns
    assert "begin_date_time_local" in df.columns
    assert "timestamp_utc" in df.columns
    assert "timezone_conversion_status" in df.columns

    est_events = df[df["cz_timezone"] == "EST-5"].copy()
    cst_events = df[df["cz_timezone"] == "CST-6"].copy()
    assert len(est_events) > 0, "No EST events found"
    assert len(cst_events) > 0, "No CST events found"

    # Verify that EST-5 events have UTC = local + 5h
    for _, r in est_events.head(20).iterrows():
        dt_local = pd.to_datetime(r["begin_date_time_local"], format="%d-%b-%y %H:%M:%S")
        dt_utc = pd.to_datetime(r["timestamp_utc"])
        diff_hours = (dt_utc - dt_local).total_seconds() / 3600.0
        assert np.isclose(diff_hours, 5.0), f"Event {r['event_id']} EST-5 offset is {diff_hours}h, expected 5.0h"

    # Verify that CST-6 events have UTC = local + 6h
    for _, r in cst_events.head(20).iterrows():
        dt_local = pd.to_datetime(r["begin_date_time_local"], format="%d-%b-%y %H:%M:%S")
        dt_utc = pd.to_datetime(r["timestamp_utc"])
        diff_hours = (dt_utc - dt_local).total_seconds() / 3600.0
        assert np.isclose(diff_hours, 6.0), f"Event {r['event_id']} CST-6 offset is {diff_hours}h, expected 6.0h"


def test_standard_and_daylight_time_consistency():
    """Verify standard time conventions across winter/summer and explicit EDT-4 events."""
    noaa_file = DATA_PROCESSED / "noaa_florida_2024.csv"
    df = pd.read_csv(noaa_file)

    # In summer (e.g. July/August), standard time NWS records (EST-5) maintain 5h offset per NWSI 10-1605
    summer_est = df[(df["cz_timezone"] == "EST-5") & (df["begin_date_time_local"].str.contains("JUL|AUG"))]
    assert len(summer_est) > 0
    sample_summer = summer_est.iloc[0]
    dt_local = pd.to_datetime(sample_summer["begin_date_time_local"], format="%d-%b-%y %H:%M:%S")
    dt_utc = pd.to_datetime(sample_summer["timestamp_utc"])
    assert np.isclose((dt_utc - dt_local).total_seconds() / 3600.0, 5.0)

    # Explicit EDT-4 events must have 4h offset
    edt_events = df[df["cz_timezone"] == "EDT-4"]
    if len(edt_events) > 0:
        for _, r in edt_events.iterrows():
            dt_local = pd.to_datetime(r["begin_date_time_local"], format="%d-%b-%y %H:%M:%S")
            dt_utc = pd.to_datetime(r["timestamp_utc"])
            assert np.isclose((dt_utc - dt_local).total_seconds() / 3600.0, 4.0)


def test_event_matching_audit_and_id_preservation():
    """Verify event matching audit records, distances, and NOAA event ID preservation."""
    audit_file = DATA_PROCESSED / "noaa_event_matching_audit_2024.csv"
    assert audit_file.exists(), f"Matching audit missing: {audit_file}"
    df_audit = pd.read_csv(audit_file)

    required_cols = {
        "event_id", "event_type", "original_local_timestamp", "cz_timezone",
        "converted_utc_timestamp", "assigned_era5_timestamp",
        "temporal_offset_minutes", "spatial_distance_km", "matching_status"
    }
    assert required_cols.issubset(set(df_audit.columns))

    # All matched events must have spatial distance <= 28 km and temporal offset <= 180 min
    matched = df_audit[df_audit["matching_status"] == "MATCHED"]
    assert len(matched) > 600, f"Expected >600 matched events, got {len(matched)}"
    assert (matched["spatial_distance_km"] <= 28.0).all()
    assert (matched["temporal_offset_minutes"] <= 180.0).all()

    # Check that master dataset contains valid event IDs
    master_file = DATA_PROCESSED / "florida_thunderstorm_2024_master.csv"
    master = pd.read_csv(master_file, usecols=["noaa_event_count", "noaa_event_ids"])
    pos = master[master["noaa_event_count"] > 0]
    assert len(pos) > 0
    assert pos["noaa_event_ids"].notna().all()


def test_primary_key_uniqueness_and_unknown_labels():
    """Verify primary key uniqueness and ensure unobserved labels remain unknown."""
    master_file = DATA_PROCESSED / "florida_thunderstorm_2024_master.csv"
    master = pd.read_csv(master_file, usecols=["valid_time_utc", "latitude", "longitude", "thunderstorm_event", "label_status"])

    # Primary key uniqueness
    keys = ["valid_time_utc", "latitude", "longitude"]
    dups = master.duplicated(subset=keys).sum()
    assert dups == 0, f"Found {dups} duplicate primary keys in master dataset"

    # Unknown labels remain unknown
    unknown_mask = master["label_status"] == "UNKNOWN"
    assert unknown_mask.sum() > 200000, "Expected >200,000 unknown observations"
    assert master.loc[unknown_mask, "thunderstorm_event"].isna().all(), "Unknown observations must have NaN target"


def test_feature_leakage_screening():
    """Verify that experimental datasets do not contain target-derived or circular features in predictors."""
    exp1_file = DATA_PROCESSED / "florida_thunderstorm_2024_training_noaa_centered.csv"
    exp3_file = DATA_PROCESSED / "florida_thunderstorm_2024_training_surface_full_year.csv"
    
    assert exp1_file.exists()
    assert exp3_file.exists()

    forbidden_cols = {
        "noaa_event_ids", "noaa_event_types", "noaa_max_magnitude",
        "label_source", "label_status", "label_confidence", "label_evidence",
        "glm_flash_count", "glm_coverage_status", "era5_thunderstorm_favorability"
    }

    df1 = pd.read_csv(exp1_file, nrows=10)
    assert forbidden_cols.isdisjoint(set(df1.columns)), f"Leakage detected in Exp 1: {forbidden_cols.intersection(set(df1.columns))}"

    df3 = pd.read_csv(exp3_file, nrows=10)
    assert forbidden_cols.isdisjoint(set(df3.columns)), f"Leakage detected in Exp 3: {forbidden_cols.intersection(set(df3.columns))}"
