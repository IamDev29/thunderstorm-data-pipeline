"""
Automated unit tests for Phase 9:
Forecast target boundaries, event ID preservation, duplicate event prevention,
historical predictor timing, target leakage prevention, unknown-label handling,
reproducible episode grouping, and GLM coverage categories.
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import gzip
import numpy as np
import pandas as pd
import pytest

DATA_PROCESSED = Path("data/processed")
CHILD_PROCESSED = Path("Florida_Thunderstorm_2024_Training/data/processed")
FORECAST_FILE = DATA_PROCESSED / "florida_thunderstorm_2024_forecasting_6h.csv"
if not FORECAST_FILE.exists() and CHILD_PROCESSED.exists():
    FORECAST_FILE = CHILD_PROCESSED / "florida_thunderstorm_2024_forecasting_6h.csv"

NOAA_FILE = Path("Florida_Thunderstorm_2024_Training/data/processed/noaa_florida_2024.csv")
RAW_NOAA_GZ = Path("data/raw/details/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz")
if not RAW_NOAA_GZ.exists():
    RAW_NOAA_GZ = Path("Florida_Thunderstorm_2024_Training/data/raw/noaa/StormEvents_details_2024.csv.gz")

GLM_FILE = Path("Florida_Thunderstorm_2024_Training/data/processed/glm_florida_2024.csv")


def test_forecast_window_boundaries():
    """Verify that every positive forecast target strictly satisfies t_init < t_event <= t_init + 6h."""
    assert FORECAST_FILE.exists(), f"Forecasting dataset missing: {FORECAST_FILE}"
    df_fc = pd.read_csv(FORECAST_FILE, low_memory=False)
    df_noaa = pd.read_csv(NOAA_FILE)
    df_noaa["timestamp_utc"] = pd.to_datetime(df_noaa["timestamp_utc"])
    
    noaa_time_map = dict(zip(df_noaa["event_id"].astype(str), df_noaa["timestamp_utc"]))
    
    pos_fc = df_fc[df_fc["future_noaa_event_6h"] == 1].copy()
    assert len(pos_fc) > 0, "No positive forecast targets found"
    
    for _, r in pos_fc.iterrows():
        t_init = pd.to_datetime(r["forecast_init_time_utc"])
        t_window_end = t_init + timedelta(hours=6)
        eids = str(r["future_noaa_event_ids"]).split(";")
        
        for eid in eids:
            if eid in noaa_time_map:
                e_time = noaa_time_map[eid]
                # Event must strictly occur within (t_init, t_init + 6h]
                assert e_time > t_init, f"Event {eid} at {e_time} is not strictly after t_init {t_init}"
                assert e_time <= t_window_end, f"Event {eid} at {e_time} is after window end {t_window_end}"


def test_event_id_preservation_and_no_duplicate_assignments():
    """Verify that every matched convective event ID is mapped to exactly one forward forecast window."""
    df_fc = pd.read_csv(FORECAST_FILE, low_memory=False)
    pos_fc = df_fc[df_fc["future_noaa_event_6h"] == 1]
    
    assigned_event_ids = []
    for _, r in pos_fc.iterrows():
        eids = str(r["future_noaa_event_ids"]).split(";")
        assigned_event_ids.extend(eids)
        
    # Check that each event ID appears only once across forecast cells/times
    total_assigned = len(assigned_event_ids)
    unique_assigned = len(set(assigned_event_ids))
    assert total_assigned == unique_assigned, (
        f"Duplicate event ID assignments found! Total: {total_assigned}, Unique: {unique_assigned}"
    )


def test_historical_predictor_timing_and_target_leakage():
    """Verify that true_prior_event_6h is strictly backward-looking and shares zero event IDs with target."""
    df_fc = pd.read_csv(FORECAST_FILE, low_memory=False)
    assert "true_prior_event_6h" in df_fc.columns
    assert "true_prior_event_ids" in df_fc.columns
    
    # Across all 319,152 observations, check intersection of true_prior_event_ids and future_noaa_event_ids
    both_pos = df_fc[(df_fc["true_prior_event_6h"] == 1) & (df_fc["future_noaa_event_6h"] == 1)]
    
    leaked_ids_count = 0
    for _, r in both_pos.iterrows():
        prior_ids = set(str(r["true_prior_event_ids"]).split(";"))
        future_ids = set(str(r["future_noaa_event_ids"]).split(";"))
        intersection = prior_ids.intersection(future_ids)
        if intersection:
            leaked_ids_count += 1
            
    assert leaked_ids_count == 0, f"Found {leaked_ids_count} rows where true_prior shares event ID with future target!"


def test_unknown_label_and_year_end_handling():
    """Verify that truncated year-end cycle is NaN and never invented as a negative label."""
    df_fc = pd.read_csv(FORECAST_FILE, low_memory=False)
    
    year_end_rows = df_fc[df_fc["forecast_init_time_utc"] == "2024-12-31 18:00:00"]
    assert len(year_end_rows) == 218, f"Expected 218 Florida cells at year-end, found {len(year_end_rows)}"
    assert (year_end_rows["forecast_window_status"] == "TRUNCATED_YEAR_END").all()
    assert year_end_rows["future_noaa_event_6h"].isna().all(), "Year-end target must be NaN"


def test_reproducible_episode_grouping():
    """Verify official NOAA EPISODE_ID preservation and zero boundary crossing between train and test."""
    assert RAW_NOAA_GZ.exists(), f"Raw NOAA details missing: {RAW_NOAA_GZ}"
    with gzip.open(RAW_NOAA_GZ, "rt") as f:
        raw_ep_df = pd.read_csv(f, usecols=["EVENT_ID", "EPISODE_ID"])
        
    df_noaa = pd.read_csv(NOAA_FILE)
    df_noaa["timestamp_utc"] = pd.to_datetime(df_noaa["timestamp_utc"])
    conv = df_noaa[df_noaa["event_type"].isin(["Thunderstorm Wind", "Hail", "Tornado", "Lightning"])].copy()
    
    merged = pd.merge(conv, raw_ep_df, left_on="event_id", right_on="EVENT_ID", how="left")
    assert merged["EPISODE_ID"].notna().all(), "Some convective events failed to map to raw EPISODE_ID"
    
    # Check that train episodes and test episodes do not overlap
    train_eps = set(merged[merged["timestamp_utc"] < pd.Timestamp("2024-09-01")]["EPISODE_ID"])
    test_eps = set(merged[merged["timestamp_utc"] >= pd.Timestamp("2024-09-01")]["EPISODE_ID"])
    
    overlap = train_eps.intersection(test_eps)
    assert len(overlap) == 0, f"Found {len(overlap)} episodes crossing the train/test split boundary: {overlap}"


def test_glm_coverage_categories():
    """Verify that GLM coverage statuses are preserved and unmonitored observations are not zero-imputed."""
    assert GLM_FILE.exists(), f"GLM processed file missing: {GLM_FILE}"
    df_glm = pd.read_csv(GLM_FILE, low_memory=False)
    
    assert "coverage_status" in df_glm.columns
    assert "glm_flash_count" in df_glm.columns
    
    # Valid coverage statuses
    valid_statuses = {"UNOBSERVED", "INCOMPLETE_COVERAGE", "COVERED_LIGHTNING", "FULL"}
    found_statuses = set(df_glm["coverage_status"].unique())
    assert found_statuses.issubset(valid_statuses), f"Unexpected coverage status: {found_statuses - valid_statuses}"
    
    # Critical rule: UNOBSERVED must have NaN or unverified detection, never confirmed zero lightning
    unobserved = df_glm[df_glm["coverage_status"] == "UNOBSERVED"]
    assert len(unobserved) > 0
    assert unobserved["glm_lightning_detected"].isna().all() or (unobserved["glm_observation_coverage"] == 0).all()
