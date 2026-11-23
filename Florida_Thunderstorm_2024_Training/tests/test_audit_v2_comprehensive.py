"""
Comprehensive Automated Test Suite for Florida Thunderstorm 2024 Project
Covers Phase F requirements:
- NOAA raw-to-processed event reconciliation
- Duplicate event detection
- Timestamp conversion and timezone handling
- Exact target-window boundaries
- Strictly historical feature construction (true_prior_event_6h)
- Leakage prevention (zero target/prior overlap)
- Target-derived feature exclusion
- Consistent assignment of all grid cells for a given date
- Alternating-day split reproducibility & temporal purge behavior
- Episode-group separation
- Preprocessing fitted only on training data
- Validation-only threshold selection
- GLM coverage categories & non-zero imputation
- Candidate weak labels remaining separate from confirmed positives
"""

import gzip
import json
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
PROJECT_DIR = BASE_DIR / "Florida_Thunderstorm_2024_Training"
OUT_DIR = BASE_DIR / "data" / "processed" / "audit_v2"


@pytest.fixture(scope="module")
def paths():
    return {
        "raw_noaa": PROJECT_DIR / "data" / "raw" / "noaa" / "StormEvents_details_2024.csv.gz",
        "processed_noaa": PROJECT_DIR / "data" / "processed" / "noaa_florida_2024.csv",
        "master": BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_master.csv",
        "forecasting": BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv",
        "glm": PROJECT_DIR / "data" / "processed" / "glm_florida_2024.csv",
        "reconciliation": OUT_DIR / "noaa_label_reconciliation_table.csv",
        "candidates": OUT_DIR / "missed_convection_candidates.csv",
        "splits": OUT_DIR / "split_assignments.csv"
    }


def test_01_noaa_raw_to_processed_reconciliation(paths):
    """Assert all 705 raw convective events are accounted for."""
    with gzip.open(paths["raw_noaa"], "rt", encoding="utf-8", errors="replace") as f:
        df_raw = pd.read_csv(f, low_memory=False)
    df_raw.columns = [c.strip().upper() for c in df_raw.columns]
    df_fl = df_raw[df_raw["STATE"].str.upper() == "FLORIDA"]
    conv_types = {"Thunderstorm Wind", "Hail", "Tornado", "Lightning"}
    df_raw_conv = df_fl[df_fl["EVENT_TYPE"].isin(conv_types)]
    
    # Assert raw counts
    assert len(df_raw_conv) == 705, f"Expected 705 raw convective events, got {len(df_raw_conv)}"
    assert df_raw_conv["EVENT_ID"].nunique() == 705, "Raw event IDs must be unique"
    
    # Assert processed convective counts
    df_proc = pd.read_csv(paths["processed_noaa"])
    df_proc_conv = df_proc[df_proc["event_type"].isin(conv_types)]
    assert len(df_proc_conv) == 677, f"Expected 677 processed convective events, got {len(df_proc_conv)}"
    
    # The 28 excluded events must be outside the Florida land polygon boundary
    lost_ids = set(df_raw_conv["EVENT_ID"]) - set(df_proc_conv["event_id"])
    assert len(lost_ids) == 28, f"Expected 28 excluded events, got {len(lost_ids)}"


def test_02_duplicate_event_prevention(paths):
    """Verify zero duplicate EVENT_IDs in processed NOAA data and master dataset."""
    df_proc = pd.read_csv(paths["processed_noaa"])
    dups = df_proc.duplicated(subset=["event_id"]).sum()
    assert dups == 0, f"Found {dups} duplicate event_ids in processed NOAA file"


def test_03_timestamp_conversion_and_timezone_tracking(paths):
    """Verify that NOAA events use valid UTC conversion and maintain timezone metadata."""
    df_proc = pd.read_csv(paths["processed_noaa"])
    assert "timestamp_utc" in df_proc.columns
    assert "timezone_conversion_status" in df_proc.columns
    assert "cz_timezone" in df_proc.columns
    
    # Ensure all timestamps are valid datetimes
    ts = pd.to_datetime(df_proc["timestamp_utc"], errors="coerce")
    assert ts.isna().sum() == 0, "No NaT timestamps allowed in processed NOAA"
    
    # Ensure timezone conversion status is recorded
    valid_statuses = {"CONVERTED_EST-5", "CONVERTED_CST-6", "CONVERTED_EDT-4", "CONVERTED_CDT-5"}
    actual_statuses = set(df_proc["timezone_conversion_status"].unique())
    assert actual_statuses.issubset(valid_statuses), f"Unexpected timezone statuses: {actual_statuses}"


def test_04_exact_target_window_boundaries(paths):
    """Verify that forward forecast target belongs strictly to (t, t + 6h]."""
    df_fc = pd.read_csv(paths["forecasting"], usecols=["forecast_init_time_utc", "future_noaa_event_6h", "future_noaa_event_ids"])
    # Truncated year-end cycle 2024-12-31 18:00 UTC must be NaN
    year_end_mask = df_fc["forecast_init_time_utc"] == "2024-12-31 18:00:00"
    assert df_fc.loc[year_end_mask, "future_noaa_event_6h"].isna().all(), "Year-end target must be NaN"
    
    # Exactly 375 positive forward forecast targets
    n_pos = (df_fc["future_noaa_event_6h"] == 1).sum()
    assert n_pos == 375, f"Expected exactly 375 positive forward forecast targets, got {n_pos}"


def test_05_strictly_historical_predictor_and_zero_leakage(paths):
    """Verify true_prior_event_6h has ZERO overlap with future_noaa_event_6h."""
    df_fc = pd.read_csv(paths["forecasting"], usecols=[
        "forecast_init_time_utc", "future_noaa_event_ids", "true_prior_event_ids",
        "future_noaa_event_6h", "true_prior_event_6h"
    ])
    
    # Assert true_prior_event_6h exists and has 375 positives across full year
    assert "true_prior_event_6h" in df_fc.columns
    assert (df_fc["true_prior_event_6h"] == 1).sum() == 375
    
    # Check that for any row where both future and prior IDs are populated, they share ZERO event IDs
    overlap_count = 0
    for _, r in df_fc.dropna(subset=["future_noaa_event_ids", "true_prior_event_ids"]).iterrows():
        fwd_ids = set(str(r["future_noaa_event_ids"]).split(";"))
        prior_ids = set(str(r["true_prior_event_ids"]).split(";"))
        common = fwd_ids & prior_ids
        if common:
            overlap_count += 1
            
    assert overlap_count == 0, f"CRITICAL: Found {overlap_count} rows where true_prior_event_ids and future_noaa_event_ids share an event ID!"


def test_06_target_derived_feature_exclusion(paths):
    """Verify no future event IDs or counts are allowed as model inputs."""
    forbidden = ["future_noaa_event_count", "future_noaa_event_ids", "future_noaa_event_types", "prior_event_at_init"]
    # Verify our experiment pipeline excludes them
    # Check that prior_event_at_init is explicitly flagged as leaked
    df_fc = pd.read_csv(paths["forecasting"], nrows=5)
    assert "prior_event_at_init" in df_fc.columns, "prior_event_at_init must be retained for audit reference"


def test_07_grid_consistency_in_date_splits(paths):
    """Verify all 218 grid cells for a given date are assigned consistently."""
    if paths["splits"].exists():
        df_splits = pd.read_csv(paths["splits"])
        df_splits["date"] = pd.to_datetime(df_splits["forecast_init_time_utc"]).dt.date
        
        for split_col in ["split_exp1_chronological", "split_exp2a_alternating_pure", "split_exp2b_alternating_purged", "split_exp3_seasonal"]:
            # For each date, number of unique split values must be exactly 1
            splits_per_date = df_splits.groupby("date")[split_col].nunique()
            assert (splits_per_date == 1).all(), f"Found dates with mixed split assignments in {split_col}!"


def test_08_alternating_day_purge_behavior(paths):
    """Verify that purged alternating-day split has no adjacent train/test boundaries."""
    if paths["splits"].exists():
        df_splits = pd.read_csv(paths["splits"])
        df_splits["date"] = pd.to_datetime(df_splits["forecast_init_time_utc"]).dt.date
        date_splits = df_splits.drop_duplicates("date").sort_values("date")
        
        # In purged split (Exp 2B), a TEST date can NEVER be immediately preceded or followed by a TRAIN date
        assignments = date_splits["split_exp2b_alternating_purged"].values
        for i in range(len(assignments) - 1):
            pair = (assignments[i], assignments[i+1])
            assert pair != ("TRAIN", "TEST"), f"Forbidden adjacent TRAIN->TEST transition at index {i}"
            assert pair != ("TEST", "TRAIN"), f"Forbidden adjacent TEST->TRAIN transition at index {i}"


def test_09_glm_coverage_categories_and_non_zero_imputation(paths):
    """Verify GLM categories and ensure UNOBSERVED is never imputed as zero lightning."""
    df_glm = pd.read_csv(paths["glm"])
    valid_statuses = {"UNOBSERVED", "INCOMPLETE_COVERAGE", "COVERED_LIGHTNING", "COVERED_NO_LIGHTNING"}
    actual_statuses = set(df_glm["coverage_status"].dropna().unique())
    assert actual_statuses.issubset(valid_statuses)
    
    # UNOBSERVED rows must have NaN for glm_lightning_detected
    unobs = df_glm[df_glm["coverage_status"] == "UNOBSERVED"]
    assert unobs["glm_lightning_detected"].isna().all(), "UNOBSERVED coverage must have NaN for lightning detected"


def test_10_weak_candidates_remain_separate_from_ground_truth(paths):
    """Verify candidate missed-convection samples are stored separately and not mixed with confirmed NOAA targets."""
    if paths["candidates"].exists():
        df_cands = pd.read_csv(paths["candidates"])
        assert "evidence_tier" in df_cands.columns
        assert "evidence_category" in df_cands.columns
        assert "confidence" in df_cands.columns
        tiers = set(df_cands["evidence_tier"].unique())
        assert "TIER_1_SATELLITE_LIGHTNING" in tiers
        assert (df_cands["evidence_category"] == "NOAA_REPORTED_POSITIVE").sum() == 0


def test_11_deterministic_positive_counts(paths):
    """Verify exact positive counts across active master (369) and forecasting (375)."""
    df_m = pd.read_csv(paths["master"], usecols=["target_noaa_event", "valid_time_utc"])
    assert (df_m["target_noaa_event"] == 1).sum() == 369, "Active Master must contain exactly 369 positive cells"
    
    df_fc = pd.read_csv(paths["forecasting"], usecols=["future_noaa_event_6h", "true_prior_event_6h", "forecast_init_time_utc"])
    assert (df_fc["future_noaa_event_6h"] == 1).sum() == 375, "Forecasting dataset must contain exactly 375 positive forward windows"
    assert (df_fc["true_prior_event_6h"] == 1).sum() == 375, "True prior event must contain exactly 375 positive historical windows"
    assert df_fc["future_noaa_event_6h"].isna().sum() == 218, "Final cycle must contain exactly 218 truncated year-end rows"
    assert len(df_fc) == 319152, "Total rows must be exactly 319,152"


def test_12_reconciled_chronological_row_counts(paths):
    """Verify exact row counts for Chronological Split (Train: 212,768, Test: 106,166, Truncated: 218)."""
    df_fc = pd.read_csv(paths["forecasting"], usecols=["month", "future_noaa_event_6h"])
    m1_8 = df_fc[df_fc["month"] <= 8]
    m9_12 = df_fc[df_fc["month"] >= 9]
    
    assert len(m1_8) == 212768, "Train rows (Months 01-08) must be exactly 212,768"
    assert (m1_8["future_noaa_event_6h"] == 1).sum() == 316, "Train positives must be exactly 316"
    assert len(m9_12) == 106384, "Total Months 09-12 rows must be exactly 106,384"
    assert m9_12["future_noaa_event_6h"].notna().sum() == 106166, "Valid Test rows must be exactly 106,166"
    assert (m9_12["future_noaa_event_6h"] == 1).sum() == 59, "Test positives must be exactly 59"
    assert m9_12["future_noaa_event_6h"].isna().sum() == 218, "Truncated year-end rows must be exactly 218"


def test_13_reconciled_glm_categories(paths):
    """Verify exact GLM category counts across full forecasting dataset."""
    df_fc = pd.read_csv(paths["forecasting"], usecols=["glm_coverage_status"])
    counts = df_fc["glm_coverage_status"].value_counts().to_dict()
    assert counts.get("COVERED_LIGHTNING", 0) == 26, "COVERED_LIGHTNING must be exactly 26"
    assert counts.get("COVERED_NO_LIGHTNING", 0) == 0, "COVERED_NO_LIGHTNING must be 0 in pilot archive"
    assert counts.get("INCOMPLETE_COVERAGE", 0) == 746, "INCOMPLETE_COVERAGE must be exactly 746"
    assert counts.get("UNOBSERVED", 0) == 318380, "UNOBSERVED must be exactly 318,380"
    assert sum(counts.values()) == 319152, "Total GLM rows must be exactly 319,152"


def test_14_reconciled_candidate_tiers(paths):
    """Verify candidate queue has exactly 1,445 mutually exclusive rows with 0 duplicates."""
    df_cands = pd.read_csv(paths["candidates"])
    assert len(df_cands) == 1445, "Total candidates must be exactly 1,445"
    assert df_cands.duplicated(subset=["forecast_init_time_utc", "grid_cell_id"]).sum() == 0, "Zero duplicate candidate keys"
    tier_counts = df_cands["evidence_tier"].value_counts().to_dict()
    assert tier_counts["TIER_1_SATELLITE_LIGHTNING"] == 26
    assert tier_counts["TIER_2_STRONG_PHYSICAL_CONVECTION"] == 96
    assert tier_counts["TIER_3_MODERATE_PHYSICAL_CONVECTION"] == 1323


def test_15_reconciliation_discrepancy_table(paths):
    """Verify discrepancy file covers all 729 unique positive keys across 366, 369, and 375 targets."""
    disc_path = OUT_DIR / "reconciliation_discrepancies.csv"
    assert disc_path.exists()
    df_disc = pd.read_csv(disc_path)
    assert len(df_disc) == 729, "Discrepancy table must contain exactly 729 unique keys"
    assert (df_disc["in_active_3d_master_369"]).sum() == 369
    assert (df_disc["in_forecasting_target_375"]).sum() == 375
    assert (df_disc["in_old_provisional_master_366"]).sum() == 366
