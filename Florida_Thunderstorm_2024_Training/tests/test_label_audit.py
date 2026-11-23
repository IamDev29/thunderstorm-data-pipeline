"""
Tests for Florida Thunderstorm 2024 Label Quality Audit and Baseline Datasets.
"""

from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

def test_label_audit_csv_integrity():
    audit_file = DATA_DIR / "florida_thunderstorm_2024_label_audit.csv"
    assert audit_file.exists(), f"Audit file not found at {audit_file}"
    
    df = pd.read_csv(audit_file)
    assert len(df) == 369, f"Expected 369 total positive rows in audit, got {len(df)}"
    
    expected_cols = {
        "valid_time_utc", "latitude", "longitude", "month",
        "noaa_event_ids", "noaa_event_types", "exclusion_rule",
        "pressure_levels_complete", "glm_coverage_status",
        "is_included_in_training", "exclusion_necessary"
    }
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in audit CSV: {expected_cols - set(df.columns)}"
    
    # Exactly 229 rows are excluded from 3D training (Months 05-12)
    excluded = df[~df["is_included_in_training"]]
    assert len(excluded) == 229, f"Expected 229 excluded positive rows, got {len(excluded)}"
    
    # 100% of these 229 rows must be from months 5-12 where pressure levels are incomplete
    assert (excluded["month"] >= 5).all(), "Found excluded rows in months 1-4 where pressure levels are complete"
    assert (excluded["pressure_levels_complete"] == False).all(), "Expected all excluded rows to lack complete pressure levels"
    assert (excluded["exclusion_rule"] == "EXCLUDED_INCOMPLETE_PRESSURE_LEVELS").all()
    
    # Exactly 140 rows are retained in provisional 3D training
    included = df[df["is_included_in_training"]]
    assert len(included) == 140, f"Expected 140 included positive rows, got {len(included)}"
    assert (included["month"] <= 4).all()
    assert (included["pressure_levels_complete"] == True).all()


def test_noaa_centered_dataset_integrity():
    noaa_file = DATA_DIR / "florida_thunderstorm_2024_training_noaa_centered.csv"
    assert noaa_file.exists(), f"NOAA centered training file not found at {noaa_file}"
    
    df = pd.read_csv(noaa_file, low_memory=False)
    assert len(df) == 105512, f"Expected 105,512 rows, got {len(df)}"
    
    # Verify primary key uniqueness
    keys = ["valid_time_utc", "latitude", "longitude"]
    dups = df.duplicated(subset=keys).sum()
    assert dups == 0, f"Found {dups} duplicate spatiotemporal keys in NOAA centered dataset"
    
    # Verify target variable and labels
    assert "target_noaa_event" in df.columns
    unique_labels = set(df["target_noaa_event"].unique())
    assert unique_labels == {0, 1}
    
    # Verify positive count in months 1-4 matches 140
    pos_count = (df["target_noaa_event"] == 1).sum()
    assert pos_count == 140, f"Expected 140 positive rows, got {pos_count}"
    
    # Verify months 1-4 coverage
    assert set(df["month"].unique()) == {1, 2, 3, 4}


def test_diagnostic_dataset_integrity():
    diag_file = DATA_DIR / "florida_thunderstorm_2024_training_diagnostic.csv"
    assert diag_file.exists(), f"Diagnostic training file not found at {diag_file}"
    
    df = pd.read_csv(diag_file, low_memory=False)
    assert len(df) == 4615, f"Expected 4,615 rows, got {len(df)}"
    
    keys = ["valid_time_utc", "latitude", "longitude"]
    dups = df.duplicated(subset=keys).sum()
    assert dups == 0, f"Found {dups} duplicate spatiotemporal keys in diagnostic dataset"
    
    assert "thunderstorm_event" in df.columns
    unique_labels = set(df["thunderstorm_event"].unique())
    assert unique_labels == {0, 1}
    assert (df["thunderstorm_event"] == 1).sum() == 140
    assert (df["thunderstorm_event"] == 0).sum() == 4475


def test_audit_report_sections():
    report_file = REPORTS_DIR / "thunderstorm_label_quality_audit.md"
    assert report_file.exists(), f"Report file not found at {report_file}"
    
    text = report_file.read_text(encoding="utf-8")
    assert "Comprehensive Florida Thunderstorm 2024 Label Quality, NOAA Timezone Correction & Baseline Experiment Report" in text
    assert "Executive Summary: The Timezone Correction & Findings" in text
    assert "Verification of Recomputed Dataset Counts" in text
    assert "Detailed Event-Level Matching Audit" in text
    assert "Feature Leakage & Circularity Audit" in text
    assert "Machine-Learning Baseline Results (Chronological Splits)" in text
    assert "Key Scientific Conclusions" in text
