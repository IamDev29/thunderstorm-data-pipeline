import pytest
from pathlib import Path
import json
import pandas as pd

AUDIT_DIR = Path("experiments/advanced_tuning_2024/audit")

def test_01_all_audit_deliverables_exist():
    expected_files = [
        "audit_summary.json",
        "feature_availability_audit.csv",
        "temporal_validation_reconciliation.csv",
        "calibration_audit.csv",
        "spatial_holdout_audit.csv",
        "bootstrap_audit.json",
        "subgroup_diagnostics.csv"
    ]
    for fname in expected_files:
        assert (AUDIT_DIR / fname).exists(), f"Missing audit deliverable: {fname}"

def test_02_audit_summary_integrity():
    with open(AUDIT_DIR / "audit_summary.json", "r") as f:
        data = json.load(f)
    assert data["dataset_sha256_verified"] == True
    assert data["total_terrestrial_rows"] == 119663
    assert data["monitored_rows"] == 112509
    assert data["pre_test_rows"] == 84436
    assert data["test_rows"] == 28073
    assert data["duplicate_keys_found"] == 0
    assert data["cohort_alignment_verified"] == True

def test_03_feature_availability_table():
    df_feat = pd.read_csv(AUDIT_DIR / "feature_availability_audit.csv")
    assert len(df_feat) == 26
    expected_cols = ["feature", "source", "available_at_initialization", "leakage_risk", "evidence"]
    for col in expected_cols:
        assert col in df_feat.columns
    assert (df_feat["leakage_risk"].isin(["Zero", "Low / Controlled"])).all()

def test_04_temporal_and_calibration_reconciliation():
    df_temp = pd.read_csv(AUDIT_DIR / "temporal_validation_reconciliation.csv")
    assert len(df_temp) == 6
    assert set(df_temp["fold_idx"]) == {1, 2, 3, 4, 5, 6}

    df_cal = pd.read_csv(AUDIT_DIR / "calibration_audit.csv")
    assert len(df_cal) == 10  # 10 bins
    assert df_cal["mean_predicted_prob"].is_monotonic_increasing

def test_05_bootstrap_audit_results():
    with open(AUDIT_DIR / "bootstrap_audit.json", "r") as f:
        boot_data = json.load(f)
    assert boot_data["status"] == "VERIFIED"
    assert boot_data["bootstrap_replicates"] == 1000
    assert boot_data["statistical_significance_confirmed"] == True
