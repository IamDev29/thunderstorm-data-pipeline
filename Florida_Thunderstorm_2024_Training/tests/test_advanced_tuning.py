import pytest
from pathlib import Path
import pandas as pd
import json

def test_01_advanced_tuning_artifacts_exist():
    exp_dir = Path("experiments/advanced_tuning_2024")
    assert exp_dir.exists(), "experiments/advanced_tuning_2024 does not exist"
    
    expected_files = [
        "reproducibility_metadata.json",
        "expanding_window_metrics.csv",
        "date_alternating_metrics.csv",
        "spatial_holdout_metrics.csv",
        "tuning_trials_lightgbm.csv",
        "tuning_trials_xgboost.csv",
        "feature_ablation_metrics.csv",
        "calibration_metrics.csv",
        "threshold_selection.json",
        "final_lightgbm_tuned.joblib",
        "final_xgboost_tuned.joblib",
        "platt_calibrator.joblib",
        "isotonic_calibrator.joblib",
        "final_test_predictions.csv",
        "final_metrics_by_month.csv",
        "final_metrics_by_station.csv",
        "bootstrap_uncertainty.csv"
    ]
    for fname in expected_files:
        assert (exp_dir / fname).exists(), f"Missing artifact: {fname}"

def test_02_reproducibility_metadata():
    meta_path = Path("experiments/advanced_tuning_2024/reproducibility_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["dataset_sha256"] == "ebaa442727daf506af3889a4a96f1a411dc68f03d11b2efeb7abbc6851d147e4"
    assert meta["pre_test_rows"] == 84436
    assert meta["test_rows"] == 28073
    assert meta["test_positives"] == 1918

def test_03_expanding_and_ablation_metrics():
    exp_csv = Path("experiments/advanced_tuning_2024/expanding_window_metrics.csv")
    df_exp = pd.read_csv(exp_csv)
    assert len(df_exp) == 24  # 6 folds * 4 models
    assert set(df_exp["fold"].unique()) == {1, 2, 3, 4, 5, 6}
    
    ab_csv = Path("experiments/advanced_tuning_2024/feature_ablation_metrics.csv")
    df_ab = pd.read_csv(ab_csv)
    assert len(df_ab) == 24  # 6 folds * 4 configurations

def test_04_bootstrap_uncertainty_significance():
    boot_csv = Path("experiments/advanced_tuning_2024/bootstrap_uncertainty.csv")
    df_boot = pd.read_csv(boot_csv)
    assert len(df_boot) == 4
    for _, row in df_boot.iterrows():
        assert row["statistically_significant"] == True

def test_05_final_predictions_untouched_test_rows():
    preds_csv = Path("experiments/advanced_tuning_2024/final_test_predictions.csv")
    df_preds = pd.read_csv(preds_csv)
    assert len(df_preds) == 28073
    assert df_preds["actual_target"].sum() == 1918
    assert set(df_preds["month"].unique()) == {10, 11, 12}
