import pytest
from pathlib import Path
import pandas as pd
import json

def test_01_model_files_exist():
    models_dir = Path("models")
    assert models_dir.exists(), "models directory does not exist"
    expected_files = [
        "logistic_regression.joblib",
        "scaler_logistic_regression.joblib",
        "lightgbm_model.joblib",
        "lightgbm_model.txt",
        "xgboost_model.joblib",
        "xgboost_model.json",
        "model_training_config.json"
    ]
    for fname in expected_files:
        assert (models_dir / fname).exists(), f"Missing model file: {fname}"

def test_02_model_training_config_integrity():
    config_path = Path("models/model_training_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    assert "splits" in config
    assert config["splits"]["test"]["samples"] == 28073
    assert config["splits"]["test"]["positives"] == 1918
    assert len(config["features"]["feature_list"]) == 26
    assert "test_benchmark_results" in config

def test_03_test_predictions_dataset():
    pred_path = Path("data/processed/model_evaluation/test_predictions.csv")
    assert pred_path.exists(), "test_predictions.csv does not exist"
    df = pd.read_csv(pred_path)
    assert len(df) == 28073, f"Expected 28,073 rows, got {len(df)}"
    assert df["actual_target"].sum() == 1918
    expected_cols = [
        "forecast_init_time_utc", "station_id", "month", "hour_utc",
        "actual_target", "pred_persistence",
        "prob_logistic_regression", "pred_bin_logistic_regression_opt",
        "prob_lightgbm", "pred_bin_lightgbm_opt",
        "prob_xgboost", "pred_bin_xgboost_opt"
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c} in test_predictions.csv"

def test_04_monthly_and_station_metrics():
    month_path = Path("data/processed/model_evaluation/metrics_by_month.csv")
    assert month_path.exists(), "metrics_by_month.csv does not exist"
    df_m = pd.read_csv(month_path)
    assert set(df_m["month"].unique()) == {10, 11, 12}
    
    st_path = Path("data/processed/model_evaluation/metrics_by_station.csv")
    assert st_path.exists(), "metrics_by_station.csv does not exist"
    df_st = pd.read_csv(st_path)
    assert len(df_st) > 50, "Expected > 50 stations in metrics_by_station.csv"
    assert "DTS" in df_st["station_id"].values
    assert "TTS" in df_st["station_id"].values
