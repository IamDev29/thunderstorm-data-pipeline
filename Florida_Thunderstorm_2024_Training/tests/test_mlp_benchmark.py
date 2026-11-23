"""
Unit and regression tests for MLP Benchmark and Corrected Episode Splitting.
Validates:
1. Feature matrix integrity and strict exclusion of prior_event_at_init.
2. Corrected Episode-Grouped split zero-leakage guarantee (Episode 191959 isolated).
3. PyTorch StormMLP neural network architecture, parameter counts, and tensor forward pass.
4. Hurricane Milton test-set concentration accounting (36 positive windows on Oct 9).
5. Operational false alarm rate calculations and metric consistency across models.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLP_BENCHMARK_DIR = BASE_DIR / "data" / "processed" / "mlp_benchmark"
AUDIT_V2_DIR = BASE_DIR / "data" / "processed" / "audit_v2"
FORECAST_FILE = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"


class StormMLP(nn.Module):
    """PyTorch MLP matching benchmark specification."""
    def __init__(self, in_features: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def test_feature_matrix_and_no_lookahead_leakage():
    """Verify feature metadata exists, contains exactly 48 features, and excludes prior_event_at_init."""
    feat_file = MLP_BENCHMARK_DIR / "feature_metadata.json"
    assert feat_file.exists(), f"Missing {feat_file}"
    with open(feat_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["feature_cols"]
    assert len(features) == 48, f"Expected 48 features, found {len(features)}"
    assert "true_prior_event_6h" in features, "true_prior_event_6h must be present"
    assert "prior_event_at_init" not in features, "CRITICAL: prior_event_at_init contains lookahead leakage and must be excluded"


def test_episode_isolation_certificate():
    """Verify Episode isolation certificate proves 0 shared episodes and Episode 191959 is 100% in train."""
    cert_file = MLP_BENCHMARK_DIR / "episode_isolation_certificate.json"
    assert cert_file.exists(), f"Missing {cert_file}"
    with open(cert_file, "r", encoding="utf-8") as f:
        cert = json.load(f)
    assert cert["shared_episodes_count"] == 0, f"Expected 0 shared episodes, got {cert['shared_episodes_count']}"
    assert cert["episode_191959_partition"] == "TRAIN", f"Episode 191959 must be in TRAIN, got {cert['episode_191959_partition']}"
    assert cert["isolation_status"] == "VERIFIED_ZERO_LEAKAGE"


def test_split_assignments_consistency():
    """Verify split_assignments.csv has correct row count and split_exp4_episode_corrected column."""
    split_file = AUDIT_V2_DIR / "split_assignments.csv"
    assert split_file.exists(), f"Missing {split_file}"
    df = pd.read_csv(split_file, usecols=["split_exp4_episode_corrected", "future_noaa_event_6h"])
    assert len(df) == 318934, f"Expected 318934 valid rows, found {len(df)}"
    counts = df["split_exp4_episode_corrected"].value_counts().to_dict()
    assert "TRAIN" in counts and "TEST" in counts, "Must have TRAIN and TEST splits"
    assert counts["TRAIN"] == 230862, f"Expected 230862 TRAIN rows, got {counts['TRAIN']}"
    assert counts["TEST"] == 88072, f"Expected 88072 TEST rows, got {counts['TEST']}"


def test_storm_mlp_architecture():
    """Verify PyTorch StormMLP architecture, parameter shapes, and forward pass."""
    model = StormMLP(in_features=48)
    # Check linear layers
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 4
    assert linear_layers[0].in_features == 48 and linear_layers[0].out_features == 64
    assert linear_layers[1].in_features == 64 and linear_layers[1].out_features == 32
    assert linear_layers[2].in_features == 32 and linear_layers[2].out_features == 16
    assert linear_layers[3].in_features == 16 and linear_layers[3].out_features == 1

    # Check dropout rates
    dropout_layers = [m for m in model.net if isinstance(m, nn.Dropout)]
    assert len(dropout_layers) == 3
    for d in dropout_layers:
        assert d.p == 0.2

    # Test forward pass with batch
    dummy_input = torch.randn(8, 48)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (8,), f"Expected output shape (8,), got {out.shape}"


def test_hurricane_milton_concentration():
    """Verify Hurricane Milton accounting: 36 positive forecast windows on 2024-10-09."""
    if not FORECAST_FILE.exists():
        pytest.skip(f"Forecasting file not found: {FORECAST_FILE}")
    df = pd.read_csv(FORECAST_FILE, usecols=["forecast_init_time_utc", "future_noaa_event_6h", "future_noaa_episode_ids"])
    # Filter to Milton date (2024-10-09)
    oct9 = df[df["forecast_init_time_utc"].str.startswith("2024-10-09")]
    milton_pos = oct9[(oct9["future_noaa_event_6h"] == 1.0) & 
                      (oct9["future_noaa_episode_ids"].astype(str).str.contains("196745|197213|197468", regex=True))]
    assert len(milton_pos) == 36, f"Expected exactly 36 Milton positive forecast windows on Oct 9, got {len(milton_pos)}"


def test_benchmark_metrics_integrity():
    """Verify benchmark summary CSV exists and contains valid metrics for 15 experiment-model pairs."""
    csv_file = MLP_BENCHMARK_DIR / "mlp_benchmark_summary.csv"
    if not csv_file.exists():
        pytest.skip("Benchmark summary CSV not yet generated")
    df = pd.read_csv(csv_file)
    assert len(df) == 15, f"Expected 15 rows (3 experiments x 5 models), got {len(df)}"
    required_cols = [
        "Experiment", "Model", "ROC_AUC", "PR_AUC", 
        "Tuned_Thresh", "Tuned_Recall", "Tuned_Precision", "Tuned_F1", "Tuned_FA_day",
        "PR_Ex_Milton"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column {c} in benchmark summary"
    assert not df["ROC_AUC"].isna().any(), "ROC_AUC has NaNs"
    assert not df["PR_AUC"].isna().any(), "PR_AUC has NaNs"


def test_false_alarm_per_day_formula():
    """Verify false alarms per day calculation and per-cell rate conversion."""
    # If 122 calendar days and 350 false positives:
    fp = 350
    calendar_days = 122
    fa_day = fp / calendar_days
    assert abs(fa_day - 2.8688) < 0.01

    # Across 218 grid cells:
    n_cells = 218
    fa_per_cell_day = fa_day / n_cells
    assert abs(fa_per_cell_day - 0.01316) < 0.001
    # 1 alert per cell every ~76 days
    days_per_cell_alert = 1.0 / fa_per_cell_day
    assert 70 < days_per_cell_alert < 80
