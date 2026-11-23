"""
evaluate_asos_baselines.py
==========================
Comprehensive model-readiness script to:
1. Verify audited ASOS forecast window dataset integrity.
2. Build modeling-ready table merging ASOS targets with ERA5 features at forecast init time.
3. Compute chronological train/validation/test splits.
4. Separate target definitions (TS, VCTS, LTG, NOAA severe events).
5. Benchmark simple baselines (Persistence, Climatology, CAPE threshold, Simple Logistic Regression).
6. Save modeling table and benchmark metrics to data/processed/asos_modeling/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss
)

ROOT_DIR = Path(".").resolve()
AUDIT_DIR = ROOT_DIR / "data" / "processed" / "asos_audit"
MODELING_DIR = ROOT_DIR / "data" / "processed" / "asos_modeling"
MODELING_DIR.mkdir(parents=True, exist_ok=True)

AUDITED_WINDOWS_PARQUET = AUDIT_DIR / "florida_asos_forecast_windows_audited.parquet"
FORECAST_6H_CSV = ROOT_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"


def run_model_readiness():
    print("=" * 80)
    print("STARTING ASOS MODEL-READINESS EVALUATION & BASELINES")
    print("=" * 80)

    # 1. Load audited parquet
    df_aud = pd.read_parquet(AUDITED_WINDOWS_PARQUET)
    print(f"Loaded audited windows: {len(df_aud):,} rows")

    # 2. Load ERA5 features
    feature_cols = [
        "forecast_init_time_utc", "grid_cell_id", "latitude", "longitude",
        "month", "hour_utc", "temperature_2m_degC", "dewpoint_2m_degC",
        "dewpoint_depression_degC", "surface_pressure_hPa", "mslp_hPa",
        "wind_speed_10m_ms", "wind_direction_10m_deg", "total_precipitation_mm",
        "convective_precipitation_mm", "cape_J_kg", "cin_J_kg", "cloud_cover_fraction",
        "tcwv_kg_m2", "temperature_850hPa_degC", "relative_humidity_850hPa_pct",
        "wind_speed_850hPa_ms", "temperature_500hPa_degC", "relative_humidity_500hPa_pct",
        "bulk_shear_850_300_ms", "temp_diff_850_500_degC", "era5_thunderstorm_favorability",
        "future_noaa_event_6h", "true_prior_event_6h"
    ]
    print("Loading ERA5 forecasting dataset...")
    df_fc = pd.read_csv(FORECAST_6H_CSV, usecols=feature_cols)
    print(f"Loaded ERA5 forecasting rows: {len(df_fc):,}")

    # 3. Join on init time and grid cell
    df_aud["init_str"] = df_aud["forecast_init_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_merged = pd.merge(
        df_aud, df_fc,
        left_on=["init_str", "grid_cell_id"],
        right_on=["forecast_init_time_utc", "grid_cell_id"],
        how="inner",
        suffixes=("", "_era5")
    )
    print(f"Merged station-windows with ERA5 features: {len(df_merged):,} rows")

    # Filter to active terrestrial grid cells
    df_terr = df_merged[df_merged["in_active_terrestrial_grid"] == True].copy()
    print(f"Terrestrial station-windows: {len(df_terr):,} rows")

    # 4. Target definition disaggregation
    # Target 1: Strict Thunderstorm (TS | VCTS | Genuine LTG) under Criterion C (Primary ASOS target)
    df_terr["target_asos_strict_crit_c"] = df_terr["target_crit_c"]
    
    # Target 2: Explicit in-station TS only (under Criterion C coverage)
    df_terr["target_asos_explicit_ts"] = np.where(
        df_terr["ts_count"] > 0, 1.0,
        np.where(df_terr["status_crit_c"] == "COVERED_NO_THUNDERSTORM", 0.0, np.nan)
    )

    # Target 3: Aerodrome/Vicinity TS (TS | VCTS)
    df_terr["target_asos_aerodrome_ts_vcts"] = np.where(
        (df_terr["ts_count"] > 0) | (df_terr["vcts_count"] > 0), 1.0,
        np.where(df_terr["status_crit_c"] == "COVERED_NO_THUNDERSTORM", 0.0, np.nan)
    )

    # Target 4: NOAA Severe Convective Event
    df_terr["target_noaa_severe_6h"] = df_terr["future_noaa_event_6h"]

    print("\n--- Target Distribution Summary (Terrestrial Monitored Windows) ---")
    for t_name in ["target_asos_strict_crit_c", "target_asos_explicit_ts", "target_asos_aerodrome_ts_vcts", "target_noaa_severe_6h"]:
        s = df_terr[t_name]
        n_pos = (s == 1.0).sum()
        n_neg = (s == 0.0).sum()
        n_nan = s.isna().sum()
        prev = n_pos / (n_pos + n_neg) * 100 if (n_pos + n_neg) > 0 else 0
        print(f"{t_name:32s}: Pos={n_pos:5,d} ({prev:5.2f}%), Neg={n_neg:6,d}, Incomplete/NaN={n_nan:5,d}")

    # 5. Chronological Train / Val / Test Split
    # Filter to monitored rows for baseline evaluation
    df_eval = df_terr[df_terr["target_asos_strict_crit_c"].notna()].copy()
    df_eval["target"] = df_eval["target_asos_strict_crit_c"].astype(int)

    # Sort to compute prior cycle persistence predictor (t-6h)
    df_eval = df_eval.sort_values(["station_id", "forecast_init_time_utc"]).reset_index(drop=True)
    df_eval["prev_init"] = df_eval.groupby("station_id")["forecast_init_time_utc"].shift(1)
    df_eval["prev_target"] = df_eval.groupby("station_id")["target"].shift(1)
    df_eval["cycle_diff_hours"] = (df_eval["forecast_init_time_utc"] - df_eval["prev_init"]).dt.total_seconds() / 3600.0
    # Exact 6-hour persistence
    df_eval["pred_persistence"] = np.where(df_eval["cycle_diff_hours"] == 6.0, df_eval["prev_target"], 0).astype(int)

    # Date split
    train_mask = df_eval["forecast_init_time_utc"] < "2024-08-01"
    val_mask = (df_eval["forecast_init_time_utc"] >= "2024-08-01") & (df_eval["forecast_init_time_utc"] < "2024-10-01")
    test_mask = df_eval["forecast_init_time_utc"] >= "2024-10-01"

    print("\n--- Chronological Split Breakdown ---")
    print(f"Train Set (Jan 1 - Jul 31): {train_mask.sum():,} rows, Pos={df_eval.loc[train_mask, 'target'].sum():,} ({df_eval.loc[train_mask, 'target'].mean()*100:.2f}%)")
    print(f"Val Set   (Aug 1 - Sep 30): {val_mask.sum():,} rows, Pos={df_eval.loc[val_mask, 'target'].sum():,} ({df_eval.loc[val_mask, 'target'].mean()*100:.2f}%)")
    print(f"Test Set  (Oct 1 - Dec 31): {test_mask.sum():,} rows, Pos={df_eval.loc[test_mask, 'target'].sum():,} ({df_eval.loc[test_mask, 'target'].mean()*100:.2f}%)")

    # 6. Baseline Models Benchmarking on Test Set
    y_test = df_eval.loc[test_mask, "target"].values
    y_train = df_eval.loc[train_mask, "target"].values

    # Baseline 1: Climatological / Constant Prevalence
    p_clim = y_train.mean()
    y_prob_clim = np.full_like(y_test, fill_value=p_clim, dtype=float)
    prauc_clim = average_precision_score(y_test, y_prob_clim)
    brier_clim = brier_score_loss(y_test, y_prob_clim)

    # Baseline 2: Persistence
    y_pred_persist = df_eval.loc[test_mask, "pred_persistence"].values
    prec_p = precision_score(y_test, y_pred_persist, zero_division=0)
    rec_p = recall_score(y_test, y_pred_persist, zero_division=0)
    f1_p = f1_score(y_test, y_pred_persist, zero_division=0)
    brier_p = brier_score_loss(y_test, y_pred_persist)

    # Baseline 3: CAPE Threshold (CAPE >= 1000 J/kg)
    cape_test = df_eval.loc[test_mask, "cape_J_kg"].fillna(0).values
    y_pred_cape = (cape_test >= 1000.0).astype(int)
    prec_c = precision_score(y_test, y_pred_cape, zero_division=0)
    rec_c = recall_score(y_test, y_pred_cape, zero_division=0)
    f1_c = f1_score(y_test, y_pred_cape, zero_division=0)
    prauc_c = average_precision_score(y_test, cape_test)
    roc_c = roc_auc_score(y_test, cape_test)

    # Baseline 4: Simple Thermodynamic & Kinematic Logistic Regression
    lr_features = [
        "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC",
        "temperature_2m_degC", "bulk_shear_850_300_ms",
        "total_precipitation_mm", "tcwv_kg_m2"
    ]
    X_train = df_eval.loc[train_mask, lr_features].fillna(0).values
    X_test = df_eval.loc[test_mask, lr_features].fillna(0).values

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)

    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    prauc_lr = average_precision_score(y_test, y_prob_lr)
    roc_lr = roc_auc_score(y_test, y_prob_lr)
    brier_lr = brier_score_loss(y_test, y_prob_lr)

    print("\n--- Baseline Model Performance on Test Set (Oct-Dec 2024) ---")
    print(f"Test Set Class Prevalence: {y_test.mean()*100:.2f}% ({y_test.sum():,} / {len(y_test):,})")
    
    baseline_records = [
        {
            "baseline": "Climatology / Constant Prevalence",
            "precision": float(p_clim),
            "recall": 1.0,
            "f1": float(2 * p_clim / (1.0 + p_clim)),
            "pr_auc": float(prauc_clim),
            "roc_auc": 0.5000,
            "brier_score": float(brier_clim),
        },
        {
            "baseline": "Persistence (Thunderstorm at t-6h)",
            "precision": float(prec_p),
            "recall": float(rec_p),
            "f1": float(f1_p),
            "pr_auc": float(prec_p * rec_p),
            "roc_auc": float(0.5 * (rec_p + (1 - (y_pred_persist & ~y_test).sum() / (~y_test).sum()))),
            "brier_score": float(brier_p),
        },
        {
            "baseline": "CAPE Threshold (CAPE >= 1000 J/kg)",
            "precision": float(prec_c),
            "recall": float(rec_c),
            "f1": float(f1_c),
            "pr_auc": float(prauc_c),
            "roc_auc": float(roc_c),
            "brier_score": float(brier_score_loss(y_test, y_pred_cape)),
        },
        {
            "baseline": "Simple Logistic Regression (7 ERA5 Predictors)",
            "precision": float(prec_lr),
            "recall": float(rec_lr),
            "f1": float(f1_lr),
            "pr_auc": float(prauc_lr),
            "roc_auc": float(roc_lr),
            "brier_score": float(brier_lr),
        },
    ]

    for rec in baseline_records:
        print(f"\n{rec['baseline']}:")
        print(f"  Precision={rec['precision']:.4f}, Recall={rec['recall']:.4f}, F1={rec['f1']:.4f}")
        print(f"  PR-AUC={rec['pr_auc']:.4f}, ROC-AUC={rec['roc_auc']:.4f}, Brier={rec['brier_score']:.4f}")

    df_base = pd.DataFrame(baseline_records)
    df_base.to_csv(MODELING_DIR / "asos_baseline_metrics.csv", index=False)

    # 7. Save modeling-ready table
    modeling_parquet = MODELING_DIR / "florida_asos_modeling_ready_2024.parquet"
    df_terr.to_parquet(modeling_parquet, index=False, engine="pyarrow")
    print(f"\nSaved modeling-ready table to {modeling_parquet} ({len(df_terr):,} rows)")

    print("\n" + "=" * 80)
    print("MODEL READINESS AUDIT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_model_readiness()
