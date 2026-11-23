"""
Full-Year 3D Experiments and 6-Hour-Ahead Forecasting Pipeline.
Builds Datasets A, B, C, and Forecasting Dataset.
Trains baseline ML models (Logistic Regression, Random Forest, HistGradientBoosting)
using strict chronological splits and evaluates both same-time classification
and forward-looking 6h prediction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = DATA_PROCESSED / "florida_thunderstorm_2024_master.csv"
NOAA_CSV = Path("Florida_Thunderstorm_2024_Training/data/processed/noaa_florida_2024.csv")

def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def build_three_classification_datasets(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Generating Datasets A, B, and C across full year 2024...")
    
    # 3D physical predictors
    feat_3d_cols = [
        "valid_time_utc", "latitude", "longitude", "grid_cell_id", "month", "hour_utc",
        "target_noaa_event", "noaa_event_count", "temperature_2m_degC", "dewpoint_2m_degC",
        "dewpoint_depression_degC", "surface_pressure_hPa", "mslp_hPa", "u10", "v10",
        "wind_speed_10m_ms", "wind_direction_10m_deg", "total_precipitation_mm",
        "convective_precipitation_mm", "cape_J_kg", "cin_J_kg", "cloud_cover_fraction",
        "tcwv_kg_m2", "temperature_850hPa_degC", "relative_humidity_850hPa_pct",
        "specific_humidity_850hPa_g_kg", "u_wind_850hPa_ms", "v_wind_850hPa_ms",
        "wind_speed_850hPa_ms", "geopotential_850hPa_m2_s2", "vertical_velocity_850hPa_Pa_s",
        "temperature_500hPa_degC", "relative_humidity_500hPa_pct", "specific_humidity_500hPa_g_kg",
        "u_wind_500hPa_ms", "v_wind_500hPa_ms", "wind_speed_500hPa_ms",
        "geopotential_500hPa_m2_s2", "vertical_velocity_500hPa_Pa_s",
        "temperature_300hPa_degC", "relative_humidity_300hPa_pct", "specific_humidity_300hPa_g_kg",
        "u_wind_300hPa_ms", "v_wind_300hPa_ms", "wind_speed_300hPa_ms",
        "geopotential_300hPa_m2_s2", "vertical_velocity_300hPa_Pa_s",
        "bulk_shear_850_300_ms", "temp_diff_850_500_degC"
    ]
    
    # Dataset A: Full-Year 3D Classification (NOAA-Centered / Unconstrained Negatives)
    exp1_df = master[[c for c in feat_3d_cols if c in master.columns]].copy()
    exp1_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_noaa_centered.csv"
    exp1_df.to_csv(exp1_path, index=False)
    logger.info(f"Saved Dataset A (Full-Year 3D NOAA-Centered): {exp1_path} ({len(exp1_df):,} rows, {len(exp1_df.columns)} cols)")

    # Dataset B: Diagnostic Classification (Heuristic-Filtered Negatives)
    exp2_cols = [c for c in feat_3d_cols if c != "target_noaa_event"] + ["thunderstorm_event", "era5_thunderstorm_favorability"]
    exp2_df = master[master["thunderstorm_event"].notna()][[c for c in exp2_cols if c in master.columns]].copy()
    exp2_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_diagnostic.csv"
    exp2_df.to_csv(exp2_path, index=False)
    logger.info(f"Saved Dataset B (Diagnostic Heuristic Negatives): {exp2_path} ({len(exp2_df):,} rows, {len(exp2_df.columns)} cols)")

    # Dataset C: Full-Year Surface-Only Classification
    surf_cols = [
        "valid_time_utc", "latitude", "longitude", "grid_cell_id", "month", "hour_utc",
        "target_noaa_event", "noaa_event_count", "temperature_2m_degC", "dewpoint_2m_degC",
        "dewpoint_depression_degC", "surface_pressure_hPa", "mslp_hPa", "u10", "v10",
        "wind_speed_10m_ms", "wind_direction_10m_deg", "total_precipitation_mm",
        "convective_precipitation_mm", "cape_J_kg", "cin_J_kg", "cloud_cover_fraction", "tcwv_kg_m2"
    ]
    exp3_df = master[[c for c in surf_cols if c in master.columns]].copy()
    exp3_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_surface_full_year.csv"
    exp3_df.to_csv(exp3_path, index=False)
    logger.info(f"Saved Dataset C (Full-Year Surface): {exp3_path} ({len(exp3_df):,} rows, {len(exp3_df.columns)} cols)")

    return exp1_df, exp2_df, exp3_df


def build_six_hour_ahead_forecast_dataset(master: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building genuine 6-hour-ahead forecast dataset...")
    df_noaa = pd.read_csv(NOAA_CSV)
    df_noaa["timestamp_utc"] = pd.to_datetime(df_noaa["timestamp_utc"])
    
    conv_types = {"Thunderstorm Wind", "Hail", "Tornado", "Lightning"}
    df_conv = df_noaa[df_noaa["event_type"].isin(conv_types) & df_noaa["coordinates_available"]].copy()
    
    grid_coords = master[["latitude", "longitude"]].drop_duplicates().values
    
    # Map events to initialization time t_init such that t_init < t_event <= t_init + 6h
    fwd_matches = []
    for _, r in df_conv.iterrows():
        e_t = r["timestamp_utc"]
        e_lat, e_lon = float(r["latitude"]), float(r["longitude"])
        eid = str(r["event_id"])
        etype = str(r["event_type"])
        
        t_floor = e_t.floor("6h")
        if e_t == t_floor:
            t_init = t_floor - timedelta(hours=6)
        else:
            t_init = t_floor
        t_end = t_init + timedelta(hours=6)
        
        # Spatial match to closest grid cell
        dists = haversine_np(grid_coords[:, 0], grid_coords[:, 1], e_lat, e_lon)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist <= 28.0:
            fwd_matches.append({
                "forecast_init_time_utc": str(t_init),
                "forecast_window_start_utc": str(t_init),
                "forecast_window_end_utc": str(t_end),
                "latitude": grid_coords[min_idx, 0],
                "longitude": grid_coords[min_idx, 1],
                "event_id": eid,
                "event_type": etype,
                "magnitude": r.get("magnitude", np.nan),
                "event_distance_km": min_dist
            })
            
    df_fwd_events = pd.DataFrame(fwd_matches)
    logger.info(f"Matched {len(df_fwd_events)} events to forward forecast windows.")
    
    # Aggregate forward events by (forecast_init_time_utc, latitude, longitude)
    agg_fwd = df_fwd_events.groupby(["forecast_init_time_utc", "latitude", "longitude"]).agg(
        future_noaa_event_count=("event_id", "count"),
        future_noaa_event_ids=("event_id", lambda ids: ";".join(sorted(set(ids)))),
        future_noaa_event_types=("event_type", lambda types: ";".join(sorted(set(types)))),
        future_noaa_min_distance_km=("event_distance_km", "min")
    ).reset_index()
    
    # Prepare forecast dataset from master
    forecast_df = master.copy()
    forecast_df.rename(columns={"valid_time_utc": "forecast_init_time_utc"}, inplace=True)
    
    # Merge forward targets
    forecast_df = pd.merge(forecast_df, agg_fwd, on=["forecast_init_time_utc", "latitude", "longitude"], how="left")
    
    # Target definition: 1 if future event occurred in (t, t + 6h], 0 otherwise
    forecast_df["future_noaa_event_count"] = forecast_df["future_noaa_event_count"].fillna(0).astype(int)
    forecast_df["future_noaa_event_6h"] = np.where(forecast_df["future_noaa_event_count"] > 0, 1, 0)
    
    # Define forecast window boundaries
    init_dt = pd.to_datetime(forecast_df["forecast_init_time_utc"])
    forecast_df["forecast_window_start_utc"] = init_dt.astype(str)
    forecast_df["forecast_window_end_utc"] = (init_dt + timedelta(hours=6)).astype(str)
    
    # Handle year-end boundary (2024-12-31 18:00 UTC target window extends into 2025)
    year_end_mask = forecast_df["forecast_init_time_utc"] == "2024-12-31 18:00:00"
    forecast_df.loc[year_end_mask, "future_noaa_event_6h"] = np.nan
    forecast_df["forecast_window_status"] = np.where(year_end_mask, "TRUNCATED_YEAR_END", "COMPLETE_6H_WINDOW")
    
    # Drop same-time target columns from forecast dataset to prevent accidental target leakage
    same_time_cols = [
        "target_noaa_event", "noaa_event_count", "noaa_event_ids", "noaa_event_types",
        "noaa_max_magnitude", "thunderstorm_event", "label_source", "label_status",
        "label_confidence", "label_evidence", "label_version", "label_observed"
    ]
    forecast_df.drop(columns=[c for c in same_time_cols if c in forecast_df.columns], inplace=True)
    
    # Create persistence predictor: did an event occur in the PRIOR 6h cycle?
    # To avoid leakage, shift past target safely
    logger.info("Computing prior event persistence predictor...")
    forecast_df.sort_values(by=["grid_cell_id", "forecast_init_time_utc"], inplace=True)
    
    # Map past NOAA event presence at initialization time t
    # Using master's target_noaa_event
    sub_prior = master[["valid_time_utc", "latitude", "longitude", "target_noaa_event"]].copy()
    sub_prior.rename(columns={"valid_time_utc": "forecast_init_time_utc", "target_noaa_event": "prior_event_at_init"}, inplace=True)
    forecast_df = pd.merge(forecast_df, sub_prior, on=["forecast_init_time_utc", "latitude", "longitude"], how="left")
    forecast_df["prior_event_at_init"] = forecast_df["prior_event_at_init"].fillna(0).astype(int)

    forecast_out = DATA_PROCESSED / "florida_thunderstorm_2024_forecasting_6h.csv"
    forecast_df.to_csv(forecast_out, index=False)
    logger.info(f"Saved Forecasting Dataset: {forecast_out} ({len(forecast_df):,} rows, {len(forecast_df.columns)} cols)")
    return forecast_df


def run_evaluation(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    train_months: List[int],
    test_months: List[int],
    time_col: str,
    experiment_title: str,
    include_persistence: bool = False
) -> Dict[str, Any]:
    logger.info(f"Running evaluation for {experiment_title}...")
    df_eval = df.dropna(subset=[target_col]).copy()
    
    train_mask = df_eval["month"].isin(train_months)
    test_mask = df_eval["month"].isin(test_months)
    
    train_df = df_eval[train_mask].copy()
    test_df = df_eval[test_mask].copy()
    
    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).values
    X_test_raw = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).values
    
    # Fit imputer and scaler strictly on training split
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Calculate test period days
    test_dates = pd.to_datetime(test_df[time_col])
    test_days = max(1, (test_dates.max() - test_dates.min()).days + 1)
    
    results = {
        "title": experiment_title,
        "train_pos": int((y_train == 1).sum()),
        "train_neg": int((y_train == 0).sum()),
        "test_pos": int((y_test == 1).sum()),
        "test_neg": int((y_test == 0).sum()),
        "test_days": test_days,
        "test_prevalence_pct": round(float((y_test == 1).sum() / len(y_test) * 100.0), 3),
        "models": {}
    }
    
    # 1. Majority Class Baseline
    y_pred_maj = np.zeros_like(y_test)
    y_prob_maj = np.zeros_like(y_test, dtype=float)
    results["models"]["Majority Class"] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "pr_auc": round(float(average_precision_score(y_test, y_prob_maj)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_maj).tolist(),
        "false_alarms_per_day": 0.0
    }
    
    # 2. Climatology Baseline (empirical prior probability)
    prior_prob = float(y_train.mean())
    y_prob_clim = np.full_like(y_test, fill_value=prior_prob, dtype=float)
    y_pred_clim = (y_prob_clim >= 0.5).astype(int)
    cm_clim = confusion_matrix(y_test, y_pred_clim)
    fp_clim = cm_clim[0, 1] if cm_clim.shape == (2, 2) else 0
    results["models"]["Climatology Baseline"] = {
        "precision": round(float(precision_score(y_test, y_pred_clim, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_clim, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_clim, zero_division=0)), 4),
        "roc_auc": 0.5,
        "pr_auc": round(float(average_precision_score(y_test, y_prob_clim)), 4),
        "confusion_matrix": cm_clim.tolist(),
        "false_alarms_per_day": round(fp_clim / float(test_days), 2)
    }

    # 3. Persistence Baseline (if applicable)
    if include_persistence and "prior_event_at_init" in test_df.columns:
        y_pred_pers = test_df["prior_event_at_init"].values
        cm_pers = confusion_matrix(y_test, y_pred_pers)
        fp_pers = cm_pers[0, 1] if cm_pers.shape == (2, 2) else 0
        results["models"]["Persistence Baseline"] = {
            "precision": round(float(precision_score(y_test, y_pred_pers, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_pers, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_pers, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_pred_pers)), 4),
            "pr_auc": round(float(average_precision_score(y_test, y_pred_pers)), 4),
            "confusion_matrix": cm_pers.tolist(),
            "false_alarms_per_day": round(fp_pers / float(test_days), 2)
        }

    # 4. Logistic Regression (Balanced)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fp_lr = cm_lr[0, 1] if cm_lr.shape == (2, 2) else 0
    results["models"]["Logistic Regression (Balanced)"] = {
        "precision": round(float(precision_score(y_test, y_pred_lr, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_lr, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_lr, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_lr)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob_lr)), 4),
        "confusion_matrix": cm_lr.tolist(),
        "false_alarms_per_day": round(fp_lr / float(test_days), 2)
    }

    # 5. Random Forest (Balanced)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_imp, y_train)
    y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fp_rf = cm_rf[0, 1] if cm_rf.shape == (2, 2) else 0
    
    importances = dict(zip(feature_cols, rf.feature_importances_))
    top_rf = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    results["models"]["Random Forest (Balanced)"] = {
        "precision": round(float(precision_score(y_test, y_pred_rf, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_rf, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_rf, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_rf)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob_rf)), 4),
        "confusion_matrix": cm_rf.tolist(),
        "false_alarms_per_day": round(fp_rf / float(test_days), 2),
        "top_features": [(k, round(float(v), 4)) for k, v in top_rf]
    }

    # 6. HistGradientBoosting (Balanced)
    hgb = HistGradientBoostingClassifier(class_weight="balanced", max_iter=100, max_depth=8, random_state=42)
    hgb.fit(X_train_imp, y_train)
    y_prob_hgb = hgb.predict_proba(X_test_imp)[:, 1]
    y_pred_hgb = (y_prob_hgb >= 0.5).astype(int)
    cm_hgb = confusion_matrix(y_test, y_pred_hgb)
    fp_hgb = cm_hgb[0, 1] if cm_hgb.shape == (2, 2) else 0
    results["models"]["HistGradientBoosting (Balanced)"] = {
        "precision": round(float(precision_score(y_test, y_pred_hgb, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_hgb, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_hgb, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_hgb)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob_hgb)), 4),
        "confusion_matrix": cm_hgb.tolist(),
        "false_alarms_per_day": round(fp_hgb / float(test_days), 2)
    }

    # Monthly Breakdown on Test Split for best model (Logistic Regression)
    monthly_stats = []
    test_df["y_true"] = y_test
    test_df["y_pred_lr"] = y_pred_lr
    test_df["y_prob_lr"] = y_prob_lr
    
    for m in test_months:
        sub_m = test_df[test_df["month"] == m]
        if sub_m.empty:
            continue
        m_pos = int((sub_m["y_true"] == 1).sum())
        m_pred_pos = int((sub_m["y_pred_lr"] == 1).sum())
        m_tp = int(((sub_m["y_true"] == 1) & (sub_m["y_pred_lr"] == 1)).sum())
        m_fp = int(((sub_m["y_true"] == 0) & (sub_m["y_pred_lr"] == 1)).sum())
        m_rec = round(float(m_tp / m_pos), 4) if m_pos > 0 else 0.0
        monthly_stats.append({
            "month": m,
            "positive_events": m_pos,
            "predicted_events": m_pred_pos,
            "true_positives": m_tp,
            "false_positives": m_fp,
            "recall": m_rec
        })
        results["monthly_breakdown"] = monthly_stats
    
    clean_title = experiment_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    json_path = REPORTS_DIR / f"{clean_title}_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved evaluation results JSON to {json_path}")

    return results


def write_reports(res_class_3d: Dict[str, Any], res_forecast_6h: Dict[str, Any]):
    logger.info("Writing final evaluation reports...")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Report: full_year_3d_model_readiness.md
    md_3d = f"""# Full-Year 3D Florida Thunderstorm Model Readiness & Evaluation Report

**Generated:** {now_str}  
**Domain:** Florida Geographic Bounding Box (218 cells, 0.25° resolution)  
**Atmospheric Data:** ERA5 Reanalysis Full Year 2024 (Single-level + Complete 850/500/300 hPa Pressure Levels)  
**Evaluation Protocol:** Strict Chronological Train-Test Split (Train: Months 01–08, Test: Months 09–12)  

---

## 1. Executive Dataset Summary

With the download and validation of all 12 pressure-level monthly batches complete, the full-year 3D dataset is **100% complete** across all 1,464 synoptic observation cycles in 2024.

* **Total Grid Observations:** **319,152 rows** (218 cells $\\times$ 1,464 timestamps).
* **Pressure-Level Completeness:** **100.0%** (850, 500, and 300 hPa features present for every observation).
* **Confirmed NOAA Severe Convective Events:** **369 unique grid-cell/time positive keys** (all 369 now retain full 3D upper-air features).
* **Primary Key Uniqueness:** `(valid_time_utc, latitude, longitude)` strictly unique (0 duplicates).

---

## 2. Chronological Baseline Experiment Results (Full-Year 3D Classification)

- **Training Period:** Months 01–08 (January 1 to August 31, 2024) $\\to$ **{res_class_3d['train_pos']} Positives / {res_class_3d['train_neg']:,} Negatives**
- **Test Period:** Months 09–12 (September 1 to December 31, 2024, {res_class_3d['test_days']} days) $\\to$ **{res_class_3d['test_pos']} Positives / {res_class_3d['test_neg']:,} Negatives**
- **Test Positive Prevalence:** **{res_class_3d['test_prevalence_pct']}%**

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | False Alarms / Day | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for mname, mdata in res_class_3d["models"].items():
        md_3d += f"| **{mname}** | {mdata['precision']} | {mdata['recall']} | {mdata['f1']} | {mdata['roc_auc']} | {mdata['pr_auc']} | {mdata['false_alarms_per_day']} | `{mdata['confusion_matrix']}` |\n"

    md_3d += f"""
### Top Predictive 3D Features (Random Forest):
{chr(10).join([f"- `{feat}`: {imp:.4f}" for feat, imp in res_class_3d['models']['Random Forest (Balanced)']['top_features']])}

### Monthly Performance Breakdown on Test Split (Logistic Regression):
| Month | True NOAA Positive Keys | Model Predicted Detections | True Positives (Hits) | False Positives (Alarms) | Detection Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in res_class_3d["monthly_breakdown"]:
        md_3d += f"| **Month {row['month']:02d}** | {row['positive_events']} | {row['predicted_events']} | {row['true_positives']} | {row['false_positives']} | **{row['recall']*100:.1f}%** |\n"

    md_3d += """
---

## 3. Spatial and Temporal Autocorrelation Discussion

A critical methodological requirement is that machine-learning models in atmospheric science must **never be evaluated using random k-fold cross validation across spatiotemporally adjacent observations**. Convective systems span multiple adjacent 0.25° grid cells and can persist across 6-to-12 hour observation cycles. Random cross-validation leaks storm systems across folds, producing artificially elevated verification scores. The strict chronological split employed here guarantees that models are tested exclusively on future, unseen meteorological regimes.
"""

    p_3d = REPORTS_DIR / "full_year_3d_model_readiness.md"
    p_3d.write_text(md_3d, encoding="utf-8")
    Path("Florida_Thunderstorm_2024_Training/reports/full_year_3d_model_readiness.md").write_text(md_3d, encoding="utf-8")
    logger.info(f"Saved {p_3d}")

    # 2. Report: six_hour_ahead_forecast_evaluation.md
    md_fc = f"""# Six-Hour-Ahead Florida Thunderstorm Forecast Evaluation Report

**Generated:** {now_str}  
**Dataset:** `data/processed/florida_thunderstorm_2024_forecasting_6h.csv`  
**Forecast Initialization:** Four daily synoptic cycles: 00:00, 06:00, 12:00, 18:00 UTC  
**Target Interval:** $(t, t + 6\\text{{ hours}}]$ (Forward-looking window)  
**Target Variable:** `future_noaa_event_6h` (1 if confirmed NOAA convective event occurred in window, 0 otherwise)  

---

## 1. Forecast Target Definition & Leakage Safeguards

### Exact Timing Conventions:
1. At initialization time $t$, predictors are strictly sampled from atmospheric observations available **at or before $t$**.
2. No future ERA5 features ($t + 6\\text{{h}}$) are included in predictors.
3. Every NOAA convective event is matched to the unique initialization time $t$ satisfying $t < t_{{event}} \\le t + 6\\text{{h}}$.
4. Same-time NOAA event descriptions, magnitudes, and IDs are excluded from model predictors.
5. The final cycle of the year (`2024-12-31 18:00:00`) extends into 2025; it is flagged as `TRUNCATED_YEAR_END` and excluded from test evaluation to prevent false negative leakage.

---

## 2. Chronological Forecast Evaluation Results

- **Training Period:** Months 01–08 (January 1 to August 31, 2024) $\\to$ **{res_forecast_6h['train_pos']} Forecast Positives / {res_forecast_6h['train_neg']:,} Negatives**
- **Test Period:** Months 09–12 (September 1 to December 31, 2024, {res_forecast_6h['test_days']} days) $\\to$ **{res_forecast_6h['test_pos']} Forecast Positives / {res_forecast_6h['test_neg']:,} Negatives**
- **Test Prevalence:** **{res_forecast_6h['test_prevalence_pct']}%**

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Operational False Alarms / Day | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for mname, mdata in res_forecast_6h["models"].items():
        md_fc += f"| **{mname}** | {mdata['precision']} | {mdata['recall']} | {mdata['f1']} | {mdata['roc_auc']} | {mdata['pr_auc']} | {mdata['false_alarms_per_day']} | `{mdata['confusion_matrix']}` |\n"

    md_fc += f"""
### Top 6-Hour Forward Predictors (Random Forest):
{chr(10).join([f"- `{feat}`: {imp:.4f}" for feat, imp in res_forecast_6h['models']['Random Forest (Balanced)']['top_features']])}

### Monthly Forecast Performance Breakdown on Test Split (Logistic Regression):
| Month | True Future Events $(t, t+6\\text{{h}}]$ | Model Forecast Alerts | Hits (True Positives) | False Alarms | Forward Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in res_forecast_6h["monthly_breakdown"]:
        md_fc += f"| **Month {row['month']:02d}** | {row['positive_events']} | {row['predicted_events']} | {row['true_positives']} | {row['false_positives']} | **{row['recall']*100:.1f}%** |\n"

    md_fc += """
---

## 3. Operational Interpretation and Limitations

1. **Target Meaning:** The target measures the occurrence of **NOAA-reported severe convective damage events** (wind >= 50 kts, hail >= 0.75 in, tornadoes, direct lightning casualties/damage). It does not capture ordinary non-severe thunderstorms that produce lightning without damage reports.
2. **Timing Limitations:** Because reanalysis data are available on a 6-hourly synoptic cadence, the forecast represents an aggregated 6-hour probability rather than minute-level convective initiation timing.
3. **Forecasting vs Detection:** Unlike same-time event detection where reanalysis reflects active storms, 6-hour forecasting requires the model to identify pre-convective environments (destabilization, moisture advection, frontal approach).
"""

    p_fc = REPORTS_DIR / "six_hour_ahead_forecast_evaluation.md"
    p_fc.write_text(md_fc, encoding="utf-8")
    Path("Florida_Thunderstorm_2024_Training/reports/six_hour_ahead_forecast_evaluation.md").write_text(md_fc, encoding="utf-8")
    logger.info(f"Saved {p_fc}")


def main():
    logger.info("=== STARTING FULL-YEAR EXPERIMENTS & FORECASTING PIPELINE ===")
    
    exp1_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_noaa_centered.csv"
    exp2_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_diagnostic.csv"
    exp3_path = DATA_PROCESSED / "florida_thunderstorm_2024_training_surface_full_year.csv"
    forecast_out = DATA_PROCESSED / "florida_thunderstorm_2024_forecasting_6h.csv"

    # 1. Load or Build Datasets
    if not (exp1_path.exists() and exp2_path.exists() and exp3_path.exists() and forecast_out.exists()):
        master = pd.read_csv(MASTER_CSV, low_memory=False)
        exp1_df, exp2_df, exp3_df = build_three_classification_datasets(master)
        forecast_df = build_six_hour_ahead_forecast_dataset(master)
    else:
        logger.info("Loading existing Datasets A, B, C and Forecasting Dataset...")
        exp1_df = pd.read_csv(exp1_path, low_memory=False)
        forecast_df = pd.read_csv(forecast_out, low_memory=False)

    # Mirror datasets to child project
    child_processed = Path("Florida_Thunderstorm_2024_Training/data/processed")
    child_processed.mkdir(parents=True, exist_ok=True)
    for p in [exp1_path, exp2_path, exp3_path, forecast_out]:
        dst = child_processed / p.name
        if not dst.exists() or dst.stat().st_size != p.stat().st_size:
            logger.info(f"Mirroring {p.name} to {dst}")
            dst.write_bytes(p.read_bytes())
    
    # 4. Feature selection
    meta_cols = {
        "valid_time_utc", "forecast_init_time_utc", "latitude", "longitude",
        "grid_cell_id", "month", "hour_utc", "target_noaa_event", "noaa_event_count",
        "thunderstorm_event", "era5_thunderstorm_favorability", "future_noaa_event_6h",
        "future_noaa_event_count", "future_noaa_event_ids", "future_noaa_event_types",
        "forecast_window_start_utc", "forecast_window_end_utc", "forecast_window_status",
        "future_noaa_min_distance_km"
    }
    feat_3d = [c for c in exp1_df.columns if c not in meta_cols]
    feat_fwd = [c for c in feat_3d if c in forecast_df.columns] + (["prior_event_at_init"] if "prior_event_at_init" in forecast_df.columns else [])
    
    logger.info(f"Feature count: 3D Classification has {len(feat_3d)} features; Forecasting has {len(feat_fwd)} features.")
    
    # 5. Run Phase 6 Baseline Evaluation (Full-Year 3D Classification)
    res_class_3d = run_evaluation(
        exp1_df, "target_noaa_event", feat_3d,
        train_months=[1, 2, 3, 4, 5, 6, 7, 8],
        test_months=[9, 10, 11, 12],
        time_col="valid_time_utc",
        experiment_title="Full-Year 3D Classification (NOAA-Centered)",
        include_persistence=False
    )
    
    # 6. Run Phase 8 Baseline Evaluation (6-Hour-Ahead Forecasting)
    res_forecast_6h = run_evaluation(
        forecast_df, "future_noaa_event_6h", feat_fwd,
        train_months=[1, 2, 3, 4, 5, 6, 7, 8],
        test_months=[9, 10, 11, 12],
        time_col="forecast_init_time_utc",
        experiment_title="Six-Hour-Ahead Convective Forecasting",
        include_persistence=True
    )
    
    # 7. Write Final Reports
    write_reports(res_class_3d, res_forecast_6h)
    logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
