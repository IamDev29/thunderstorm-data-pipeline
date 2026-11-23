import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import lightgbm as lgb
import xgboost as xgb
import joblib

# Paths
ROOT_DIR = Path(".").resolve()
EXP_DIR = ROOT_DIR / "experiments" / "advanced_tuning_2024"
EXP_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"

# Feature definitions
ERA5_FEATURES = [
    "cape_J_kg", "cin_J_kg", "dewpoint_depression_degC", "tcwv_kg_m2",
    "total_precipitation_mm", "convective_precipitation_mm", "cloud_cover_fraction",
    "temperature_2m_degC", "dewpoint_2m_degC", "surface_pressure_hPa", "mslp_hPa",
    "wind_speed_10m_ms", "wind_direction_10m_deg",
    "temperature_850hPa_degC", "relative_humidity_850hPa_pct", "wind_speed_850hPa_ms",
    "temperature_500hPa_degC", "relative_humidity_500hPa_pct",
    "bulk_shear_850_300_ms", "temp_diff_850_500_degC", "era5_thunderstorm_favorability"
]

SPATIAL_DIURNAL_FEATURES = ["latitude", "longitude", "sin_hour", "cos_hour"]
PERSISTENCE_FEATURE = ["prior_strict_ts_6h"]

MODEL_A_FEATURES = ERA5_FEATURES
MODEL_B_FEATURES = ERA5_FEATURES + SPATIAL_DIURNAL_FEATURES
MODEL_C_FEATURES = ERA5_FEATURES + SPATIAL_DIURNAL_FEATURES + PERSISTENCE_FEATURE

def calculate_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Probability metrics
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Clip probabilities for robust log loss calculation
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    ll = float(log_loss(y_true, y_prob_clipped))
    
    return {
        "threshold": round(threshold, 4),
        "sample_count": len(y_true),
        "positives": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        "pr_auc": round(pr_auc, 6), "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6), "log_loss": round(ll, 6)
    }

def find_best_threshold(y_true, y_prob, metric="f1"):
    best_thresh = 0.5
    best_score = -1.0
    for thresh in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return float(best_thresh), float(best_score)

def load_and_prepare_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter strictly to monitored rows
    df_mon = df[df["target_asos_strict_crit_c"].notna()].copy()
    df_mon["target"] = df_mon["target_asos_strict_crit_c"].astype(int)
    
    # Diurnal encoding
    df_mon["sin_hour"] = np.sin(2.0 * np.pi * df_mon["hour_utc"] / 24.0)
    df_mon["cos_hour"] = np.cos(2.0 * np.pi * df_mon["hour_utc"] / 24.0)
    
    # Ensure correct month
    df_mon["month"] = df_mon["forecast_init_time_utc"].dt.month
    df_mon["date_str"] = df_mon["forecast_init_time_utc"].dt.strftime("%Y-%m-%d")
    
    return df, df_mon

# ==============================================================================
# PHASE 1: REPRODUCIBLE EVALUATION VERIFICATION
# ==============================================================================
def phase_1_audit(df_raw, df_mon):
    print("\n" + "=" * 80)
    print("PHASE 1: AUDIT & REPRODUCIBILITY METADATA")
    print("=" * 80)
    
    with open(DATA_PATH, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    
    print(f"Dataset SHA256: {file_sha256}")
    print(f"Total raw dataset rows: {len(df_raw):,}")
    print(f"Monitored Criterion C rows: {len(df_mon):,}")
    
    # Split counts
    pre_test_mask = df_mon["forecast_init_time_utc"] < "2024-10-01"
    test_mask = df_mon["forecast_init_time_utc"] >= "2024-10-01"
    
    n_pre = pre_test_mask.sum()
    n_test = test_mask.sum()
    pos_pre = int(df_mon.loc[pre_test_mask, "target"].sum())
    pos_test = int(df_mon.loc[test_mask, "target"].sum())
    
    print(f"Pre-test rows (Jan-Sep): {n_pre:,} rows, {pos_pre:,} positives ({pos_pre/n_pre*100:.2f}%)")
    print(f"Test rows (Oct-Dec):     {n_test:,} rows, {pos_test:,} positives ({pos_test/n_test*100:.2f}%)")
    assert n_test == 28073, f"Expected 28,073 test rows, got {n_test}"
    assert pos_test == 1918, f"Expected 1,918 test positives, got {pos_test}"
    
    # Zero leakage verification
    era5_diff = (pd.to_datetime(df_raw["forecast_init_time_utc_era5"], utc=True) - df_raw["forecast_init_time_utc"]).dt.total_seconds().abs().max()
    assert era5_diff == 0.0, "ERA5 forecast init time mismatch!"
    
    # Persistence continuity verification
    assert (df_raw.loc[df_raw["cycle_diff_hours"] != 6.0, "prior_strict_ts_6h"] == 0).all(), "Persistence bridged missing window!"
    
    metadata = {
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(DATA_PATH),
        "dataset_sha256": file_sha256,
        "total_terrestrial_rows": len(df_raw),
        "monitored_rows": len(df_mon),
        "pre_test_rows": int(n_pre),
        "pre_test_positives": pos_pre,
        "test_rows": int(n_test),
        "test_positives": pos_test,
        "random_seed": 42,
        "software_versions": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "target_definition": "target_asos_strict_crit_c (strict TS / VCTS / LTG in (t, t+6h] under Criterion C)",
        "features": {
            "model_a_era5_count": len(MODEL_A_FEATURES),
            "model_b_spatial_diurnal_count": len(MODEL_B_FEATURES),
            "model_c_full_count": len(MODEL_C_FEATURES),
            "full_feature_list": MODEL_C_FEATURES
        }
    }
    
    meta_path = EXP_DIR / "reproducibility_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    return metadata

# ==============================================================================
# PHASE 2: EXPANDING-WINDOW TEMPORAL VALIDATION
# ==============================================================================
def phase_2_expanding_window(df_mon):
    print("\n" + "=" * 80)
    print("PHASE 2: EXPANDING-WINDOW CHRONOLOGICAL VALIDATION (JAN - SEP 2024)")
    print("=" * 80)
    
    folds = [
        {"fold_idx": 1, "train_months": [1, 2, 3], "val_month": 4, "name": "Train Jan-Mar, Val Apr"},
        {"fold_idx": 2, "train_months": [1, 2, 3, 4], "val_month": 5, "name": "Train Jan-Apr, Val May"},
        {"fold_idx": 3, "train_months": [1, 2, 3, 4, 5], "val_month": 6, "name": "Train Jan-May, Val Jun"},
        {"fold_idx": 4, "train_months": [1, 2, 3, 4, 5, 6], "val_month": 7, "name": "Train Jan-Jun, Val Jul"},
        {"fold_idx": 5, "train_months": [1, 2, 3, 4, 5, 6, 7], "val_month": 8, "name": "Train Jan-Jul, Val Aug"},
        {"fold_idx": 6, "train_months": [1, 2, 3, 4, 5, 6, 7, 8], "val_month": 9, "name": "Train Jan-Aug, Val Sep"},
    ]
    
    results = []
    
    for f in folds:
        print(f"\n--- Running Fold {f['fold_idx']}: {f['name']} ---")
        train_mask = df_mon["month"].isin(f["train_months"])
        val_mask = df_mon["month"] == f["val_month"]
        
        df_tr = df_mon[train_mask].copy()
        df_va = df_mon[val_mask].copy()
        
        y_tr = df_tr["target"].values
        y_va = df_va["target"].values
        
        n_tr, pos_tr = len(y_tr), int(y_tr.sum())
        n_va, pos_va = len(y_va), int(y_va.sum())
        print(f"Train: {n_tr:,} rows, {pos_tr:,} pos ({pos_tr/n_tr*100:.2f}%) | Val: {n_va:,} rows, {pos_va:,} pos ({pos_va/n_va*100:.2f}%)")
        
        # 1. Persistence Baseline
        p_va = df_va["prior_strict_ts_6h"].values
        m_p = calculate_all_metrics(y_va, p_va, threshold=0.5)
        m_p.update({"fold": f["fold_idx"], "fold_name": f["name"], "model": "Prior 6h Persistence"})
        results.append(m_p)
        
        # 2. Logistic Regression (fitted with StandardScaler strictly on train)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(df_tr[MODEL_C_FEATURES])
        X_va_scaled = scaler.transform(df_va[MODEL_C_FEATURES])
        
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
        lr.fit(X_tr_scaled, y_tr)
        prob_va_lr = lr.predict_proba(X_va_scaled)[:, 1]
        
        m_lr = calculate_all_metrics(y_va, prob_va_lr, threshold=0.5)
        m_lr.update({"fold": f["fold_idx"], "fold_name": f["name"], "model": "Logistic Regression"})
        results.append(m_lr)
        
        # 3. LightGBM (baseline)
        lgb_clf = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss", learning_rate=0.05,
            num_leaves=31, max_depth=6, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_estimators=300, verbose=-1
        )
        lgb_clf.fit(df_tr[MODEL_C_FEATURES], y_tr)
        prob_va_lgb = lgb_clf.predict_proba(df_va[MODEL_C_FEATURES])[:, 1]
        
        m_lgb = calculate_all_metrics(y_va, prob_va_lgb, threshold=0.5)
        m_lgb.update({"fold": f["fold_idx"], "fold_name": f["name"], "model": "LightGBM (Baseline)"})
        results.append(m_lgb)
        
        # 4. XGBoost (baseline)
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", learning_rate=0.05,
            max_depth=5, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_estimators=300
        )
        xgb_clf.fit(df_tr[MODEL_C_FEATURES], y_tr, verbose=False)
        prob_va_xgb = xgb_clf.predict_proba(df_va[MODEL_C_FEATURES])[:, 1]
        
        m_xgb = calculate_all_metrics(y_va, prob_va_xgb, threshold=0.5)
        m_xgb.update({"fold": f["fold_idx"], "fold_name": f["name"], "model": "XGBoost (Baseline)"})
        results.append(m_xgb)
        
        print(f"  Persistence PR-AUC: {m_p['pr_auc']:.4f}, ROC-AUC: {m_p['roc_auc']:.4f}, Brier: {m_p['brier_score']:.4f}")
        print(f"  LightGBM    PR-AUC: {m_lgb['pr_auc']:.4f}, ROC-AUC: {m_lgb['roc_auc']:.4f}, Brier: {m_lgb['brier_score']:.4f}")
        print(f"  XGBoost     PR-AUC: {m_xgb['pr_auc']:.4f}, ROC-AUC: {m_xgb['roc_auc']:.4f}, Brier: {m_xgb['brier_score']:.4f}")

    df_res = pd.DataFrame(results)
    out_csv = EXP_DIR / "expanding_window_metrics.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved expanding-window metrics to {out_csv}")
    
    # Compute aggregate summary across folds
    agg = df_res.groupby("model").agg(
        mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
        mean_brier=("brier_score", "mean"), std_brier=("brier_score", "std"),
        mean_f1=("f1", "mean"), std_f1=("f1", "std")
    ).reset_index()
    print("\n--- Expanding-Window Aggregate Summary (Mean +/- Std across 6 Folds) ---")
    print(agg.to_string(index=False))
    
    return df_res, agg

# ==============================================================================
# PHASE 3: SECONDARY SPLIT ROBUSTNESS EXPERIMENTS
# ==============================================================================
def phase_3_secondary_splits(df_mon):
    print("\n" + "=" * 80)
    print("PHASE 3: SECONDARY SPLIT ROBUSTNESS EXPERIMENTS")
    print("=" * 80)
    
    pre_test_mask = df_mon["forecast_init_time_utc"] < "2024-10-01"
    df_pre = df_mon[pre_test_mask].copy()
    
    # -------------------------------------------------------------
    # Experiment A: Alternating Calendar Dates
    # -------------------------------------------------------------
    print("\n--- Experiment 3A: Alternating Calendar Dates (Jan - Sep) ---")
    # Deterministic assignment by day of year (even DOY = Train, odd DOY = Val)
    doy = df_pre["forecast_init_time_utc"].dt.dayofyear
    train_dates_mask = (doy % 2 == 0)
    val_dates_mask = (doy % 2 == 1)
    
    df_d_tr = df_pre[train_dates_mask].copy()
    df_d_va = df_pre[val_dates_mask].copy()
    
    y_d_tr = df_d_tr["target"].values
    y_d_va = df_d_va["target"].values
    
    print(f"Date Split: Train {len(df_d_tr):,} rows ({y_d_tr.sum():,} pos) | Val {len(df_d_va):,} rows ({y_d_va.sum():,} pos)")
    
    # Train LightGBM & XGBoost on Date-Split
    lgb_date = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=31, max_depth=6, random_state=42, n_estimators=300, verbose=-1)
    lgb_date.fit(df_d_tr[MODEL_C_FEATURES], y_d_tr)
    prob_d_lgb = lgb_date.predict_proba(df_d_va[MODEL_C_FEATURES])[:, 1]
    
    xgb_date = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.05, max_depth=5, random_state=42, n_estimators=300)
    xgb_date.fit(df_d_tr[MODEL_C_FEATURES], y_d_tr, verbose=False)
    prob_d_xgb = xgb_date.predict_proba(df_d_va[MODEL_C_FEATURES])[:, 1]
    
    p_d_va = df_d_va["prior_strict_ts_6h"].values
    
    m_p_d = calculate_all_metrics(y_d_va, p_d_va, threshold=0.5)
    m_p_d.update({"experiment": "Alternating Dates", "model": "Persistence"})
    m_lgb_d = calculate_all_metrics(y_d_va, prob_d_lgb, threshold=0.5)
    m_lgb_d.update({"experiment": "Alternating Dates", "model": "LightGBM"})
    m_xgb_d = calculate_all_metrics(y_d_va, prob_d_xgb, threshold=0.5)
    m_xgb_d.update({"experiment": "Alternating Dates", "model": "XGBoost"})
    
    date_results = pd.DataFrame([m_p_d, m_lgb_d, m_xgb_d])
    date_csv = EXP_DIR / "date_alternating_metrics.csv"
    date_results.to_csv(date_csv, index=False)
    print(f"Saved date-alternating metrics to {date_csv}")
    print(date_results[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier_score"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # Experiment B: Spatial Holdout (5-Fold Station CV)
    # -------------------------------------------------------------
    print("\n--- Experiment 3B: Spatial Holdout (5 Station Groups) ---")
    stations = sorted(df_pre["station_id"].unique())
    n_stations = len(stations)
    np.random.seed(42)
    shuffled_stations = np.random.permutation(stations)
    station_groups = np.array_split(shuffled_stations, 5)
    
    spatial_results = []
    
    for g_idx, val_stations in enumerate(station_groups):
        train_st_mask = ~df_pre["station_id"].isin(val_stations)
        val_st_mask = df_pre["station_id"].isin(val_stations)
        
        df_s_tr = df_pre[train_st_mask].copy()
        df_s_va = df_pre[val_st_mask].copy()
        
        y_s_tr = df_s_tr["target"].values
        y_s_va = df_s_va["target"].values
        
        # Evaluate Persistence on unseen stations
        p_s_va = df_s_va["prior_strict_ts_6h"].values
        m_s_p = calculate_all_metrics(y_s_va, p_s_va, threshold=0.5)
        m_s_p.update({"repeat": g_idx + 1, "model": "Persistence", "held_out_stations": len(val_stations)})
        spatial_results.append(m_s_p)
        
        # Evaluate LightGBM on unseen stations
        lgb_sp = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=31, max_depth=6, random_state=42, n_estimators=300, verbose=-1)
        lgb_sp.fit(df_s_tr[MODEL_C_FEATURES], y_s_tr)
        prob_s_lgb = lgb_sp.predict_proba(df_s_va[MODEL_C_FEATURES])[:, 1]
        
        m_s_lgb = calculate_all_metrics(y_s_va, prob_s_lgb, threshold=0.5)
        m_s_lgb.update({"repeat": g_idx + 1, "model": "LightGBM", "held_out_stations": len(val_stations)})
        spatial_results.append(m_s_lgb)
        
        # Evaluate XGBoost on unseen stations
        xgb_sp = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.05, max_depth=5, random_state=42, n_estimators=300)
        xgb_sp.fit(df_s_tr[MODEL_C_FEATURES], y_s_tr, verbose=False)
        prob_s_xgb = xgb_sp.predict_proba(df_s_va[MODEL_C_FEATURES])[:, 1]
        
        m_s_xgb = calculate_all_metrics(y_s_va, prob_s_xgb, threshold=0.5)
        m_s_xgb.update({"repeat": g_idx + 1, "model": "XGBoost", "held_out_stations": len(val_stations)})
        spatial_results.append(m_s_xgb)
        
        print(f"  Group {g_idx+1} ({len(val_stations)} stations, {len(df_s_va):,} rows): LightGBM PR-AUC={m_s_lgb['pr_auc']:.4f}, ROC={m_s_lgb['roc_auc']:.4f} | Persistence PR-AUC={m_s_p['pr_auc']:.4f}")

    df_sp = pd.DataFrame(spatial_results)
    sp_csv = EXP_DIR / "spatial_holdout_metrics.csv"
    df_sp.to_csv(sp_csv, index=False)
    print(f"\nSaved spatial holdout metrics to {sp_csv}")
    
    agg_sp = df_sp.groupby("model").agg(
        mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
        mean_f1=("f1", "mean"), std_f1=("f1", "std"),
        mean_brier=("brier_score", "mean"), std_brier=("brier_score", "std")
    ).reset_index()
    print("\n--- Spatial Holdout Aggregate Summary (Mean +/- Std across 5 Station Groups) ---")
    print(agg_sp.to_string(index=False))
    
    return date_results, df_sp

# ==============================================================================
# PHASE 4: HYPERPARAMETER OPTIMIZATION (50 TRIALS EACH)
# ==============================================================================
def phase_4_hyperparameter_optimization(df_mon):
    print("\n" + "=" * 80)
    print("PHASE 4: HYPERPARAMETER OPTIMIZATION (50 TRIALS LIGHTGBM & XGBOOST)")
    print("=" * 80)
    
    # We evaluate trials across expanding-window validation folds
    # To maintain exceptional speed and thoroughness, evaluate each trial across 3 diverse expanding folds:
    # Fold 2 (May onset), Fold 4 (July peak), Fold 6 (September tropical)
    eval_folds = [
        {"train_months": [1, 2, 3, 4], "val_month": 5},
        {"train_months": [1, 2, 3, 4, 5, 6], "val_month": 7},
        {"train_months": [1, 2, 3, 4, 5, 6, 7, 8], "val_month": 9}
    ]
    
    # Pre-slice data for fast evaluation
    fold_data = []
    for f in eval_folds:
        tr_m = df_mon["month"].isin(f["train_months"])
        va_m = df_mon["month"] == f["val_month"]
        X_tr = df_mon.loc[tr_m, MODEL_C_FEATURES].values
        y_tr = df_mon.loc[tr_m, "target"].values
        X_va = df_mon.loc[va_m, MODEL_C_FEATURES].values
        y_va = df_mon.loc[va_m, "target"].values
        fold_data.append((X_tr, y_tr, X_va, y_va))
        
    np.random.seed(42)
    n_trials = 50
    
    # 1. LightGBM Search
    print(f"\n--- Running {n_trials} LightGBM Hyperparameter Trials ---")
    lgb_trials = []
    
    for t_idx in range(n_trials):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": float(np.exp(np.random.uniform(np.log(0.01), np.log(0.10)))),
            "n_estimators": int(np.random.choice([300, 500, 800, 1200])),
            "num_leaves": int(np.random.randint(15, 128)),
            "max_depth": int(np.random.choice([3, 5, 6, 8, 10, -1])),
            "min_child_samples": int(np.random.randint(20, 201)),
            "subsample": float(np.random.uniform(0.70, 1.00)),
            "colsample_bytree": float(np.random.uniform(0.70, 1.00)),
            "reg_alpha": float(np.random.uniform(0.0, 5.0)),
            "reg_lambda": float(np.random.uniform(0.0, 10.0)),
            "scale_pos_weight": float(np.random.choice([1.0, 1.0, 1.5, 2.0, 3.0])),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        
        pr_scores = []
        roc_scores = []
        brier_scores = []
        
        for X_tr, y_tr, X_va, y_va in fold_data:
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_va)[:, 1]
            pr_scores.append(average_precision_score(y_va, probs))
            roc_scores.append(roc_auc_score(y_va, probs))
            brier_scores.append(brier_score_loss(y_va, probs))
            
        mean_pr = float(np.mean(pr_scores))
        std_pr = float(np.std(pr_scores))
        mean_roc = float(np.mean(roc_scores))
        mean_brier = float(np.mean(brier_scores))
        
        trial_record = {
            "trial_id": t_idx + 1,
            "model_family": "LightGBM",
            "mean_pr_auc": round(mean_pr, 6),
            "std_pr_auc": round(std_pr, 6),
            "mean_roc_auc": round(mean_roc, 6),
            "mean_brier": round(mean_brier, 6),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in params.items() if k not in ["objective", "metric", "verbose", "n_jobs", "boosting_type"]}
        }
        lgb_trials.append(trial_record)
        if (t_idx + 1) % 10 == 0 or t_idx == 0:
            print(f"  LightGBM Trial {t_idx+1:2d}/{n_trials}: Mean PR-AUC = {mean_pr:.4f} (std={std_pr:.4f}), ROC = {mean_roc:.4f}, Brier = {mean_brier:.4f}")

    df_lgb_trials = pd.DataFrame(lgb_trials).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    lgb_csv = EXP_DIR / "tuning_trials_lightgbm.csv"
    df_lgb_trials.to_csv(lgb_csv, index=False)
    print(f"Saved LightGBM trials to {lgb_csv}")
    print("Top 3 LightGBM Trials:")
    print(df_lgb_trials.head(3)[["trial_id", "mean_pr_auc", "mean_roc_auc", "mean_brier", "learning_rate", "num_leaves", "max_depth", "scale_pos_weight"]].to_string(index=False))
    
    # 2. XGBoost Search
    print(f"\n--- Running {n_trials} XGBoost Hyperparameter Trials ---")
    xgb_trials = []
    
    for t_idx in range(n_trials):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": float(np.exp(np.random.uniform(np.log(0.01), np.log(0.10)))),
            "n_estimators": int(np.random.choice([300, 500, 800])),
            "max_depth": int(np.random.randint(3, 11)),
            "min_child_weight": float(np.random.uniform(1.0, 20.0)),
            "subsample": float(np.random.uniform(0.70, 1.00)),
            "colsample_bytree": float(np.random.uniform(0.70, 1.00)),
            "gamma": float(np.random.uniform(0.0, 5.0)),
            "reg_alpha": float(np.random.uniform(0.0, 5.0)),
            "reg_lambda": float(np.random.uniform(1.0, 20.0)),
            "scale_pos_weight": float(np.random.choice([1.0, 1.0, 1.5, 2.0, 3.0])),
            "random_state": 42,
            "n_jobs": -1
        }
        
        pr_scores = []
        roc_scores = []
        brier_scores = []
        
        for X_tr, y_tr, X_va, y_va in fold_data:
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr, verbose=False)
            probs = clf.predict_proba(X_va)[:, 1]
            pr_scores.append(average_precision_score(y_va, probs))
            roc_scores.append(roc_auc_score(y_va, probs))
            brier_scores.append(brier_score_loss(y_va, probs))
            
        mean_pr = float(np.mean(pr_scores))
        std_pr = float(np.std(pr_scores))
        mean_roc = float(np.mean(roc_scores))
        mean_brier = float(np.mean(brier_scores))
        
        trial_record = {
            "trial_id": t_idx + 1,
            "model_family": "XGBoost",
            "mean_pr_auc": round(mean_pr, 6),
            "std_pr_auc": round(std_pr, 6),
            "mean_roc_auc": round(mean_roc, 6),
            "mean_brier": round(mean_brier, 6),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in params.items() if k not in ["objective", "eval_metric", "n_jobs"]}
        }
        xgb_trials.append(trial_record)
        if (t_idx + 1) % 10 == 0 or t_idx == 0:
            print(f"  XGBoost Trial {t_idx+1:2d}/{n_trials}: Mean PR-AUC = {mean_pr:.4f} (std={std_pr:.4f}), ROC = {mean_roc:.4f}, Brier = {mean_brier:.4f}")

    df_xgb_trials = pd.DataFrame(xgb_trials).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    xgb_csv = EXP_DIR / "tuning_trials_xgboost.csv"
    df_xgb_trials.to_csv(xgb_csv, index=False)
    print(f"Saved XGBoost trials to {xgb_csv}")
    print("Top 3 XGBoost Trials:")
    print(df_xgb_trials.head(3)[["trial_id", "mean_pr_auc", "mean_roc_auc", "mean_brier", "learning_rate", "max_depth", "min_child_weight", "scale_pos_weight"]].to_string(index=False))
    
    return df_lgb_trials, df_xgb_trials

# ==============================================================================
# PHASE 5: FEATURE ABLATION EXPERIMENTS
# ==============================================================================
def phase_5_feature_ablation(df_mon, best_lgb_params):
    print("\n" + "=" * 80)
    print("PHASE 5: FEATURE ABLATION (ACROSS ALL 6 EXPANDING-WINDOW FOLDS)")
    print("=" * 80)
    
    # Feature configurations:
    # Model A: ERA5 atmospheric predictors only (21 features)
    # Model B: Model A + Spatial & Diurnal (25 features)
    # Model C: Model B + Persistence lag (26 features)
    # Baseline: Persistence only
    
    feature_configs = [
        ("Model A: ERA5 Atmospheric Only", MODEL_A_FEATURES),
        ("Model B: ERA5 + Spatial/Diurnal", MODEL_B_FEATURES),
        ("Model C: Full (ERA5 + Spatial/Diurnal + Persistence)", MODEL_C_FEATURES)
    ]
    
    folds = [
        {"fold_idx": 1, "train_months": [1, 2, 3], "val_month": 4},
        {"fold_idx": 2, "train_months": [1, 2, 3, 4], "val_month": 5},
        {"fold_idx": 3, "train_months": [1, 2, 3, 4, 5], "val_month": 6},
        {"fold_idx": 4, "train_months": [1, 2, 3, 4, 5, 6], "val_month": 7},
        {"fold_idx": 5, "train_months": [1, 2, 3, 4, 5, 6, 7], "val_month": 8},
        {"fold_idx": 6, "train_months": [1, 2, 3, 4, 5, 6, 7, 8], "val_month": 9},
    ]
    
    ablation_records = []
    
    # Clean up parameters for LightGBM
    lgb_clean_params = {k: v for k, v in best_lgb_params.items() if k not in ["trial_id", "model_family", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_brier"]}
    lgb_clean_params.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1, "n_jobs": -1})
    
    for f in folds:
        tr_mask = df_mon["month"].isin(f["train_months"])
        va_mask = df_mon["month"] == f["val_month"]
        df_tr = df_mon[tr_mask]
        df_va = df_mon[va_mask]
        y_tr = df_tr["target"].values
        y_va = df_va["target"].values
        
        # Persistence only
        p_va = df_va["prior_strict_ts_6h"].values
        m_p = calculate_all_metrics(y_va, p_va, threshold=0.5)
        m_p.update({"fold": f["fold_idx"], "ablation_config": "Baseline: Persistence Only", "feature_count": 1})
        ablation_records.append(m_p)
        
        for cfg_name, feat_list in feature_configs:
            clf = lgb.LGBMClassifier(**lgb_clean_params)
            clf.fit(df_tr[feat_list], y_tr)
            probs = clf.predict_proba(df_va[feat_list])[:, 1]
            m = calculate_all_metrics(y_va, probs, threshold=0.5)
            m.update({"fold": f["fold_idx"], "ablation_config": cfg_name, "feature_count": len(feat_list)})
            ablation_records.append(m)

    df_ab = pd.DataFrame(ablation_records)
    ab_csv = EXP_DIR / "feature_ablation_metrics.csv"
    df_ab.to_csv(ab_csv, index=False)
    print(f"Saved feature ablation metrics to {ab_csv}")
    
    agg_ab = df_ab.groupby("ablation_config").agg(
        features=("feature_count", "first"),
        mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
        mean_f1=("f1", "mean"), std_f1=("f1", "std"),
        mean_brier=("brier_score", "mean"), std_brier=("brier_score", "std")
    ).sort_values("mean_pr_auc").reset_index()
    print("\n--- Feature Ablation Summary (Mean across 6 Folds) ---")
    print(agg_ab.to_string(index=False))
    
    return df_ab, agg_ab

# ==============================================================================
# PHASE 6: PROBABILITY CALIBRATION & THRESHOLD SELECTION
# ==============================================================================
def phase_6_calibration_and_thresholds(df_mon, best_lgb_params):
    print("\n" + "=" * 80)
    print("PHASE 6: PROBABILITY CALIBRATION & THRESHOLD SELECTION (PRE-TEST PERIOD)")
    print("=" * 80)
    
    # Generate Out-Of-Fold (OOF) predictions across the 6 expanding-window folds
    folds = [
        {"fold_idx": 1, "train_months": [1, 2, 3], "val_month": 4},
        {"fold_idx": 2, "train_months": [1, 2, 3, 4], "val_month": 5},
        {"fold_idx": 3, "train_months": [1, 2, 3, 4, 5], "val_month": 6},
        {"fold_idx": 4, "train_months": [1, 2, 3, 4, 5, 6], "val_month": 7},
        {"fold_idx": 5, "train_months": [1, 2, 3, 4, 5, 6, 7], "val_month": 8},
        {"fold_idx": 6, "train_months": [1, 2, 3, 4, 5, 6, 7, 8], "val_month": 9},
    ]
    
    lgb_clean_params = {k: v for k, v in best_lgb_params.items() if k not in ["trial_id", "model_family", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_brier"]}
    lgb_clean_params.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1, "n_jobs": -1})
    
    oof_y_true = []
    oof_y_prob = []
    
    print("Generating OOF predictions across 6 expanding validation folds...")
    for f in folds:
        tr_m = df_mon["month"].isin(f["train_months"])
        va_m = df_mon["month"] == f["val_month"]
        clf = lgb.LGBMClassifier(**lgb_clean_params)
        clf.fit(df_mon.loc[tr_m, MODEL_C_FEATURES], df_mon.loc[tr_m, "target"])
        probs = clf.predict_proba(df_mon.loc[va_m, MODEL_C_FEATURES])[:, 1]
        oof_y_true.extend(df_mon.loc[va_m, "target"].values)
        oof_y_prob.extend(probs)
        
    oof_y_true = np.array(oof_y_true)
    oof_y_prob = np.array(oof_y_prob)
    print(f"OOF samples collected: {len(oof_y_true):,} (Positives: {np.sum(oof_y_true):,}, Prev: {np.mean(oof_y_true)*100:.2f}%)")
    
    # 1. Uncalibrated Metrics
    m_uncal = calculate_all_metrics(oof_y_true, oof_y_prob, threshold=0.5)
    m_uncal["calibrator"] = "Uncalibrated"
    
    # 2. Platt Scaling (Sigmoidal Logistic Calibration)
    # Fit logistic regression on log-odds
    eps = 1e-15
    oof_prob_clipped = np.clip(oof_y_prob, eps, 1.0 - eps)
    oof_logits = np.log(oof_prob_clipped / (1.0 - oof_prob_clipped)).reshape(-1, 1)
    
    platt_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt_calibrator.fit(oof_logits, oof_y_true)
    prob_platt = platt_calibrator.predict_proba(oof_logits)[:, 1]
    m_platt = calculate_all_metrics(oof_y_true, prob_platt, threshold=0.5)
    m_platt["calibrator"] = "Platt Scaling (Sigmoid)"
    joblib.dump(platt_calibrator, EXP_DIR / "platt_calibrator.joblib")
    
    # 3. Isotonic Regression
    iso_calibrator = IsotonicRegression(out_of_bounds="clip")
    iso_calibrator.fit(oof_y_prob, oof_y_true)
    prob_iso = iso_calibrator.predict(oof_y_prob)
    m_iso = calculate_all_metrics(oof_y_true, prob_iso, threshold=0.5)
    m_iso["calibrator"] = "Isotonic Regression"
    joblib.dump(iso_calibrator, EXP_DIR / "isotonic_calibrator.joblib")
    
    df_cal = pd.DataFrame([m_uncal, m_platt, m_iso])
    cal_csv = EXP_DIR / "calibration_metrics.csv"
    df_cal.to_csv(cal_csv, index=False)
    print(f"\nSaved calibration comparison to {cal_csv}")
    print(df_cal[["calibrator", "brier_score", "log_loss", "pr_auc", "roc_auc", "f1"]].to_string(index=False))
    
    # Threshold Sensitivity Curve on OOF Validation Data
    threshold_records = []
    for thresh in np.linspace(0.10, 0.90, 81):
        y_pred = (oof_y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(oof_y_true, y_pred).ravel()
        p = precision_score(oof_y_true, y_pred, zero_division=0)
        r = recall_score(oof_y_true, y_pred, zero_division=0)
        f = f1_score(oof_y_true, y_pred, zero_division=0)
        threshold_records.append({
            "threshold": round(thresh, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        })
        
    df_thresh = pd.DataFrame(threshold_records)
    best_f1_row = df_thresh.loc[df_thresh["f1"].idxmax()]
    print(f"\nValidation F1-maximizing threshold: tau = {best_f1_row['threshold']:.2f} (F1 = {best_f1_row['f1']:.4f}, Prec = {best_f1_row['precision']:.4f}, Rec = {best_f1_row['recall']:.4f})")
    
    # Document documented operational objectives:
    # 1. Balanced F1 threshold
    # 2. High-Sensitivity Public Safety threshold (Recall >= 75%)
    # 3. High-Precision Economic Alert threshold (Precision >= 60%)
    high_sens_row = df_thresh[df_thresh["recall"] >= 0.75].iloc[-1]
    high_prec_row = df_thresh[df_thresh["precision"] >= 0.60].iloc[0]
    
    thresh_selection = {
        "validation_samples": len(oof_y_true),
        "validation_positives": int(np.sum(oof_y_true)),
        "operational_objectives": {
            "balanced_f1_optimal": {
                "threshold": float(best_f1_row["threshold"]),
                "validation_precision": float(best_f1_row["precision"]),
                "validation_recall": float(best_f1_row["recall"]),
                "validation_f1": float(best_f1_row["f1"])
            },
            "high_sensitivity_safety": {
                "threshold": float(high_sens_row["threshold"]),
                "validation_precision": float(high_sens_row["precision"]),
                "validation_recall": float(high_sens_row["recall"]),
                "validation_f1": float(high_sens_row["f1"])
            },
            "high_precision_economic": {
                "threshold": float(high_prec_row["threshold"]),
                "validation_precision": float(high_prec_row["precision"]),
                "validation_recall": float(high_prec_row["recall"]),
                "validation_f1": float(high_prec_row["f1"])
            },
            "standard_operational_default": {
                "threshold": 0.50,
                "validation_precision": float(df_thresh.loc[df_thresh["threshold"] == 0.50, "precision"].iloc[0]),
                "validation_recall": float(df_thresh.loc[df_thresh["threshold"] == 0.50, "recall"].iloc[0]),
                "validation_f1": float(df_thresh.loc[df_thresh["threshold"] == 0.50, "f1"].iloc[0])
            }
        }
    }
    
    thresh_json = EXP_DIR / "threshold_selection.json"
    with open(thresh_json, "w") as f:
        json.dump(thresh_selection, f, indent=2)
    print(f"Saved threshold selection records to {thresh_json}")
    
    return platt_calibrator, iso_calibrator, thresh_selection

# ==============================================================================
# PHASE 7: FINAL MODEL SELECTION & TEST EVALUATION (OCT - DEC 2024 LOCKED TEST)
# ==============================================================================
def phase_7_final_test_evaluation(df_mon, best_lgb_params, best_xgb_params, platt_calibrator, thresh_selection):
    print("\n" + "=" * 80)
    print("PHASE 7: FINAL MODEL SELECTION & ONE-TIME TEST EVALUATION (OCT - DEC 2024)")
    print("=" * 80)
    
    # 1. Refit candidates on the entire permitted pre-test period (Jan 1 - Sep 30, 2024; N = 84,436)
    pre_test_mask = df_mon["forecast_init_time_utc"] < "2024-10-01"
    test_mask = df_mon["forecast_init_time_utc"] >= "2024-10-01"
    
    df_pre = df_mon[pre_test_mask].copy()
    df_test = df_mon[test_mask].copy()
    
    X_pre = df_pre[MODEL_C_FEATURES]
    y_pre = df_pre["target"].values
    
    X_test = df_test[MODEL_C_FEATURES]
    y_test = df_test["target"].values
    
    print(f"Refitting models on complete pre-test data: {len(df_pre):,} rows ({y_pre.sum():,} pos, {y_pre.mean()*100:.2f}%)")
    print(f"Lock-box holdout test set: {len(df_test):,} rows ({y_test.sum():,} pos, {y_test.mean()*100:.2f}%)")
    
    # Fit Tuned LightGBM
    lgb_clean_params = {k: v for k, v in best_lgb_params.items() if k not in ["trial_id", "model_family", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_brier"]}
    lgb_clean_params.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1, "n_jobs": -1})
    
    final_lgb = lgb.LGBMClassifier(**lgb_clean_params)
    final_lgb.fit(X_pre, y_pre)
    joblib.dump(final_lgb, EXP_DIR / "final_lightgbm_tuned.joblib")
    final_lgb.booster_.save_model(str(EXP_DIR / "final_lightgbm_tuned.txt"))
    
    # Fit Tuned XGBoost
    xgb_clean_params = {k: v for k, v in best_xgb_params.items() if k not in ["trial_id", "model_family", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_brier"]}
    xgb_clean_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "n_jobs": -1})
    
    final_xgb = xgb.XGBClassifier(**xgb_clean_params)
    final_xgb.fit(X_pre, y_pre, verbose=False)
    joblib.dump(final_xgb, EXP_DIR / "final_xgboost_tuned.joblib")
    final_xgb.save_model(str(EXP_DIR / "final_xgboost_tuned.json"))
    
    # Evaluate raw test probabilities
    test_prob_lgb = final_lgb.predict_proba(X_test)[:, 1]
    test_prob_xgb = final_xgb.predict_proba(X_test)[:, 1]
    
    # Calibrated probabilities via Platt scaling
    eps = 1e-15
    lgb_logits = np.log(np.clip(test_prob_lgb, eps, 1.0 - eps) / (1.0 - np.clip(test_prob_lgb, eps, 1.0 - eps))).reshape(-1, 1)
    test_prob_lgb_platt = platt_calibrator.predict_proba(lgb_logits)[:, 1]
    
    # Persistence
    p_test = df_test["prior_strict_ts_6h"].values
    
    # Operational thresholds chosen on validation
    tau_opt = thresh_selection["operational_objectives"]["balanced_f1_optimal"]["threshold"]
    tau_safe = thresh_selection["operational_objectives"]["high_sensitivity_safety"]["threshold"]
    
    # Save predictions
    df_preds = pd.DataFrame({
        "forecast_init_time_utc": df_test["forecast_init_time_utc"].values,
        "station_id": df_test["station_id"].values,
        "month": df_test["month"].values,
        "hour_utc": df_test["hour_utc"].values,
        "actual_target": y_test,
        "pred_persistence": p_test,
        "prob_lightgbm_tuned": test_prob_lgb,
        "prob_lightgbm_calibrated": test_prob_lgb_platt,
        "pred_bin_lightgbm_050": (test_prob_lgb >= 0.50).astype(int),
        "pred_bin_lightgbm_opt": (test_prob_lgb >= tau_opt).astype(int),
        "prob_xgboost_tuned": test_prob_xgb,
        "pred_bin_xgboost_050": (test_prob_xgb >= 0.50).astype(int)
    })
    preds_csv = EXP_DIR / "final_test_predictions.csv"
    df_preds.to_csv(preds_csv, index=False)
    print(f"Saved final test predictions to {preds_csv}")
    
    # Compile Test Set Benchmark Summary
    test_benchmarks = [
        {"model": "Prior 6h Persistence", "threshold": "Prior TS == 1", **calculate_all_metrics(y_test, p_test, threshold=0.5)},
        {"model": "Tuned LightGBM (tau=0.50)", "threshold": "Prob >= 0.50", **calculate_all_metrics(y_test, test_prob_lgb, threshold=0.50)},
        {"model": f"Tuned LightGBM (Val-Opt tau={tau_opt:.2f})", "threshold": f"Prob >= {tau_opt:.2f}", **calculate_all_metrics(y_test, test_prob_lgb, threshold=tau_opt)},
        {"model": f"Tuned LightGBM (Safety tau={tau_safe:.2f})", "threshold": f"Prob >= {tau_safe:.2f}", **calculate_all_metrics(y_test, test_prob_lgb, threshold=tau_safe)},
        {"model": "Platt Calibrated LightGBM (tau=0.50)", "threshold": "Cal Prob >= 0.50", **calculate_all_metrics(y_test, test_prob_lgb_platt, threshold=0.50)},
        {"model": "Tuned XGBoost (tau=0.50)", "threshold": "Prob >= 0.50", **calculate_all_metrics(y_test, test_prob_xgb, threshold=0.50)},
    ]
    
    df_tb = pd.DataFrame(test_benchmarks)
    print("\n--- FINAL TEST SET BENCHMARKS (OCT - DEC 2024; N = 28,073) ---")
    print(df_tb[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier_score", "log_loss"]].to_string(index=False))
    
    # Monthly Breakdown
    month_records = []
    month_names = {10: "October", 11: "November", 12: "December"}
    for m in [10, 11, 12]:
        m_mask = df_preds["month"] == m
        sub_m = df_preds[m_mask]
        y_m = sub_m["actual_target"].values
        p_m = sub_m["pred_persistence"].values
        lgb_m = sub_m["prob_lightgbm_tuned"].values
        
        m_p = calculate_all_metrics(y_m, p_m, threshold=0.5)
        m_p.update({"month": m, "month_name": month_names[m], "model": "Persistence"})
        month_records.append(m_p)
        
        m_lgb_050 = calculate_all_metrics(y_m, lgb_m, threshold=0.50)
        m_lgb_050.update({"month": m, "month_name": month_names[m], "model": "Tuned LightGBM (tau=0.50)"})
        month_records.append(m_lgb_050)
        
        m_lgb_opt = calculate_all_metrics(y_m, lgb_m, threshold=tau_opt)
        m_lgb_opt.update({"month": m, "month_name": month_names[m], "model": f"Tuned LightGBM (tau={tau_opt:.2f})"})
        month_records.append(m_lgb_opt)
        
    df_mb = pd.DataFrame(month_records)
    mb_csv = EXP_DIR / "final_metrics_by_month.csv"
    df_mb.to_csv(mb_csv, index=False)
    print(f"Saved final monthly metrics to {mb_csv}")
    
    # Station Breakdown
    station_records = []
    for st in sorted(df_preds["station_id"].unique()):
        sub_st = df_preds[df_preds["station_id"] == st]
        y_st = sub_st["actual_target"].values
        n_st = len(y_st)
        n_pos = int(y_st.sum())
        
        if n_pos >= 5:
            p_st = sub_st["pred_persistence"].values
            lgb_st = sub_st["prob_lightgbm_tuned"].values
            
            f1_p = f1_score(y_st, p_st, zero_division=0)
            brier_p = brier_score_loss(y_st, p_st)
            
            f1_lgb = f1_score(y_st, (lgb_st >= tau_opt).astype(int), zero_division=0)
            pr_lgb = average_precision_score(y_st, lgb_st)
            roc_lgb = roc_auc_score(y_st, lgb_st) if len(np.unique(y_st)) > 1 else 0.5
            brier_lgb = brier_score_loss(y_st, lgb_st)
            
            station_records.append({
                "station_id": st, "samples": n_st, "positives": n_pos, "prevalence": round(n_pos/n_st, 4),
                "f1_persistence": round(f1_p, 4), "brier_persistence": round(brier_p, 4),
                "f1_lightgbm": round(f1_lgb, 4), "pr_lightgbm": round(pr_lgb, 4),
                "roc_lightgbm": round(roc_lgb, 4), "brier_lightgbm": round(brier_lgb, 4)
            })
            
    df_sb = pd.DataFrame(station_records)
    sb_csv = EXP_DIR / "final_metrics_by_station.csv"
    df_sb.to_csv(sb_csv, index=False)
    print(f"Saved final station metrics ({len(df_sb)} qualifying stations) to {sb_csv}")
    
    return df_preds, df_tb, final_lgb

# ==============================================================================
# PHASE 8: BLOCK BOOTSTRAP UNCERTAINTY (DATE-LEVEL RESAMPLING)
# ==============================================================================
def phase_8_bootstrap_uncertainty(df_preds, tau_opt):
    print("\n" + "=" * 80)
    print("PHASE 8: STATISTICAL UNCERTAINTY (1,000 DATE-BLOCK BOOTSTRAP RESAMPLES)")
    print("=" * 80)
    
    df_preds["date_str"] = df_preds["forecast_init_time_utc"].dt.strftime("%Y-%m-%d")
    unique_dates = df_preds["date_str"].unique()
    n_dates = len(unique_dates)
    print(f"Resampling across {n_dates} calendar dates in holdout test set...")
    
    np.random.seed(42)
    n_bootstraps = 1000
    
    # Store metrics for each bootstrap draw
    delta_pr_list = []
    delta_roc_list = []
    delta_f1_list = []
    delta_brier_list = []
    
    lgb_pr_list = []
    p_pr_list = []
    
    # Pre-index dates for speed
    date_groups = {d: df_preds[df_preds["date_str"] == d] for d in unique_dates}
    
    for b in range(n_bootstraps):
        sampled_dates = np.random.choice(unique_dates, size=n_dates, replace=True)
        sub_df = pd.concat([date_groups[d] for d in sampled_dates], ignore_index=True)
        
        y_b = sub_b = sub_df["actual_target"].values
        if np.sum(y_b) == 0 or np.sum(y_b) == len(y_b):
            continue
            
        p_b = sub_df["pred_persistence"].values
        lgb_probs_b = sub_df["prob_lightgbm_tuned"].values
        lgb_preds_b = (lgb_probs_b >= tau_opt).astype(int)
        
        # Persistence metrics
        pr_p = average_precision_score(y_b, p_b)
        roc_p = roc_auc_score(y_b, p_b)
        f1_p = f1_score(y_b, p_b, zero_division=0)
        brier_p = brier_score_loss(y_b, p_b)
        
        # LightGBM metrics
        pr_lgb = average_precision_score(y_b, lgb_probs_b)
        roc_lgb = roc_auc_score(y_b, lgb_probs_b)
        f1_lgb = f1_score(y_b, lgb_preds_b, zero_division=0)
        brier_lgb = brier_score_loss(y_b, lgb_probs_b)
        
        delta_pr_list.append(pr_lgb - pr_p)
        delta_roc_list.append(roc_lgb - roc_p)
        delta_f1_list.append(f1_lgb - f1_p)
        delta_brier_list.append(brier_lgb - brier_p)
        
        lgb_pr_list.append(pr_lgb)
        p_pr_list.append(pr_p)

    def ci95(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    boot_summary = {
        "resamples": len(delta_pr_list),
        "lightgbm_pr_auc_mean": float(np.mean(lgb_pr_list)),
        "lightgbm_pr_auc_ci95": ci95(lgb_pr_list),
        "persistence_pr_auc_mean": float(np.mean(p_pr_list)),
        "persistence_pr_auc_ci95": ci95(p_pr_list),
        "delta_pr_auc_mean": float(np.mean(delta_pr_list)),
        "delta_pr_auc_ci95": ci95(delta_pr_list),
        "delta_roc_auc_mean": float(np.mean(delta_roc_list)),
        "delta_roc_auc_ci95": ci95(delta_roc_list),
        "delta_f1_mean": float(np.mean(delta_f1_list)),
        "delta_f1_ci95": ci95(delta_f1_list),
        "delta_brier_mean": float(np.mean(delta_brier_list)),
        "delta_brier_ci95": ci95(delta_brier_list),
    }
    
    df_boot = pd.DataFrame([{
        "metric": "PR-AUC Difference (LightGBM - Persistence)",
        "mean_difference": round(boot_summary["delta_pr_auc_mean"], 4),
        "ci_95_lower": round(boot_summary["delta_pr_auc_ci95"][0], 4),
        "ci_95_upper": round(boot_summary["delta_pr_auc_ci95"][1], 4),
        "statistically_significant": boot_summary["delta_pr_auc_ci95"][0] > 0
    }, {
        "metric": "ROC-AUC Difference (LightGBM - Persistence)",
        "mean_difference": round(boot_summary["delta_roc_auc_mean"], 4),
        "ci_95_lower": round(boot_summary["delta_roc_auc_ci95"][0], 4),
        "ci_95_upper": round(boot_summary["delta_roc_auc_ci95"][1], 4),
        "statistically_significant": boot_summary["delta_roc_auc_ci95"][0] > 0
    }, {
        "metric": "F1 Score Difference (LightGBM - Persistence)",
        "mean_difference": round(boot_summary["delta_f1_mean"], 4),
        "ci_95_lower": round(boot_summary["delta_f1_ci95"][0], 4),
        "ci_95_upper": round(boot_summary["delta_f1_ci95"][1], 4),
        "statistically_significant": boot_summary["delta_f1_ci95"][0] > 0
    }, {
        "metric": "Brier Score Difference (LightGBM - Persistence)",
        "mean_difference": round(boot_summary["delta_brier_mean"], 4),
        "ci_95_lower": round(boot_summary["delta_brier_ci95"][0], 4),
        "ci_95_upper": round(boot_summary["delta_brier_ci95"][1], 4),
        "statistically_significant": boot_summary["delta_brier_ci95"][1] < 0
    }])
    
    boot_csv = EXP_DIR / "bootstrap_uncertainty.csv"
    df_boot.to_csv(boot_csv, index=False)
    print(f"Saved bootstrap uncertainty analysis to {boot_csv}")
    print(df_boot.to_string(index=False))
    
    return boot_summary

def main():
    start_time = time.time()
    df_raw, df_mon = load_and_prepare_data()
    
    # Phase 1
    phase_1_audit(df_raw, df_mon)
    
    # Phase 2
    phase_2_expanding_window(df_mon)
    
    # Phase 3
    phase_3_secondary_splits(df_mon)
    
    # Phase 4
    df_lgb_trials, df_xgb_trials = phase_4_hyperparameter_optimization(df_mon)
    best_lgb_params = df_lgb_trials.iloc[0].to_dict()
    best_xgb_params = df_xgb_trials.iloc[0].to_dict()
    
    # Phase 5
    phase_5_feature_ablation(df_mon, best_lgb_params)
    
    # Phase 6
    platt_calibrator, iso_calibrator, thresh_selection = phase_6_calibration_and_thresholds(df_mon, best_lgb_params)
    
    # Phase 7
    df_preds, df_tb, final_lgb = phase_7_final_test_evaluation(
        df_mon, best_lgb_params, best_xgb_params, platt_calibrator, thresh_selection
    )
    
    # Phase 8
    tau_opt = thresh_selection["operational_objectives"]["balanced_f1_optimal"]["threshold"]
    phase_8_bootstrap_uncertainty(df_preds, tau_opt)
    
    elapsed = time.time() - start_time
    print(f"\nALL 8 ADVANCED TUNING PHASES COMPLETED IN {elapsed/60:.2f} MINUTES!")

if __name__ == "__main__":
    main()
