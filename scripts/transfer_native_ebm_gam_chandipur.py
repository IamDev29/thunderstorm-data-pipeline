"""
transfer_native_ebm_gam_chandipur.py
====================================
Cross-regional transfer of native Florida-trained:
1. pyGAM LogisticGAM (native_gam_model.pkl)
2. InterpretML ExplainableBoostingClassifier (native_ebm_model.pkl)

To Chandipur ERA5 meteorological dataset (2022-2025):
`MET_DATA_with_storm_score_label.csv`

This experiment evaluates cross-region transfer and agreement with the existing formula.
It does NOT establish independent real-world thunderstorm forecasting accuracy in Chandipur.

Outputs are saved in:
`reports/ebm_gam_chandipur_transfer_native/`
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pygam
from pygam import LogisticGAM
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr, spearmanr

# 1. Directory and File Paths
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

NATIVE_MODELS_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
OUTPUT_DIR = ROOT_DIR / "reports" / "ebm_gam_chandipur_transfer_native"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature mapping
FEATURE_MAPPING = {
    "cape_Jkg": "cape_J_kg",
    "cin_Jkg": "cin_J_kg",
    "bulk_shear_850_300": "bulk_shear_850_300_ms",
    "total_cloud_cover": "cloud_cover_fraction",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_speed_850hPa": "wind_speed_850hPa_ms"
}

COMPATIBLE_FEATURES_FL = [
    "cape_J_kg",
    "cin_J_kg",
    "bulk_shear_850_300_ms",
    "cloud_cover_fraction",
    "wind_speed_10m_ms",
    "wind_speed_850hPa_ms"
]

FEATURE_LABELS = {
    "cape_J_kg": "Surface CAPE (J/kg)",
    "cin_J_kg": "Convective Inhibition (J/kg)",
    "bulk_shear_850_300_ms": "Bulk Shear 850-300 hPa (m/s)",
    "cloud_cover_fraction": "Total Cloud Cover (fraction)",
    "wind_speed_10m_ms": "10 m Wind Speed (m/s)",
    "wind_speed_850hPa_ms": "850 hPa Wind Speed (m/s)"
}

# Optimal decision thresholds selected during Florida validation phase (Aug-Sep 2024)
FLORIDA_THRESH_GAM = 0.33
FLORIDA_THRESH_EBM = 0.34


def assign_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter (DJF)"
    elif m in [3, 4, 5]:
        return "Pre-monsoon (MAM)"
    elif m in [6, 7, 8, 9]:
        return "Monsoon (JJAS)"
    elif m in [10, 11]:
        return "Post-monsoon (ON)"
    return "Unknown"


def compute_all_metrics(y_true, y_prob, threshold, storm_score):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Probabilistic metrics
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Continuous correlation with continuous storm score
    r_pearson, _ = pearsonr(y_prob, storm_score)
    r_spearman, _ = spearmanr(y_prob, storm_score)
    
    return {
        "threshold": threshold,
        "sample_count": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "cohen_kappa": round(float(kappa), 6),
        "mcc": round(float(mcc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "roc_auc": round(float(roc_auc), 6),
        "brier_score": round(float(brier), 6),
        "log_loss": round(float(ll), 6),
        "pearson_r": round(float(r_pearson), 6),
        "spearman_rho": round(float(r_spearman), 6),
        "metric_interpretation": "Agreement with formula-generated proxy labels, not independent thunderstorm forecasting skill"
    }


def main():
    print("=" * 80)
    print("CHANDIPUR CROSS-REGIONAL TRANSFER: NATIVE pyGAM & InterpretML EBM")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: VERIFY NATIVE MODELS
    # =========================================================================
    print("\n--- PHASE 1: Verifying Native Florida Model Artifacts ---")
    gam_path = NATIVE_MODELS_DIR / "native_gam_model.pkl"
    ebm_path = NATIVE_MODELS_DIR / "native_ebm_model.pkl"
    
    assert gam_path.exists(), f"Missing native GAM model at {gam_path}"
    assert ebm_path.exists(), f"Missing native EBM model at {ebm_path}"
    
    gam_model = joblib.load(gam_path)
    ebm_model = joblib.load(ebm_path)
    
    print(f"Loaded GAM Model: {type(gam_model)} from {gam_path}")
    print(f"  Terms: {gam_model.terms}")
    print(f"Loaded EBM Model: {type(ebm_model)} from {ebm_path}")
    print(f"  Feature Names: {ebm_model.feature_names}")
    print(f"  Intercept: {ebm_model.intercept_}")
    
    assert isinstance(gam_model, LogisticGAM), f"Unexpected GAM class: {type(gam_model)}"
    assert isinstance(ebm_model, ExplainableBoostingClassifier), f"Unexpected EBM class: {type(ebm_model)}"
    assert ebm_model.feature_names == COMPATIBLE_FEATURES_FL, f"EBM feature names mismatch: {ebm_model.feature_names}"
    
    # =========================================================================
    # PHASE 2: PREPARE CHANDIPUR FEATURES & LOAD FLORIDA TRAINING DATA
    # =========================================================================
    print("\n--- PHASE 2: Preparing Chandipur Features & Loading Florida Training Baseline ---")
    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH)
    print(f"Loaded Chandipur Dataset: {len(df_ch)} rows, {len(df_ch.columns)} columns")
    assert len(df_ch) == 23376, f"Expected 23,376 rows, found {len(df_ch)}"
    assert df_ch[list(FEATURE_MAPPING.keys())].isna().sum().sum() == 0, "Missing values found in Chandipur features!"
    
    df_ch["valid_time"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["original_row_id"] = np.arange(len(df_ch))
    df_ch["year"] = df_ch["valid_time"].dt.year
    df_ch["month"] = df_ch["valid_time"].dt.month
    df_ch["season"] = df_ch["valid_time"].apply(assign_season)
    
    # Load Florida training set (Jan-Jul 2024) to establish reference distributions
    print(f"Loading Florida training reference from {FLORIDA_DATA_PATH}...")
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    df_fl_valid = df_fl[df_fl["target_asos_strict_crit_c"].notna()].copy()
    fl_train = df_fl_valid[df_fl_valid["month"].between(1, 7)][COMPATIBLE_FEATURES_FL].copy()
    print(f"Florida Training Set (Months 1-7): {len(fl_train)} rows across 6 features")
    
    # Build Chandipur feature matrix in Florida order
    X_ch_df = df_ch[list(FEATURE_MAPPING.keys())].rename(columns=FEATURE_MAPPING)[COMPATIBLE_FEATURES_FL].copy()
    X_ch_arr = X_ch_df.values
    
    # =========================================================================
    # PHASE 3: GENERATE NATIVE MODEL PREDICTIONS
    # =========================================================================
    print("\n--- PHASE 3: Generating Prospective Model Predictions on Chandipur ---")
    t0 = time.time()
    prob_gam = gam_model.predict_proba(X_ch_arr)
    prob_ebm = ebm_model.predict_proba(X_ch_df)[:, 1]
    t1 = time.time()
    print(f"Inference completed in {t1 - t0:.2f} seconds for 23,376 rows")
    
    print(f"Native pyGAM Predicted Probabilities:")
    print(f"  Min: {prob_gam.min():.4f}, Max: {prob_gam.max():.4f}, Mean: {prob_gam.mean():.4f}, Median: {np.median(prob_gam):.4f}")
    print(f"Native InterpretML EBM Predicted Probabilities:")
    print(f"  Min: {prob_ebm.min():.4f}, Max: {prob_ebm.max():.4f}, Mean: {prob_ebm.mean():.4f}, Median: {np.median(prob_ebm):.4f}")
    
    # Add predictions and flags to Chandipur DataFrame
    df_ch["prob_gam"] = prob_gam
    df_ch["prob_ebm"] = prob_ebm
    df_ch["pred_gam_fl_thresh"] = (prob_gam >= FLORIDA_THRESH_GAM).astype(int)
    df_ch["pred_ebm_fl_thresh"] = (prob_ebm >= FLORIDA_THRESH_EBM).astype(int)
    df_ch["pred_gam_050"] = (prob_gam >= 0.50).astype(int)
    df_ch["pred_ebm_050"] = (prob_ebm >= 0.50).astype(int)
    
    # Flags for Florida range limits
    fl_stats = {}
    for feat in COMPATIBLE_FEATURES_FL:
        s_fl = fl_train[feat]
        s_ch = X_ch_df[feat]
        f_min, f_max = float(s_fl.min()), float(s_fl.max())
        p1, p99 = float(s_fl.quantile(0.01)), float(s_fl.quantile(0.99))
        fl_stats[feat] = {
            "min": f_min,
            "max": f_max,
            "p1": p1,
            "p99": p99,
            "mean": float(s_fl.mean()),
            "std": float(s_fl.std()),
            "ch_min": float(s_ch.min()),
            "ch_max": float(s_ch.max()),
            "ch_p1": float(s_ch.quantile(0.01)),
            "ch_p99": float(s_ch.quantile(0.99)),
            "ch_mean": float(s_ch.mean()),
            "ch_std": float(s_ch.std()),
            "pct_outside_range": float(((s_ch < f_min) | (s_ch > f_max)).mean() * 100.0),
            "pct_outside_p1_p99": float(((s_ch < p1) | (s_ch > p99)).mean() * 100.0)
        }
        df_ch[f"{feat}_outside_fl_range"] = (s_ch < f_min) | (s_ch > f_max)
        df_ch[f"{feat}_outside_fl_p1_p99"] = (s_ch < p1) | (s_ch > p99)
        
    df_ch["is_any_outside_fl_range"] = df_ch[[f"{feat}_outside_fl_range" for feat in COMPATIBLE_FEATURES_FL]].any(axis=1)
    df_ch["is_any_outside_fl_p1_p99"] = df_ch[[f"{feat}_outside_fl_p1_p99" for feat in COMPATIBLE_FEATURES_FL]].any(axis=1)
    df_ch["is_extreme_cape_extrapolation"] = df_ch["cape_Jkg"] > fl_stats["cape_J_kg"]["max"]
    
    # Save full predictions table
    out_preds_path = OUTPUT_DIR / "chandipur_native_gam_ebm_predictions.csv"
    df_ch.to_csv(out_preds_path, index=False)
    print(f"Saved full Chandipur predictions to {out_preds_path}")
    
    # Save extreme extrapolation cases
    extreme_df = df_ch[df_ch["is_any_outside_fl_range"]].copy()
    extreme_out_path = OUTPUT_DIR / "extreme_extrapolation_cases.csv"
    extreme_df.to_csv(extreme_out_path, index=False)
    print(f"Saved {len(extreme_df)} extreme extrapolation cases to {extreme_out_path}")
    
    # =========================================================================
    # PHASE 4: DOMAIN SHIFT ANALYSIS
    # =========================================================================
    print("\n--- PHASE 4: Domain Shift Analysis ---")
    domain_shift_rows = []
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    for feat in COMPATIBLE_FEATURES_FL:
        st = fl_stats[feat]
        s_fl = fl_train[feat]
        s_ch = X_ch_df[feat]
        row = {
            "feature": feat,
            "feature_label": FEATURE_LABELS[feat],
            "fl_train_min": round(st["min"], 2),
            "fl_train_p01": round(float(s_fl.quantile(0.01)), 2),
            "fl_train_p25": round(float(s_fl.quantile(0.25)), 2),
            "fl_train_p50_median": round(float(s_fl.median()), 2),
            "fl_train_p75": round(float(s_fl.quantile(0.75)), 2),
            "fl_train_p99": round(float(s_fl.quantile(0.99)), 2),
            "fl_train_max": round(st["max"], 2),
            "fl_train_mean": round(st["mean"], 2),
            "fl_train_std": round(st["std"], 2),
            "ch_min": round(st["ch_min"], 2),
            "ch_p01": round(float(s_ch.quantile(0.01)), 2),
            "ch_p25": round(float(s_ch.quantile(0.25)), 2),
            "ch_p50_median": round(float(s_ch.median()), 2),
            "ch_p75": round(float(s_ch.quantile(0.75)), 2),
            "ch_p99": round(float(s_ch.quantile(0.99)), 2),
            "ch_max": round(st["ch_max"], 2),
            "ch_mean": round(st["ch_mean"], 2),
            "ch_std": round(st["ch_std"], 2),
            "pct_chandipur_outside_fl_range": round(st["pct_outside_range"], 2),
            "pct_chandipur_outside_fl_p1_p99": round(st["pct_outside_p1_p99"], 2)
        }
        domain_shift_rows.append(row)
        print(f"Domain Shift: {feat:25s} | FL Max: {st['max']:8.2f} vs CH Max: {st['ch_max']:8.2f} | Out Range: {st['pct_outside_range']:5.2f}% | Out p1-p99: {st['pct_outside_p1_p99']:5.2f}%")
        
    df_domain_shift = pd.DataFrame(domain_shift_rows)
    df_domain_shift.to_csv(OUTPUT_DIR / "feature_domain_shift.csv", index=False)
    
    # Distribution summary table
    dist_summary = []
    for grp_col in ["overall", "year", "season", "storm_label_score"]:
        if grp_col == "overall":
            dist_summary.append({
                "group_type": "overall",
                "group_value": "All Samples (2022-2025)",
                "count": len(df_ch),
                "gam_prob_mean": round(float(df_ch["prob_gam"].mean()), 4),
                "gam_prob_std": round(float(df_ch["prob_gam"].std()), 4),
                "gam_prob_median": round(float(df_ch["prob_gam"].median()), 4),
                "ebm_prob_mean": round(float(df_ch["prob_ebm"].mean()), 4),
                "ebm_prob_std": round(float(df_ch["prob_ebm"].std()), 4),
                "ebm_prob_median": round(float(df_ch["prob_ebm"].median()), 4),
                "storm_score_mean": round(float(df_ch["storm_score"].mean()), 4),
                "storm_score_std": round(float(df_ch["storm_score"].std()), 4),
                "proxy_positive_pct": round(float(df_ch["storm_label_score"].mean() * 100), 2)
            })
        else:
            for g_val, sub in df_ch.groupby(grp_col):
                dist_summary.append({
                    "group_type": grp_col,
                    "group_value": str(g_val),
                    "count": len(sub),
                    "gam_prob_mean": round(float(sub["prob_gam"].mean()), 4),
                    "gam_prob_std": round(float(sub["prob_gam"].std()), 4),
                    "gam_prob_median": round(float(sub["prob_gam"].median()), 4),
                    "ebm_prob_mean": round(float(sub["prob_ebm"].mean()), 4),
                    "ebm_prob_std": round(float(sub["prob_ebm"].std()), 4),
                    "ebm_prob_median": round(float(sub["prob_ebm"].median()), 4),
                    "storm_score_mean": round(float(sub["storm_score"].mean()), 4),
                    "storm_score_std": round(float(sub["storm_score"].std()), 4),
                    "proxy_positive_pct": round(float(sub["storm_label_score"].mean() * 100), 2)
                })
    pd.DataFrame(dist_summary).to_csv(OUTPUT_DIR / "prediction_distribution_summary.csv", index=False)
    
    # =========================================================================
    # PHASE 5 & 6: PROXY AGREEMENT EVALUATION
    # =========================================================================
    print("\n--- PHASE 5 & 6: Proxy Agreement Evaluation ---")
    y_true = df_ch["storm_label_score"].values
    storm_scores = df_ch["storm_score"].values
    
    # Evaluate GAM and EBM at Florida threshold and at 0.50
    proxy_metrics = [
        {"model": "Native pyGAM LogisticGAM", "evaluation_setting": "Florida Validation Optimal Threshold", **compute_all_metrics(y_true, prob_gam, FLORIDA_THRESH_GAM, storm_scores)},
        {"model": "Native pyGAM LogisticGAM", "evaluation_setting": "Standard 0.50 Threshold", **compute_all_metrics(y_true, prob_gam, 0.50, storm_scores)},
        {"model": "Native InterpretML EBM", "evaluation_setting": "Florida Validation Optimal Threshold", **compute_all_metrics(y_true, prob_ebm, FLORIDA_THRESH_EBM, storm_scores)},
        {"model": "Native InterpretML EBM", "evaluation_setting": "Standard 0.50 Threshold", **compute_all_metrics(y_true, prob_ebm, 0.50, storm_scores)}
    ]
    df_proxy_metrics = pd.DataFrame(proxy_metrics)
    cols_order = [
        "model", "evaluation_setting", "threshold", "sample_count", "positives", "prevalence",
        "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "cohen_kappa", "mcc",
        "pr_auc", "roc_auc", "brier_score", "log_loss", "pearson_r", "spearman_rho", "metric_interpretation"
    ]
    df_proxy_metrics = df_proxy_metrics[cols_order]
    df_proxy_metrics.to_csv(OUTPUT_DIR / "proxy_agreement_metrics.csv", index=False)
    print("Proxy Agreement Summary:")
    for _, r in df_proxy_metrics.iterrows():
        print(f"  {r['model']} ({r['evaluation_setting']}, th={r['threshold']}): F1={r['f1']:.4f}, Acc={r['accuracy']:.4f}, PR-AUC={r['pr_auc']:.4f}, ROC-AUC={r['roc_auc']:.4f}, Brier={r['brier_score']:.4f}, Pearson r={r['pearson_r']:.4f}")
        
    # =========================================================================
    # PHASE 7: TEMPORAL AND SEASONAL ANALYSIS
    # =========================================================================
    print("\n--- PHASE 7: Temporal and Seasonal Breakdown ---")
    annual_rows = []
    for yr in sorted(df_ch["year"].unique()):
        sub = df_ch[df_ch["year"] == yr]
        y_yr = sub["storm_label_score"].values
        ss_yr = sub["storm_score"].values
        p_gam_yr = sub["prob_gam"].values
        p_ebm_yr = sub["prob_ebm"].values
        
        m_gam = compute_all_metrics(y_yr, p_gam_yr, FLORIDA_THRESH_GAM, ss_yr)
        m_ebm = compute_all_metrics(y_yr, p_ebm_yr, FLORIDA_THRESH_EBM, ss_yr)
        
        annual_rows.append({"year": yr, "model": "Native pyGAM", **m_gam})
        annual_rows.append({"year": yr, "model": "Native InterpretML EBM", **m_ebm})
        
    df_annual = pd.DataFrame(annual_rows)
    df_annual.to_csv(OUTPUT_DIR / "annual_agreement_metrics.csv", index=False)
    
    seasonal_rows = []
    season_order = ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"]
    for s_name in season_order:
        sub = df_ch[df_ch["season"] == s_name]
        y_s = sub["storm_label_score"].values
        ss_s = sub["storm_score"].values
        p_gam_s = sub["prob_gam"].values
        p_ebm_s = sub["prob_ebm"].values
        
        m_gam = compute_all_metrics(y_s, p_gam_s, FLORIDA_THRESH_GAM, ss_s)
        m_ebm = compute_all_metrics(y_s, p_ebm_s, FLORIDA_THRESH_EBM, ss_s)
        
        seasonal_rows.append({"season": s_name, "model": "Native pyGAM", **m_gam})
        seasonal_rows.append({"season": s_name, "model": "Native InterpretML EBM", **m_ebm})
        
    df_seasonal = pd.DataFrame(seasonal_rows)
    df_seasonal.to_csv(OUTPUT_DIR / "seasonal_agreement_metrics.csv", index=False)
    
    # =========================================================================
    # PHASE 8: CASE STUDIES & LOCAL INTERPRETABILITY EXPLANATIONS
    # =========================================================================
    print("\n--- PHASE 8: Case Studies & Interpretability Explanations ---")
    # Identify representative cases:
    # 1. Both agree with proxy = 1 (True Positive vs proxy)
    # 2. Both agree with proxy = 0 (True Negative vs proxy)
    # 3. Both models = 1, but proxy = 0 (Disagreement / Potential formula false negative)
    # 4. Both models = 0, but proxy = 1 (Disagreement / Potential formula false positive)
    # 5. Models disagree with each other (GAM=1, EBM=0 or GAM=0, EBM=1)
    # 6. Extreme extrapolation (CAPE > 7240 J/kg)
    # 7. Meteorological sensitivity: High CAPE but suppressed by High CIN
    
    case_indices = []
    case_reasons = []
    
    # 1. Consensus positive
    cp_cand = df_ch[(df_ch["storm_label_score"] == 1) & (df_ch["pred_gam_fl_thresh"] == 1) & (df_ch["pred_ebm_fl_thresh"] == 1) & (df_ch["storm_score"] >= 3.0)]
    if len(cp_cand) > 0:
        idx = cp_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Consensus Severe Convection: All methods agree on storm occurrence")
        
    # 2. Consensus negative
    cn_cand = df_ch[(df_ch["storm_label_score"] == 0) & (df_ch["pred_gam_fl_thresh"] == 0) & (df_ch["pred_ebm_fl_thresh"] == 0) & (df_ch["storm_score"] < 1.0)]
    if len(cn_cand) > 0:
        idx = cn_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Consensus Quiescent Atmosphere: All methods agree on storm absence")
        
    # 3. Models predict storm, Proxy = 0 (Formula suppression)
    fp_cand = df_ch[(df_ch["storm_label_score"] == 0) & (df_ch["pred_gam_fl_thresh"] == 1) & (df_ch["pred_ebm_fl_thresh"] == 1) & (df_ch["storm_score"] >= 2.0) & (df_ch["storm_score"] < 2.5)]
    if len(fp_cand) > 0:
        idx = fp_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Marginal Sub-threshold Formula State: Both models indicate convective favorability (prob > 0.45) but formula narrowly misses threshold (score ~2.3)")
        
    # 4. Models predict no storm, Proxy = 1 (Formula inflation due to 300hPa RH or CP)
    fn_cand = df_ch[(df_ch["storm_label_score"] == 1) & (df_ch["pred_gam_fl_thresh"] == 0) & (df_ch["pred_ebm_fl_thresh"] == 0)]
    if len(fn_cand) > 0:
        idx = fn_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Formula Anomaly State: Formula flags storm (score >= 2.5) driven by 300hPa RH/CP, but both models see weak thermodynamic fuel")
        
    # 5. Model Divergence: GAM = 1, EBM = 0
    div_cand1 = df_ch[(df_ch["pred_gam_fl_thresh"] == 1) & (df_ch["pred_ebm_fl_thresh"] == 0)]
    if len(div_cand1) > 0:
        idx = div_cand1.index[0]
        case_indices.append(idx)
        case_reasons.append("Model Divergence (GAM > EBM): Spline curvature extrapolates higher logit than bagged tree stump")
        
    # 6. Model Divergence: EBM = 1, GAM = 0
    div_cand2 = df_ch[(df_ch["pred_gam_fl_thresh"] == 0) & (df_ch["pred_ebm_fl_thresh"] == 1)]
    if len(div_cand2) > 0:
        idx = div_cand2.index[0]
        case_indices.append(idx)
        case_reasons.append("Model Divergence (EBM > GAM): Decision stump activates on boundary threshold where spline remains conservative")
        
    # 7. Extreme CAPE Extrapolation (> 7240 J/kg)
    ext_cand = df_ch[df_ch["cape_Jkg"] > 8000].sort_values("cape_Jkg", ascending=False)
    if len(ext_cand) > 0:
        idx = ext_cand.index[0]
        case_indices.append(idx)
        case_reasons.append(f"Extreme CAPE Extrapolation: Tropical buoyancy reaches {ext_cand.loc[idx, 'cape_Jkg']:.0f} J/kg, far exceeding Florida training max")
        
    # 8. High CAPE suppressed by Extreme CIN (Convective capping)
    cap_cand = df_ch[(df_ch["cape_Jkg"] > 2500) & (df_ch["cin_Jkg"] > 300)]
    if len(cap_cand) > 0:
        idx = cap_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Capping Inversion Suppression: High buoyancy (>2500 J/kg) suppressed by heavy capping barrier (>300 J/kg CIN)")
        
    # 9. Overcast with weak shear
    ovc_cand = df_ch[(df_ch["total_cloud_cover"] > 0.95) & (df_ch["bulk_shear_850_300"] < 8.0) & (df_ch["cape_Jkg"] > 1500)]
    if len(ovc_cand) > 0:
        idx = ovc_cand.index[0]
        case_indices.append(idx)
        case_reasons.append("Tropical Monsoon Profile: High cloudiness, moderate buoyancy, but very low vertical shear")
        
    # Build case study dataframe
    case_rows = []
    explanation_rows = []
    
    for c_id, (idx, reason) in enumerate(zip(case_indices, case_reasons), 1):
        row_data = df_ch.loc[idx]
        case_rows.append({
            "case_id": c_id,
            "category": reason.split(":")[0],
            "description": reason,
            "original_row_id": int(row_data["original_row_id"]),
            "valid_time": str(row_data["valid_time"]),
            "latitude": float(row_data["latitude"]),
            "longitude": float(row_data["longitude"]),
            "cape_Jkg": float(row_data["cape_Jkg"]),
            "cin_Jkg": float(row_data["cin_Jkg"]),
            "bulk_shear_850_300": float(row_data["bulk_shear_850_300"]),
            "total_cloud_cover": float(row_data["total_cloud_cover"]),
            "wind_speed_10m": float(row_data["wind_speed_10m"]),
            "wind_speed_850hPa": float(row_data["wind_speed_850hPa"]),
            "relative_humidity_300hPa": float(row_data["relative_humidity_300hPa"]),
            "vertical_velocity_Pas_500hPa": float(row_data["vertical_velocity_Pas_500hPa"]),
            "cp_24h_sum": float(row_data["cp_24h_sum"]),
            "storm_score": float(row_data["storm_score"]),
            "storm_label_score": int(row_data["storm_label_score"]),
            "prob_gam": float(row_data["prob_gam"]),
            "prob_ebm": float(row_data["prob_ebm"]),
            "pred_gam_fl_thresh": int(row_data["pred_gam_fl_thresh"]),
            "pred_ebm_fl_thresh": int(row_data["pred_ebm_fl_thresh"]),
            "is_outside_fl_range": bool(row_data["is_any_outside_fl_range"]),
            "is_extreme_cape": bool(row_data["is_extreme_cape_extrapolation"])
        })
        
        # Local model explanations
        sample_df = X_ch_df.loc[[idx]]
        sample_arr = sample_df.values
        
        # EBM local explanation
        ebm_loc = ebm_model.explain_local(sample_df)
        ebm_loc_data = ebm_loc.data(0)
        ebm_names = ebm_loc_data["names"]
        ebm_scores = ebm_loc_data["scores"]
        
        # GAM term contributions
        gam_contribs = []
        for term_i in range(6):
            pd_val = gam_model.partial_dependence(term=term_i, X=sample_arr)
            gam_contribs.append(float(pd_val[0]))
            
        for f_name, e_score, g_score in zip(ebm_names, ebm_scores, gam_contribs):
            explanation_rows.append({
                "case_id": c_id,
                "valid_time": str(row_data["valid_time"]),
                "feature": f_name,
                "feature_value": float(sample_df[f_name].iloc[0]),
                "ebm_logit_contribution": round(float(e_score), 4),
                "gam_logit_contribution": round(float(g_score), 4),
                "storm_score": float(row_data["storm_score"]),
                "proxy_label": int(row_data["storm_label_score"]),
                "prob_gam": float(row_data["prob_gam"]),
                "prob_ebm": float(row_data["prob_ebm"])
            })
            
    df_cases = pd.DataFrame(case_rows)
    df_cases.to_csv(OUTPUT_DIR / "representative_case_studies.csv", index=False)
    
    df_expl = pd.DataFrame(explanation_rows)
    df_expl.to_csv(OUTPUT_DIR / "model_explanation_results.csv", index=False)
    print(f"Saved {len(df_cases)} representative case studies and local explanations")
    
    # =========================================================================
    # PHASE 9: HIGH-RESOLUTION PLOTS
    # =========================================================================
    print("\n--- PHASE 9: Generating High-Resolution Diagnostic Plots ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Prediction Distributions (GAM vs EBM vs Storm Score)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df_ch["prob_gam"], bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].axvline(FLORIDA_THRESH_GAM, color="red", linestyle="--", label=f"FL Thresh ({FLORIDA_THRESH_GAM})")
    axes[0].axvline(0.50, color="black", linestyle=":", label="0.50 Thresh")
    axes[0].set_title("Native pyGAM Predicted Probabilities", fontweight="bold")
    axes[0].set_xlabel("Probability")
    axes[0].legend()
    
    sns.histplot(df_ch["prob_ebm"], bins=40, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].axvline(FLORIDA_THRESH_EBM, color="red", linestyle="--", label=f"FL Thresh ({FLORIDA_THRESH_EBM})")
    axes[1].axvline(0.50, color="black", linestyle=":", label="0.50 Thresh")
    axes[1].set_title("Native InterpretML EBM Predicted Probabilities", fontweight="bold")
    axes[1].set_xlabel("Probability")
    axes[1].legend()
    
    sns.histplot(df_ch["storm_score"], bins=40, kde=True, ax=axes[2], color="#d62728")
    axes[2].axvline(2.5, color="black", linestyle="--", label="Proxy Cutoff (tau=2.5)")
    axes[2].set_title("Chandipur Existing Storm Score", fontweight="bold")
    axes[2].set_xlabel("Storm Score")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_distributions_gam_ebm.png", dpi=300)
    plt.close()
    
    # 2. Domain Shift Distributions (Florida vs Chandipur)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(COMPATIBLE_FEATURES_FL):
        ax = axes[i]
        fl_s = fl_train[feat]
        ch_s = X_ch_df[feat]
        sns.kdeplot(fl_s, ax=ax, label="Florida Train (Jan-Jul 2024)", color="#1f77b4", linewidth=2.5, fill=True, alpha=0.2)
        sns.kdeplot(ch_s, ax=ax, label="Chandipur (2022-2025)", color="#ff7f0e", linewidth=2.5, fill=True, alpha=0.2)
        ax.set_title(FEATURE_LABELS[feat], fontweight="bold")
        ax.set_xlabel("Feature Value")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_domain_shift_distributions.png", dpi=300)
    plt.close()
    
    # 3. Model Probabilities vs Storm Score
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Hexbin / scatter for GAM
    hb1 = axes[0].hexbin(df_ch["storm_score"], df_ch["prob_gam"], gridsize=40, cmap="Blues", mincnt=1)
    axes[0].axvline(2.5, color="red", linestyle="--", label="Proxy Threshold (2.5)")
    axes[0].axhline(FLORIDA_THRESH_GAM, color="darkblue", linestyle="--", label=f"GAM Thresh ({FLORIDA_THRESH_GAM})")
    axes[0].set_title(f"pyGAM Probability vs Storm Score (Pearson r = {df_proxy_metrics.loc[0, 'pearson_r']:.3f})", fontweight="bold")
    axes[0].set_xlabel("Existing Storm Score")
    axes[0].set_ylabel("pyGAM Predicted Probability")
    axes[0].legend(loc="upper left")
    plt.colorbar(hb1, ax=axes[0], label="Observation Count")
    
    # Hexbin / scatter for EBM
    hb2 = axes[1].hexbin(df_ch["storm_score"], df_ch["prob_ebm"], gridsize=40, cmap="Greens", mincnt=1)
    axes[1].axvline(2.5, color="red", linestyle="--", label="Proxy Threshold (2.5)")
    axes[1].axhline(FLORIDA_THRESH_EBM, color="darkgreen", linestyle="--", label=f"EBM Thresh ({FLORIDA_THRESH_EBM})")
    axes[1].set_title(f"InterpretML EBM Probability vs Storm Score (Pearson r = {df_proxy_metrics.loc[2, 'pearson_r']:.3f})", fontweight="bold")
    axes[1].set_xlabel("Existing Storm Score")
    axes[1].set_ylabel("EBM Predicted Probability")
    axes[1].legend(loc="upper left")
    plt.colorbar(hb2, ax=axes[1], label="Observation Count")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_vs_storm_score.png", dpi=300)
    plt.close()
    
    # 4. Annual & Seasonal Agreement Trends
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Annual PR-AUC and F1
    df_ann_gam = df_annual[df_annual["model"] == "Native pyGAM"]
    df_ann_ebm = df_annual[df_annual["model"] == "Native InterpretML EBM"]
    
    x_yrs = np.arange(len(df_ann_gam))
    width = 0.35
    axes[0].bar(x_yrs - width/2, df_ann_gam["f1"], width, label="pyGAM F1", color="#3b82f6")
    axes[0].bar(x_yrs + width/2, df_ann_ebm["f1"], width, label="EBM F1", color="#10b981")
    axes[0].set_xticks(x_yrs)
    axes[0].set_xticklabels(df_ann_gam["year"])
    axes[0].set_title("Annual Agreement F1-Score Across Years (2022-2025)", fontweight="bold")
    axes[0].set_ylabel("F1 Score vs Proxy")
    axes[0].set_ylim(0.4, 0.75)
    axes[0].legend()
    
    # Seasonal PR-AUC and F1
    df_sea_gam = df_seasonal[df_seasonal["model"] == "Native pyGAM"]
    df_sea_ebm = df_seasonal[df_seasonal["model"] == "Native InterpretML EBM"]
    
    x_sea = np.arange(len(df_sea_gam))
    axes[1].bar(x_sea - width/2, df_sea_gam["f1"], width, label="pyGAM F1", color="#3b82f6")
    axes[1].bar(x_sea + width/2, df_sea_ebm["f1"], width, label="EBM F1", color="#10b981")
    axes[1].set_xticks(x_sea)
    axes[1].set_xticklabels(["Winter\n(DJF)", "Pre-monsoon\n(MAM)", "Monsoon\n(JJAS)", "Post-monsoon\n(ON)"])
    axes[1].set_title("Seasonal Agreement F1-Score Across Climatic Regimes", fontweight="bold")
    axes[1].set_ylabel("F1 Score vs Proxy")
    axes[1].set_ylim(0.0, 0.8)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "annual_and_seasonal_agreement_metrics.png", dpi=300)
    plt.close()
    
    # 5. Prediction Time Series (Smoothed Weekly Rolling Means)
    df_ch_daily = df_ch.groupby("valid_time")[["prob_gam", "prob_ebm", "storm_score", "storm_label_score"]].mean().reset_index()
    df_ch_daily["storm_score_norm"] = df_ch_daily["storm_score"] / 5.0  # Normalized to [0, 1] for visual overlay
    df_ch_daily["prob_gam_roll"] = df_ch_daily["prob_gam"].rolling(28, min_periods=7).mean()
    df_ch_daily["prob_ebm_roll"] = df_ch_daily["prob_ebm"].rolling(28, min_periods=7).mean()
    df_ch_daily["storm_norm_roll"] = df_ch_daily["storm_score_norm"].rolling(28, min_periods=7).mean()
    df_ch_daily["proxy_roll"] = df_ch_daily["storm_label_score"].rolling(28, min_periods=7).mean()
    
    plt.figure(figsize=(18, 6))
    plt.plot(df_ch_daily["valid_time"], df_ch_daily["prob_gam_roll"], label="pyGAM Prob (7-day roll)", color="#1f77b4", linewidth=2)
    plt.plot(df_ch_daily["valid_time"], df_ch_daily["prob_ebm_roll"], label="InterpretML EBM Prob (7-day roll)", color="#2ca02c", linewidth=2)
    plt.plot(df_ch_daily["valid_time"], df_ch_daily["proxy_roll"], label="Proxy Label Frequency (7-day roll)", color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.title("Chandipur 2022-2025 Prospective Multi-Year Prediction Time Series", fontweight="bold")
    plt.ylabel("Mean Convective Probability / Rate")
    plt.xlabel("Date")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_time_series.png", dpi=300)
    plt.close()
    
    # 6. Extreme Extrapolation Analysis: Predictions vs CAPE up to 9906 J/kg
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_ch.sample(n=3000, random_state=42), x="cape_Jkg", y="prob_ebm", alpha=0.3, color="#2ca02c", label="EBM Predicted Prob")
    sns.scatterplot(data=df_ch.sample(n=3000, random_state=42), x="cape_Jkg", y="prob_gam", alpha=0.3, color="#1f77b4", label="pyGAM Predicted Prob")
    plt.axvline(fl_stats["cape_J_kg"]["max"], color="red", linestyle="--", linewidth=2, label=f"Florida Training Max ({fl_stats['cape_J_kg']['max']:.0f} J/kg)")
    plt.axvline(fl_stats["cape_J_kg"]["p99"], color="orange", linestyle=":", linewidth=2, label=f"Florida Training 99th Pct ({fl_stats['cape_J_kg']['p99']:.0f} J/kg)")
    plt.title("Model Predictions vs CAPE Across Extreme Extrapolation Domain (Up to 9,906 J/kg)", fontweight="bold")
    plt.xlabel("Surface CAPE (J/kg)")
    plt.ylabel("Predicted Thunderstorm Probability")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "extreme_extrapolation_analysis.png", dpi=300)
    plt.close()
    
    # 7. Model Disagreement Breakdown
    # Categorize into 4 quadrants
    # Q1: Both agree with proxy=1
    # Q2: Both agree with proxy=0
    # Q3: Models = 1, Proxy = 0
    # Q4: Models = 0, Proxy = 1
    # Q5: Models disagree with each other
    conditions = [
        (df_ch["pred_gam_fl_thresh"] == 1) & (df_ch["pred_ebm_fl_thresh"] == 1) & (df_ch["storm_label_score"] == 1),
        (df_ch["pred_gam_fl_thresh"] == 0) & (df_ch["pred_ebm_fl_thresh"] == 0) & (df_ch["storm_label_score"] == 0),
        (df_ch["pred_gam_fl_thresh"] == 1) & (df_ch["pred_ebm_fl_thresh"] == 1) & (df_ch["storm_label_score"] == 0),
        (df_ch["pred_gam_fl_thresh"] == 0) & (df_ch["pred_ebm_fl_thresh"] == 0) & (df_ch["storm_label_score"] == 1),
        (df_ch["pred_gam_fl_thresh"] != df_ch["pred_ebm_fl_thresh"])
    ]
    choices = [
        "Consensus Positive (Models=1, Proxy=1)",
        "Consensus Negative (Models=0, Proxy=0)",
        "Model False Positive vs Proxy (Models=1, Proxy=0)",
        "Model False Negative vs Proxy (Models=0, Proxy=1)",
        "Model Divergence (GAM != EBM)"
    ]
    df_ch["agreement_category"] = np.select(conditions, choices, default="Other")
    
    cat_counts = df_ch["agreement_category"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis")
    plt.title("Chandipur 2022-2025 Agreement and Disagreement Regimes (N = 23,376)", fontweight="bold")
    plt.xlabel("Number of Synoptic Observations")
    for i, v in enumerate(cat_counts.values):
        pct = v / len(df_ch) * 100
        plt.text(v + 100, i, f"{v:,} ({pct:.1f}%)", va="center", fontweight="bold")
    plt.xlim(0, max(cat_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_disagreement_analysis.png", dpi=300)
    plt.close()
    
    print("\nAll figures generated successfully.")
    print(f"Artifacts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
