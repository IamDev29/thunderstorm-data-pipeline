"""
run_step_03_domain_shift_analysis.py
====================================
Thunderstorm Domain Adaptation Framework (TDAF) - Step 3:
Comprehensive Domain Shift Analysis and Adaptation Strategy Design.

This analytical pipeline executes:
1. Marginal covariate shift metrics (Mean, Std, Median, IQR, Min, Max, P1, P5, P95, P99,
   SMD, Wasserstein, KS statistic, Jensen-Shannon divergence, PSI).
2. Correlation shift metrics (Pearson, Spearman, and difference matrices).
3. Seasonal distribution shift analysis (DJF, MAM, JJAS, ON).
4. Feature support and multivariate overlap analysis.
5. Multivariate Domain Classifier (Logistic Regression & LightGBM) with block splitting.
6. Verification & reconciliation of EBM and pyGAM CAPE extrapolation behavior.
7. Generates 8 publication-grade diagnostic figures.
8. Generates all required CSVs and tables for Step 3.

NO MODEL RETRAINING OR TARGET-SUPERVISED ADAPTATION IS PERFORMED.
"""

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance, ks_2samp, pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import lightgbm as lgb
import pygam
from pygam import LogisticGAM
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

# ---------------------------------------------------------
# 1. Directories and Configuration
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

NATIVE_MODELS_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
GAM_MODEL_PATH = NATIVE_MODELS_DIR / "native_gam_model.pkl"
EBM_MODEL_PATH = NATIVE_MODELS_DIR / "native_ebm_model.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_03"
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

COMPATIBLE_FEATURES = [
    "cape_J_kg",
    "cin_J_kg",
    "bulk_shear_850_300_ms",
    "cloud_cover_fraction",
    "wind_speed_10m_ms",
    "wind_speed_850hPa_ms"
]

FEATURE_MAPPING = {
    "cape_Jkg": "cape_J_kg",
    "cin_Jkg": "cin_J_kg",
    "bulk_shear_850_300": "bulk_shear_850_300_ms",
    "total_cloud_cover": "cloud_cover_fraction",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_speed_850hPa": "wind_speed_850hPa_ms"
}

FEATURE_LABELS = {
    "cape_J_kg": "Surface CAPE (J/kg)",
    "cin_J_kg": "CIN (J/kg)",
    "bulk_shear_850_300_ms": "Bulk Shear 850-300 hPa (m/s)",
    "cloud_cover_fraction": "Total Cloud Cover (fraction)",
    "wind_speed_10m_ms": "10m Wind Speed (m/s)",
    "wind_speed_850hPa_ms": "850hPa Wind Speed (m/s)"
}

def assign_season(month):
    if month in [12, 1, 2]:
        return "DJF (Winter)"
    elif month in [3, 4, 5]:
        return "MAM (Pre-monsoon)"
    elif month in [6, 7, 8, 9]:
        return "JJAS (Monsoon)"
    elif month in [10, 11]:
        return "ON (Post-monsoon)"
    return "Unknown"

def compute_psi(source_arr, target_arr, num_bins=10):
    """Compute Population Stability Index (PSI) using source deciles."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(source_arr, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) < 3:
        return 0.0
        
    s_counts, _ = np.histogram(source_arr, bins=bin_edges)
    t_counts, _ = np.histogram(target_arr, bins=bin_edges)
    
    s_pct = s_counts / len(source_arr)
    t_pct = t_counts / len(target_arr)
    
    eps = 1e-4
    s_pct = np.where(s_pct == 0, eps, s_pct)
    t_pct = np.where(t_pct == 0, eps, t_pct)
    
    psi = np.sum((t_pct - s_pct) * np.log(t_pct / s_pct))
    return float(psi)

def compute_js_divergence(source_arr, target_arr, num_bins=50):
    """Compute Jensen-Shannon divergence over joint support range."""
    min_val = min(np.min(source_arr), np.min(target_arr))
    max_val = max(np.max(source_arr), np.max(target_arr))
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, num_bins + 1)
    p, _ = np.histogram(source_arr, bins=bins, density=True)
    q, _ = np.histogram(target_arr, bins=bins, density=True)
    p = p / np.sum(p) if np.sum(p) > 0 else np.ones(num_bins)/num_bins
    q = q / np.sum(q) if np.sum(q) > 0 else np.ones(num_bins)/num_bins
    # Add smoothing
    eps = 1e-6
    p = (p + eps) / np.sum(p + eps)
    q = (q + eps) / np.sum(q + eps)
    return float(jensenshannon(p, q, base=2))

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("TDAF STEP 3: COMPREHENSIVE DOMAIN SHIFT ANALYSIS & STRATEGY DESIGN")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 1. Load and Harmonize Datasets
    # ---------------------------------------------------------
    print("\n[1/8] Loading and harmonizing datasets...")
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    fl_valid = df_fl[df_fl["target_asos_strict_crit_c"].notna()].copy()
    fl_train = fl_valid[fl_valid["month"].between(1, 7)].copy()
    
    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH)
    df_ch["valid_time"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["month"] = df_ch["valid_time"].dt.month
    df_ch["year"] = df_ch["valid_time"].dt.year
    df_ch["season"] = df_ch["month"].apply(assign_season)
    fl_train["season"] = fl_train["month"].apply(assign_season)
    fl_valid["season"] = fl_valid["month"].apply(assign_season)
    
    df_ch_renamed = df_ch.rename(columns=FEATURE_MAPPING)
    
    print(f"  Florida Train Set: {len(fl_train)} rows")
    print(f"  Florida Full Set:  {len(fl_valid)} rows")
    print(f"  Chandipur Set:     {len(df_ch_renamed)} rows")
    
    # ---------------------------------------------------------
    # 2. Marginal Covariate Shift Metrics
    # ---------------------------------------------------------
    print("\n[2/8] Computing marginal covariate shift metrics...")
    marginal_rows = []
    
    for comp_name, src_df in [("Florida_Train_vs_Chandipur", fl_train), ("Florida_Full_vs_Chandipur", fl_valid)]:
        for feat in COMPATIBLE_FEATURES:
            s_vals = src_df[feat].values
            t_vals = df_ch_renamed[feat].values
            
            s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals, ddof=1))
            t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals, ddof=1))
            
            s_med = float(np.median(s_vals))
            t_med = float(np.median(t_vals))
            s_iqr = float(np.percentile(s_vals, 75) - np.percentile(s_vals, 25))
            t_iqr = float(np.percentile(t_vals, 75) - np.percentile(t_vals, 25))
            
            s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
            t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
            
            s_p1, s_p5, s_p95, s_p99 = [float(np.percentile(s_vals, p)) for p in [1, 5, 95, 99]]
            t_p1, t_p5, t_p95, t_p99 = [float(np.percentile(t_vals, p)) for p in [1, 5, 95, 99]]
            
            # Pooled std for SMD
            pooled_std = np.sqrt((s_std**2 + t_std**2) / 2.0)
            smd = (t_mean - s_mean) / pooled_std if pooled_std > 0 else 0.0
            
            # Distances
            w_dist = wasserstein_distance(s_vals, t_vals)
            ks_stat, ks_pval = ks_2samp(s_vals, t_vals)
            js_div = compute_js_divergence(s_vals, t_vals, num_bins=50)
            psi = compute_psi(s_vals, t_vals, num_bins=10)
            
            marginal_rows.append({
                "comparison": comp_name,
                "feature": feat,
                "feature_label": FEATURE_LABELS[feat],
                "source_mean": round(s_mean, 4),
                "source_std": round(s_std, 4),
                "target_mean": round(t_mean, 4),
                "target_std": round(t_std, 4),
                "source_median": round(s_med, 4),
                "source_iqr": round(s_iqr, 4),
                "target_median": round(t_med, 4),
                "target_iqr": round(t_iqr, 4),
                "source_min": round(s_min, 4),
                "source_max": round(s_max, 4),
                "target_min": round(t_min, 4),
                "target_max": round(t_max, 4),
                "source_p1": round(s_p1, 4),
                "source_p5": round(s_p5, 4),
                "source_p95": round(s_p95, 4),
                "source_p99": round(s_p99, 4),
                "target_p1": round(t_p1, 4),
                "target_p5": round(t_p5, 4),
                "target_p95": round(t_p95, 4),
                "target_p99": round(t_p99, 4),
                "smd_cohens_d": round(float(smd), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(ks_pval),
                "jensen_shannon_divergence": round(float(js_div), 4),
                "population_stability_index": round(float(psi), 4)
            })
            
    df_marginal = pd.DataFrame(marginal_rows)
    df_marginal.to_csv(OUTPUT_DIR / "marginal_shift_metrics.csv", index=False)
    print("  Saved marginal_shift_metrics.csv")
    
    # ---------------------------------------------------------
    # 3. Correlation Structure & Joint Shift
    # ---------------------------------------------------------
    print("\n[3/8] Computing correlation shift matrices...")
    fl_train_feat = fl_train[COMPATIBLE_FEATURES]
    ch_feat = df_ch_renamed[COMPATIBLE_FEATURES]
    
    corr_p_fl = fl_train_feat.corr(method="pearson")
    corr_p_ch = ch_feat.corr(method="pearson")
    diff_p = corr_p_ch - corr_p_fl
    
    corr_s_fl = fl_train_feat.corr(method="spearman")
    corr_s_ch = ch_feat.corr(method="spearman")
    diff_s = corr_s_ch - corr_s_fl
    
    corr_rows = []
    n_fl, n_ch = len(fl_train_feat), len(ch_feat)
    for i in range(len(COMPATIBLE_FEATURES)):
        for j in range(i + 1, len(COMPATIBLE_FEATURES)):
            f1 = COMPATIBLE_FEATURES[i]
            f2 = COMPATIBLE_FEATURES[j]
            
            rp_fl = corr_p_fl.loc[f1, f2]
            rp_ch = corr_p_ch.loc[f1, f2]
            delta_rp = diff_p.loc[f1, f2]
            
            rs_fl = corr_s_fl.loc[f1, f2]
            rs_ch = corr_s_ch.loc[f1, f2]
            delta_rs = diff_s.loc[f1, f2]
            
            # Fisher z-transformation test for Pearson correlation difference
            z_fl = np.arctanh(np.clip(rp_fl, -0.9999, 0.9999))
            z_ch = np.arctanh(np.clip(rp_ch, -0.9999, 0.9999))
            se_diff = np.sqrt(1.0 / (n_fl - 3) + 1.0 / (n_ch - 3))
            z_stat = (z_ch - z_fl) / se_diff
            
            corr_rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "pearson_source_train": round(float(rp_fl), 4),
                "pearson_target": round(float(rp_ch), 4),
                "delta_pearson": round(float(delta_rp), 4),
                "fisher_z_stat": round(float(z_stat), 2),
                "spearman_source_train": round(float(rs_fl), 4),
                "spearman_target": round(float(rs_ch), 4),
                "delta_spearman": round(float(delta_rs), 4)
            })
            
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(OUTPUT_DIR / "correlation_shift_metrics.csv", index=False)
    print("  Saved correlation_shift_metrics.csv")
    
    # ---------------------------------------------------------
    # 4. Seasonal Shift Analysis
    # ---------------------------------------------------------
    print("\n[4/8] Computing seasonal shift metrics...")
    seasons = ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]
    seasonal_rows = []
    
    for s in seasons:
        s_src = fl_valid[fl_valid["season"] == s]
        s_tgt = df_ch_renamed[df_ch_renamed["season"] == s]
        
        for feat in COMPATIBLE_FEATURES:
            if len(s_src) == 0 or len(s_tgt) == 0:
                continue
            src_vals = s_src[feat].values
            tgt_vals = s_tgt[feat].values
            
            s_mean, s_std = float(np.mean(src_vals)), float(np.std(src_vals, ddof=1))
            t_mean, t_std = float(np.mean(tgt_vals)), float(np.std(tgt_vals, ddof=1))
            s_med, t_med = float(np.median(src_vals)), float(np.median(tgt_vals))
            
            pooled_std = np.sqrt((s_std**2 + t_std**2) / 2.0)
            smd = (t_mean - s_mean) / pooled_std if pooled_std > 0 else 0.0
            w_dist = wasserstein_distance(src_vals, tgt_vals)
            ks_stat, _ = ks_2samp(src_vals, tgt_vals)
            
            seasonal_rows.append({
                "season": s,
                "feature": feat,
                "source_count": len(s_src),
                "target_count": len(s_tgt),
                "source_mean": round(s_mean, 4),
                "source_std": round(s_std, 4),
                "target_mean": round(t_mean, 4),
                "target_std": round(t_std, 4),
                "source_median": round(s_med, 4),
                "target_median": round(t_med, 4),
                "smd_cohens_d": round(float(smd), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "ks_statistic": round(float(ks_stat), 4)
            })
            
    df_seasonal = pd.DataFrame(seasonal_rows)
    df_seasonal.to_csv(OUTPUT_DIR / "seasonal_shift_metrics.csv", index=False)
    print("  Saved seasonal_shift_metrics.csv")
    
    # ---------------------------------------------------------
    # 5. Feature Support & Extrapolation Analysis
    # ---------------------------------------------------------
    print("\n[5/8] Computing feature support and extrapolation analysis...")
    support_rows = []
    
    for feat in COMPATIBLE_FEATURES:
        s_vals = fl_train[feat].values
        t_vals = df_ch_renamed[feat].values
        
        s_min, s_max = np.min(s_vals), np.max(s_vals)
        s_p1, s_p99 = np.percentile(s_vals, 1), np.percentile(s_vals, 99)
        
        n_below_min = int(np.sum(t_vals < s_min))
        n_above_max = int(np.sum(t_vals > s_max))
        n_outside_range = n_below_min + n_above_max
        pct_outside_range = float(n_outside_range / len(t_vals) * 100)
        
        n_below_p1 = int(np.sum(t_vals < s_p1))
        n_above_p99 = int(np.sum(t_vals > s_p99))
        n_outside_p1_p99 = n_below_p1 + n_above_p99
        pct_outside_p1_p99 = float(n_outside_p1_p99 / len(t_vals) * 100)
        
        support_rows.append({
            "feature": feat,
            "source_train_min": round(float(s_min), 4),
            "source_train_max": round(float(s_max), 4),
            "target_min": round(float(np.min(t_vals)), 4),
            "target_max": round(float(np.max(t_vals)), 4),
            "target_below_source_min_count": n_below_min,
            "target_above_source_max_count": n_above_max,
            "target_outside_source_range_count": n_outside_range,
            "target_outside_source_range_pct": round(pct_outside_range, 4),
            "source_train_p1": round(float(s_p1), 4),
            "source_train_p99": round(float(s_p99), 4),
            "target_outside_p1_p99_count": n_outside_p1_p99,
            "target_outside_p1_p99_pct": round(pct_outside_p1_p99, 4)
        })
        
    df_support = pd.DataFrame(support_rows)
    df_support.to_csv(OUTPUT_DIR / "feature_support_analysis.csv", index=False)
    print("  Saved feature_support_analysis.csv")
    
    # ---------------------------------------------------------
    # 6. Multivariate Domain Separation Diagnostic (Domain Classifier)
    # ---------------------------------------------------------
    print("\n[6/8] Training multivariate domain separation classifiers...")
    # Label source as 0, target as 1
    # To prevent seasonal confounding and cross-day leakage:
    # Use interleaved calendar day split (Even days for training, Odd days for testing).
    # This guarantees that all four seasons are equally represented in both train and test partitions.
    fl_valid["init_time"] = pd.to_datetime(fl_valid["forecast_init_time_utc"])
    fl_dc_train = fl_valid[fl_valid["init_time"].dt.day % 2 == 0][COMPATIBLE_FEATURES].copy()
    fl_dc_test = fl_valid[fl_valid["init_time"].dt.day % 2 == 1][COMPATIBLE_FEATURES].copy()
    
    ch_dc_train = df_ch_renamed[df_ch_renamed["valid_time"].dt.day % 2 == 0][COMPATIBLE_FEATURES].copy()
    ch_dc_test = df_ch_renamed[df_ch_renamed["valid_time"].dt.day % 2 == 1][COMPATIBLE_FEATURES].copy()
    
    X_dc_train = np.vstack([fl_dc_train.values, ch_dc_train.values])
    y_dc_train = np.hstack([np.zeros(len(fl_dc_train)), np.ones(len(ch_dc_train))])
    
    X_dc_test = np.vstack([fl_dc_test.values, ch_dc_test.values])
    y_dc_test = np.hstack([np.zeros(len(fl_dc_test)), np.ones(len(ch_dc_test))])
    
    # 1. Linear Domain Classifier (StandardScaler + LogisticRegression)
    scaler = StandardScaler()
    X_dc_train_scaled = scaler.fit_transform(X_dc_train)
    X_dc_test_scaled = scaler.transform(X_dc_test)
    
    clf_lr = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr.fit(X_dc_train_scaled, y_dc_train)
    prob_lr = clf_lr.predict_proba(X_dc_test_scaled)[:, 1]
    
    # 2. Non-linear Domain Classifier (LightGBM)
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    clf_lgb.fit(X_dc_train, y_dc_train)
    prob_lgb = clf_lgb.predict_proba(X_dc_test)[:, 1]
    
    dc_results = []
    for m_name, probs in [("Logistic Regression (Linear)", prob_lr), ("LightGBM (Non-linear)", prob_lgb)]:
        auc_roc = roc_auc_score(y_dc_test, probs)
        auc_pr = average_precision_score(y_dc_test, probs)
        pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_dc_test, pred)
        f1 = f1_score(y_dc_test, pred)
        prec = precision_score(y_dc_test, pred)
        rec = recall_score(y_dc_test, pred)
        tn, fp, fn, tp = confusion_matrix(y_dc_test, pred).ravel()
        
        dc_results.append({
            "model": m_name,
            "test_sample_count": len(y_dc_test),
            "target_domain_positives": int(np.sum(y_dc_test)),
            "roc_auc": round(float(auc_roc), 6),
            "pr_auc": round(float(auc_pr), 6),
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "interpretation": "Quantifies multivariate separability of Florida vs Chandipur feature distributions (P(Domain=Chandipur|X))"
        })
        
    df_dc = pd.DataFrame(dc_results)
    df_dc.to_csv(OUTPUT_DIR / "domain_classifier_results.csv", index=False)
    print("  Saved domain_classifier_results.csv")
    
    # ---------------------------------------------------------
    # 7. Adaptation Methods Comparative Assessment
    # ---------------------------------------------------------
    print("\n[7/8] Compiling adaptation methods comparative evaluation...")
    adaptation_rows = [
        {
            "method_name": "1. Feature Standardization & Normalization (Z-Score / RobustScaler)",
            "mathematical_principle": "Z_s = (X_s - μ_s)/σ_s; Z_t = (X_t - μ_t)/σ_t. Centers and rescales each feature independently to zero mean and unit variance per domain.",
            "shift_addressed": "Marginal Covariate Shift (Scale and offset disparities).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Yes (Linear affine transformations preserve exact additive term interpretability).",
            "compatibility_with_gam_ebm": "High (Directly compatible; transforms input coordinates before spline/stump evaluation).",
            "key_risks_and_limitations": "Does not address correlation shift or covariance cross-coupling; assumes feature shapes are identical across domains.",
            "recommended_evaluation": "Phase A baseline diagnostic."
        },
        {
            "method_name": "2. Correlation Alignment (CORAL)",
            "mathematical_principle": "Linear transformation A = C_s^{1/2} C_t^{-1/2} applied to target data (or reverse) to align second-order statistics (covariance matrices): C_adapted = C_target.",
            "shift_addressed": "Joint Covariate Shift (Covariance and linear correlation structure alignment).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Partial / Moderate (Linear mixing creates feature rotations; can be interpreted via transformed feature weights).",
            "compatibility_with_gam_ebm": "High (Transformed feature matrix X_coral can be fed into frozen GAM/EBM or re-trained on aligned source).",
            "key_risks_and_limitations": "Only aligns second-order moments; does not fix higher-order non-linear discrepancies or extreme boundary support gaps.",
            "recommended_evaluation": "Phase B core experiment."
        },
        {
            "method_name": "3. Zero-Phase Component Analysis (ZCA) Whitening",
            "mathematical_principle": "W_zca = V Λ^{-1/2} V^T. Whitens both source and target into decorrelated spherical spaces while minimizing Euclidean distortion to original features.",
            "shift_addressed": "Multivariate Covariate Shift (Removes multicollinearity and aligns principal axes).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Moderate (Preserves maximal correlation to original coordinate axes).",
            "compatibility_with_gam_ebm": "Moderate (GAM/EBM can operate on whitened coordinates).",
            "key_risks_and_limitations": "Sensible to eigenvalue truncation; extreme outliers in CAPE can distort spherical transformation.",
            "recommended_evaluation": "Phase B comparative variant."
        },
        {
            "method_name": "4. Quantile Mapping / Empirical CDF Matching",
            "mathematical_principle": "X_adapted = F_source^{-1}(F_target(X_target)). Non-linear monotonic warping aligning the marginal empirical CDF of target to source training range.",
            "shift_addressed": "Marginal Covariate Shift & Boundary Extrapolation (Completely eliminates out-of-distribution tail collapse).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "High (Monotonic rank-preserving transformation; retains physical variable order and preserves individual spline response).",
            "compatibility_with_gam_ebm": "Very High (Solves the pyGAM spline collapse by mapping extreme Chandipur CAPE back onto the Florida training knot support [0, 7240 J/kg]).",
            "key_risks_and_limitations": "Distorts physical units; operates marginally without altering the joint correlation structure.",
            "recommended_evaluation": "Phase D essential safety safeguard."
        },
        {
            "method_name": "5. Density-Ratio Importance Weighting (KMM / RuLSIF)",
            "mathematical_principle": "w(x) = P_target(x) / P_source(x). Re-weights source training loss during model fitting: L_adapted = Σ w(x_i) L(y_i, f(x_i)).",
            "shift_addressed": "Covariate Shift (Under strict assumption that conditional P(Y|X) is invariant).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "High (Model retains original functional form; only fitting sample weights change).",
            "compatibility_with_gam_ebm": "High for EBM (native sample_weight support); Moderate for pyGAM (requires sample-weighted PIRLS).",
            "key_risks_and_limitations": "Severe weight explosion when support does not overlap (w(x) -> ∞ for CAPE > 7,240 J/kg); high variance in weights.",
            "recommended_evaluation": "Phase C experiment (requires weight truncation / clipping)."
        },
        {
            "method_name": "6. Maximum Mean Discrepancy (MMD) Kernel Alignment",
            "mathematical_principle": "Minimizes RKHS distance between source and target empirical embeddings: D_MMD^2(P_s, P_t) = ||E[φ(X_s)] - E[φ(X_t)]||_H^2 via optimal transport or reweighting.",
            "shift_addressed": "Higher-Order Covariate Shift (Matches all statistical moments implicitly).",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Low if learning latent projections; High if used for Kernel Mean Matching (sample weights).",
            "compatibility_with_gam_ebm": "Moderate (Best implemented via KMM sample weighting for tabular Glassbox models).",
            "key_risks_and_limitations": "Computational scaling O(N^2) over large meteorological datasets (N=60,637 and N=23,376); choice of kernel bandwidth.",
            "recommended_evaluation": "Phase E advanced experiment."
        },
        {
            "method_name": "7. Deep CORAL (Deep Neural Network Covariance Loss)",
            "mathematical_principle": "Loss = L_classification(Y_source, f(X_source)) + λ L_coral(f(X_source), f(X_target)). Aligns deep feature covariances.",
            "shift_addressed": "Latent Feature Covariate Shift.",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Very Low (Replaces Glassbox GAM/EBM with black-box MLP / ResNet representations).",
            "compatibility_with_gam_ebm": "Incompatible (Requires deep neural network; cannot be integrated into native pyGAM or InterpretML EBM).",
            "key_risks_and_limitations": "Destroys exact explainability, which is a foundational requirement of TDAF; risk of overfitting on tabular data.",
            "recommended_evaluation": "Phase G exploratory comparison."
        },
        {
            "method_name": "8. Domain-Adversarial Neural Network (DANN)",
            "mathematical_principle": "Min-max adversarial game via Gradient Reversal Layer (GRL): feature extractor fools domain discriminator while minimizing source task loss.",
            "shift_addressed": "Non-linear Domain Invariance.",
            "requires_target_labels": "No (Strictly Unsupervised).",
            "preserves_interpretability": "Very Low (Latent representations are completely opaque).",
            "compatibility_with_gam_ebm": "Incompatible (Neural architecture only).",
            "key_risks_and_limitations": "Instability during minimax optimization; destroys meteorological physical interpretability.",
            "recommended_evaluation": "Phase G exploratory comparison."
        },
        {
            "method_name": "9. Target-Supervised Fine-Tuning (Proxy Labels)",
            "mathematical_principle": "Fine-tuning or retraining model directly on Chandipur storm score proxy labels (Y_proxy = I(storm_score >= 2.50)).",
            "shift_addressed": "Supervised Target Adaptation.",
            "requires_target_labels": "Yes (Requires Target Proxy Labels).",
            "preserves_interpretability": "High (Maintains Glassbox structure).",
            "compatibility_with_gam_ebm": "High.",
            "key_risks_and_limitations": "Circular validation failure! Optimizes model to fit a deterministic heuristic formula rather than real physical thunderstorms.",
            "recommended_evaluation": "Phase F strictly separated controlled experiment."
        },
        {
            "method_name": "10. Pseudo-Labeling / Confidence-Thresholded Self-Training",
            "mathematical_principle": "Iteratively assigns pseudo-labels to high-confidence target predictions (P > τ_high or P < τ_low) and adds them to the training set.",
            "shift_addressed": "Transductive Domain Adaptation.",
            "requires_target_labels": "No (Self-generated).",
            "preserves_interpretability": "High (Re-trains standard GAM/EBM).",
            "compatibility_with_gam_ebm": "High.",
            "key_risks_and_limitations": "Confirmation bias! If out-of-distribution cases (e.g. extreme CAPE in pyGAM) predict 0% probability, pseudo-labeling reinforces the false negative.",
            "recommended_evaluation": "Not recommended without prior quantile / extrapolation safeguards."
        }
    ]
    df_adapt = pd.DataFrame(adaptation_rows)
    df_adapt.to_csv(OUTPUT_DIR / "adaptation_method_comparison.csv", index=False)
    print("  Saved adaptation_method_comparison.csv")
    
    # ---------------------------------------------------------
    # 8. Verification & Reconciliation of CAPE Extrapolation
    # ---------------------------------------------------------
    print("\n[8/8] Reconciling and verifying exact EBM & pyGAM CAPE extrapolation...")
    gam_model = joblib.load(GAM_MODEL_PATH)
    ebm_model = joblib.load(EBM_MODEL_PATH)
    
    gam_sha = get_file_sha256(GAM_MODEL_PATH)
    ebm_sha = get_file_sha256(EBM_MODEL_PATH)
    
    fl_medians = fl_train[COMPATIBLE_FEATURES].median().to_dict()
    
    # Detailed sweep of CAPE from 0 to 10,000 J/kg
    cape_sweep = np.linspace(0, 10000, 500)
    sweep_df = pd.DataFrame({f: [fl_medians[f]] * len(cape_sweep) for f in COMPATIBLE_FEATURES})
    sweep_df["cape_J_kg"] = cape_sweep
    
    # EBM predictions:
    ebm_total_logits = ebm_model.decision_function(sweep_df[COMPATIBLE_FEATURES])
    ebm_total_probs = ebm_model.predict_proba(sweep_df[COMPATIBLE_FEATURES])[:, 1]
    
    # EBM marginal term score for CAPE:
    # Bin indices
    cape_bins = ebm_model.bins_[0][0]
    ebm_bin_indices = np.digitize(cape_sweep, cape_bins)
    ebm_marginal_scores = ebm_model.term_scores_[0][ebm_bin_indices]
    
    # pyGAM predictions:
    gam_total_logits = gam_model._linear_predictor(sweep_df[COMPATIBLE_FEATURES].values)
    gam_total_probs = gam_model.predict_proba(sweep_df[COMPATIBLE_FEATURES].values)
    gam_partial_dep = gam_model.partial_dependence(term=0, X=sweep_df[COMPATIBLE_FEATURES].values)
    
    # Benchmark check at exact checkpoints: 7240, 9906, 10000
    checkpoints = [7240, 9906, 10000]
    check_rows = []
    for cp in checkpoints:
        row = pd.DataFrame([{**fl_medians, "cape_J_kg": cp}])
        b_idx = np.digitize(cp, cape_bins)
        
        ebm_m_score = float(ebm_model.term_scores_[0][b_idx])
        ebm_t_logit = float(ebm_model.decision_function(row)[0])
        ebm_t_prob = float(ebm_model.predict_proba(row)[0, 1])
        
        gam_p_dep = float(gam_model.partial_dependence(term=0, X=row[COMPATIBLE_FEATURES].values)[0])
        gam_t_logit = float(gam_model._linear_predictor(row[COMPATIBLE_FEATURES].values)[0])
        gam_t_prob = float(gam_model.predict_proba(row[COMPATIBLE_FEATURES].values)[0])
        
        check_rows.append({
            "cape_checkpoint_J_kg": cp,
            "ebm_marginal_term_score": round(ebm_m_score, 4),
            "ebm_total_logit_at_medians": round(ebm_t_logit, 4),
            "ebm_total_prob_at_medians": round(ebm_t_prob, 4),
            "gam_partial_dependence_term": round(gam_p_dep, 4),
            "gam_total_logit_at_medians": round(gam_t_logit, 4),
            "gam_total_prob_at_medians": round(gam_t_prob, 4)
        })
    df_check = pd.DataFrame(check_rows)
    print("Checkpoint Verification Table:")
    print(df_check.to_string())
    
    # ---------------------------------------------------------
    # 9. Generate Publication-Grade Diagnostic Figures
    # ---------------------------------------------------------
    print("\n[9/9] Generating 8 diagnostic figures...")
    
    # Figure 1: Marginal Distribution Comparison (KDE & Boxplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, feat in enumerate(COMPATIBLE_FEATURES):
        ax = axes[idx]
        sns.kdeplot(fl_train[feat], label="Florida Train (2024)", color="#2563eb", fill=True, alpha=0.25, linewidth=2, ax=ax)
        sns.kdeplot(df_ch_renamed[feat], label="Chandipur (2022-2025)", color="#dc2626", fill=True, alpha=0.25, linewidth=2, ax=ax)
        ax.set_title(f"{FEATURE_LABELS[feat]}\nKS={df_marginal[(df_marginal['feature']==feat)&(df_marginal['comparison']=='Florida_Train_vs_Chandipur')]['ks_statistic'].values[0]:.3f} | SMD={df_marginal[(df_marginal['feature']==feat)&(df_marginal['comparison']=='Florida_Train_vs_Chandipur')]['smd_cohens_d'].values[0]:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "marginal_distribution_comparison.png", dpi=300)
    plt.close()
    
    # Figure 2: Correlation Difference Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.heatmap(corr_p_fl, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Source (Florida Train) Pearson Matrix", fontsize=12, fontweight="bold")
    sns.heatmap(corr_p_ch, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar=False)
    axes[1].set_title("Target (Chandipur) Pearson Matrix", fontsize=12, fontweight="bold")
    sns.heatmap(diff_p, annot=True, fmt=".2f", cmap="vlag", vmin=-0.8, vmax=0.8, ax=axes[2])
    axes[2].set_title("Correlation Difference (Target - Source)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_difference_heatmap.png", dpi=300)
    plt.close()
    
    # Figure 3: Joint Feature Distribution (Bivariate KDE / Scatter)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Subplot 1: CAPE vs CIN
    fl_sample = fl_train.sample(n=min(3000, len(fl_train)), random_state=42)
    ch_sample = df_ch_renamed.sample(n=min(3000, len(df_ch_renamed)), random_state=42)
    axes[0].scatter(fl_sample["cape_J_kg"], fl_sample["cin_J_kg"], color="#2563eb", alpha=0.3, s=15, label="Florida Train")
    axes[0].scatter(ch_sample["cape_J_kg"], ch_sample["cin_J_kg"], color="#dc2626", alpha=0.3, s=15, label="Chandipur")
    axes[0].axvline(7240, color="black", linestyle="--", label="FL Max CAPE (7,240)")
    axes[0].set_title("Joint Distribution: CAPE vs CIN", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Surface CAPE (J/kg)", fontsize=10)
    axes[0].set_ylabel("CIN (J/kg)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: CAPE vs Shear
    axes[1].scatter(fl_sample["cape_J_kg"], fl_sample["bulk_shear_850_300_ms"], color="#2563eb", alpha=0.3, s=15, label="Florida Train")
    axes[1].scatter(ch_sample["cape_J_kg"], ch_sample["bulk_shear_850_300_ms"], color="#dc2626", alpha=0.3, s=15, label="Chandipur")
    axes[1].axvline(7240, color="black", linestyle="--", label="FL Max CAPE (7,240)")
    axes[1].set_title("Joint Distribution: CAPE vs Deep Shear", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Surface CAPE (J/kg)", fontsize=10)
    axes[1].set_ylabel("Bulk Shear 850-300 hPa (m/s)", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: Wind 10m vs Wind 850hPa
    axes[2].scatter(fl_sample["wind_speed_10m_ms"], fl_sample["wind_speed_850hPa_ms"], color="#2563eb", alpha=0.3, s=15, label="Florida Train")
    axes[2].scatter(ch_sample["wind_speed_10m_ms"], ch_sample["wind_speed_850hPa_ms"], color="#dc2626", alpha=0.3, s=15, label="Chandipur")
    axes[2].set_title("Joint Distribution: Surface Wind vs 850hPa Wind", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("10m Wind Speed (m/s)", fontsize=10)
    axes[2].set_ylabel("850hPa Wind Speed (m/s)", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "joint_feature_distribution.png", dpi=300)
    plt.close()
    
    # Figure 4: PCA Domain Projection
    pca_scaler = StandardScaler()
    X_fl_scaled = pca_scaler.fit_transform(fl_train[COMPATIBLE_FEATURES].values)
    X_ch_scaled = pca_scaler.transform(df_ch_renamed[COMPATIBLE_FEATURES].values)
    
    pca = PCA(n_components=2)
    fl_pca = pca.fit_transform(X_fl_scaled)
    ch_pca = pca.transform(X_ch_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(fl_pca[::10, 0], fl_pca[::10, 1], color="#2563eb", alpha=0.25, s=12, label="Florida Train (10% Subsample)")
    plt.scatter(ch_pca[::4, 0], ch_pca[::4, 1], color="#dc2626", alpha=0.25, s=12, label="Chandipur Target (25% Subsample)")
    plt.title(f"PCA Domain Projection (2D)\nPC1 ({pca.explained_variance_ratio_[0]:.1%}) vs PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)", fontsize=11)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_domain_projection.png", dpi=300)
    plt.close()
    
    # Figure 5: Seasonal Domain Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    seasonal_order = ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]
    
    # CAPE across seasons
    sns.boxplot(data=pd.concat([
        fl_valid.assign(Domain="Florida Full"),
        df_ch_renamed.assign(Domain="Chandipur")
    ]), x="season", y="cape_J_kg", hue="Domain", order=seasonal_order, palette=["#2563eb", "#dc2626"], ax=axes[0])
    axes[0].set_title("Surface CAPE Distribution Across Seasons", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Season", fontsize=10)
    axes[0].set_ylabel("Surface CAPE (J/kg)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Bulk shear across seasons
    sns.boxplot(data=pd.concat([
        fl_valid.assign(Domain="Florida Full"),
        df_ch_renamed.assign(Domain="Chandipur")
    ]), x="season", y="bulk_shear_850_300_ms", hue="Domain", order=seasonal_order, palette=["#2563eb", "#dc2626"], ax=axes[1])
    axes[1].set_title("Bulk Shear (850-300 hPa) Distribution Across Seasons", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Season", fontsize=10)
    axes[1].set_ylabel("Bulk Shear (m/s)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Cloud cover across seasons
    sns.boxplot(data=pd.concat([
        fl_valid.assign(Domain="Florida Full"),
        df_ch_renamed.assign(Domain="Chandipur")
    ]), x="season", y="cloud_cover_fraction", hue="Domain", order=seasonal_order, palette=["#2563eb", "#dc2626"], ax=axes[2])
    axes[2].set_title("Total Cloud Cover Fraction Across Seasons", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Season", fontsize=10)
    axes[2].set_ylabel("Cloud Cover Fraction", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # 850 hPa wind across seasons
    sns.boxplot(data=pd.concat([
        fl_valid.assign(Domain="Florida Full"),
        df_ch_renamed.assign(Domain="Chandipur")
    ]), x="season", y="wind_speed_850hPa_ms", hue="Domain", order=seasonal_order, palette=["#2563eb", "#dc2626"], ax=axes[3])
    axes[3].set_title("850 hPa Wind Speed Across Seasons", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Season", fontsize=10)
    axes[3].set_ylabel("850 hPa Wind Speed (m/s)", fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_domain_comparison.png", dpi=300)
    plt.close()
    
    # Figure 6: Feature Support Overlap Bar Chart
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(COMPATIBLE_FEATURES))
    width = 0.35
    
    plt.bar(x_pos - width/2, df_support["target_outside_source_range_pct"], width, label="Target % Outside Source Training Range [Min, Max]", color="#ef4444")
    plt.bar(x_pos + width/2, df_support["target_outside_p1_p99_pct"], width, label="Target % Outside Source 1st-99th Percentiles", color="#f97316")
    plt.xticks(x_pos, [f.replace("_J_kg", "").replace("_ms", "").replace("_fraction", "") for f in COMPATIBLE_FEATURES], rotation=15, fontsize=10)
    plt.ylabel("Percentage of Chandipur Observations (%)", fontsize=11)
    plt.title("Chandipur Feature Support Deficit Relative to Florida Training Range", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_support_overlap.png", dpi=300)
    plt.close()
    
    # Figure 7: Domain Classifier ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_dc_test, prob_lr)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_dc_test, prob_lgb)
    
    axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_dc_test, prob_lr):.4f})", color="#2563eb", linewidth=2.5)
    axes[0].plot(fpr_lgb, tpr_lgb, label=f"LightGBM Non-linear (AUC = {roc_auc_score(y_dc_test, prob_lgb):.4f})", color="#059669", linewidth=2.5)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    axes[0].set_title("Multivariate Domain Separation: ROC Curves\n(Predicting Domain: Florida=0 vs Chandipur=1)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    p_lr, r_lr, _ = precision_recall_curve(y_dc_test, prob_lr)
    p_lgb, r_lgb, _ = precision_recall_curve(y_dc_test, prob_lgb)
    prev_dc = np.mean(y_dc_test)
    
    axes[1].plot(r_lr, p_lr, label=f"Logistic Regression (PR-AUC = {average_precision_score(y_dc_test, prob_lr):.4f})", color="#2563eb", linewidth=2.5)
    axes[1].plot(r_lgb, p_lgb, label=f"LightGBM Non-linear (PR-AUC = {average_precision_score(y_dc_test, prob_lgb):.4f})", color="#059669", linewidth=2.5)
    axes[1].axhline(prev_dc, color="gray", linestyle="--", label=f"Baseline Prevalence ({prev_dc:.1%})")
    axes[1].set_title("Domain Separation: Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "domain_classifier_roc.png", dpi=300)
    plt.close()
    
    # Figure 8: Verified CAPE Extrapolation (Reconciling Step 1 and Step 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Univariate Feature Term Score (Component Function) -> Explaining Step 1
    axes[0].plot(cape_sweep, gam_partial_dep, label="pyGAM Partial Dependence s(CAPE)", color="#2563eb", linewidth=2.8)
    axes[0].plot(cape_sweep, ebm_marginal_scores, label="EBM Marginal Term Score f(CAPE)", color="#059669", linewidth=2.8)
    axes[0].axvline(7240, color="#dc2626", linestyle="--", linewidth=1.8, label="FL Train Max (7,240 J/kg)")
    axes[0].axvline(9906, color="#7c3aed", linestyle="-.", linewidth=1.8, label="CH Target Max (9,906 J/kg)")
    
    axes[0].annotate("Step 1 Observation:\nEBM Term Clamps at +1.79 logit\n(Peak: +1.7868, Saturation: +1.6754)",
                     xy=(7240, 1.6754), xytext=(4500, 2.5),
                     arrowprops=dict(facecolor="#059669", shrink=0.08, width=2, headwidth=8),
                     fontsize=9, fontweight="bold", color="#065f46",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", alpha=0.9))
    
    axes[0].annotate("Step 1 Observation:\npyGAM Spline Dives to -4.36 logit\nat 9,906 J/kg (-4.51 at 10k)",
                     xy=(9906, -4.3633), xytext=(6500, -2.5),
                     arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=2, headwidth=8),
                     fontsize=9, fontweight="bold", color="#1e40af",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#eff6ff", ec="#2563eb", alpha=0.9))
    
    axes[0].set_title("Univariate Component Scores (Step 1 Focus)\nf_j(CAPE) and s_j(CAPE) in Logit Space", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[0].set_ylabel("Marginal Term Score (Log-Odds Contribution)", fontsize=11)
    axes[0].legend(loc="lower left", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Multivariate Total Predicted Probability at Medians -> Explaining Step 2
    axes[1].plot(cape_sweep, gam_total_probs, label="pyGAM Total Probability P(Y=1)", color="#2563eb", linewidth=2.8)
    axes[1].plot(cape_sweep, ebm_total_probs, label="EBM Total Probability P(Y=1)", color="#059669", linewidth=2.8)
    axes[1].axvline(7240, color="#dc2626", linestyle="--", linewidth=1.8, label="FL Train Max (7,240 J/kg)")
    axes[1].axvline(9906, color="#7c3aed", linestyle="-.", linewidth=1.8, label="CH Target Max (9,906 J/kg)")
    
    axes[1].annotate("Step 2 Observation:\nEBM Total Probability\nPlateaus at 0.3778 (37.8%)\n(Total Logit = -0.4989)",
                     xy=(8000, 0.3778), xytext=(5200, 0.22),
                     arrowprops=dict(facecolor="#059669", shrink=0.08, width=2, headwidth=8),
                     fontsize=9, fontweight="bold", color="#065f46",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", alpha=0.9))
    
    axes[1].annotate("Step 2 Observation:\npyGAM Total Probability\nCollapses to 0.0039 (0.39%)\nat 9,906 J/kg (0.0034 at 10k)",
                     xy=(9906, 0.0039), xytext=(6500, 0.50),
                     arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=2, headwidth=8),
                     fontsize=9, fontweight="bold", color="#1e40af",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#eff6ff", ec="#2563eb", alpha=0.9))
    
    axes[1].set_title("Full Model Prediction at Medians (Step 2 Focus)\nTotal Probability P(Y=1|CAPE, Medians)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[1].set_ylabel("Predicted Thunderstorm Probability", fontsize=11)
    axes[1].set_ylim([-0.02, 1.02])
    axes[1].legend(loc="upper left", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "verified_cape_extrapolation.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 10. Mirror to Secondary Directory
    # ---------------------------------------------------------
    print("\n[10/10] Mirroring outputs to Florida_Thunderstorm_2024_Training...")
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, MIRROR_DIR / f.name)
    print("  Mirroring complete.")
    
    print("\n" + "=" * 80)
    print("STEP 3 DOMAIN SHIFT ANALYSIS SCRIPT COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
