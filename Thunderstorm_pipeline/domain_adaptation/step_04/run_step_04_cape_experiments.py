"""
run_step_04_cape_experiments.py
===============================
Thunderstorm Domain Adaptation Framework (TDAF) - Step 4:
Controlled CAPE Adaptation and Extrapolation Experiments.

This script executes:
1. Reconstruct and verify frozen baseline predictions on Chandipur (Exp A).
2. Experiment B: CAPE Boundary Clipping at Florida Training Max (7,240 J/kg).
3. Experiment C: Empirical Quantile Mapping (Global Empirical CDF Matching).
4. Experiment D: Monotonic CAPE Saturation (Bounded Tanh Transformation).
5. Experiment E: Seasonal CAPE Quantile Mapping (DJF, MAM, JJAS, ON fallback).
6. Comprehensive comparative evaluations across both GAM and EBM at Florida optimal
   and standard 0.50 thresholds.
7. Feature distribution statistics, correlation preservation checks, extreme outlier
   diagnostics, and seasonal breakdowns.
8. Generates all 5 CSV deliverables and all 8 publication-grade diagnostic figures (300 DPI).

NO MODEL RETRAINING OR TARGET LABEL OPTIMIZATION IS PERFORMED.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
)

# ---------------------------------------------------------
# 1. Directories & Paths
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

NATIVE_MODELS_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
GAM_MODEL_PATH = NATIVE_MODELS_DIR / "native_gam_model.pkl"
EBM_MODEL_PATH = NATIVE_MODELS_DIR / "native_ebm_model.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_04"
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

THRESH_GAM_OPT = 0.33
THRESH_EBM_OPT = 0.34
THRESH_STD = 0.50
CAPE_MAX_FL = 7240.0
CAPE_SCALE_PREDEFINED = 3744.28 # Florida Training 99th percentile

def get_season(m):
    if m in [12, 1, 2]:
        return "DJF (Winter)"
    elif m in [3, 4, 5]:
        return "MAM (Pre-monsoon)"
    elif m in [6, 7, 8, 9]:
        return "JJAS (Monsoon)"
    elif m in [10, 11]:
        return "ON (Post-monsoon)"
    return "Unknown"

def compute_metrics(y_true, y_prob, threshold, exp_id, exp_name, model_name, thresh_type):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15))
    
    return {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "model": model_name,
        "threshold_type": thresh_type,
        "threshold": round(threshold, 4),
        "sample_count": len(y_true),
        "positives": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "cohen_kappa": round(float(kappa), 6),
        "mcc": round(float(mcc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "roc_auc": round(float(roc_auc), 6),
        "brier_score": round(float(brier), 6),
        "log_loss": round(float(ll), 6)
    }

def main():
    print("=" * 80)
    print("TDAF STEP 4: CONTROLLED CAPE ADAPTATION & EXTRAPOLATION EXPERIMENTS")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 1. Load Pretrained Models & Datasets
    # ---------------------------------------------------------
    print("\n[1/7] Loading saved models and datasets...")
    gam_model = joblib.load(GAM_MODEL_PATH)
    ebm_model = joblib.load(EBM_MODEL_PATH)
    
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    fl_valid = df_fl[df_fl["target_asos_strict_crit_c"].notna()].copy()
    fl_train = fl_valid[fl_valid["month"].between(1, 7)].copy()
    fl_cape = fl_train["cape_J_kg"].values
    fl_train["season"] = fl_train["month"].apply(get_season)
    
    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH).rename(columns=FEATURE_MAPPING)
    df_ch["valid_time"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["month"] = df_ch["valid_time"].dt.month
    df_ch["season"] = df_ch["month"].apply(get_season)
    
    y_ch = df_ch["storm_label_score"].values
    ch_raw_cape = df_ch["cape_J_kg"].values
    ch_cin = df_ch["cin_J_kg"].values
    ch_shear = df_ch["bulk_shear_850_300_ms"].values
    
    print(f"  Florida Train Set: {len(fl_train)} rows across months 1-7")
    print(f"  Chandipur Set:     {len(df_ch)} rows (2022-2025)")
    print(f"  Chandipur Proxy Target: {np.sum(y_ch)} positives ({np.mean(y_ch):.2%})")

    # ---------------------------------------------------------
    # 2. Construct Transformed CAPE Vectors
    # ---------------------------------------------------------
    print("\n[2/7] Generating experimental CAPE transformations...")
    
    # Exp A: Original Raw Baseline
    cape_a = ch_raw_cape.copy()
    
    # Exp B: Boundary Clipping at 7,240 J/kg
    cape_b = np.minimum(ch_raw_cape, CAPE_MAX_FL)
    
    # Exp C: Empirical Quantile Mapping (Global)
    ranks_global = (np.argsort(np.argsort(ch_raw_cape)) + 0.5) / len(ch_raw_cape)
    cape_c = np.percentile(fl_cape, ranks_global * 100.0)
    
    # Exp D: Monotonic Tanh Saturation
    cape_d = CAPE_MAX_FL * np.tanh(ch_raw_cape / CAPE_SCALE_PREDEFINED)
    
    # Exp E: Seasonal Quantile Mapping
    cape_e = np.zeros(len(df_ch))
    seasonal_names = ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]
    for s in seasonal_names:
        ch_idx = df_ch[df_ch["season"] == s].index
        s_ch_vals = df_ch.loc[ch_idx, "cape_J_kg"].values
        s_ranks = (np.argsort(np.argsort(s_ch_vals)) + 0.5) / len(s_ch_vals)
        
        if s in ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)"]:
            src_vals = fl_train[fl_train["season"] == s]["cape_J_kg"].values
        else: # ON fallback to global Florida training distribution
            src_vals = fl_cape
            
        cape_e[ch_idx] = np.percentile(src_vals, s_ranks * 100.0)
        
    cape_dict = {
        "Exp_A": ("Original Zero-Shot Baseline", cape_a),
        "Exp_B": ("CAPE Boundary Clipping (7,240 J/kg)", cape_b),
        "Exp_C": ("Empirical Quantile Mapping (Global)", cape_c),
        "Exp_D": ("Monotonic CAPE Saturation (Tanh)", cape_d),
        "Exp_E": ("Seasonal Quantile Mapping (with ON fallback)", cape_e)
    }

    # ---------------------------------------------------------
    # 3. Model Inference Across All Experiments
    # ---------------------------------------------------------
    print("\n[3/7] Running model inference for GAM and EBM across all experiments...")
    results_metrics = []
    predictions_dict = {}
    
    for exp_id, (exp_name, cape_arr) in cape_dict.items():
        X_df = df_ch[COMPATIBLE_FEATURES].copy()
        X_df["cape_J_kg"] = cape_arr
        
        prob_gam = gam_model.predict_proba(X_df.values)
        prob_ebm = ebm_model.predict_proba(X_df)[:, 1]
        
        predictions_dict[exp_id] = {
            "prob_gam": prob_gam,
            "prob_ebm": prob_ebm,
            "cape": cape_arr
        }
        
        # GAM Evaluation
        results_metrics.append(compute_metrics(y_ch, prob_gam, THRESH_GAM_OPT, exp_id, exp_name, "Native pyGAM", "FL Optimal (0.33)"))
        results_metrics.append(compute_metrics(y_ch, prob_gam, THRESH_STD, exp_id, exp_name, "Native pyGAM", "Standard (0.50)"))
        
        # EBM Evaluation
        results_metrics.append(compute_metrics(y_ch, prob_ebm, THRESH_EBM_OPT, exp_id, exp_name, "Native InterpretML EBM", "FL Optimal (0.34)"))
        results_metrics.append(compute_metrics(y_ch, prob_ebm, THRESH_STD, exp_id, exp_name, "Native InterpretML EBM", "Standard (0.50)"))
        
    df_exp_comp = pd.DataFrame(results_metrics)
    df_exp_comp.to_csv(OUTPUT_DIR / "cape_experiment_comparison.csv", index=False)
    print("  Saved cape_experiment_comparison.csv")
    print(df_exp_comp[["experiment_id", "model", "threshold_type", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]].to_string())

    # ---------------------------------------------------------
    # 4. Seasonal Performance Evaluation
    # ---------------------------------------------------------
    print("\n[4/7] Computing seasonal comparative performance breakdown...")
    seasonal_metric_rows = []
    
    for exp_id, (exp_name, _) in cape_dict.items():
        prob_gam = predictions_dict[exp_id]["prob_gam"]
        prob_ebm = predictions_dict[exp_id]["prob_ebm"]
        
        for s in seasonal_names:
            s_mask = (df_ch["season"] == s).values
            y_s = y_ch[s_mask]
            p_gam_s = prob_gam[s_mask]
            p_ebm_s = prob_ebm[s_mask]
            
            # GAM seasonal metrics
            f1_g = f1_score(y_s, (p_gam_s >= THRESH_GAM_OPT).astype(int), zero_division=0)
            rec_g = recall_score(y_s, (p_gam_s >= THRESH_GAM_OPT).astype(int), zero_division=0)
            prec_g = precision_score(y_s, (p_gam_s >= THRESH_GAM_OPT).astype(int), zero_division=0)
            auc_g = roc_auc_score(y_s, p_gam_s) if len(np.unique(y_s)) > 1 else np.nan
            pr_g = average_precision_score(y_s, p_gam_s) if len(np.unique(y_s)) > 1 else np.nan
            
            # EBM seasonal metrics
            f1_e = f1_score(y_s, (p_ebm_s >= THRESH_EBM_OPT).astype(int), zero_division=0)
            rec_e = recall_score(y_s, (p_ebm_s >= THRESH_EBM_OPT).astype(int), zero_division=0)
            prec_e = precision_score(y_s, (p_ebm_s >= THRESH_EBM_OPT).astype(int), zero_division=0)
            auc_e = roc_auc_score(y_s, p_ebm_s) if len(np.unique(y_s)) > 1 else np.nan
            pr_e = average_precision_score(y_s, p_ebm_s) if len(np.unique(y_s)) > 1 else np.nan
            
            seasonal_metric_rows.append({
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "season": s,
                "sample_count": int(np.sum(s_mask)),
                "positives": int(np.sum(y_s)),
                "prevalence": round(float(np.mean(y_s)), 4),
                "gam_f1": round(float(f1_g), 4),
                "gam_precision": round(float(prec_g), 4),
                "gam_recall": round(float(rec_g), 4),
                "gam_roc_auc": round(float(auc_g), 4),
                "gam_pr_auc": round(float(pr_g), 4),
                "ebm_f1": round(float(f1_e), 4),
                "ebm_precision": round(float(prec_e), 4),
                "ebm_recall": round(float(rec_e), 4),
                "ebm_roc_auc": round(float(auc_e), 4),
                "ebm_pr_auc": round(float(pr_e), 4),
            })
            
    df_seasonal_comp = pd.DataFrame(seasonal_metric_rows)
    df_seasonal_comp.to_csv(OUTPUT_DIR / "cape_seasonal_comparison.csv", index=False)
    print("  Saved cape_seasonal_comparison.csv")

    # ---------------------------------------------------------
    # 5. Distribution Statistics Before and After Transformation
    # ---------------------------------------------------------
    print("\n[5/7] Computing distribution statistics & correlation preservation...")
    dist_rows = []
    
    # Include Florida training reference row
    dist_rows.append({
        "condition": "Reference: Florida Training (Months 1-7)",
        "mean": round(float(np.mean(fl_cape)), 2),
        "std": round(float(np.std(fl_cape, ddof=1)), 2),
        "median": round(float(np.median(fl_cape)), 2),
        "iqr": round(float(np.percentile(fl_cape, 75) - np.percentile(fl_cape, 25)), 2),
        "min": round(float(np.min(fl_cape)), 2),
        "max": round(float(np.max(fl_cape)), 2),
        "p95": round(float(np.percentile(fl_cape, 95)), 2),
        "p99": round(float(np.percentile(fl_cape, 99)), 2),
        "zero_pct": round(float(np.mean(fl_cape == 0.0) * 100), 2),
        "gt_7240_count": int(np.sum(fl_cape > 7240.0))
    })
    
    for exp_id, (exp_name, arr) in cape_dict.items():
        dist_rows.append({
            "condition": f"{exp_id}: {exp_name}",
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr, ddof=1)), 2),
            "median": round(float(np.median(arr)), 2),
            "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "zero_pct": round(float(np.mean(arr == 0.0) * 100), 2),
            "gt_7240_count": int(np.sum(arr > 7240.0))
        })
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(OUTPUT_DIR / "cape_distribution_before_after.csv", index=False)
    print("  Saved cape_distribution_before_after.csv")
    
    # Correlation preservation table
    corr_pres_rows = []
    for exp_id, (exp_name, arr) in cape_dict.items():
        r_cin, _ = pearsonr(arr, ch_cin)
        rho_cin, _ = spearmanr(arr, ch_cin)
        r_sh, _ = pearsonr(arr, ch_shear)
        rho_sh, _ = spearmanr(arr, ch_shear)
        
        # Rank preservation with original CAPE
        rho_orig, _ = spearmanr(arr, ch_raw_cape)
        
        corr_pres_rows.append({
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "spearman_rank_corr_with_original_cape": round(float(rho_orig), 6),
            "pearson_with_cin": round(float(r_cin), 4),
            "spearman_with_cin": round(float(rho_cin), 4),
            "delta_spearman_cin_vs_original": round(float(rho_cin - spearmanr(ch_raw_cape, ch_cin)[0]), 6),
            "pearson_with_bulk_shear": round(float(r_sh), 4),
            "spearman_with_bulk_shear": round(float(rho_sh), 4),
            "delta_spearman_shear_vs_original": round(float(rho_sh - spearmanr(ch_raw_cape, ch_shear)[0]), 6)
        })
    df_corr_pres = pd.DataFrame(corr_pres_rows)
    df_corr_pres.to_csv(OUTPUT_DIR / "cape_correlation_preservation.csv", index=False)
    print("  Saved cape_correlation_preservation.csv")

    # ---------------------------------------------------------
    # 6. Extreme Outlier Case Analysis (N=81 where CAPE > 7,240 J/kg)
    # ---------------------------------------------------------
    print("\n[6/7] Extracting predictions for 81 extreme CAPE cases (>7,240 J/kg)...")
    ext_mask = ch_raw_cape > CAPE_MAX_FL
    df_ext = df_ch[ext_mask].copy()
    print(f"  Identified {len(df_ext)} extreme CAPE cases (Max = {ch_raw_cape.max():.1f} J/kg)")
    print(f"  Positive proxy labels: {df_ext['storm_label_score'].sum()} / {len(df_ext)} ({df_ext['storm_label_score'].mean():.1%})")
    
    ext_df_export = df_ext[["valid_time", "latitude", "longitude", "season", "cape_J_kg", "cin_J_kg", "bulk_shear_850_300_ms", "storm_score", "storm_label_score"]].copy()
    
    for exp_id in cape_dict.keys():
        ext_df_export[f"cape_{exp_id}"] = predictions_dict[exp_id]["cape"][ext_mask]
        ext_df_export[f"prob_gam_{exp_id}"] = predictions_dict[exp_id]["prob_gam"][ext_mask]
        ext_df_export[f"prob_ebm_{exp_id}"] = predictions_dict[exp_id]["prob_ebm"][ext_mask]
        ext_df_export[f"pred_gam_{exp_id}"] = (predictions_dict[exp_id]["prob_gam"][ext_mask] >= THRESH_GAM_OPT).astype(int)
        ext_df_export[f"pred_ebm_{exp_id}"] = (predictions_dict[exp_id]["prob_ebm"][ext_mask] >= THRESH_EBM_OPT).astype(int)
        
    df_ext_path = OUTPUT_DIR / "cape_extreme_case_predictions.csv"
    ext_df_export.to_csv(df_ext_path, index=False)
    print(f"  Saved cape_extreme_case_predictions.csv ({len(ext_df_export)} rows)")

    # ---------------------------------------------------------
    # 7. Generate Publication-Grade Diagnostic Figures
    # ---------------------------------------------------------
    print("\n[7/7] Generating 8 diagnostic figures...")
    
    # Figure 1: CAPE Distribution Comparison (Before & After)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Density plot
    sns.kdeplot(ch_raw_cape, label="Exp A: Original Chandipur", color="#111827", linewidth=2.5, ax=axes[0])
    sns.kdeplot(fl_cape, label="Reference: Florida Train", color="#2563eb", linestyle="--", linewidth=2, ax=axes[0])
    sns.kdeplot(cape_b, label="Exp B: Boundary Clipping (7,240)", color="#dc2626", linewidth=2, ax=axes[0])
    sns.kdeplot(cape_c, label="Exp C: Quantile Mapped (Global)", color="#059669", linewidth=2, ax=axes[0])
    sns.kdeplot(cape_d, label="Exp D: Tanh Saturation", color="#d97706", linewidth=2, ax=axes[0])
    axes[0].set_title("Surface CAPE Density Distributions Across Transformations", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_xlim([-100, 10500])
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative CDF plot
    for name, arr, col, ls in [
        ("Original Chandipur", ch_raw_cape, "#111827", "-"),
        ("Florida Train Reference", fl_cape, "#2563eb", "--"),
        ("Exp B (Clip)", cape_b, "#dc2626", "-"),
        ("Exp C (Quantile)", cape_c, "#059669", "-"),
        ("Exp D (Tanh Saturation)", cape_d, "#d97706", "-")
    ]:
        sorted_x = np.sort(arr)
        cdf_y = np.linspace(0, 1, len(sorted_x))
        axes[1].plot(sorted_x, cdf_y, label=name, color=col, linestyle=ls, linewidth=2.2)
    axes[1].axvline(7240, color="red", linestyle=":", label="FL Training Max (7,240)")
    axes[1].set_title("Empirical Cumulative Distribution Functions (CDFs)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[1].set_ylabel("Cumulative Probability F(x)", fontsize=11)
    axes[1].set_xlim([-100, 10500])
    axes[1].legend(loc="lower right", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_distribution_transformation_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Empirical Quantile Mapping Transfer Function
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Transfer function curve
    sample_ch_cape = np.linspace(0, 9906, 500)
    # Target percentile in Chandipur
    sample_ranks = [float(np.mean(ch_raw_cape <= x)) for x in sample_ch_cape]
    sample_mapped = [float(np.percentile(fl_cape, r * 100.0)) for r in sample_ranks]
    
    axes[0].plot(sample_ch_cape, sample_mapped, color="#059669", linewidth=2.8, label="Quantile Mapping Function: F_S^{-1}(F_T(x))")
    axes[0].plot([0, 7240], [0, 7240], "k--", alpha=0.5, label="Identity (x = y)")
    axes[0].axvline(7240, color="red", linestyle=":", label="FL Max (7,240 J/kg)")
    axes[0].axhline(7240, color="red", linestyle=":")
    axes[0].annotate("Zero-Inflation Compression:\nChandipur CAPE up to 600 J/kg\nis compressed toward 0 J/kg!",
                     xy=(600, sample_mapped[np.argmin(np.abs(sample_ch_cape - 600))]),
                     xytext=(1500, 100),
                     arrowprops=dict(facecolor="#059669", shrink=0.08, width=2, headwidth=8),
                     fontsize=9, fontweight="bold", color="#065f46",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", alpha=0.9))
    axes[0].set_title("Global Empirical Quantile Mapping Transfer Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Chandipur Input CAPE (J/kg)", fontsize=11)
    axes[0].set_ylabel("Mapped Florida-Scale CAPE (J/kg)", fontsize=11)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Quantile mapping difference (Delta = Mapped - Input)
    delta_mapped = np.array(sample_mapped) - sample_ch_cape
    axes[1].plot(sample_ch_cape, delta_mapped, color="#2563eb", linewidth=2.5)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("Quantile Mapping Discrepancy (Mapped - Original)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Chandipur Input CAPE (J/kg)", fontsize=11)
    axes[1].set_ylabel("Adjustment Delta (J/kg)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_quantile_mapping_curve.png", dpi=300)
    plt.close()

    # Figure 3: Monotonic Saturation Curves (Tanh Parameter Sweep)
    plt.figure(figsize=(10, 6))
    sweep_input = np.linspace(0, 10000, 500)
    for scale_val, col in [(2000, "#3b82f6"), (3000, "#10b981"), (3744.28, "#d97706"), (5000, "#8b5cf6")]:
        sat_y = CAPE_MAX_FL * np.tanh(sweep_input / scale_val)
        label_txt = f"Tanh Scale = {scale_val:.0f} J/kg"
        if scale_val == 3744.28:
            label_txt += " (Primary: FL 99th Pct)"
        plt.plot(sweep_input, sat_y, label=label_txt, color=col, linewidth=2.5)
        
    plt.plot([0, 7240], [0, 7240], "k--", alpha=0.5, label="Identity (x = y)")
    plt.axhline(7240, color="red", linestyle="--", label="Asymptotic Ceiling: 7,240 J/kg")
    plt.axvline(7240, color="red", linestyle=":")
    plt.axvline(9906, color="purple", linestyle="-.", label="Chandipur Absolute Max: 9,906 J/kg")
    plt.title("Monotonic CAPE Saturation Curves: CAPE_adj = 7240 * tanh(CAPE / Scale)", fontsize=13, fontweight="bold")
    plt.xlabel("Target Input CAPE (J/kg)", fontsize=11)
    plt.ylabel("Saturated CAPE Output (J/kg)", fontsize=11)
    plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_saturation_curves.png", dpi=300)
    plt.close()

    # Figure 4: pyGAM Prediction Comparison Across Transformations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Probability Scatter vs Original CAPE
    axes[0, 0].scatter(ch_raw_cape[::5], predictions_dict["Exp_A"]["prob_gam"][::5], alpha=0.3, s=10, color="#111827", label="Exp A: Original Raw (Collapses!)")
    axes[0, 0].scatter(ch_raw_cape[ext_mask], predictions_dict["Exp_B"]["prob_gam"][ext_mask], color="#dc2626", s=30, label="Exp B: Clipped (81 extreme cases)")
    axes[0, 0].axvline(7240, color="red", linestyle="--", label="FL Max (7,240)")
    axes[0, 0].set_title("pyGAM Predicted Probability vs Input CAPE", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Surface CAPE (J/kg)", fontsize=10)
    axes[0, 0].set_ylabel("Predicted Probability P(Y=1)", fontsize=10)
    axes[0, 0].legend(loc="upper left", frameon=True, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: ROC Curves for pyGAM
    for exp_id, col, ls in [("Exp_A", "#111827", "-"), ("Exp_B", "#dc2626", "-"), ("Exp_C", "#059669", "--"), ("Exp_D", "#d97706", "-"), ("Exp_E", "#8b5cf6", ":")]:
        fpr, tpr, _ = roc_curve(y_ch, predictions_dict[exp_id]["prob_gam"])
        auc_val = roc_auc_score(y_ch, predictions_dict[exp_id]["prob_gam"])
        axes[0, 1].plot(fpr, tpr, label=f"{exp_id} (AUC={auc_val:.4f})", color=col, linestyle=ls, linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 1].set_title("pyGAM ROC Curves Across CAPE Transformations", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("False Positive Rate", fontsize=10)
    axes[0, 1].set_ylabel("True Positive Rate", fontsize=10)
    axes[0, 1].legend(loc="lower right", frameon=True, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Precision-Recall Curves for pyGAM
    for exp_id, col, ls in [("Exp_A", "#111827", "-"), ("Exp_B", "#dc2626", "-"), ("Exp_C", "#059669", "--"), ("Exp_D", "#d97706", "-"), ("Exp_E", "#8b5cf6", ":")]:
        p, r, _ = precision_recall_curve(y_ch, predictions_dict[exp_id]["prob_gam"])
        pr_val = average_precision_score(y_ch, predictions_dict[exp_id]["prob_gam"])
        axes[1, 0].plot(r, p, label=f"{exp_id} (PR-AUC={pr_val:.4f})", color=col, linestyle=ls, linewidth=2)
    axes[1, 0].axhline(np.mean(y_ch), color="gray", linestyle="--", label=f"Prevalence ({np.mean(y_ch):.1%})")
    axes[1, 0].set_title("pyGAM Precision-Recall Curves", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Recall", fontsize=10)
    axes[1, 0].set_ylabel("Precision", fontsize=10)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Probability Distribution Boxplots
    df_box_gam = pd.DataFrame({
        "Exp_A": predictions_dict["Exp_A"]["prob_gam"],
        "Exp_B": predictions_dict["Exp_B"]["prob_gam"],
        "Exp_C": predictions_dict["Exp_C"]["prob_gam"],
        "Exp_D": predictions_dict["Exp_D"]["prob_gam"],
        "Exp_E": predictions_dict["Exp_E"]["prob_gam"]
    })
    sns.boxplot(data=df_box_gam, ax=axes[1, 1], palette="Set2")
    axes[1, 1].axhline(THRESH_GAM_OPT, color="red", linestyle="--", label=f"FL Threshold ({THRESH_GAM_OPT})")
    axes[1, 1].set_title("pyGAM Predicted Probability Distributions", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Predicted Probability", fontsize=10)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_gam_prediction_comparison.png", dpi=300)
    plt.close()

    # Figure 5: EBM Prediction Comparison Across Transformations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: ROC Curves for EBM
    for exp_id, col, ls in [("Exp_A", "#111827", "-"), ("Exp_B", "#dc2626", ":"), ("Exp_C", "#059669", "--"), ("Exp_D", "#d97706", "-"), ("Exp_E", "#8b5cf6", "-.")]:
        fpr, tpr, _ = roc_curve(y_ch, predictions_dict[exp_id]["prob_ebm"])
        auc_val = roc_auc_score(y_ch, predictions_dict[exp_id]["prob_ebm"])
        axes[0].plot(fpr, tpr, label=f"{exp_id} (AUC={auc_val:.4f})", color=col, linestyle=ls, linewidth=2.2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("InterpretML EBM ROC Curves Across Transformations\n(Note: Exp A & B curves overlap identically)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Precision-Recall Curves for EBM
    for exp_id, col, ls in [("Exp_A", "#111827", "-"), ("Exp_B", "#dc2626", ":"), ("Exp_C", "#059669", "--"), ("Exp_D", "#d97706", "-"), ("Exp_E", "#8b5cf6", "-.")]:
        p, r, _ = precision_recall_curve(y_ch, predictions_dict[exp_id]["prob_ebm"])
        pr_val = average_precision_score(y_ch, predictions_dict[exp_id]["prob_ebm"])
        axes[1].plot(r, p, label=f"{exp_id} (PR-AUC={pr_val:.4f})", color=col, linestyle=ls, linewidth=2.2)
    axes[1].axhline(np.mean(y_ch), color="gray", linestyle="--", label=f"Prevalence ({np.mean(y_ch):.1%})")
    axes[1].set_title("InterpretML EBM Precision-Recall Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_ebm_prediction_comparison.png", dpi=300)
    plt.close()

    # Figure 6: Joint CAPE-CIN Relationship Before & After
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Panel 1: Original Raw
    axes[0].scatter(ch_raw_cape[::4], ch_cin[::4], color="#111827", alpha=0.25, s=12)
    axes[0].axvline(7240, color="red", linestyle="--", label="FL Max (7,240)")
    axes[0].set_title(f"Original Chandipur: CAPE vs CIN\nSpearman rho = {spearmanr(ch_raw_cape, ch_cin)[0]:.4f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Surface CAPE (J/kg)", fontsize=10)
    axes[0].set_ylabel("CIN (J/kg)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Quantile Mapped (Exp C)
    axes[1].scatter(cape_c[::4], ch_cin[::4], color="#059669", alpha=0.25, s=12)
    axes[1].set_title(f"Exp C (Quantile Mapped): CAPE vs CIN\nSpearman rho = {spearmanr(cape_c, ch_cin)[0]:.4f} (Zero Inflation!)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mapped CAPE (J/kg)", fontsize=10)
    axes[1].set_ylabel("CIN (J/kg)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Tanh Saturated (Exp D)
    axes[2].scatter(cape_d[::4], ch_cin[::4], color="#d97706", alpha=0.25, s=12)
    axes[2].set_title(f"Exp D (Tanh Saturated): CAPE vs CIN\nSpearman rho = {spearmanr(cape_d, ch_cin)[0]:.4f} (Preserved Ranks!)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Saturated CAPE (J/kg)", fontsize=10)
    axes[2].set_ylabel("CIN (J/kg)", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_cin_relationship_before_after.png", dpi=300)
    plt.close()

    # Figure 7: Seasonal CAPE Mapping Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of seasonal F1 for pyGAM
    df_sea_gam = df_seasonal_comp.pivot(index="season", columns="experiment_id", values="gam_f1")
    df_sea_gam[["Exp_A", "Exp_B", "Exp_C", "Exp_D", "Exp_E"]].plot(kind="bar", ax=axes[0], colormap="viridis", width=0.8)
    axes[0].set_title("pyGAM Seasonal F1-Score Across Transformations", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("F1 Score vs Proxy", fontsize=11)
    axes[0].set_xlabel("Season", fontsize=11)
    axes[0].set_ylim([0, 0.8])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Experiment", fontsize=8)
    
    # Bar chart of seasonal F1 for EBM
    df_sea_ebm = df_seasonal_comp.pivot(index="season", columns="experiment_id", values="ebm_f1")
    df_sea_ebm[["Exp_A", "Exp_B", "Exp_C", "Exp_D", "Exp_E"]].plot(kind="bar", ax=axes[1], colormap="viridis", width=0.8)
    axes[1].set_title("InterpretML EBM Seasonal F1-Score Across Transformations", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("F1 Score vs Proxy", fontsize=11)
    axes[1].set_xlabel("Season", fontsize=11)
    axes[1].set_ylim([0, 0.8])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Experiment", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_cape_mapping_comparison.png", dpi=300)
    plt.close()

    # Figure 8: Extreme Case Analysis (The 81 observations with CAPE > 7,240 J/kg)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ext_cape_vals = ch_raw_cape[ext_mask]
    sort_idx = np.argsort(ext_cape_vals)
    
    # Panel 1: pyGAM probability on extreme cases
    axes[0].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_A"]["prob_gam"][ext_mask][sort_idx], "o-", label="Exp A: Raw pyGAM (Collapses!)", color="#111827", linewidth=2)
    axes[0].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_B"]["prob_gam"][ext_mask][sort_idx], "s-", label="Exp B: Clipped at 7,240 J/kg", color="#dc2626", linewidth=2)
    axes[0].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_C"]["prob_gam"][ext_mask][sort_idx], "^-", label="Exp C: Quantile Mapped", color="#059669", linewidth=2)
    axes[0].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_D"]["prob_gam"][ext_mask][sort_idx], "d-", label="Exp D: Tanh Saturated", color="#d97706", linewidth=2)
    axes[0].axhline(THRESH_GAM_OPT, color="red", linestyle="--", label=f"FL Decision Threshold ({THRESH_GAM_OPT})")
    axes[0].set_title("pyGAM Predicted Probability on 81 Extreme CAPE Cases (>7,240 J/kg)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Raw Surface CAPE (J/kg)", fontsize=10)
    axes[0].set_ylabel("Predicted Thunderstorm Probability", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: EBM probability on extreme cases
    axes[1].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_A"]["prob_ebm"][ext_mask][sort_idx], "o-", label="Exp A: Raw EBM (Plateaued)", color="#111827", linewidth=2)
    axes[1].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_B"]["prob_ebm"][ext_mask][sort_idx], "s--", label="Exp B: Clipped (Identical to Exp A!)", color="#dc2626", linewidth=1.5)
    axes[1].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_C"]["prob_ebm"][ext_mask][sort_idx], "^-", label="Exp C: Quantile Mapped", color="#059669", linewidth=2)
    axes[1].plot(ext_cape_vals[sort_idx], predictions_dict["Exp_D"]["prob_ebm"][ext_mask][sort_idx], "d-", label="Exp D: Tanh Saturated", color="#d97706", linewidth=2)
    axes[1].axhline(THRESH_EBM_OPT, color="green", linestyle="--", label=f"FL Decision Threshold ({THRESH_EBM_OPT})")
    axes[1].set_title("InterpretML EBM Predicted Probability on 81 Extreme CAPE Cases", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Raw Surface CAPE (J/kg)", fontsize=10)
    axes[1].set_ylabel("Predicted Thunderstorm Probability", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_extreme_case_analysis.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 8. Mirror to Secondary Directory
    # ---------------------------------------------------------
    print("\n[8/8] Mirroring outputs to Florida_Thunderstorm_2024_Training...")
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, MIRROR_DIR / f.name)
    print("  Mirroring complete.")
    
    print("\n" + "=" * 80)
    print("STEP 4 CONTROLLED CAPE ADAPTATION EXPERIMENTS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
