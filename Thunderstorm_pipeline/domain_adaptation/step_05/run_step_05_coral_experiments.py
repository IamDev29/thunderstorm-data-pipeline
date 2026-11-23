"""
run_step_05_coral_experiments.py
================================
Thunderstorm Domain Adaptation Framework (TDAF) - Step 5:
Correlation Alignment (CORAL) and Multivariate Feature Adaptation.

This script executes:
1. Reconstruct frozen reference and original Step 2 & Step 4 baselines.
2. Condition A: Original zero-shot baseline (unmodified).
3. Condition B: CAPE clipping only (CAPE <= 7240 J/kg).
4. Condition C: Standard CORAL (Target-to-Source, unconstrained).
5. Condition D: CORAL followed by Physical Boundary Clamping (CAPE in [0, 7240], CIN >= 0, Wind >= 0, Cloud in [0, 1]).
6. Condition E: Regularized Covariance CORAL (Ridge shrinkage lambda = 0.05).
7. Condition F: Partial Alignment CORAL (alpha in [0.25, 0.50, 0.75, 1.00]).
8. Condition G: Seasonal CORAL (DJF, MAM, JJAS, with ON fallback).
9. Multivariate domain alignment diagnostics:
   - Frobenius covariance distances
   - Correlation difference matrices
   - Wasserstein distribution distances
   - Domain Classifier ROC-AUC before and after CORAL
10. Model evaluations across pyGAM and InterpretML EBM at frozen thresholds (0.33, 0.34, 0.50).
11. Interpretability and extreme case inspections.
12. Generates all 5 CSV deliverables and 8 publication-grade diagnostic figures (300 DPI).

NO MODEL RETRAINING OR TARGET-SUPERVISED TUNING IS PERFORMED.
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

from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
)

# ---------------------------------------------------------
# 1. Directories & Configuration
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

NATIVE_MODELS_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
GAM_MODEL_PATH = NATIVE_MODELS_DIR / "native_gam_model.pkl"
EBM_MODEL_PATH = NATIVE_MODELS_DIR / "native_ebm_model.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_05"
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

THRESH_GAM_OPT = 0.33
THRESH_EBM_OPT = 0.34
THRESH_STD = 0.50
CAPE_MAX_FL = 7240.0

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

def matrix_power(C, p, min_eval=1e-8):
    """Numerically stable symmetric matrix fractional power via eigendecomposition."""
    evals, evecs = eigh(C)
    evals = np.maximum(evals, min_eval)
    return evecs @ np.diag(evals ** p) @ evecs.T

def compute_metrics(y_true, y_prob, threshold, cond_id, cond_name, model_name, thresh_type):
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
        "condition_id": cond_id,
        "condition_name": cond_name,
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
    print("TDAF STEP 5: CORRELATION ALIGNMENT (CORAL) & MULTIVARIATE FEATURE ADAPTATION")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 1. Load Pretrained Models & Datasets
    # ---------------------------------------------------------
    print("\n[1/8] Loading models and datasets...")
    gam_model = joblib.load(GAM_MODEL_PATH)
    ebm_model = joblib.load(EBM_MODEL_PATH)
    
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    fl_valid = df_fl[df_fl["target_asos_strict_crit_c"].notna()].copy()
    fl_train = fl_valid[fl_valid["month"].between(1, 7)].copy()
    fl_train["season"] = fl_train["month"].apply(get_season)
    
    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH).rename(columns=FEATURE_MAPPING)
    df_ch["valid_time"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["month"] = df_ch["valid_time"].dt.month
    df_ch["season"] = df_ch["month"].apply(get_season)
    
    y_ch = df_ch["storm_label_score"].values
    
    X_s = fl_train[COMPATIBLE_FEATURES].values
    X_t = df_ch[COMPATIBLE_FEATURES].values
    
    mu_s = np.mean(X_s, axis=0)
    mu_t = np.mean(X_t, axis=0)
    
    # Regularized empirical covariance (eps = 1e-5 for inversion safety)
    eps_ridge = 1e-5
    C_s = np.cov(X_s, rowvar=False) + eps_ridge * np.eye(len(COMPATIBLE_FEATURES))
    C_t = np.cov(X_t, rowvar=False) + eps_ridge * np.eye(len(COMPATIBLE_FEATURES))
    
    print(f"  Florida Train Set: {len(fl_train)} rows")
    print(f"  Chandipur Target Set: {len(df_ch)} rows")

    # ---------------------------------------------------------
    # 2. Compute CORAL Transformation Matrix & Conditions
    # ---------------------------------------------------------
    print("\n[2/8] Generating CORAL transformation matrices and conditions...")
    
    # Standard CORAL Transformation Matrix: A = C_T^(-1/2) * C_S^(1/2)
    C_t_inv_sqrt = matrix_power(C_t, -0.5)
    C_s_sqrt = matrix_power(C_s, 0.5)
    A_standard = C_t_inv_sqrt @ C_s_sqrt
    
    # Condition A: Original Raw Baseline
    X_cond_A = X_t.copy()
    
    # Condition B: CAPE Clipping Only (Step 4 Exp B)
    X_cond_B = X_t.copy()
    X_cond_B[:, 0] = np.minimum(X_cond_B[:, 0], CAPE_MAX_FL)
    
    # Condition C: Standard CORAL (Unconstrained)
    X_cond_C = (X_t - mu_t) @ A_standard + mu_s
    
    # Condition D: CORAL + Physical Clamping
    # Enforces: CAPE in [0, 7240], CIN >= 0, Wind >= 0, Cloud Cover in [0, 1]
    X_cond_D = X_cond_C.copy()
    X_cond_D[:, 0] = np.clip(X_cond_D[:, 0], 0.0, CAPE_MAX_FL) # CAPE
    X_cond_D[:, 1] = np.maximum(X_cond_D[:, 1], 0.0)          # CIN
    X_cond_D[:, 2] = np.maximum(X_cond_D[:, 2], 0.05)         # Bulk shear
    X_cond_D[:, 3] = np.clip(X_cond_D[:, 3], 0.0, 1.0)        # Cloud cover
    X_cond_D[:, 4] = np.maximum(X_cond_D[:, 4], 0.0)          # Wind 10m
    X_cond_D[:, 5] = np.maximum(X_cond_D[:, 5], 0.0)          # Wind 850hPa
    
    # Condition E: Regularized Covariance CORAL (Shrinkage lambda = 0.05 towards diagonal)
    lam_shrink = 0.05
    C_s_reg = (1.0 - lam_shrink) * C_s + lam_shrink * np.diag(np.diag(C_s))
    C_t_reg = (1.0 - lam_shrink) * C_t + lam_shrink * np.diag(np.diag(C_t))
    A_reg = matrix_power(C_t_reg, -0.5) @ matrix_power(C_s_reg, 0.5)
    X_cond_E = (X_t - mu_t) @ A_reg + mu_s
    
    # Condition F: Partial Alignment Suite
    X_cond_F_025 = X_t + 0.25 * (X_cond_C - X_t)
    X_cond_F_050 = X_t + 0.50 * (X_cond_C - X_t)
    X_cond_F_075 = X_t + 0.75 * (X_cond_C - X_t)
    
    # Condition G: Seasonal CORAL (with documented fallback for Season ON)
    X_cond_G = np.zeros_like(X_t)
    seasonal_names = ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]
    for s in seasonal_names:
        ch_idx = df_ch[df_ch["season"] == s].index
        X_t_s = df_ch.loc[ch_idx, COMPATIBLE_FEATURES].values
        mu_t_s = np.mean(X_t_s, axis=0)
        C_t_s = np.cov(X_t_s, rowvar=False) + eps_ridge * np.eye(len(COMPATIBLE_FEATURES))
        
        if s in ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)"]:
            X_s_s = fl_train[fl_train["season"] == s][COMPATIBLE_FEATURES].values
            mu_s_s = np.mean(X_s_s, axis=0)
            C_s_s = np.cov(X_s_s, rowvar=False) + eps_ridge * np.eye(len(COMPATIBLE_FEATURES))
        else: # Season ON fallback to global Florida training distribution
            mu_s_s = mu_s
            C_s_s = C_s
            
        A_s = matrix_power(C_t_s, -0.5) @ matrix_power(C_s_s, 0.5)
        X_cond_G[ch_idx] = (X_t_s - mu_t_s) @ A_s + mu_s_s
        
    conditions_dict = {
        "Cond_A": ("Original Zero-Shot Baseline", X_cond_A),
        "Cond_B": ("CAPE Boundary Clipping Only (7,240 J/kg)", X_cond_B),
        "Cond_C": ("Standard CORAL (Unconstrained)", X_cond_C),
        "Cond_D": ("CORAL + Physical Domain Clamping", X_cond_D),
        "Cond_E": ("Regularized Covariance CORAL (λ=0.05)", X_cond_E),
        "Cond_F_025": ("Partial CORAL (α=0.25)", X_cond_F_025),
        "Cond_F_050": ("Partial CORAL (α=0.50)", X_cond_F_050),
        "Cond_F_075": ("Partial CORAL (α=0.75)", X_cond_F_075),
        "Cond_G": ("Seasonal CORAL (with ON Fallback)", X_cond_G)
    }

    # ---------------------------------------------------------
    # 3. Model Inference & Evaluation Across All Conditions
    # ---------------------------------------------------------
    print("\n[3/8] Running model inference across all CORAL conditions...")
    eval_metrics = []
    predictions_dict = {}
    
    for cond_id, (cond_name, X_mat) in conditions_dict.items():
        df_mat = pd.DataFrame(X_mat, columns=COMPATIBLE_FEATURES)
        
        prob_gam = gam_model.predict_proba(X_mat)
        prob_ebm = ebm_model.predict_proba(df_mat)[:, 1]
        
        predictions_dict[cond_id] = {
            "prob_gam": prob_gam,
            "prob_ebm": prob_ebm,
            "X_mat": X_mat
        }
        
        # GAM Evaluations
        eval_metrics.append(compute_metrics(y_ch, prob_gam, THRESH_GAM_OPT, cond_id, cond_name, "Native pyGAM", "FL Optimal (0.33)"))
        eval_metrics.append(compute_metrics(y_ch, prob_gam, THRESH_STD, cond_id, cond_name, "Native pyGAM", "Standard (0.50)"))
        
        # EBM Evaluations
        eval_metrics.append(compute_metrics(y_ch, prob_ebm, THRESH_EBM_OPT, cond_id, cond_name, "Native InterpretML EBM", "FL Optimal (0.34)"))
        eval_metrics.append(compute_metrics(y_ch, prob_ebm, THRESH_STD, cond_id, cond_name, "Native InterpretML EBM", "Standard (0.50)"))
        
    df_exp_comp = pd.DataFrame(eval_metrics)
    df_exp_comp.to_csv(OUTPUT_DIR / "coral_experiment_comparison.csv", index=False)
    print("  Saved coral_experiment_comparison.csv")
    print(df_exp_comp[["condition_id", "model", "threshold_type", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]].to_string())

    # ---------------------------------------------------------
    # 4. Domain Alignment Diagnostics & Covariance Distances
    # ---------------------------------------------------------
    print("\n[4/8] Computing multivariate domain alignment diagnostics...")
    diag_rows = []
    
    # Train non-linear domain classifier (LightGBM) on each condition
    # Using the interleaved day split (Even = Train, Odd = Test)
    fl_valid["init_time"] = pd.to_datetime(fl_valid["forecast_init_time_utc"])
    fl_dc_train = fl_valid[fl_valid["init_time"].dt.day % 2 == 0][COMPATIBLE_FEATURES].values
    fl_dc_test = fl_valid[fl_valid["init_time"].dt.day % 2 == 1][COMPATIBLE_FEATURES].values
    
    even_mask = (df_ch["valid_time"].dt.day % 2 == 0).values
    odd_mask = (df_ch["valid_time"].dt.day % 2 == 1).values
    
    y_dc_train = np.hstack([np.zeros(len(fl_dc_train)), np.ones(np.sum(even_mask))])
    y_dc_test = np.hstack([np.zeros(len(fl_dc_test)), np.ones(np.sum(odd_mask))])
    
    for cond_id, (cond_name, X_mat) in conditions_dict.items():
        C_adapted = np.cov(X_mat, rowvar=False)
        mu_adapted = np.mean(X_mat, axis=0)
        
        # Covariance Frobenius distance to Florida Training
        frob_cov_dist = float(np.linalg.norm(C_adapted - C_s, "fro"))
        
        # Mean absolute correlation difference
        corr_s = np.corrcoef(X_s, rowvar=False)
        corr_adapted = np.corrcoef(X_mat, rowvar=False)
        mean_abs_corr_diff = float(np.mean(np.abs(corr_adapted - corr_s)))
        
        # Mean absolute mean difference
        mean_diff_l1 = float(np.mean(np.abs(mu_adapted - mu_s)))
        
        # Mean Wasserstein distance across 6 features
        w_dists = [wasserstein_distance(X_s[:, j], X_mat[:, j]) for j in range(len(COMPATIBLE_FEATURES))]
        mean_w_dist = float(np.mean(w_dists))
        
        # Domain Classifier Separability (LightGBM)
        X_dc_tr = np.vstack([fl_dc_train, X_mat[even_mask]])
        X_dc_te = np.vstack([fl_dc_test, X_mat[odd_mask]])
        
        clf_dc = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)
        clf_dc.fit(X_dc_tr, y_dc_train)
        probs_dc = clf_dc.predict_proba(X_dc_te)[:, 1]
        dc_roc_auc = float(roc_auc_score(y_dc_test, probs_dc))
        
        diag_rows.append({
            "condition_id": cond_id,
            "condition_name": cond_name,
            "frobenius_covariance_distance": round(frob_cov_dist, 2),
            "mean_abs_correlation_difference": round(mean_abs_corr_diff, 4),
            "mean_feature_bias_l1": round(mean_diff_l1, 4),
            "mean_wasserstein_distance": round(mean_w_dist, 4),
            "domain_classifier_roc_auc": round(dc_roc_auc, 4),
            "cape_min": round(float(np.min(X_mat[:, 0])), 2),
            "cape_max": round(float(np.max(X_mat[:, 0])), 2),
            "negative_cape_count": int(np.sum(X_mat[:, 0] < 0.0)),
            "negative_cin_count": int(np.sum(X_mat[:, 1] < 0.0)),
            "negative_wind10m_count": int(np.sum(X_mat[:, 4] < 0.0)),
            "cloud_cover_out_of_bounds_count": int(np.sum((X_mat[:, 3] < 0.0) | (X_mat[:, 3] > 1.0)))
        })
        
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(OUTPUT_DIR / "coral_alignment_diagnostics.csv", index=False)
    print("  Saved coral_alignment_diagnostics.csv")
    print(df_diag[["condition_id", "frobenius_covariance_distance", "domain_classifier_roc_auc", "negative_cape_count", "negative_cin_count"]].to_string())

    # ---------------------------------------------------------
    # 5. Seasonal Performance Breakdown
    # ---------------------------------------------------------
    print("\n[5/8] Computing seasonal comparative performance breakdown...")
    sea_rows = []
    
    for cond_id, (cond_name, _) in conditions_dict.items():
        prob_gam = predictions_dict[cond_id]["prob_gam"]
        prob_ebm = predictions_dict[cond_id]["prob_ebm"]
        
        for s in seasonal_names:
            s_mask = (df_ch["season"] == s).values
            y_s = y_ch[s_mask]
            pg_s = prob_gam[s_mask]
            pe_s = prob_ebm[s_mask]
            
            f1_g = f1_score(y_s, (pg_s >= THRESH_GAM_OPT).astype(int), zero_division=0)
            f1_e = f1_score(y_s, (pe_s >= THRESH_EBM_OPT).astype(int), zero_division=0)
            auc_g = roc_auc_score(y_s, pg_s) if len(np.unique(y_s)) > 1 else np.nan
            auc_e = roc_auc_score(y_s, pe_s) if len(np.unique(y_s)) > 1 else np.nan
            
            sea_rows.append({
                "condition_id": cond_id,
                "condition_name": cond_name,
                "season": s,
                "sample_count": int(np.sum(s_mask)),
                "positives": int(np.sum(y_s)),
                "prevalence": round(float(np.mean(y_s)), 4),
                "gam_f1": round(float(f1_g), 4),
                "gam_roc_auc": round(float(auc_g), 4),
                "ebm_f1": round(float(f1_e), 4),
                "ebm_roc_auc": round(float(auc_e), 4)
            })
            
    df_seasonal_comp = pd.DataFrame(sea_rows)
    df_seasonal_comp.to_csv(OUTPUT_DIR / "coral_seasonal_comparison.csv", index=False)
    print("  Saved coral_seasonal_comparison.csv")

    # ---------------------------------------------------------
    # 6. Feature Transformation Summary Statistics
    # ---------------------------------------------------------
    print("\n[6/8] Compiling feature transformation summaries...")
    feat_summary_rows = []
    
    for j, f in enumerate(COMPATIBLE_FEATURES):
        feat_summary_rows.append({
            "feature": f,
            "source_train_mean": round(float(mu_s[j]), 2),
            "target_raw_mean": round(float(mu_t[j]), 2),
            "coral_mean": round(float(np.mean(X_cond_C[:, j])), 2),
            "coral_clamped_mean": round(float(np.mean(X_cond_D[:, j])), 2),
            "source_train_std": round(float(np.sqrt(C_s[j, j])), 2),
            "target_raw_std": round(float(np.sqrt(C_t[j, j])), 2),
            "coral_std": round(float(np.std(X_cond_C[:, j], ddof=1)), 2),
            "coral_clamped_std": round(float(np.std(X_cond_D[:, j], ddof=1)), 2),
            "coral_min": round(float(np.min(X_cond_C[:, j])), 2),
            "coral_max": round(float(np.max(X_cond_C[:, j])), 2),
            "coral_clamped_min": round(float(np.min(X_cond_D[:, j])), 2),
            "coral_clamped_max": round(float(np.max(X_cond_D[:, j])), 2)
        })
    df_feat_sum = pd.DataFrame(feat_summary_rows)
    df_feat_sum.to_csv(OUTPUT_DIR / "coral_feature_transformation_summary.csv", index=False)
    print("  Saved coral_feature_transformation_summary.csv")

    # ---------------------------------------------------------
    # 7. Extreme Case Case-Study Analysis
    # ---------------------------------------------------------
    print("\n[7/8] Generating extreme CAPE case analysis...")
    ext_mask = df_ch["cape_J_kg"] > CAPE_MAX_FL
    df_ext = df_ch[ext_mask].copy()
    
    ext_export = df_ext[["valid_time", "latitude", "longitude", "season", "cape_J_kg", "cin_J_kg", "bulk_shear_850_300_ms", "storm_score", "storm_label_score"]].copy()
    for cond_id in ["Cond_A", "Cond_B", "Cond_C", "Cond_D", "Cond_G"]:
        ext_export[f"cape_{cond_id}"] = predictions_dict[cond_id]["X_mat"][ext_mask, 0]
        ext_export[f"prob_gam_{cond_id}"] = predictions_dict[cond_id]["prob_gam"][ext_mask]
        ext_export[f"prob_ebm_{cond_id}"] = predictions_dict[cond_id]["prob_ebm"][ext_mask]
        
    df_ext_path = OUTPUT_DIR / "coral_extreme_case_analysis.csv"
    ext_export.to_csv(df_ext_path, index=False)
    print(f"  Saved coral_extreme_case_analysis.csv ({len(ext_export)} rows)")

    # ---------------------------------------------------------
    # 8. Publication-Grade Diagnostic Figures
    # ---------------------------------------------------------
    print("\n[8/8] Generating 8 diagnostic figures...")
    
    # Figure 1: Covariance Matrices Before and After CORAL
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.heatmap(C_s, annot=False, cmap="viridis", ax=axes[0])
    axes[0].set_title(f"Source (Florida Train) Covariance C_S\nTr={np.trace(C_s):.0f}", fontsize=12, fontweight="bold")
    sns.heatmap(C_t, annot=False, cmap="viridis", ax=axes[1])
    axes[1].set_title(f"Target Raw (Chandipur) Covariance C_T\nFrob Dist to C_S = {np.linalg.norm(C_t - C_s, 'fro'):.0f}", fontsize=12, fontweight="bold")
    sns.heatmap(np.cov(X_cond_C, rowvar=False), annot=False, cmap="viridis", ax=axes[2])
    axes[2].set_title(f"Target Aligned (CORAL) Covariance C_CORAL\nFrob Dist to C_S = {np.linalg.norm(np.cov(X_cond_C, rowvar=False) - C_s, 'fro'):.1f} (Aligned!)", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xticklabels([f[:8] for f in COMPATIBLE_FEATURES], rotation=30)
        ax.set_yticklabels([f[:8] for f in COMPATIBLE_FEATURES], rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_covariance_before_after.png", dpi=300)
    plt.close()

    # Figure 2: Correlation Heatmaps & Inversion Alignment
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    corr_s_mat = np.corrcoef(X_s, rowvar=False)
    corr_t_mat = np.corrcoef(X_t, rowvar=False)
    corr_c_mat = np.corrcoef(X_cond_C, rowvar=False)
    
    sns.heatmap(corr_s_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Source (Florida Train) Correlation R_S", fontsize=12, fontweight="bold")
    sns.heatmap(corr_t_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar=False)
    axes[1].set_title("Target Raw (Chandipur) Correlation R_T", fontsize=12, fontweight="bold")
    sns.heatmap(corr_c_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title("Target Aligned (CORAL) Correlation R_CORAL", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xticklabels([f[:8] for f in COMPATIBLE_FEATURES], rotation=30)
        ax.set_yticklabels([f[:8] for f in COMPATIBLE_FEATURES], rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_correlation_heatmaps.png", dpi=300)
    plt.close()

    # Figure 3: Feature Distributions Before & After CORAL (Highlighting Negative Bounds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for j, f in enumerate(COMPATIBLE_FEATURES):
        ax = axes[j]
        sns.kdeplot(X_s[:, j], label="Florida Train Reference", color="#2563eb", fill=True, alpha=0.15, linewidth=2, ax=ax)
        sns.kdeplot(X_t[:, j], label="Chandipur Raw", color="#111827", linewidth=2, ax=ax)
        sns.kdeplot(X_cond_C[:, j], label="CORAL Unconstrained", color="#dc2626", linestyle="--", linewidth=2, ax=ax)
        sns.kdeplot(X_cond_D[:, j], label="CORAL Clamped", color="#059669", linewidth=2, ax=ax)
        
        # Mark physical bounds
        if f in ["cape_J_kg", "cin_J_kg", "wind_speed_10m_ms", "wind_speed_850hPa_ms"]:
            ax.axvline(0.0, color="red", linestyle=":", label="Physical Lower Bound (0.0)")
        if f == "cloud_cover_fraction":
            ax.axvline(0.0, color="red", linestyle=":")
            ax.axvline(1.0, color="red", linestyle=":")
            
        ax.set_title(f"{FEATURE_LABELS[f]}", fontsize=11, fontweight="bold")
        ax.set_xlabel(FEATURE_LABELS[f], fontsize=10)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_feature_distributions.png", dpi=300)
    plt.close()

    # Figure 4: PCA Domain Alignment
    pca = PCA(n_components=2)
    X_s_std = StandardScaler().fit_transform(X_s)
    X_t_std = StandardScaler().fit_transform(X_t)
    X_c_std = StandardScaler().fit_transform(X_cond_C)
    
    s_pca = pca.fit_transform(X_s_std)
    t_pca = pca.transform(X_t_std)
    c_pca = pca.transform(X_c_std)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(s_pca[::10, 0], s_pca[::10, 1], color="#2563eb", alpha=0.25, s=12, label="Florida Train")
    axes[0].scatter(t_pca[::4, 0], t_pca[::4, 1], color="#111827", alpha=0.25, s=12, label="Chandipur Raw")
    axes[0].set_title("Before CORAL: PCA Space Overlap", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(s_pca[::10, 0], s_pca[::10, 1], color="#2563eb", alpha=0.25, s=12, label="Florida Train")
    axes[1].scatter(c_pca[::4, 0], c_pca[::4, 1], color="#dc2626", alpha=0.25, s=12, label="Chandipur CORAL Aligned")
    axes[1].set_title("After CORAL: PCA Space Alignment\n(Centroids and Principal Axes Superimposed)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_pca_alignment.png", dpi=300)
    plt.close()

    # Figure 5: Domain Classifier Separability Before & After CORAL
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cond_id, col, ls in [("Cond_A", "#111827", "-"), ("Cond_C", "#dc2626", "-"), ("Cond_D", "#059669", "--"), ("Cond_E", "#d97706", ":")]:
        X_mat = conditions_dict[cond_id][1]
        X_dc_tr = np.vstack([fl_dc_train, X_mat[even_mask]])
        X_dc_te = np.vstack([fl_dc_test, X_mat[odd_mask]])
        clf_dc = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)
        clf_dc.fit(X_dc_tr, y_dc_train)
        probs_dc = clf_dc.predict_proba(X_dc_te)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_dc_test, probs_dc)
        auc_val = roc_auc_score(y_dc_test, probs_dc)
        axes[0].plot(fpr, tpr, label=f"{cond_id} (AUC={auc_val:.4f})", color=col, linestyle=ls, linewidth=2)
        
        p, r, _ = precision_recall_curve(y_dc_test, probs_dc)
        pr_val = average_precision_score(y_dc_test, probs_dc)
        axes[1].plot(r, p, label=f"{cond_id} (PR-AUC={pr_val:.4f})", color=col, linestyle=ls, linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("Domain Classifier ROC Curves\n(Reduced AUC = Lower Domain Distinguishability)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axhline(np.mean(y_dc_test), color="gray", linestyle="--")
    axes[1].set_title("Domain Classifier Precision-Recall Curves", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_domain_classifier_comparison.png", dpi=300)
    plt.close()

    # Figure 6: Partial Alignment Alpha Sensitivity Curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    frob_vals = []
    gam_f1s = []
    ebm_f1s = []
    
    for a in alphas:
        X_p = X_t + a * (X_cond_C - X_t)
        C_p = np.cov(X_p, rowvar=False)
        frob_vals.append(float(np.linalg.norm(C_p - C_s, "fro")))
        
        pg = gam_model.predict_proba(X_p)
        pe = ebm_model.predict_proba(pd.DataFrame(X_p, columns=COMPATIBLE_FEATURES))[:, 1]
        gam_f1s.append(f1_score(y_ch, (pg >= THRESH_GAM_OPT).astype(int)))
        ebm_f1s.append(f1_score(y_ch, (pe >= THRESH_EBM_OPT).astype(int)))
        
    axes[0].plot(alphas, frob_vals, "o-", color="#2563eb", linewidth=2.5)
    axes[0].set_title("Covariance Frobenius Distance vs Alignment Fraction α", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("CORAL Alignment Fraction (α)", fontsize=11)
    axes[0].set_ylabel("Frobenius Distance to C_S", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(alphas, gam_f1s, "s-", label="pyGAM F1 (τ=0.33)", color="#2563eb", linewidth=2.5)
    axes[1].plot(alphas, ebm_f1s, "o-", label="InterpretML EBM F1 (τ=0.34)", color="#059669", linewidth=2.5)
    axes[1].set_title("Model F1 Performance vs Alignment Fraction α\n(Illustrating Monotonic Degradation Against Proxy)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("CORAL Alignment Fraction (α)", fontsize=11)
    axes[1].set_ylabel("F1 Score vs Proxy", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_alpha_sensitivity.png", dpi=300)
    plt.close()

    # Figure 7: Seasonal Alignment Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_sea_pivot_gam = df_seasonal_comp.pivot(index="season", columns="condition_id", values="gam_f1")
    df_sea_pivot_gam[["Cond_A", "Cond_B", "Cond_C", "Cond_D", "Cond_G"]].plot(kind="bar", ax=axes[0], colormap="viridis", width=0.8)
    axes[0].set_title("pyGAM Seasonal F1-Score Across CORAL Conditions", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("F1 Score", fontsize=11)
    axes[0].set_ylim([0, 0.8])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Condition", fontsize=8)
    
    df_sea_pivot_ebm = df_seasonal_comp.pivot(index="season", columns="condition_id", values="ebm_f1")
    df_sea_pivot_ebm[["Cond_A", "Cond_B", "Cond_C", "Cond_D", "Cond_G"]].plot(kind="bar", ax=axes[1], colormap="viridis", width=0.8)
    axes[1].set_title("InterpretML EBM Seasonal F1-Score Across CORAL Conditions", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("F1 Score", fontsize=11)
    axes[1].set_ylim([0, 0.8])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Condition", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_seasonal_alignment.png", dpi=300)
    plt.close()

    # Figure 8: Extreme CAPE Case Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ext_cape_raw = df_ch.loc[ext_mask, "cape_J_kg"].values
    sort_idx = np.argsort(ext_cape_raw)
    
    axes[0].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_A"]["prob_gam"][ext_mask][sort_idx], "o-", label="Cond A: Raw (Spline Collapse)", color="#111827", linewidth=2)
    axes[0].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_B"]["prob_gam"][ext_mask][sort_idx], "s-", label="Cond B: Clipped (7,240)", color="#dc2626", linewidth=2)
    axes[0].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_C"]["prob_gam"][ext_mask][sort_idx], "^-", label="Cond C: CORAL Unconstrained", color="#3b82f6", linewidth=2)
    axes[0].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_D"]["prob_gam"][ext_mask][sort_idx], "d-", label="Cond D: CORAL Clamped", color="#059669", linewidth=2)
    axes[0].axhline(THRESH_GAM_OPT, color="red", linestyle="--", label=f"FL Threshold ({THRESH_GAM_OPT})")
    axes[0].set_title("pyGAM Predicted Probability on 81 Extreme Cases (>7,240 J/kg)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Raw Input CAPE (J/kg)", fontsize=10)
    axes[0].set_ylabel("Predicted Probability", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_A"]["prob_ebm"][ext_mask][sort_idx], "o-", label="Cond A: Raw (Plateaued)", color="#111827", linewidth=2)
    axes[1].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_B"]["prob_ebm"][ext_mask][sort_idx], "s--", label="Cond B: Clipped (Identical)", color="#dc2626", linewidth=1.5)
    axes[1].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_C"]["prob_ebm"][ext_mask][sort_idx], "^-", label="Cond C: CORAL Unconstrained", color="#3b82f6", linewidth=2)
    axes[1].plot(ext_cape_raw[sort_idx], predictions_dict["Cond_D"]["prob_ebm"][ext_mask][sort_idx], "d-", label="Cond D: CORAL Clamped", color="#059669", linewidth=2)
    axes[1].axhline(THRESH_EBM_OPT, color="green", linestyle="--", label=f"FL Threshold ({THRESH_EBM_OPT})")
    axes[1].set_title("InterpretML EBM Predicted Probability on 81 Extreme Cases", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Raw Input CAPE (J/kg)", fontsize=10)
    axes[1].set_ylabel("Predicted Probability", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coral_extreme_cape_analysis.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 9. Mirror Outputs to Secondary Directory
    # ---------------------------------------------------------
    print("\n[9/9] Mirroring outputs to Florida_Thunderstorm_2024_Training...")
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, MIRROR_DIR / f.name)
    print("  Mirroring complete.")
    
    print("\n" + "=" * 80)
    print("STEP 5 CORAL EXPERIMENTS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
