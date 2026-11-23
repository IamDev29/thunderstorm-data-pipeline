"""
reconstruct_and_freeze_baseline.py
==================================
Thunderstorm Domain Adaptation Framework (TDAF) - Step 2:
Reconstruct and Freeze the Baseline Reference.

This script executes a strictly read-only reconstruction of:
1. Native Florida-trained pyGAM LogisticGAM model.
2. Native Florida-trained InterpretML ExplainableBoostingClassifier (EBM).
3. Predictions on Florida Test Set (Oct-Dec 2024, N=28,073).
4. Zero-shot transfer predictions on Chandipur Target Set (2022-2025, N=23,376).
5. Comprehensive metrics across both domains at both optimal and default 0.50 thresholds.
6. Publication-grade figures:
   - baseline_confusion_matrices.png
   - baseline_roc_curves.png
   - baseline_precision_recall_curves.png
   - cape_extrapolation_curves.png
7. Complete frozen artifacts saved in reports/domain_adaptation/step_02/

NO DOMAIN ADAPTATION OR MODEL RETRAINING IS PERFORMED IN THIS SCRIPT.
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

import pygam
from pygam import LogisticGAM
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
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
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. Directories and Paths
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

NATIVE_MODELS_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
GAM_MODEL_PATH = NATIVE_MODELS_DIR / "native_gam_model.pkl"
EBM_MODEL_PATH = NATIVE_MODELS_DIR / "native_ebm_model.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Secondary mirror directory
MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_02"

# ---------------------------------------------------------
# 2. Features and Operational Constants
# ---------------------------------------------------------
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
THRESH_DEFAULT = 0.50

FL_TARGET_COL = "target_asos_strict_crit_c"
CH_TARGET_COL = "storm_label_score"
CH_SCORE_COL = "storm_score"

def evaluate_predictions(y_true, y_prob, threshold, dataset_name, model_name, thresh_type, target_desc):
    """Compute complete suite of classification and calibration metrics."""
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
        "dataset": dataset_name,
        "model": model_name,
        "threshold_type": thresh_type,
        "threshold": round(threshold, 4),
        "sample_count": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 6),
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
        "target_description": target_desc
    }

def main():
    print("=" * 80)
    print("TDAF STEP 2: RECONSTRUCT AND FREEZE BASELINE REFERENCE")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 1. Load Pretrained Models
    # ---------------------------------------------------------
    print("\n[1/6] Loading pre-trained Florida models...")
    assert GAM_MODEL_PATH.exists(), f"Missing GAM model: {GAM_MODEL_PATH}"
    assert EBM_MODEL_PATH.exists(), f"Missing EBM model: {EBM_MODEL_PATH}"
    
    gam_model = joblib.load(GAM_MODEL_PATH)
    ebm_model = joblib.load(EBM_MODEL_PATH)
    print(f"  pyGAM loaded: {type(gam_model)}")
    print(f"  InterpretML EBM loaded: {type(ebm_model)}")
    assert ebm_model.feature_names == COMPATIBLE_FEATURES, "EBM feature mismatch!"

    # ---------------------------------------------------------
    # 2. Load Florida Test Set & Generate Predictions
    # ---------------------------------------------------------
    print("\n[2/6] Loading Florida 2024 dataset and extracting Test Set (Oct-Dec)...")
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    fl_valid = df_fl[df_fl[FL_TARGET_COL].notna()].copy()
    fl_valid["target"] = fl_valid[FL_TARGET_COL].astype(int)
    
    # Training set for distribution medians
    fl_train = fl_valid[fl_valid["month"].between(1, 7)].copy()
    fl_test = fl_valid[fl_valid["month"].between(10, 12)].copy()
    
    print(f"  Florida Test Set: {len(fl_test)} rows, {fl_test['target'].sum()} positive thunderstorms ({fl_test['target'].mean():.4%})")
    assert len(fl_test) == 28073, f"Expected 28,073 Florida test rows, got {len(fl_test)}"
    assert fl_test['target'].sum() == 1918, f"Expected 1,918 positive Florida test rows, got {fl_test['target'].sum()}"
    
    X_fl_test_df = fl_test[COMPATIBLE_FEATURES].copy()
    X_fl_test_arr = X_fl_test_df.values
    y_fl_test = fl_test["target"].values
    
    print("  Generating Florida test predictions...")
    fl_prob_gam = gam_model.predict_proba(X_fl_test_arr)
    fl_prob_ebm = ebm_model.predict_proba(X_fl_test_df)[:, 1]
    
    # Save Florida test predictions dataframe
    fl_test_preds_df = fl_test[[
        "forecast_init_time_utc", "station_id", "latitude", "longitude", "month", "hour_utc",
        *COMPATIBLE_FEATURES, "target"
    ]].copy()
    fl_test_preds_df["prob_gam"] = fl_prob_gam
    fl_test_preds_df["prob_ebm"] = fl_prob_ebm
    fl_test_preds_df["pred_gam_opt_thresh"] = (fl_prob_gam >= THRESH_GAM_OPT).astype(int)
    fl_test_preds_df["pred_ebm_opt_thresh"] = (fl_prob_ebm >= THRESH_EBM_OPT).astype(int)
    fl_test_preds_df["pred_gam_050"] = (fl_prob_gam >= THRESH_DEFAULT).astype(int)
    fl_test_preds_df["pred_ebm_050"] = (fl_prob_ebm >= THRESH_DEFAULT).astype(int)
    
    fl_preds_path = OUTPUT_DIR / "baseline_predictions_florida.csv"
    fl_test_preds_df.to_csv(fl_preds_path, index=False)
    print(f"  Saved Florida test predictions to {fl_preds_path} ({len(fl_test_preds_df)} rows)")

    # ---------------------------------------------------------
    # 3. Load Chandipur Target Set & Generate Predictions
    # ---------------------------------------------------------
    print("\n[3/6] Loading Chandipur ERA5 dataset (2022-2025)...")
    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH)
    assert len(df_ch) == 23376, f"Expected 23,376 Chandipur rows, got {len(df_ch)}"
    print(f"  Chandipur Set: {len(df_ch)} rows, {df_ch[CH_TARGET_COL].sum()} proxy thunderstorms ({df_ch[CH_TARGET_COL].mean():.4%})")
    
    df_ch_renamed = df_ch.rename(columns=FEATURE_MAPPING)
    X_ch_df = df_ch_renamed[COMPATIBLE_FEATURES].copy()
    X_ch_arr = X_ch_df.values
    y_ch = df_ch[CH_TARGET_COL].values
    storm_score_ch = df_ch[CH_SCORE_COL].values
    
    print("  Generating Chandipur zero-shot transfer predictions...")
    ch_prob_gam = gam_model.predict_proba(X_ch_arr)
    ch_prob_ebm = ebm_model.predict_proba(X_ch_df)[:, 1]
    
    # Save Chandipur predictions dataframe
    ch_preds_df = df_ch.copy()
    ch_preds_df["prob_gam"] = ch_prob_gam
    ch_preds_df["prob_ebm"] = ch_prob_ebm
    ch_preds_df["pred_gam_opt_thresh"] = (ch_prob_gam >= THRESH_GAM_OPT).astype(int)
    ch_preds_df["pred_ebm_opt_thresh"] = (ch_prob_ebm >= THRESH_EBM_OPT).astype(int)
    ch_preds_df["pred_gam_050"] = (ch_prob_gam >= THRESH_DEFAULT).astype(int)
    ch_preds_df["pred_ebm_050"] = (ch_prob_ebm >= THRESH_DEFAULT).astype(int)
    
    ch_preds_path = OUTPUT_DIR / "baseline_predictions_chandipur.csv"
    ch_preds_df.to_csv(ch_preds_path, index=False)
    print(f"  Saved Chandipur predictions to {ch_preds_path} ({len(ch_preds_df)} rows)")

    # ---------------------------------------------------------
    # 4. Compute and Compile Metrics
    # ---------------------------------------------------------
    print("\n[4/6] Computing baseline metrics across both domains...")
    metrics_list = []
    
    # Florida Test Metrics
    metrics_list.append(evaluate_predictions(
        y_fl_test, fl_prob_gam, THRESH_GAM_OPT, "Florida_Test_2024", "Native pyGAM", "Optimal_Val (0.33)", "NOAA ASOS Ground Truth"
    ))
    metrics_list.append(evaluate_predictions(
        y_fl_test, fl_prob_gam, THRESH_DEFAULT, "Florida_Test_2024", "Native pyGAM", "Default (0.50)", "NOAA ASOS Ground Truth"
    ))
    metrics_list.append(evaluate_predictions(
        y_fl_test, fl_prob_ebm, THRESH_EBM_OPT, "Florida_Test_2024", "Native InterpretML EBM", "Optimal_Val (0.34)", "NOAA ASOS Ground Truth"
    ))
    metrics_list.append(evaluate_predictions(
        y_fl_test, fl_prob_ebm, THRESH_DEFAULT, "Florida_Test_2024", "Native InterpretML EBM", "Default (0.50)", "NOAA ASOS Ground Truth"
    ))
    
    # Chandipur Zero-Shot Metrics
    metrics_list.append(evaluate_predictions(
        y_ch, ch_prob_gam, THRESH_GAM_OPT, "Chandipur_Zero_Shot_2022_2025", "Native pyGAM", "Florida_Optimal (0.33)", "ERA5 Heuristic Storm Score >= 2.50"
    ))
    metrics_list.append(evaluate_predictions(
        y_ch, ch_prob_gam, THRESH_DEFAULT, "Chandipur_Zero_Shot_2022_2025", "Native pyGAM", "Default (0.50)", "ERA5 Heuristic Storm Score >= 2.50"
    ))
    metrics_list.append(evaluate_predictions(
        y_ch, ch_prob_ebm, THRESH_EBM_OPT, "Chandipur_Zero_Shot_2022_2025", "Native InterpretML EBM", "Florida_Optimal (0.34)", "ERA5 Heuristic Storm Score >= 2.50"
    ))
    metrics_list.append(evaluate_predictions(
        y_ch, ch_prob_ebm, THRESH_DEFAULT, "Chandipur_Zero_Shot_2022_2025", "Native InterpretML EBM", "Default (0.50)", "ERA5 Heuristic Storm Score >= 2.50"
    ))
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved baseline metrics to {metrics_path}")
    print("\nBaseline Metrics Summary Table:")
    print(metrics_df[["dataset", "model", "threshold_type", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]].to_string())

    # ---------------------------------------------------------
    # 5. Generate Figures
    # ---------------------------------------------------------
    print("\n[5/6] Generating figures...")
    
    # Figure 1: Confusion Matrices
    print("  Creating baseline_confusion_matrices.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    cm_configs = [
        (0, 0, y_fl_test, (fl_prob_gam >= THRESH_GAM_OPT).astype(int), 
         f"Florida Test: Native pyGAM (τ={THRESH_GAM_OPT})\nTarget: ASOS Observed Thunderstorm", "Blues"),
        (0, 1, y_fl_test, (fl_prob_ebm >= THRESH_EBM_OPT).astype(int), 
         f"Florida Test: Native InterpretML EBM (τ={THRESH_EBM_OPT})\nTarget: ASOS Observed Thunderstorm", "Greens"),
        (1, 0, y_ch, (ch_prob_gam >= THRESH_GAM_OPT).astype(int), 
         f"Chandipur Zero-Shot: Native pyGAM (τ={THRESH_GAM_OPT})\nTarget: Proxy Storm Score ≥ 2.50", "Purples"),
        (1, 1, y_ch, (ch_prob_ebm >= THRESH_EBM_OPT).astype(int), 
         f"Chandipur Zero-Shot: Native InterpretML EBM (τ={THRESH_EBM_OPT})\nTarget: Proxy Storm Score ≥ 2.50", "Oranges"),
    ]
    
    for r, c, y_true_cm, y_pred_cm, title, cmap in cm_configs:
        ax = axes[r, c]
        cm = confusion_matrix(y_true_cm, y_pred_cm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = len(y_true_cm)
        prec = precision_score(y_true_cm, y_pred_cm, zero_division=0)
        rec = recall_score(y_true_cm, y_pred_cm, zero_division=0)
        f1 = f1_score(y_true_cm, y_pred_cm, zero_division=0)
        
        annot = np.array([
            [f"TN: {tn:,}\n({tn/total:.1%})", f"FP: {fp:,}\n({fp/total:.1%})"],
            [f"FN: {fn:,}\n({fn/total:.1%})", f"TP: {tp:,}\n({tp/total:.1%})"]
        ])
        
        sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
                    xticklabels=["Pred 0 (No)", "Pred 1 (Yes)"],
                    yticklabels=["True 0 (No)", "True 1 (Yes)"],
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"{title}\nPrecision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}", fontsize=12, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True / Reference Label", fontsize=11)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_confusion_matrices.png", dpi=300)
    plt.close()
    
    # Figure 2: ROC Curves
    print("  Creating baseline_roc_curves.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Florida ROC
    fpr_fl_gam, tpr_fl_gam, _ = roc_curve(y_fl_test, fl_prob_gam)
    fpr_fl_ebm, tpr_fl_ebm, _ = roc_curve(y_fl_test, fl_prob_ebm)
    auc_fl_gam = roc_auc_score(y_fl_test, fl_prob_gam)
    auc_fl_ebm = roc_auc_score(y_fl_test, fl_prob_ebm)
    
    axes[0].plot(fpr_fl_gam, tpr_fl_gam, label=f"pyGAM (AUC = {auc_fl_gam:.4f})", color="#2563eb", linewidth=2.5)
    axes[0].plot(fpr_fl_ebm, tpr_fl_ebm, label=f"InterpretML EBM (AUC = {auc_fl_ebm:.4f})", color="#059669", linewidth=2.5)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    axes[0].set_title("Florida Test Set (Oct-Dec 2024)\nGround Truth: NOAA ASOS Thunderstorm", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Chandipur ROC
    fpr_ch_gam, tpr_ch_gam, _ = roc_curve(y_ch, ch_prob_gam)
    fpr_ch_ebm, tpr_ch_ebm, _ = roc_curve(y_ch, ch_prob_ebm)
    auc_ch_gam = roc_auc_score(y_ch, ch_prob_gam)
    auc_ch_ebm = roc_auc_score(y_ch, ch_prob_ebm)
    
    axes[1].plot(fpr_ch_gam, tpr_ch_gam, label=f"pyGAM (AUC = {auc_ch_gam:.4f})", color="#2563eb", linewidth=2.5)
    axes[1].plot(fpr_ch_ebm, tpr_ch_ebm, label=f"InterpretML EBM (AUC = {auc_ch_ebm:.4f})", color="#059669", linewidth=2.5)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    axes[1].set_title("Chandipur Zero-Shot Transfer (2022-2025)\nReference: Storm Score Heuristic Proxy ≥ 2.50", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_roc_curves.png", dpi=300)
    plt.close()
    
    # Figure 3: Precision-Recall Curves
    print("  Creating baseline_precision_recall_curves.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Florida PR
    p_fl_gam, r_fl_gam, _ = precision_recall_curve(y_fl_test, fl_prob_gam)
    p_fl_ebm, r_fl_ebm, _ = precision_recall_curve(y_fl_test, fl_prob_ebm)
    pr_auc_fl_gam = average_precision_score(y_fl_test, fl_prob_gam)
    pr_auc_fl_ebm = average_precision_score(y_fl_test, fl_prob_ebm)
    prev_fl = np.mean(y_fl_test)
    
    axes[0].plot(r_fl_gam, p_fl_gam, label=f"pyGAM (PR-AUC = {pr_auc_fl_gam:.4f})", color="#2563eb", linewidth=2.5)
    axes[0].plot(r_fl_ebm, p_fl_ebm, label=f"InterpretML EBM (PR-AUC = {pr_auc_fl_ebm:.4f})", color="#059669", linewidth=2.5)
    axes[0].axhline(prev_fl, color="gray", linestyle="--", label=f"Baseline Prevalence ({prev_fl:.2%})")
    
    # Mark operating points
    axes[0].scatter([recall_score(y_fl_test, (fl_prob_gam >= THRESH_GAM_OPT).astype(int))],
                    [precision_score(y_fl_test, (fl_prob_gam >= THRESH_GAM_OPT).astype(int))],
                    color="#1d4ed8", s=100, zorder=5, label=f"pyGAM τ={THRESH_GAM_OPT}")
    axes[0].scatter([recall_score(y_fl_test, (fl_prob_ebm >= THRESH_EBM_OPT).astype(int))],
                    [precision_score(y_fl_test, (fl_prob_ebm >= THRESH_EBM_OPT).astype(int))],
                    color="#047857", s=100, zorder=5, label=f"EBM τ={THRESH_EBM_OPT}")
    
    axes[0].set_title("Florida Test Set (Oct-Dec 2024)\nGround Truth: NOAA ASOS Thunderstorm", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Recall", fontsize=11)
    axes[0].set_ylabel("Precision", fontsize=11)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].legend(loc="upper right", frameon=True, fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Chandipur PR
    p_ch_gam, r_ch_gam, _ = precision_recall_curve(y_ch, ch_prob_gam)
    p_ch_ebm, r_ch_ebm, _ = precision_recall_curve(y_ch, ch_prob_ebm)
    pr_auc_ch_gam = average_precision_score(y_ch, ch_prob_gam)
    pr_auc_ch_ebm = average_precision_score(y_ch, ch_prob_ebm)
    prev_ch = np.mean(y_ch)
    
    axes[1].plot(r_ch_gam, p_ch_gam, label=f"pyGAM (PR-AUC = {pr_auc_ch_gam:.4f})", color="#2563eb", linewidth=2.5)
    axes[1].plot(r_ch_ebm, p_ch_ebm, label=f"InterpretML EBM (PR-AUC = {pr_auc_ch_ebm:.4f})", color="#059669", linewidth=2.5)
    axes[1].axhline(prev_ch, color="gray", linestyle="--", label=f"Baseline Prevalence ({prev_ch:.2%})")
    
    axes[1].scatter([recall_score(y_ch, (ch_prob_gam >= THRESH_GAM_OPT).astype(int))],
                    [precision_score(y_ch, (ch_prob_gam >= THRESH_GAM_OPT).astype(int))],
                    color="#1d4ed8", s=100, zorder=5, label=f"pyGAM τ={THRESH_GAM_OPT}")
    axes[1].scatter([recall_score(y_ch, (ch_prob_ebm >= THRESH_EBM_OPT).astype(int))],
                    [precision_score(y_ch, (ch_prob_ebm >= THRESH_EBM_OPT).astype(int))],
                    color="#047857", s=100, zorder=5, label=f"EBM τ={THRESH_EBM_OPT}")
    
    axes[1].set_title("Chandipur Zero-Shot Transfer (2022-2025)\nReference: Storm Score Heuristic Proxy ≥ 2.50", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.0])
    axes[1].legend(loc="upper right", frameon=True, fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_precision_recall_curves.png", dpi=300)
    plt.close()
    
    # Figure 4: CAPE Extrapolation Curves (Spline Collapse vs Tree Plateau)
    print("  Creating cape_extrapolation_curves.png...")
    fl_medians = fl_train[COMPATIBLE_FEATURES].median().to_dict()
    fl_cape_p99 = float(fl_train["cape_J_kg"].quantile(0.99))
    fl_cape_max = float(fl_train["cape_J_kg"].max())
    ch_cape_max = float(df_ch_renamed["cape_J_kg"].max())
    
    cape_sweep = np.linspace(0, 10000, 500)
    synth_df = pd.DataFrame({f: [fl_medians[f]] * len(cape_sweep) for f in COMPATIBLE_FEATURES})
    synth_df["cape_J_kg"] = cape_sweep
    
    # GAM probabilities and log-odds
    gam_synth_prob = gam_model.predict_proba(synth_df[COMPATIBLE_FEATURES].values)
    gam_synth_logit = gam_model._linear_predictor(synth_df[COMPATIBLE_FEATURES].values)
    
    # EBM probabilities and log-odds
    ebm_synth_prob = ebm_model.predict_proba(synth_df[COMPATIBLE_FEATURES])[:, 1]
    ebm_synth_logit = ebm_model.decision_function(synth_df[COMPATIBLE_FEATURES])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Probability vs CAPE
    axes[0].plot(cape_sweep, gam_synth_prob, label="pyGAM (Cubic Splines)", color="#2563eb", linewidth=2.8)
    axes[0].plot(cape_sweep, ebm_synth_prob, label="InterpretML EBM (Decision Stumps)", color="#059669", linewidth=2.8)
    
    axes[0].axvline(fl_cape_p99, color="#d97706", linestyle=":", linewidth=2, label=f"FL Train 99th Pct ({fl_cape_p99:.0f} J/kg)")
    axes[0].axvline(fl_cape_max, color="#dc2626", linestyle="--", linewidth=2, label=f"FL Train Max Domain Boundary ({fl_cape_max:.0f} J/kg)")
    axes[0].axvline(ch_cape_max, color="#7c3aed", linestyle="-.", linewidth=2, label=f"CH Max Target Observation ({ch_cape_max:.0f} J/kg)")
    
    # Highlight extrapolation zone
    axes[0].axvspan(fl_cape_max, 10000, color="#fee2e2", alpha=0.4, label="Out-of-Distribution Extrapolation Zone")
    
    # Annotation on pyGAM collapse
    axes[0].annotate("Spline Boundary Collapse:\nProb drops to 0.003\nat 10,000 J/kg!",
                     xy=(9000, gam_synth_prob[np.argmin(np.abs(cape_sweep - 9000))]),
                     xytext=(6800, 0.45),
                     arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=2, headwidth=8),
                     fontsize=10, fontweight="bold", color="#1e40af",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#eff6ff", ec="#2563eb", alpha=0.9))
    
    # Annotation on EBM plateau
    axes[0].annotate("Tree Stump Clamping:\nFlat plateau at 0.378\nabove 7,240 J/kg",
                     xy=(8500, ebm_synth_prob[np.argmin(np.abs(cape_sweep - 8500))]),
                     xytext=(6800, 0.20),
                     arrowprops=dict(facecolor="#059669", shrink=0.08, width=2, headwidth=8),
                     fontsize=10, fontweight="bold", color="#065f46",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", alpha=0.9))
    
    axes[0].set_title("Predicted Convective Probability vs CAPE (0–10,000 J/kg)\n(All other features held at Florida training medians)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[0].set_ylabel("Predicted Thunderstorm Probability", fontsize=11)
    axes[0].set_ylim([-0.02, 1.02])
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Linear Predictor (Log-Odds) vs CAPE
    axes[1].plot(cape_sweep, gam_synth_logit, label="pyGAM Log-Odds (Link Function η)", color="#2563eb", linewidth=2.8)
    axes[1].plot(cape_sweep, ebm_synth_logit, label="InterpretML EBM Log-Odds (Score Function)", color="#059669", linewidth=2.8)
    
    axes[1].axvline(fl_cape_p99, color="#d97706", linestyle=":", linewidth=2, label=f"FL Train 99th Pct ({fl_cape_p99:.0f} J/kg)")
    axes[1].axvline(fl_cape_max, color="#dc2626", linestyle="--", linewidth=2, label=f"FL Train Max Domain Boundary ({fl_cape_max:.0f} J/kg)")
    axes[1].axvline(ch_cape_max, color="#7c3aed", linestyle="-.", linewidth=2, label=f"CH Max Target Observation ({ch_cape_max:.0f} J/kg)")
    axes[1].axvspan(fl_cape_max, 10000, color="#fee2e2", alpha=0.4, label="Extrapolation Zone")
    
    axes[1].set_title("Model Link Function / Log-Odds vs CAPE\n(Revealing Mathematical Spline Extrapolation Divergence)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    axes[1].set_ylabel("Log-Odds (Logit Scale)", fontsize=11)
    axes[1].legend(loc="lower left", frameon=True, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cape_extrapolation_curves.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 6. Mirror to Secondary Directory
    # ---------------------------------------------------------
    print("\n[6/6] Mirroring outputs to Florida_Thunderstorm_2024_Training...")
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, MIRROR_DIR / f.name)
    print(f"  Successfully mirrored files to {MIRROR_DIR}")
    
    print("\n" + "=" * 80)
    print("STEP 2 BASELINE RECONSTRUCTION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
