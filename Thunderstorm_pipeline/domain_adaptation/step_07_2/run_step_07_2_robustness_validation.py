"""
run_step_07_2_robustness_validation.py
======================================
Thunderstorm Domain Adaptation Framework (TDAF)
Step 7.2 — Robustness, Statistical Significance and Generalization Validation

Comprehensive validation script:
1. Audit Transformation Fitting (unsupervised transductive parameters, JSON config).
2. Audit Florida Threshold Selection (validation tuning, test holdout revalidation).
3. Experiment A: Day-level block bootstrap confidence intervals (B=1,000) for 5 models.
4. Experiment B: Paired model comparison (Symmetric B4 GAM vs Original Raw GAM) with 95% CIs and p-values.
5. Experiment C: Parameter sensitivity analysis (0.80x to 1.20x alpha scaling).
6. Experiment D: Seasonal generalization with uncertainty bounds across all convective regimes.
7. Experiment E: Temporal, spatial, leave-one-out, and CAPE quantile subgroup analyses.
8. Experiment F: Interpretability and spline knot redistribution analysis.
9. Deliver all 6 CSVs and 6 publication-quality diagnostic figures (300 DPI).
"""

import os
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

import pygam
from pygam import LogisticGAM, s
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

# -------------------------------------------------------------
# Configuration and Paths
# -------------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

FROZEN_GAM_PATH = ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_gam_model.pkl"
FROZEN_EBM_PATH = ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_ebm_model.pkl"
SYM_GAM_PATH = ROOT_DIR / "reports" / "domain_adaptation" / "step_07" / "audit" / "symmetric_gam_model.pkl"
SYM_EBM_PATH = ROOT_DIR / "reports" / "domain_adaptation" / "step_07" / "audit" / "symmetric_ebm_model.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_07_2"
MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_07_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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
    "cloud_cover_fraction": "Total Cloud Cover Fraction",
    "wind_speed_10m_ms": "10 m Wind Speed (m/s)",
    "wind_speed_850hPa_ms": "850 hPa Wind Speed (m/s)"
}

CHANDIPUR_RENAME = {
    "cape_Jkg": "cape_J_kg",
    "cin_Jkg": "cin_J_kg",
    "bulk_shear_850_300": "bulk_shear_850_300_ms",
    "total_cloud_cover": "cloud_cover_fraction",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_speed_850hPa": "wind_speed_850hPa_ms"
}

CAPE_CLIP_THRESHOLD = 7240.0
TARGET_COL_FLORIDA = "target_asos_strict_crit_c"
TARGET_COL_CHANDIPUR = "storm_label_score"

def get_season(m):
    if m in [12, 1, 2]:
        return "DJF (Winter)"
    elif m in [3, 4, 5]:
        return "MAM (Pre-monsoon)"
    elif m in [6, 7, 8, 9]:
        return "JJAS (Monsoon)"
    else:
        return "ON (Post-monsoon)"

def evaluate_metrics_fast(y_true, y_prob, threshold):
    """Compute fast evaluation metrics for bootstrapping."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    n_unique = len(np.unique(y_true))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if n_unique > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_unique > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "brier_score": brier
    }

def find_optimal_threshold(y_val, y_prob_val):
    best_f1 = -1.0
    best_tau = 0.5
    for tau in np.arange(0.10, 0.90, 0.01):
        y_pred = (y_prob_val >= tau).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_tau = tau
    return round(float(best_tau), 2)

# -------------------------------------------------------------
# Main Script Execution
# -------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 80)
    print("STEP 7.2: ROBUSTNESS, STATISTICAL SIGNIFICANCE AND GENERALIZATION VALIDATION")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/8] Loading Florida ASOS and Chandipur ERA5 datasets...", flush=True)
    df_fl = pd.read_parquet(FLORIDA_DATA_PATH)
    df_fl_valid = df_fl[df_fl[TARGET_COL_FLORIDA].notna()].copy()
    df_fl_valid["target"] = df_fl_valid[TARGET_COL_FLORIDA].astype(int)

    fl_train = df_fl_valid[df_fl_valid["month"].between(1, 7)].copy()
    fl_val = df_fl_valid[df_fl_valid["month"].between(8, 9)].copy()
    fl_test = df_fl_valid[df_fl_valid["month"].between(10, 12)].copy()

    X_fl_train_raw = fl_train[FEATURES].values
    y_fl_train = fl_train["target"].values
    X_fl_val_raw = fl_val[FEATURES].values
    y_fl_val = fl_val["target"].values
    X_fl_test_raw = fl_test[FEATURES].values
    y_fl_test = fl_test["target"].values

    df_ch = pd.read_csv(CHANDIPUR_DATA_PATH)
    df_ch = df_ch.rename(columns=CHANDIPUR_RENAME)
    df_ch["valid_time_dt"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["date"] = df_ch["valid_time_dt"].dt.date
    df_ch["year"] = df_ch["valid_time_dt"].dt.year
    df_ch["month"] = df_ch["valid_time_dt"].dt.month
    df_ch["season"] = df_ch["month"].apply(get_season)
    df_ch["grid_id"] = df_ch["latitude"].astype(str) + "N_" + df_ch["longitude"].astype(str) + "E"

    X_ch_raw = df_ch[FEATURES].values
    y_ch = df_ch[TARGET_COL_CHANDIPUR].values
    print(f"  Florida Train: N={len(fl_train)}, Val: N={len(fl_val)}, Test: N={len(fl_test)}")
    print(f"  Chandipur Target: N={len(df_ch)}, Unique Days={df_ch['date'].nunique()}, Grid Points={df_ch['grid_id'].nunique()}")

    # 2. Section 3: Audit Transformation Fitting
    print("\n[2/8] Section 3: Auditing transformation parameters and creating configuration JSON...", flush=True)
    s_cape_p95 = float(fl_train["cape_J_kg"].quantile(0.95))
    t_cape_p95 = float(df_ch["cape_J_kg"].quantile(0.95))
    alpha_cape = float(s_cape_p95 / np.log1p(t_cape_p95 / 1000.0))

    s_cin_p95 = float(fl_train["cin_J_kg"].quantile(0.95))
    t_cin_p95 = float(df_ch["cin_J_kg"].quantile(0.95))
    alpha_cin = float(s_cin_p95 / np.log1p(t_cin_p95 / 100.0))

    trans_config = {
        "paradigm": "Unsupervised Transductive Adaptation",
        "method_name": "Symmetric B4 Selective Transformation",
        "parameters": {
            "cape": {
                "source_p95_J_kg": s_cape_p95,
                "target_p95_J_kg": t_cape_p95,
                "scale_denominator": 1000.0,
                "alpha_cape": alpha_cape,
                "formula": "alpha_cape * ln(1 + x / 1000.0)"
            },
            "cin": {
                "source_p95_J_kg": s_cin_p95,
                "target_p95_J_kg": t_cin_p95,
                "scale_denominator": 100.0,
                "alpha_cin": alpha_cin,
                "formula": "alpha_cin * ln(1 + x / 100.0)"
            },
            "untouched_features": [
                "bulk_shear_850_300_ms",
                "cloud_cover_fraction",
                "wind_speed_10m_ms",
                "wind_speed_850hPa_ms"
            ]
        },
        "target_label_access": False,
        "florida_test_label_access": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(OUTPUT_DIR / "transformation_config.json", "w") as f:
        json.dump(trans_config, f, indent=2)
    print(f"  Stored {OUTPUT_DIR / 'transformation_config.json'}")

    def apply_b4(X_arr, a_cape=alpha_cape, a_cin=alpha_cin):
        X_out = X_arr.copy()
        X_out[:, 0] = a_cape * np.log1p(X_arr[:, 0] / 1000.0)
        X_out[:, 1] = a_cin * np.log1p(X_arr[:, 1] / 100.0)
        return X_out

    # Transformed data arrays
    X_fl_train_b4 = apply_b4(X_fl_train_raw)
    X_fl_val_b4 = apply_b4(X_fl_val_raw)
    X_fl_test_b4 = apply_b4(X_fl_test_raw)
    X_ch_b4 = apply_b4(X_ch_raw)

    # 3. Load Models
    orig_gam = joblib.load(FROZEN_GAM_PATH)
    orig_ebm = joblib.load(FROZEN_EBM_PATH)
    sym_gam = joblib.load(SYM_GAM_PATH)
    sym_ebm = joblib.load(SYM_EBM_PATH)
    print("  Models loaded: Original pyGAM, Original EBM, Symmetric pyGAM, Symmetric EBM")

    # 4. Section 4: Audit Florida Threshold Selection
    print("\n[3/8] Section 4: Auditing Florida threshold selection and evaluating on Florida holdout test...", flush=True)
    p_sym_gam_fl_val = sym_gam.predict_proba(X_fl_val_b4)
    tau_opt_sym_gam = find_optimal_threshold(y_fl_val, p_sym_gam_fl_val)

    p_sym_ebm_fl_val = sym_ebm.predict_proba(X_fl_val_b4)[:, 1]
    tau_opt_sym_ebm = find_optimal_threshold(y_fl_val, p_sym_ebm_fl_val)

    print(f"  Selected Florida Validation Thresholds: Symmetric pyGAM = {tau_opt_sym_gam:.2f}, Symmetric EBM = {tau_opt_sym_ebm:.2f}")

    p_orig_gam_fl_test = orig_gam.predict_proba(X_fl_test_raw)
    p_orig_ebm_fl_test = orig_ebm.predict_proba(X_fl_test_raw)[:, 1]
    p_sym_gam_fl_test = sym_gam.predict_proba(X_fl_test_b4)
    p_sym_ebm_fl_test = sym_ebm.predict_proba(X_fl_test_b4)[:, 1]

    thresh_records = []
    # Evaluate Florida Test
    for m_name, p_orig, p_sym, t_orig, t_sym in [
        ("pyGAM", p_orig_gam_fl_test, p_sym_gam_fl_test, 0.33, tau_opt_sym_gam),
        ("InterpretML EBM", p_orig_ebm_fl_test, p_sym_ebm_fl_test, 0.34, tau_opt_sym_ebm)
    ]:
        # Original model at original threshold
        m_orig_opt = evaluate_metrics_fast(y_fl_test, p_orig, t_orig)
        m_orig_50 = evaluate_metrics_fast(y_fl_test, p_orig, 0.50)
        # Symmetric model at original threshold
        m_sym_orig_t = evaluate_metrics_fast(y_fl_test, p_sym, t_orig)
        # Symmetric model at newly tuned threshold
        m_sym_opt = evaluate_metrics_fast(y_fl_test, p_sym, t_sym)
        m_sym_50 = evaluate_metrics_fast(y_fl_test, p_sym, 0.50)

        thresh_records.append({
            "model": m_name,
            "architecture": "Original Raw",
            "threshold_type": f"Original Florida Val ({t_orig:.2f})",
            "threshold_value": t_orig,
            "f1": round(m_orig_opt["f1"], 4),
            "precision": round(m_orig_opt["precision"], 4),
            "recall": round(m_orig_opt["recall"], 4),
            "pr_auc": round(m_orig_opt["pr_auc"], 4),
            "roc_auc": round(m_orig_opt["roc_auc"], 4),
            "brier_score": round(m_orig_opt["brier_score"], 4)
        })
        thresh_records.append({
            "model": m_name,
            "architecture": "Original Raw",
            "threshold_type": "Standard (0.50)",
            "threshold_value": 0.50,
            "f1": round(m_orig_50["f1"], 4),
            "precision": round(m_orig_50["precision"], 4),
            "recall": round(m_orig_50["recall"], 4),
            "pr_auc": round(m_orig_50["pr_auc"], 4),
            "roc_auc": round(m_orig_50["roc_auc"], 4),
            "brier_score": round(m_orig_50["brier_score"], 4)
        })
        thresh_records.append({
            "model": m_name,
            "architecture": "Symmetric B4 Retrained",
            "threshold_type": f"Tuned Florida Val ({t_sym:.2f})",
            "threshold_value": t_sym,
            "f1": round(m_sym_opt["f1"], 4),
            "precision": round(m_sym_opt["precision"], 4),
            "recall": round(m_sym_opt["recall"], 4),
            "pr_auc": round(m_sym_opt["pr_auc"], 4),
            "roc_auc": round(m_sym_opt["roc_auc"], 4),
            "brier_score": round(m_sym_opt["brier_score"], 4)
        })
        thresh_records.append({
            "model": m_name,
            "architecture": "Symmetric B4 Retrained",
            "threshold_type": "Standard (0.50)",
            "threshold_value": 0.50,
            "f1": round(m_sym_50["f1"], 4),
            "precision": round(m_sym_50["precision"], 4),
            "recall": round(m_sym_50["recall"], 4),
            "pr_auc": round(m_sym_50["pr_auc"], 4),
            "roc_auc": round(m_sym_50["roc_auc"], 4),
            "brier_score": round(m_sym_50["brier_score"], 4)
        })

    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv(OUTPUT_DIR / "threshold_revalidation.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'threshold_revalidation.csv'}")

    # 5. Precompute Predictions on Full Chandipur Dataset
    print("\n[4/8] Precomputing model probabilities on Chandipur...", flush=True)
    p_orig_gam_ch = orig_gam.predict_proba(X_ch_raw)
    
    X_ch_b5 = X_ch_raw.copy()
    X_ch_b5[:, 0] = np.minimum(X_ch_b5[:, 0], CAPE_CLIP_THRESHOLD)
    p_clip_gam_ch = orig_gam.predict_proba(X_ch_b5)

    p_sym_gam_ch = sym_gam.predict_proba(X_ch_b4)
    p_orig_ebm_ch = orig_ebm.predict_proba(X_ch_raw)[:, 1]
    p_sym_ebm_ch = sym_ebm.predict_proba(X_ch_b4)[:, 1]

    # 6. Experiment A & B: Block Bootstrap & Paired Differences (B = 1,000)
    print("\n[5/8] Running Day-Level Block Bootstrap & Paired Comparison (B = 1,000 iterations)...", flush=True)
    np.random.seed(42)
    B = 1000
    
    unique_dates = df_ch["date"].unique()
    K = len(unique_dates)
    # Pre-map dates to indices for speed
    date_to_idx = {d: np.where(df_ch["date"].values == d)[0] for d in unique_dates}

    models_eval = [
        ("Original_Raw_GAM", "1. Original Raw GAM", p_orig_gam_ch, 0.33),
        ("CAPE_Clipped_GAM", "2. CAPE-Clipped GAM (B5)", p_clip_gam_ch, 0.33),
        ("Symmetric_B4_GAM", "3. Symmetric B4 GAM", p_sym_gam_ch, tau_opt_sym_gam),
        ("Original_Raw_EBM", "4. Original Raw EBM", p_orig_ebm_ch, 0.34),
        ("Symmetric_B4_EBM", "5. Symmetric B4 EBM", p_sym_ebm_ch, tau_opt_sym_ebm)
    ]

    boot_metrics = {m_id: {"pr_auc": [], "roc_auc": [], "f1": [], "precision": [], "recall": [], "brier_score": []} for m_id, _, _, _ in models_eval}
    paired_diffs = {
        "diff_pr_auc": [],
        "diff_roc_auc": [],
        "diff_brier_score": [],
        "diff_f1_frozen_thresh": [],
        "diff_f1_tuned_thresh": []
    }

    for b in range(B):
        # Sample K days with replacement
        sample_days = np.random.choice(unique_dates, size=K, replace=True)
        # Concatenate row indices
        boot_idx = np.concatenate([date_to_idx[d] for d in sample_days])
        y_b = y_ch[boot_idx]

        # Evaluate each model on resample
        res_m = {}
        for m_id, _, p_vec, tau in models_eval:
            p_b = p_vec[boot_idx]
            m_dict = evaluate_metrics_fast(y_b, p_b, tau)
            res_m[m_id] = m_dict
            for k_met in boot_metrics[m_id]:
                boot_metrics[m_id][k_met].append(m_dict[k_met])

        # Paired differences: Symmetric B4 GAM vs Original Raw GAM
        p_orig_b = p_orig_gam_ch[boot_idx]
        p_sym_b = p_sym_gam_ch[boot_idx]

        d_pr = res_m["Symmetric_B4_GAM"]["pr_auc"] - res_m["Original_Raw_GAM"]["pr_auc"]
        d_roc = res_m["Symmetric_B4_GAM"]["roc_auc"] - res_m["Original_Raw_GAM"]["roc_auc"]
        d_brier = res_m["Symmetric_B4_GAM"]["brier_score"] - res_m["Original_Raw_GAM"]["brier_score"]
        
        # F1 at frozen threshold (0.33)
        f1_orig_frozen = f1_score(y_b, (p_orig_b >= 0.33).astype(int), zero_division=0)
        f1_sym_frozen = f1_score(y_b, (p_sym_b >= 0.33).astype(int), zero_division=0)
        d_f1_frozen = f1_sym_frozen - f1_orig_frozen

        # F1 at tuned threshold (tau_opt_sym_gam vs 0.33)
        d_f1_tuned = res_m["Symmetric_B4_GAM"]["f1"] - res_m["Original_Raw_GAM"]["f1"]

        paired_diffs["diff_pr_auc"].append(d_pr)
        paired_diffs["diff_roc_auc"].append(d_roc)
        paired_diffs["diff_brier_score"].append(d_brier)
        paired_diffs["diff_f1_frozen_thresh"].append(d_f1_frozen)
        paired_diffs["diff_f1_tuned_thresh"].append(d_f1_tuned)

    # Save Bootstrap Confidence Intervals CSV
    boot_records = []
    for m_id, m_label, _, tau in models_eval:
        for met_name in ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier_score"]:
            vals = np.array(boot_metrics[m_id][met_name])
            boot_records.append({
                "model_id": m_id,
                "model_name": m_label,
                "threshold": tau,
                "metric": met_name,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "ci_lower_95": round(float(np.percentile(vals, 2.5)), 4),
                "median": round(float(np.median(vals)), 4),
                "ci_upper_95": round(float(np.percentile(vals, 97.5)), 4)
            })

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'bootstrap_confidence_intervals.csv'}")

    # Save Paired Model Comparison CSV
    paired_records = []
    for diff_key, label in [
        ("diff_pr_auc", "PR-AUC Difference (Sym B4 GAM - Raw GAM)"),
        ("diff_roc_auc", "ROC-AUC Difference (Sym B4 GAM - Raw GAM)"),
        ("diff_brier_score", "Brier Score Difference (Sym B4 GAM - Raw GAM)"),
        ("diff_f1_frozen_thresh", "F1 Difference at Frozen Threshold 0.33"),
        ("diff_f1_tuned_thresh", f"F1 Difference at Source Tuned Threshold ({tau_opt_sym_gam:.2f} vs 0.33)")
    ]:
        diff_vals = np.array(paired_diffs[diff_key])
        mean_d = float(np.mean(diff_vals))
        std_d = float(np.std(diff_vals))
        ci_low = float(np.percentile(diff_vals, 2.5))
        ci_high = float(np.percentile(diff_vals, 97.5))
        # Two-sided empirical bootstrap p-value
        if mean_d >= 0:
            p_val = 2.0 * float(np.mean(diff_vals <= 0))
        else:
            p_val = 2.0 * float(np.mean(diff_vals >= 0))
        p_val = min(1.0, max(0.0, p_val))

        is_sig = (ci_low > 0 or ci_high < 0)
        paired_records.append({
            "comparison_metric": diff_key,
            "metric_description": label,
            "mean_difference": round(mean_d, 4),
            "std_error": round(std_d, 4),
            "ci_lower_95": round(ci_low, 4),
            "ci_upper_95": round(ci_high, 4),
            "p_value": round(p_val, 4),
            "excludes_zero": bool(is_sig),
            "statistical_significance": "STATISTICALLY SIGNIFICANT (p < 0.05)" if is_sig else "NOT STATISTICALLY SIGNIFICANT (Interval includes zero)"
        })

    paired_df = pd.DataFrame(paired_records)
    paired_df.to_csv(OUTPUT_DIR / "paired_model_comparison.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'paired_model_comparison.csv'}")

    # 7. Experiment C: Transformation Parameter Sensitivity
    print("\n[6/8] Experiment C: Parameter sensitivity sweep (0.80x to 1.20x alpha)...", flush=True)
    scale_factors = [0.80, 0.90, 1.00, 1.10, 1.20]
    sens_records = []

    for k in scale_factors:
        a_cape_k = k * alpha_cape
        a_cin_k = k * alpha_cin
        print(f"  Testing scale factor k = {k:.2f} (alpha_cape={a_cape_k:.1f}, alpha_cin={a_cin_k:.1f})...", flush=True)

        X_train_k = apply_b4(X_fl_train_raw, a_cape_k, a_cin_k)
        X_val_k = apply_b4(X_fl_val_raw, a_cape_k, a_cin_k)
        X_test_k = apply_b4(X_fl_test_raw, a_cape_k, a_cin_k)
        X_ch_k = apply_b4(X_ch_raw, a_cape_k, a_cin_k)

        # Retrain GAM
        gam_k = LogisticGAM(
            s(0, n_splines=15, lam=1.0) +
            s(1, n_splines=15, lam=1.0) +
            s(2, n_splines=15, lam=1.0) +
            s(3, n_splines=15, lam=1.0) +
            s(4, n_splines=15, lam=1.0) +
            s(5, n_splines=15, lam=1.0)
        )
        gam_k.fit(X_train_k, y_fl_train)

        # Tune threshold on Florida val
        p_val_k = gam_k.predict_proba(X_val_k)
        tau_k = find_optimal_threshold(y_fl_val, p_val_k)

        # Evaluate Florida test
        p_fl_test_k = gam_k.predict_proba(X_test_k)
        m_fl_test_k = evaluate_metrics_fast(y_fl_test, p_fl_test_k, tau_k)

        # Evaluate Chandipur
        p_ch_k = gam_k.predict_proba(X_ch_k)
        m_ch_k = evaluate_metrics_fast(y_ch, p_ch_k, tau_k)

        sens_records.append({
            "scale_factor_k": k,
            "alpha_cape": round(a_cape_k, 2),
            "alpha_cin": round(a_cin_k, 2),
            "florida_val_optimal_thresh": tau_k,
            "fl_test_pr_auc": round(m_fl_test_k["pr_auc"], 4),
            "fl_test_roc_auc": round(m_fl_test_k["roc_auc"], 4),
            "fl_test_f1": round(m_fl_test_k["f1"], 4),
            "fl_test_brier": round(m_fl_test_k["brier_score"], 4),
            "chandipur_pr_auc": round(m_ch_k["pr_auc"], 4),
            "chandipur_roc_auc": round(m_ch_k["roc_auc"], 4),
            "chandipur_f1": round(m_ch_k["f1"], 4),
            "chandipur_brier": round(m_ch_k["brier_score"], 4),
            "pr_auc_gain_vs_raw": round(m_ch_k["pr_auc"] - 0.5523, 4)
        })

    sens_df = pd.DataFrame(sens_records)
    sens_df.to_csv(OUTPUT_DIR / "transformation_parameter_sensitivity.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'transformation_parameter_sensitivity.csv'}")

    # 8. Experiment D: Seasonal Generalization & Uncertainty Bounds
    print("\n[7/8] Experiment D & E: Seasonal, Temporal, and Spatial Subgroup Analyses...", flush=True)
    seasonal_records = []
    for s_name in ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]:
        s_mask = (df_ch["season"] == s_name).values
        y_s = y_ch[s_mask]
        p_orig_s = p_orig_gam_ch[s_mask]
        p_sym_s = p_sym_gam_ch[s_mask]

        m_orig = evaluate_metrics_fast(y_s, p_orig_s, 0.33)
        m_sym = evaluate_metrics_fast(y_s, p_sym_s, tau_opt_sym_gam)

        # Seasonal bootstrap for PR-AUC and ROC-AUC
        s_dates = df_ch.loc[s_mask, "date"].unique()
        s_date_map = {d: np.where((df_ch["date"].values == d) & s_mask)[0] for d in s_dates}
        s_boot_pr_diff = []
        s_boot_roc_diff = []
        for _ in range(500):
            sample_d = np.random.choice(s_dates, size=len(s_dates), replace=True)
            idx_sb = np.concatenate([s_date_map[d] for d in sample_d])
            y_sb = y_ch[idx_sb]
            if len(np.unique(y_sb)) > 1:
                pr_orig_sb = average_precision_score(y_sb, p_orig_gam_ch[idx_sb])
                pr_sym_sb = average_precision_score(y_sb, p_sym_gam_ch[idx_sb])
                roc_orig_sb = roc_auc_score(y_sb, p_orig_gam_ch[idx_sb])
                roc_sym_sb = roc_auc_score(y_sb, p_sym_gam_ch[idx_sb])
                s_boot_pr_diff.append(pr_sym_sb - pr_orig_sb)
                s_boot_roc_diff.append(roc_sym_sb - roc_orig_sb)

        pr_diff_ci = [np.percentile(s_boot_pr_diff, 2.5), np.percentile(s_boot_pr_diff, 97.5)] if len(s_boot_pr_diff) > 0 else [0.0, 0.0]

        seasonal_records.append({
            "season": s_name,
            "sample_count": int(np.sum(s_mask)),
            "positive_count": int(np.sum(y_s)),
            "positive_prevalence": round(float(np.mean(y_s)), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "pr_auc_diff_ci_95": f"[{pr_diff_ci[0]:.4f}, {pr_diff_ci[1]:.4f}]",
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "orig_gam_brier": round(m_orig["brier_score"], 4),
            "sym_b4_gam_brier": round(m_sym["brier_score"], 4)
        })

    seasonal_df = pd.DataFrame(seasonal_records)
    seasonal_df.to_csv(OUTPUT_DIR / "seasonal_robustness_comparison.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'seasonal_robustness_comparison.csv'}")

    # 9. Experiment E: Temporal & Spatial Subgroup Analyses
    subgroup_records = []
    
    # 1. By Year
    for yr in sorted(df_ch["year"].unique()):
        mask = (df_ch["year"] == yr).values
        m_orig = evaluate_metrics_fast(y_ch[mask], p_orig_gam_ch[mask], 0.33)
        m_sym = evaluate_metrics_fast(y_ch[mask], p_sym_gam_ch[mask], tau_opt_sym_gam)
        subgroup_records.append({
            "subgroup_type": "Year",
            "subgroup_name": str(yr),
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y_ch[mask])),
            "positive_prevalence": round(float(np.mean(y_ch[mask])), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "sym_b4_superior": bool(m_sym["pr_auc"] >= m_orig["pr_auc"])
        })

    # 2. Leave-One-Year-Out
    for yr in sorted(df_ch["year"].unique()):
        mask = (df_ch["year"] != yr).values
        m_orig = evaluate_metrics_fast(y_ch[mask], p_orig_gam_ch[mask], 0.33)
        m_sym = evaluate_metrics_fast(y_ch[mask], p_sym_gam_ch[mask], tau_opt_sym_gam)
        subgroup_records.append({
            "subgroup_type": "Leave-One-Year-Out",
            "subgroup_name": f"All except {yr}",
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y_ch[mask])),
            "positive_prevalence": round(float(np.mean(y_ch[mask])), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "sym_b4_superior": bool(m_sym["pr_auc"] >= m_orig["pr_auc"])
        })

    # 3. By Grid Point
    for gp in sorted(df_ch["grid_id"].unique()):
        mask = (df_ch["grid_id"] == gp).values
        m_orig = evaluate_metrics_fast(y_ch[mask], p_orig_gam_ch[mask], 0.33)
        m_sym = evaluate_metrics_fast(y_ch[mask], p_sym_gam_ch[mask], tau_opt_sym_gam)
        subgroup_records.append({
            "subgroup_type": "Grid Point",
            "subgroup_name": gp,
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y_ch[mask])),
            "positive_prevalence": round(float(np.mean(y_ch[mask])), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "sym_b4_superior": bool(m_sym["pr_auc"] >= m_orig["pr_auc"])
        })

    # 4. Leave-One-Grid-Point-Out
    for gp in sorted(df_ch["grid_id"].unique()):
        mask = (df_ch["grid_id"] != gp).values
        m_orig = evaluate_metrics_fast(y_ch[mask], p_orig_gam_ch[mask], 0.33)
        m_sym = evaluate_metrics_fast(y_ch[mask], p_sym_gam_ch[mask], tau_opt_sym_gam)
        subgroup_records.append({
            "subgroup_type": "Leave-One-GridPoint-Out",
            "subgroup_name": f"All except {gp}",
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y_ch[mask])),
            "positive_prevalence": round(float(np.mean(y_ch[mask])), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "sym_b4_superior": bool(m_sym["pr_auc"] >= m_orig["pr_auc"])
        })

    # 5. By CAPE Range
    cape_bins = [
        ("CAPE 0-500 J/kg", (X_ch_raw[:, 0] >= 0) & (X_ch_raw[:, 0] < 500)),
        ("CAPE 500-1500 J/kg", (X_ch_raw[:, 0] >= 500) & (X_ch_raw[:, 0] < 1500)),
        ("CAPE 1500-3000 J/kg", (X_ch_raw[:, 0] >= 1500) & (X_ch_raw[:, 0] < 3000)),
        ("CAPE 3000-5000 J/kg", (X_ch_raw[:, 0] >= 3000) & (X_ch_raw[:, 0] < 5000)),
        ("CAPE >5000 J/kg (Extreme)", (X_ch_raw[:, 0] >= 5000))
    ]
    for c_label, mask in cape_bins:
        y_c = y_ch[mask]
        p_orig_c = p_orig_gam_ch[mask]
        p_sym_c = p_sym_gam_ch[mask]
        m_orig = evaluate_metrics_fast(y_c, p_orig_c, 0.33)
        m_sym = evaluate_metrics_fast(y_c, p_sym_c, tau_opt_sym_gam)
        subgroup_records.append({
            "subgroup_type": "CAPE Range",
            "subgroup_name": c_label,
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y_c)),
            "positive_prevalence": round(float(np.mean(y_c)), 4),
            "orig_gam_pr_auc": round(m_orig["pr_auc"], 4),
            "sym_b4_gam_pr_auc": round(m_sym["pr_auc"], 4),
            "pr_auc_difference": round(m_sym["pr_auc"] - m_orig["pr_auc"], 4),
            "orig_gam_roc_auc": round(m_orig["roc_auc"], 4),
            "sym_b4_gam_roc_auc": round(m_sym["roc_auc"], 4),
            "orig_gam_f1": round(m_orig["f1"], 4),
            "sym_b4_gam_f1": round(m_sym["f1"], 4),
            "sym_b4_superior": bool(m_sym["pr_auc"] >= m_orig["pr_auc"])
        })

    subgroup_df = pd.DataFrame(subgroup_records)
    subgroup_df.to_csv(OUTPUT_DIR / "year_gridpoint_robustness.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'year_gridpoint_robustness.csv'}")

    # 10. Render 6 Diagnostic Figures (300 DPI)
    print("\n[8/8] Rendering Diagnostic Figures at 300 DPI...", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Figure 1: Bootstrap Confidence Intervals Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    # PR-AUC
    ax = axes[0]
    pr_data = boot_df[boot_df["metric"] == "pr_auc"].copy()
    y_pos = np.arange(len(pr_data))
    ax.errorbar(pr_data["mean"], y_pos, xerr=[pr_data["mean"] - pr_data["ci_lower_95"], pr_data["ci_upper_95"] - pr_data["mean"]],
                fmt='o', color="#1f77b4", ecolor="#1f77b4", elinewidth=2.5, capsize=5, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pr_data["model_name"], fontsize=9)
    ax.set_xlabel("PR-AUC (95% Block Bootstrap CI)", fontsize=10)
    ax.set_title("Chandipur PR-AUC: 95% Bootstrap Confidence Intervals", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    # ROC-AUC
    ax = axes[1]
    roc_data = boot_df[boot_df["metric"] == "roc_auc"].copy()
    y_pos = np.arange(len(roc_data))
    ax.errorbar(roc_data["mean"], y_pos, xerr=[roc_data["mean"] - roc_data["ci_lower_95"], roc_data["ci_upper_95"] - roc_data["mean"]],
                fmt='s', color="#2ca02c", ecolor="#2ca02c", elinewidth=2.5, capsize=5, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(roc_data["model_name"], fontsize=9)
    ax.set_xlabel("ROC-AUC (95% Block Bootstrap CI)", fontsize=10)
    ax.set_title("Chandipur ROC-AUC: 95% Bootstrap Confidence Intervals", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bootstrap_confidence_intervals_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Paired Model Difference Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax = axes[0]
    sns.histplot(paired_diffs["diff_pr_auc"], bins=35, kde=True, ax=ax, color="#1f77b4", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2.0, label="Zero Difference (Null)")
    ax.axvline(np.mean(paired_diffs["diff_pr_auc"]), color="black", linestyle="-", linewidth=2.0, label=f"Mean Diff = +{np.mean(paired_diffs['diff_pr_auc']):.4f}")
    ax.set_title("Paired Difference in PR-AUC (Symmetric B4 - Raw GAM)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Delta PR-AUC", fontsize=10)
    ax.set_ylabel("Bootstrap Replications", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    sns.histplot(paired_diffs["diff_roc_auc"], bins=35, kde=True, ax=ax, color="#2ca02c", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2.0, label="Zero Difference (Null)")
    ax.axvline(np.mean(paired_diffs["diff_roc_auc"]), color="black", linestyle="-", linewidth=2.0, label=f"Mean Diff = +{np.mean(paired_diffs['diff_roc_auc']):.4f}")
    ax.set_title("Paired Difference in ROC-AUC (Symmetric B4 - Raw GAM)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Delta ROC-AUC", fontsize=10)
    ax.set_ylabel("Bootstrap Replications", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "paired_model_difference_distributions.png", dpi=300)
    plt.close()

    # Figure 3: Parameter Sensitivity Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax = axes[0]
    ax.plot(sens_df["scale_factor_k"], sens_df["chandipur_pr_auc"], marker='o', linewidth=2.2, color="#1f77b4", label="Chandipur Target PR-AUC")
    ax.plot(sens_df["scale_factor_k"], sens_df["fl_test_pr_auc"], marker='s', linewidth=2.0, linestyle="--", color="#ff7f0e", label="Florida Test Holdout PR-AUC")
    ax.axhline(0.5523, color="red", linestyle=":", label="Raw pyGAM Target Baseline (0.5523)")
    ax.set_title("Sensitivity of PR-AUC to Scale Multiplier k", fontsize=11, fontweight="bold")
    ax.set_xlabel("Scale Factor k (around alpha)", fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(sens_df["scale_factor_k"], sens_df["chandipur_roc_auc"], marker='o', linewidth=2.2, color="#2ca02c", label="Chandipur Target ROC-AUC")
    ax.plot(sens_df["scale_factor_k"], sens_df["fl_test_roc_auc"], marker='s', linewidth=2.0, linestyle="--", color="#9467bd", label="Florida Test Holdout ROC-AUC")
    ax.axhline(0.8494, color="red", linestyle=":", label="Raw pyGAM Target Baseline (0.8494)")
    ax.set_title("Sensitivity of ROC-AUC to Scale Multiplier k", fontsize=11, fontweight="bold")
    ax.set_xlabel("Scale Factor k (around alpha)", fontsize=10)
    ax.set_ylabel("ROC-AUC", fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_sensitivity_curves.png", dpi=300)
    plt.close()

    # Figure 4: Seasonal Robustness with Error Bars
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x_pos = np.arange(len(seasonal_df))
    width = 0.35
    ax.bar(x_pos - width/2, seasonal_df["orig_gam_pr_auc"], width, label="Original Raw pyGAM", color="#7f7f7f", alpha=0.8)
    ax.bar(x_pos + width/2, seasonal_df["sym_b4_gam_pr_auc"], width, label="Symmetric B4 pyGAM", color="#1f77b4")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seasonal_df["season"], fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_title("Seasonal Generalization: Original vs. Symmetric B4 pyGAM", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.8)
    for i, row in seasonal_df.iterrows():
        diff = row["pr_auc_difference"]
        sign = "+" if diff >= 0 else ""
        ax.annotate(f"{sign}{diff:.3f}", (i + width/2, row["sym_b4_gam_pr_auc"]),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_robustness_with_ci.png", dpi=300)
    plt.close()

    # Figure 5: Subgroup Analysis (Year and Grid Point)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # Year
    ax = axes[0]
    yr_data = subgroup_df[subgroup_df["subgroup_type"] == "Year"]
    x_yr = np.arange(len(yr_data))
    ax.bar(x_yr - width/2, yr_data["orig_gam_pr_auc"], width, label="Raw pyGAM", color="#7f7f7f")
    ax.bar(x_yr + width/2, yr_data["sym_b4_gam_pr_auc"], width, label="Symmetric B4 pyGAM", color="#1f77b4")
    ax.set_xticks(x_yr)
    ax.set_xticklabels(yr_data["subgroup_name"], fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_title("Subgroup Robustness by Target Year (2022-2025)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=9)

    # Grid Point
    ax = axes[1]
    gp_data = subgroup_df[subgroup_df["subgroup_type"] == "Grid Point"]
    x_gp = np.arange(len(gp_data))
    ax.bar(x_gp - width/2, gp_data["orig_gam_pr_auc"], width, label="Raw pyGAM", color="#7f7f7f")
    ax.bar(x_gp + width/2, gp_data["sym_b4_gam_pr_auc"], width, label="Symmetric B4 pyGAM", color="#2ca02c")
    ax.set_xticks(x_gp)
    ax.set_xticklabels(gp_data["subgroup_name"], fontsize=9, rotation=15)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_title("Subgroup Robustness across 4 Spatial Grid Points", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "year_and_gridpoint_subgroup_analysis.png", dpi=300)
    plt.close()

    # Figure 6: Spline Knot Redistribution & Partial Dependence Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 6A: CAPE Spline Knot Allocation in Physical Space
    ax = axes[0, 0]
    raw_knots_cape = np.linspace(0, 7240, 15)
    sym_knots_b4 = np.linspace(0, alpha_cape * np.log1p(7240.0 / 1000.0), 15)
    # Map symmetric knots back to physical space: x = 1000 * (exp(x_b4 / alpha) - 1)
    sym_knots_physical = 1000.0 * (np.expm1(sym_knots_b4 / alpha_cape))

    y_raw_line = np.ones_like(raw_knots_cape) * 1.0
    y_sym_line = np.ones_like(sym_knots_physical) * 0.5
    ax.scatter(raw_knots_cape, y_raw_line, color="#d62728", s=60, label="Raw pyGAM Spline Knots (Uniform in [0, 7240])", marker='|', linewidth=3)
    ax.scatter(sym_knots_physical, y_sym_line, color="#1f77b4", s=60, label="Symmetric B4 Spline Knots (Log-distributed in physical scale)", marker='|', linewidth=3)
    sns.kdeplot(fl_train["cape_J_kg"].values, ax=ax.twinx(), color="gray", alpha=0.3, label="Florida Training CAPE Density")
    ax.set_title("Spline Knot Allocation across Physical CAPE (J/kg)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Physical CAPE (J/kg)", fontsize=10)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["Symmetric B4", "Raw pyGAM"], fontsize=9)
    ax.set_xlim(-100, 7500)
    ax.legend(fontsize=8, loc="upper right")

    # 6B: Partial Dependence across Full Range (0 - 10,000 J/kg)
    ax = axes[0, 1]
    cape_grid = np.linspace(0, 10000, 500)
    dummy_raw = np.zeros((500, 6))
    dummy_raw[:, 0] = cape_grid
    p_orig_resp = orig_gam.partial_dependence(term=0, X=dummy_raw)

    dummy_sym = np.zeros((500, 6))
    dummy_sym[:, 0] = alpha_cape * np.log1p(cape_grid / 1000.0)
    p_sym_resp = sym_gam.partial_dependence(term=0, X=dummy_sym)

    ax.plot(cape_grid, p_orig_resp, color="#d62728", linestyle="--", linewidth=2.0, label="Original pyGAM (Raw Spline - Collapses at >7,240)")
    ax.plot(cape_grid, p_sym_resp, color="#1f77b4", linewidth=2.2, label="Symmetric B4 pyGAM (Trained on Log-Compressed CAPE)")
    ax.axvline(7240, color="#8c564b", linestyle=":", label="Florida Raw Knot Boundary (7,240 J/kg)")
    ax.set_title("Partial Dependence f(CAPE) in Physical Energy Units", fontsize=11, fontweight="bold")
    ax.set_xlabel("Physical CAPE (J/kg)", fontsize=10)
    ax.set_ylabel("Log-Odds Contribution", fontsize=10)
    ax.set_ylim(-3.5, 3.0)
    ax.legend(fontsize=8, loc="lower right")

    # 6C: CIN Spline Knot Allocation
    ax = axes[1, 0]
    raw_knots_cin = np.linspace(0, 997.4, 15)
    sym_knots_cin_b4 = np.linspace(0, alpha_cin * np.log1p(997.4 / 100.0), 15)
    sym_knots_cin_phys = 100.0 * (np.expm1(sym_knots_cin_b4 / alpha_cin))

    ax.scatter(raw_knots_cin, np.ones_like(raw_knots_cin) * 1.0, color="#d62728", s=60, label="Raw pyGAM Knots (Uniform in [0, 997.4])", marker='|', linewidth=3)
    ax.scatter(sym_knots_cin_phys, np.ones_like(sym_knots_cin_phys) * 0.5, color="#1f77b4", s=60, label="Symmetric B4 Knots (Log-distributed)", marker='|', linewidth=3)
    ax.set_title("Spline Knot Allocation across Physical CIN (J/kg)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Physical CIN (J/kg)", fontsize=10)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["Symmetric B4", "Raw pyGAM"], fontsize=9)
    ax.set_xlim(-10, 1050)
    ax.legend(fontsize=8, loc="upper right")

    # 6D: Partial Dependence across CIN Range (0 - 1,000 J/kg)
    ax = axes[1, 1]
    cin_grid = np.linspace(0, 1000, 500)
    dummy_cin_raw = np.zeros((500, 6))
    dummy_cin_raw[:, 1] = cin_grid
    p_orig_cin = orig_gam.partial_dependence(term=1, X=dummy_cin_raw)

    dummy_cin_sym = np.zeros((500, 6))
    dummy_cin_sym[:, 1] = alpha_cin * np.log1p(cin_grid / 100.0)
    p_sym_cin = sym_gam.partial_dependence(term=1, X=dummy_cin_sym)

    ax.plot(cin_grid, p_orig_cin, color="#d62728", linestyle="--", linewidth=2.0, label="Original pyGAM (Raw Spline)")
    ax.plot(cin_grid, p_sym_cin, color="#1f77b4", linewidth=2.2, label="Symmetric B4 pyGAM (Trained on Log-Compressed CIN)")
    ax.set_title("Partial Dependence f(CIN) in Physical Energy Units", fontsize=11, fontweight="bold")
    ax.set_xlabel("Physical CIN (J/kg)", fontsize=10)
    ax.set_ylabel("Log-Odds Contribution", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpretability_spline_redistribution.png", dpi=300)
    plt.close()

    print(f"  All 6 diagnostic figures rendered successfully at 300 DPI in {OUTPUT_DIR}")
    print(f"Execution completed in {time.time() - t_start:.2f} seconds.")
    print("STEP 7.2 EXECUTION COMPLETE.")

if __name__ == "__main__":
    main()
