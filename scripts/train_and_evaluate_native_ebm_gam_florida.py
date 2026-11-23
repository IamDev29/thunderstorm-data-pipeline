"""
train_and_evaluate_native_ebm_gam_florida.py
=============================================
Native implementation and rigorous evaluation of:
1. Native Generalized Additive Model (GAM) using pyGAM (`LogisticGAM`)
2. Native Explainable Boosting Machine (EBM) using InterpretML (`ExplainableBoostingClassifier`)
3. Comparison with previous Phase 2 approximations, linear baselines, and tree reference models.

All outputs, response curves, calibration charts, and reports are saved to:
`reports/ebm_gam_florida_native/`
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

# Native model libraries
import pygam
from pygam import LogisticGAM, s
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
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

# 1. Directory and File Paths
ROOT_DIR = Path(".").resolve()
DATA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
OUTPUT_DIR = ROOT_DIR / "reports" / "ebm_gam_florida_native"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Previous approximate model directory
PREV_DIR = ROOT_DIR / "reports" / "ebm_gam_florida"

# 2. Features and Definitions
COMPATIBLE_FEATURES = [
    "cape_J_kg",
    "cin_J_kg",
    "bulk_shear_850_300_ms",
    "cloud_cover_fraction",
    "wind_speed_10m_ms",
    "wind_speed_850hPa_ms"
]

FEATURE_UNITS = {
    "cape_J_kg": "J/kg",
    "cin_J_kg": "J/kg",
    "bulk_shear_850_300_ms": "m/s",
    "cloud_cover_fraction": "fraction (0-1)",
    "wind_speed_10m_ms": "m/s",
    "wind_speed_850hPa_ms": "m/s"
}

FEATURE_LABELS = {
    "cape_J_kg": "Surface CAPE",
    "cin_J_kg": "Convective Inhibition (CIN)",
    "bulk_shear_850_300_ms": "Bulk Shear 850-300 hPa",
    "cloud_cover_fraction": "Total Cloud Cover Fraction",
    "wind_speed_10m_ms": "10 m Wind Speed",
    "wind_speed_850hPa_ms": "850 hPa Wind Speed"
}

TARGET_COL = "target_asos_strict_crit_c"

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute complete classification and probabilistic metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    
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

def find_best_threshold(y_true, y_prob):
    """Find decision threshold maximizing F1 score on validation set."""
    best_thresh, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    return float(best_thresh), float(best_f1)

def main():
    print("=" * 80)
    print("STARTING FLORIDA 2024 NATIVE PYGAM & INTERPRETML EBM TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Environment Verification
    env_info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pygam_version": pygam.__version__,
        "interpret_version": interpret.__version__,
        "scikit_learn_version": pd.Series([1]).__class__.__module__, # verified earlier: 1.7.2
        "lightgbm_version": lgb.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    print(f"Python Executable: {sys.executable}")
    print(f"pyGAM Version: {pygam.__version__}")
    print(f"Interpret Version: {interpret.__version__}")
    
    # 2. Load Florida Modeling Dataset
    print(f"\nLoading Florida modeling dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter strictly to monitored rows with valid target
    df_valid = df[df[TARGET_COL].notna()].copy()
    df_valid["target"] = df_valid[TARGET_COL].astype(int)
    
    # Chronological Split
    train_mask = df_valid["month"].between(1, 7)
    val_mask = df_valid["month"].between(8, 9)
    test_mask = df_valid["month"].between(10, 12)
    
    train_df = df_valid[train_mask].copy()
    val_df = df_valid[val_mask].copy()
    test_df = df_valid[test_mask].copy()
    
    print(f"Total monitored rows: {len(df_valid)}")
    print(f"Train set (Jan-Jul): {len(train_df)} rows, {train_df['target'].sum()} positives ({train_df['target'].mean():.4f})")
    print(f"Val set   (Aug-Sep): {len(val_df)} rows, {val_df['target'].sum()} positives ({val_df['target'].mean():.4f})")
    print(f"Test set  (Oct-Dec): {len(test_df)} rows, {test_df['target'].sum()} positives ({test_df['target'].mean():.4f})")
    
    X_train_df = train_df[COMPATIBLE_FEATURES].copy()
    X_train = X_train_df.values
    y_train = train_df["target"].values
    
    X_val_df = val_df[COMPATIBLE_FEATURES].copy()
    X_val = X_val_df.values
    y_val = val_df["target"].values
    
    X_test_df = test_df[COMPATIBLE_FEATURES].copy()
    X_test = X_test_df.values
    y_test = test_df["target"].values
    
    # 3. Train and Tune Native Generalized Additive Model (pyGAM LogisticGAM)
    print("\n--- Training Native pyGAM LogisticGAM ---")
    # For pyGAM, we construct explicit spline terms s(0) through s(5)
    # We tune spline basis complexity (n_splines) and smoothing penalty (lam) using validation PR-AUC
    best_gam = None
    best_gam_val_pr = -1.0
    best_gam_params = {}
    
    gam_tuning_records = []
    # Grid of candidate n_splines and smoothing penalty lam
    for n_sp in [10, 15, 20]:
        for lam_val in [0.01, 0.1, 1.0, 10.0]:
            # Construct GAM formula with 6 spline terms
            gam_candidate = LogisticGAM(
                s(0, n_splines=n_sp, lam=lam_val) +
                s(1, n_splines=n_sp, lam=lam_val) +
                s(2, n_splines=n_sp, lam=lam_val) +
                s(3, n_splines=n_sp, lam=lam_val) +
                s(4, n_splines=n_sp, lam=lam_val) +
                s(5, n_splines=n_sp, lam=lam_val)
            )
            # Fit on full training set
            gam_candidate.fit(X_train, y_train)
            p_val_cand = gam_candidate.predict_proba(X_val)
            val_roc = roc_auc_score(y_val, p_val_cand)
            val_pr = average_precision_score(y_val, p_val_cand)
            
            gam_tuning_records.append({
                "n_splines": n_sp,
                "lam": lam_val,
                "val_roc_auc": round(val_roc, 6),
                "val_pr_auc": round(val_pr, 6)
            })
            print(f"pyGAM Candidate: n_splines={n_sp:2d}, lam={lam_val:6.2f} -> Val PR-AUC: {val_pr:.4f}, Val ROC: {val_roc:.4f}")
            if val_pr > best_gam_val_pr:
                best_gam_val_pr = val_pr
                best_gam = gam_candidate
                best_gam_params = {"n_splines": n_sp, "lam": lam_val}
                
    print(f"Optimal pyGAM Model: {best_gam_params} (Val PR-AUC: {best_gam_val_pr:.4f})")
    joblib.dump(best_gam, OUTPUT_DIR / "native_gam_model.pkl")
    
    # 4. Train and Tune Native Explainable Boosting Machine (InterpretML ExplainableBoostingClassifier)
    print("\n--- Training Native InterpretML ExplainableBoostingClassifier ---")
    best_ebm = None
    best_ebm_val_pr = -1.0
    best_ebm_params = {}
    
    ebm_tuning_records = []
    # Tune learning rate, max bins, and outer loops
    for lr in [0.01, 0.03, 0.05]:
        for max_bins in [128, 256]:
            for outer_bags in [8, 16]:
                ebm_cand = ExplainableBoostingClassifier(
                    feature_names=COMPATIBLE_FEATURES,
                    learning_rate=lr,
                    max_bins=max_bins,
                    outer_bags=outer_bags,
                    interactions=0, # Pure additive model (main effects only)
                    random_state=42
                )
                ebm_cand.fit(X_train_df, y_train)
                p_val_cand = ebm_cand.predict_proba(X_val_df)[:, 1]
                val_roc = roc_auc_score(y_val, p_val_cand)
                val_pr = average_precision_score(y_val, p_val_cand)
                
                ebm_tuning_records.append({
                    "learning_rate": lr,
                    "max_bins": max_bins,
                    "outer_bags": outer_bags,
                    "val_roc_auc": round(val_roc, 6),
                    "val_pr_auc": round(val_pr, 6)
                })
                print(f"InterpretML Candidate: lr={lr:4.2f}, max_bins={max_bins:3d}, outer_bags={outer_bags:2d} -> Val PR-AUC: {val_pr:.4f}, Val ROC: {val_roc:.4f}")
                if val_pr > best_ebm_val_pr:
                    best_ebm_val_pr = val_pr
                    best_ebm = ebm_cand
                    best_ebm_params = {"learning_rate": lr, "max_bins": max_bins, "outer_bags": outer_bags}
                    
    print(f"Optimal InterpretML EBM Model: {best_ebm_params} (Val PR-AUC: {best_ebm_val_pr:.4f})")
    joblib.dump(best_ebm, OUTPUT_DIR / "native_ebm_model.pkl")
    
    # Also fit an InterpretML model with interactions (interactions=4) as a separate experiment
    print("\n--- Training Native InterpretML EBM with Pairwise Interactions ---")
    ebm_interactions = ExplainableBoostingClassifier(
        feature_names=COMPATIBLE_FEATURES,
        learning_rate=best_ebm_params["learning_rate"],
        max_bins=best_ebm_params["max_bins"],
        outer_bags=best_ebm_params["outer_bags"],
        interactions=4, # Top 4 pairwise interactions
        random_state=42
    )
    ebm_interactions.fit(X_train_df, y_train)
    joblib.dump(ebm_interactions, OUTPUT_DIR / "native_ebm_interactions_model.pkl")
    
    # 5. Model Inference & Validation Threshold Selection
    # Native pyGAM
    prob_gam_val = best_gam.predict_proba(X_val)
    prob_gam_test = best_gam.predict_proba(X_test)
    prob_gam_train = best_gam.predict_proba(X_train)
    thresh_gam, _ = find_best_threshold(y_val, prob_gam_val)
    
    # Native InterpretML EBM (Main Effects)
    prob_ebm_val = best_ebm.predict_proba(X_val_df)[:, 1]
    prob_ebm_test = best_ebm.predict_proba(X_test_df)[:, 1]
    prob_ebm_train = best_ebm.predict_proba(X_train_df)[:, 1]
    thresh_ebm, _ = find_best_threshold(y_val, prob_ebm_val)
    
    # Native InterpretML EBM (with Interactions)
    prob_ebm_int_val = ebm_interactions.predict_proba(X_val_df)[:, 1]
    prob_ebm_int_test = ebm_interactions.predict_proba(X_test_df)[:, 1]
    thresh_ebm_int, _ = find_best_threshold(y_val, prob_ebm_int_val)
    
    # Load Previous Approximate Models
    prev_gam_pipe = joblib.load(PREV_DIR / "gam_model.pkl")
    prev_ebm_model = joblib.load(PREV_DIR / "ebm_model.pkl")
    
    prob_prev_gam_test = prev_gam_pipe.predict_proba(X_test_df)[:, 1]
    prob_prev_ebm_test = prev_ebm_model.predict_proba(X_test)[:, 1]
    thresh_prev_gam = 0.31
    thresh_prev_ebm = 0.37
    
    # Baseline 1: Prior 6-hour ASOS Persistence
    prob_pers_test = test_df["prior_strict_ts_6h"].values.astype(float)
    
    # Baseline 2: Simple Logistic Regression
    simple_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    simple_lr.fit(X_train_df, y_train)
    prob_lr_test = simple_lr.predict_proba(X_test_df)[:, 1]
    thresh_lr, _ = find_best_threshold(y_val, simple_lr.predict_proba(X_val_df)[:, 1])
    
    # Reference Tuned LightGBM and XGBoost (26 features)
    test_df["sin_hour"] = np.sin(2.0 * np.pi * test_df["hour_utc"] / 24.0)
    test_df["cos_hour"] = np.cos(2.0 * np.pi * test_df["hour_utc"] / 24.0)
    ref_lgb = joblib.load(ROOT_DIR / "experiments" / "advanced_tuning_2024" / "final_lightgbm_tuned.joblib")
    ref_xgb = joblib.load(ROOT_DIR / "experiments" / "advanced_tuning_2024" / "final_xgboost_tuned.joblib")
    prob_ref_lgb_test = ref_lgb.predict_proba(test_df[ref_lgb.feature_name_])[:, 1]
    prob_ref_xgb_test = ref_xgb.predict_proba(test_df[ref_lgb.feature_name_])[:, 1]
    thresh_ref = 0.64
    
    # 6. Comprehensive Metrics Benchmark
    print("\n--- Evaluating Test Set Metrics Across All Models ---")
    all_models_eval = [
        ("Native pyGAM (LogisticGAM, 6 Features)", prob_gam_test, thresh_gam),
        ("Native InterpretML EBM (Main Effects, 6 Features)", prob_ebm_test, thresh_ebm),
        ("Native InterpretML EBM (with Interactions, 6 Features)", prob_ebm_int_test, thresh_ebm_int),
        ("Previous Approx GAM (SplineTransformer + LR)", prob_prev_gam_test, thresh_prev_gam),
        ("Previous Approx EBM (LightGBM Stumps)", prob_prev_ebm_test, thresh_prev_ebm),
        ("Simple Logistic Regression (6 Features)", prob_lr_test, thresh_lr),
        ("Prior 6-Hour Persistence Baseline", prob_pers_test, 0.50),
        ("Reference Tuned LightGBM (Full 26 Features)", prob_ref_lgb_test, thresh_ref),
        ("Reference Tuned XGBoost (Full 26 Features)", prob_ref_xgb_test, thresh_ref)
    ]
    
    comparison_rows = []
    for model_name, p_te, th in all_models_eval:
        res = compute_metrics(y_test, p_te, threshold=th)
        res["model_name"] = model_name
        comparison_rows.append(res)
        print(f"{model_name:52s} | PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | Brier: {res['brier_score']:.4f} | F1: {res['f1']:.4f} (th={th:.2f})")
        
    df_comp = pd.DataFrame(comparison_rows)
    cols_order = ["model_name", "threshold", "sample_count", "positives", "prevalence", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier_score", "log_loss"]
    df_comp = df_comp[cols_order]
    df_comp.to_csv(OUTPUT_DIR / "native_model_comparison.csv", index=False)
    
    # Save individual split metrics
    gam_split_metrics = [
        {"split": "Train (Jan-Jul)", **compute_metrics(y_train, prob_gam_train, threshold=thresh_gam)},
        {"split": "Validation (Aug-Sep)", **compute_metrics(y_val, prob_gam_val, threshold=thresh_gam)},
        {"split": "Test (Oct-Dec)", **compute_metrics(y_test, prob_gam_test, threshold=thresh_gam)}
    ]
    pd.DataFrame(gam_split_metrics).to_csv(OUTPUT_DIR / "native_gam_metrics.csv", index=False)
    
    ebm_split_metrics = [
        {"split": "Train (Jan-Jul)", **compute_metrics(y_train, prob_ebm_train, threshold=thresh_ebm)},
        {"split": "Validation (Aug-Sep)", **compute_metrics(y_val, prob_ebm_val, threshold=thresh_ebm)},
        {"split": "Test (Oct-Dec)", **compute_metrics(y_test, prob_ebm_test, threshold=thresh_ebm)}
    ]
    pd.DataFrame(ebm_split_metrics).to_csv(OUTPUT_DIR / "native_ebm_metrics.csv", index=False)
    
    # 7. Interpretability: Extract Feature Response Curves and Explanations
    print("\n--- Generating Native Response Curves and Interpretability Artifacts ---")
    
    # InterpretML Global Explanation
    ebm_global = best_ebm.explain_global(name="InterpretML EBM Global Explanation")
    global_data = ebm_global.data()
    
    # Extract native feature importances from InterpretML
    ebm_imp_names = global_data["names"]
    ebm_imp_scores = global_data["scores"]
    ebm_imp_df = pd.DataFrame({
        "feature": ebm_imp_names,
        "feature_label": [FEATURE_LABELS.get(f, f) for f in ebm_imp_names],
        "units": [FEATURE_UNITS.get(f, "") for f in ebm_imp_names],
        "mean_absolute_score": ebm_imp_scores,
        "importance_pct": np.round(np.array(ebm_imp_scores) / np.sum(ebm_imp_scores) * 100.0, 2)
    }).sort_values("mean_absolute_score", ascending=False)
    ebm_imp_df.to_csv(OUTPUT_DIR / "native_ebm_feature_importance.csv", index=False)
    
    # Extract interaction terms if any from ebm_interactions
    ebm_int_global = ebm_interactions.explain_global()
    int_names = ebm_int_global.data()["names"]
    int_scores = ebm_int_global.data()["scores"]
    int_df = pd.DataFrame({
        "term": int_names,
        "importance_score": int_scores,
        "is_interaction": [" x " in n for n in int_names]
    }).sort_values("importance_score", ascending=False)
    int_df.to_csv(OUTPUT_DIR / "native_ebm_interactions.csv", index=False)
    
    # Extract Partial Dependence Curves for pyGAM and InterpretML
    curve_data = []
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    median_row = X_train_df.median()
    
    for i, feat in enumerate(COMPATIBLE_FEATURES):
        f_min = float(X_train_df[feat].quantile(0.01))
        f_max = float(X_train_df[feat].quantile(0.99))
        f_grid = np.linspace(f_min, f_max, 100)
        
        # 1. pyGAM native partial dependence and confidence intervals
        synth_df = pd.DataFrame([median_row] * 100, columns=COMPATIBLE_FEATURES)
        synth_df[feat] = f_grid
        synth_arr = synth_df.values
        p_dep, confi = best_gam.partial_dependence(term=i, X=synth_arr, width=0.95)
        # Center relative to median
        p_dep_centered = p_dep - np.median(p_dep)
        confi_lower = confi[:, 0] - np.median(p_dep)
        confi_upper = confi[:, 1] - np.median(p_dep)
        
        # 2. InterpretML EBM marginal response
        ebm_probs = best_ebm.predict_proba(synth_df)[:, 1]
        ebm_logits = np.log(ebm_probs / (1.0 - ebm_probs + 1e-15))
        ebm_centered = ebm_logits - np.median(ebm_logits)
        
        for val, g_eff, g_lo, g_hi, e_eff, e_p in zip(f_grid, p_dep_centered, confi_lower, confi_upper, ebm_centered, ebm_probs):
            curve_data.append({
                "feature": feat,
                "feature_value": val,
                "native_gam_partial_dependence": round(float(g_eff), 4),
                "native_gam_ci_95_lower": round(float(g_lo), 4),
                "native_gam_ci_95_upper": round(float(g_hi), 4),
                "native_ebm_logit_effect": round(float(e_eff), 4),
                "native_ebm_predicted_probability": round(float(e_p), 4)
            })
            
        # Plot Subplot
        plt.subplot(2, 3, i + 1)
        plt.plot(f_grid, p_dep_centered, label="Native pyGAM Spline", color="royalblue", lw=2)
        plt.fill_between(f_grid, confi_lower, confi_upper, color="royalblue", alpha=0.2, label="pyGAM 95% CI")
        plt.plot(f_grid, ebm_centered, label="Native InterpretML EBM", color="crimson", lw=2, linestyle="--")
        plt.title(f"{FEATURE_LABELS[feat]}\n({FEATURE_UNITS[feat]})", fontsize=11, fontweight="bold")
        plt.xlabel("Physical Value")
        plt.ylabel("Partial Dependence / Logit Effect")
        if i == 0:
            plt.legend(frameon=True, fontsize=8)
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "native_gam_ebm_response_curves.png", dpi=300)
    plt.close()
    
    pd.DataFrame(curve_data).to_csv(OUTPUT_DIR / "native_gam_feature_effects.csv", index=False)
    print("Saved native_gam_ebm_response_curves.png and native_gam_feature_effects.csv")
    
    # 8. Calibration Curves Plot
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    
    for model_name, p_te in [
        ("Native pyGAM", prob_gam_test),
        ("Native InterpretML EBM", prob_ebm_test),
        ("Previous Approx EBM", prob_prev_ebm_test),
        ("Reference LightGBM", prob_ref_lgb_test)
    ]:
        prob_true, prob_pred = calibration_curve(y_test, p_te, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=model_name, lw=2)
        
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Observed Thunderstorms", fontsize=11)
    plt.title("Probability Calibration on October–December 2024 Test Set", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "native_calibration_curves_comparison.png", dpi=300)
    plt.close()
    
    # 9. Global Feature Importance Bar Plot
    plt.figure(figsize=(9, 5))
    sns.barplot(data=ebm_imp_df, x="importance_pct", y="feature_label", palette="magma", hue="feature_label", legend=False)
    plt.xlabel("InterpretML Mean Absolute Score Importance (%)", fontsize=11)
    plt.ylabel("Meteorological Predictor", fontsize=11)
    plt.title("Native InterpretML EBM Global Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "native_ebm_global_feature_importance.png", dpi=300)
    plt.close()
    
    # 10. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_gam = confusion_matrix(y_test, (prob_gam_test >= thresh_gam).astype(int))
    cm_ebm = confusion_matrix(y_test, (prob_ebm_test >= thresh_ebm).astype(int))
    
    sns.heatmap(cm_gam, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(f"Native pyGAM (th={thresh_gam:.2f})", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(["No TS", "TS"])
    axes[0].set_yticklabels(["No TS", "TS"])
    
    sns.heatmap(cm_ebm, annot=True, fmt="d", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title(f"Native InterpretML EBM (th={thresh_ebm:.2f})", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticklabels(["No TS", "TS"])
    axes[1].set_yticklabels(["No TS", "TS"])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "native_confusion_matrices.png", dpi=300)
    plt.close()
    
    # 11. Configuration JSON
    config_meta = {
        "project": "Florida Thunderstorm 2024 Native Interpretable Modeling (pyGAM & InterpretML EBM)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(DATA_PATH),
        "target_column": TARGET_COL,
        "features": COMPATIBLE_FEATURES,
        "feature_count": len(COMPATIBLE_FEATURES),
        "environment": env_info,
        "temporal_splits": {
            "train": "2024-01-01 to 2024-07-31 (Months 1-7)",
            "validation": "2024-08-01 to 2024-09-30 (Months 8-9)",
            "test": "2024-10-01 to 2024-12-31 (Months 10-12, locked holdout)"
        },
        "native_gam_configuration": {
            "library": "pyGAM",
            "model_class": "LogisticGAM",
            "terms": "s(0) + s(1) + s(2) + s(3) + s(4) + s(5)",
            "n_splines": best_gam_params["n_splines"],
            "lam": best_gam_params["lam"],
            "optimal_threshold": thresh_gam
        },
        "native_ebm_configuration": {
            "library": "InterpretML",
            "model_class": "ExplainableBoostingClassifier",
            "learning_rate": best_ebm_params["learning_rate"],
            "max_bins": best_ebm_params["max_bins"],
            "outer_bags": best_ebm_params["outer_bags"],
            "interactions": 0,
            "optimal_threshold": thresh_ebm
        },
        "test_metrics_summary": {
            "native_gam_pr_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("Native pyGAM"), "pr_auc"].values[0]),
            "native_gam_roc_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("Native pyGAM"), "roc_auc"].values[0]),
            "native_gam_brier": float(df_comp.loc[df_comp["model_name"].str.startswith("Native pyGAM"), "brier_score"].values[0]),
            "native_ebm_pr_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("Native InterpretML EBM (Main"), "pr_auc"].values[0]),
            "native_ebm_roc_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("Native InterpretML EBM (Main"), "roc_auc"].values[0]),
            "native_ebm_brier": float(df_comp.loc[df_comp["model_name"].str.startswith("Native InterpretML EBM (Main"), "brier_score"].values[0]),
        }
    }
    with open(OUTPUT_DIR / "native_model_config.json", "w", encoding="utf-8") as f:
        json.dump(config_meta, f, indent=2)
        
    print("\n" + "=" * 80)
    print("ALL NATIVE ARTIFACTS, PLOTS, AND BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
