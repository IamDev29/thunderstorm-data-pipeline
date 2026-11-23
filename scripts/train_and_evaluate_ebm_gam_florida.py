"""
train_and_evaluate_ebm_gam_florida.py
======================================
Production script to train, tune, evaluate, and interpret:
1. Generalized Additive Model (GAM) using cubic splines and penalized logistic regression
2. Explainable Boosting Machine (EBM) using pure additive tree stumps (max_depth=1)
3. Full-feature benchmarks and comparison against existing reference baselines.

Saves all artifacts, evaluation metrics, feature response curves, and plots into:
`reports/ebm_gam_florida/`
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
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
OUTPUT_DIR = ROOT_DIR / "reports" / "ebm_gam_florida"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 2. Feature Sets
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
    """Find decision threshold maximizing F1 score."""
    best_thresh, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    return float(best_thresh), float(best_f1)

def main():
    print("=" * 80)
    print("STARTING FLORIDA 2024 GAM & EBM TRAINING AND EVALUATION PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    print(f"\nLoading Florida modeling dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter strictly to monitored rows with valid target
    df_valid = df[df[TARGET_COL].notna()].copy()
    df_valid["target"] = df_valid[TARGET_COL].astype(int)
    
    # Established chronological split:
    # Train: Months 1-7 (Jan-Jul)
    # Val:   Months 8-9 (Aug-Sep)
    # Test:  Months 10-12 (Oct-Dec)
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
    
    X_train = train_df[COMPATIBLE_FEATURES].copy()
    y_train = train_df["target"].values
    
    X_val = val_df[COMPATIBLE_FEATURES].copy()
    y_val = val_df["target"].values
    
    X_test = test_df[COMPATIBLE_FEATURES].copy()
    y_test = test_df["target"].values
    
    # 2. Train and Tune Generalized Additive Model (GAM)
    print("\n--- Training Generalized Additive Model (Spline Logistic GAM) ---")
    best_gam_pipe = None
    best_gam_val_pr = -1.0
    best_gam_params = {}
    
    gam_tuning_records = []
    for n_knots in [4, 6, 8, 10]:
        for C_reg in [0.01, 0.05, 0.1, 0.5, 1.0]:
            pipe = Pipeline([
                ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C_reg, max_iter=500, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            p_val = pipe.predict_proba(X_val)[:, 1]
            val_roc = roc_auc_score(y_val, p_val)
            val_pr = average_precision_score(y_val, p_val)
            gam_tuning_records.append({
                "n_knots": n_knots,
                "C_regularization": C_reg,
                "val_roc_auc": round(val_roc, 6),
                "val_pr_auc": round(val_pr, 6)
            })
            if val_pr > best_gam_val_pr:
                best_gam_val_pr = val_pr
                best_gam_pipe = pipe
                best_gam_params = {"n_knots": n_knots, "C_regularization": C_reg}
                
    print(f"Best GAM Hyperparameters: {best_gam_params} (Val PR-AUC: {best_gam_val_pr:.4f})")
    joblib.dump(best_gam_pipe, OUTPUT_DIR / "gam_model.pkl")
    
    # 3. Train and Tune Explainable Boosting Machine (EBM)
    print("\n--- Training Explainable Boosting Machine (Tree-based Additive Model) ---")
    best_ebm_model = None
    best_ebm_val_pr = -1.0
    best_ebm_params = {}
    
    ebm_tuning_records = []
    for lr in [0.01, 0.03, 0.05]:
        for n_est in [100, 200, 300, 500]:
            for l2 in [0.1, 1.0, 5.0]:
                clf = lgb.LGBMClassifier(
                    max_depth=1, # Exact pure additive model (no interactions)
                    learning_rate=lr,
                    n_estimators=n_est,
                    reg_lambda=l2,
                    subsample=0.8,
                    colsample_bytree=1.0,
                    random_state=42,
                    verbose=-1
                )
                clf.fit(X_train, y_train)
                p_val = clf.predict_proba(X_val)[:, 1]
                val_roc = roc_auc_score(y_val, p_val)
                val_pr = average_precision_score(y_val, p_val)
                ebm_tuning_records.append({
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "reg_lambda": l2,
                    "val_roc_auc": round(val_roc, 6),
                    "val_pr_auc": round(val_pr, 6)
                })
                if val_pr > best_ebm_val_pr:
                    best_ebm_val_pr = val_pr
                    best_ebm_model = clf
                    best_ebm_params = {"learning_rate": lr, "n_estimators": n_est, "reg_lambda": l2}
                    
    print(f"Best EBM Hyperparameters: {best_ebm_params} (Val PR-AUC: {best_ebm_val_pr:.4f})")
    joblib.dump(best_ebm_model, OUTPUT_DIR / "ebm_model.pkl")
    
    # Also fit EBM with depth=2 to check for important pairwise interactions
    ebm_interactions_model = lgb.LGBMClassifier(
        max_depth=2, # Captures pairwise feature interactions
        learning_rate=best_ebm_params["learning_rate"],
        n_estimators=best_ebm_params["n_estimators"],
        reg_lambda=best_ebm_params["reg_lambda"],
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    ebm_interactions_model.fit(X_train, y_train)
    joblib.dump(ebm_interactions_model, OUTPUT_DIR / "ebm_interactions_model.pkl")
    
    # 4. Model Predictions & Optimal Threshold Selection on Validation Data
    # GAM
    prob_gam_val = best_gam_pipe.predict_proba(X_val)[:, 1]
    prob_gam_test = best_gam_pipe.predict_proba(X_test)[:, 1]
    prob_gam_train = best_gam_pipe.predict_proba(X_train)[:, 1]
    thresh_gam, _ = find_best_threshold(y_val, prob_gam_val)
    
    # EBM
    prob_ebm_val = best_ebm_model.predict_proba(X_val)[:, 1]
    prob_ebm_test = best_ebm_model.predict_proba(X_test)[:, 1]
    prob_ebm_train = best_ebm_model.predict_proba(X_train)[:, 1]
    thresh_ebm, _ = find_best_threshold(y_val, prob_ebm_val)
    
    # EBM with interactions
    prob_ebm_int_test = ebm_interactions_model.predict_proba(X_test)[:, 1]
    thresh_ebm_int, _ = find_best_threshold(y_val, ebm_interactions_model.predict_proba(X_val)[:, 1])
    
    # 5. Baseline Predictions on Test Data
    # Baseline 1: Prior 6-hour ASOS Persistence
    prob_pers_test = test_df["prior_strict_ts_6h"].values.astype(float)
    
    # Baseline 2: Simple Logistic Regression (Linear on the 6 compatible features)
    simple_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    simple_lr.fit(X_train, y_train)
    prob_lr_test = simple_lr.predict_proba(X_test)[:, 1]
    thresh_lr, _ = find_best_threshold(y_val, simple_lr.predict_proba(X_val)[:, 1])
    
    # Baseline 3: Reference Tuned LightGBM (Full 26 features)
    ref_lgb = joblib.load(ROOT_DIR / "experiments" / "advanced_tuning_2024" / "final_lightgbm_tuned.joblib")
    # Add sin_hour / cos_hour to test_df if not present
    test_df["sin_hour"] = np.sin(2.0 * np.pi * test_df["hour_utc"] / 24.0)
    test_df["cos_hour"] = np.cos(2.0 * np.pi * test_df["hour_utc"] / 24.0)
    prob_ref_lgb_test = ref_lgb.predict_proba(test_df[ref_lgb.feature_name_])[:, 1]
    
    # Baseline 4: Reference Tuned XGBoost (Full 26 features)
    ref_xgb = joblib.load(ROOT_DIR / "experiments" / "advanced_tuning_2024" / "final_xgboost_tuned.joblib")
    prob_ref_xgb_test = ref_xgb.predict_proba(test_df[ref_lgb.feature_name_])[:, 1]
    
    # Thresholds for reference models from existing threshold_selection.json (0.64)
    thresh_ref = 0.64
    
    # 6. Compute Comprehensive Metrics
    print("\n--- Evaluating Test Set Metrics Across All Models ---")
    models_eval = [
        ("GAM (Spline, 6 Compatible Features)", prob_gam_test, thresh_gam),
        ("EBM (Additive Trees, 6 Compatible Features)", prob_ebm_test, thresh_ebm),
        ("EBM with Interactions (Depth=2, 6 Features)", prob_ebm_int_test, thresh_ebm_int),
        ("Prior 6-Hour Persistence Baseline", prob_pers_test, 0.5),
        ("Simple Logistic Regression (6 Features)", prob_lr_test, thresh_lr),
        ("Reference Tuned LightGBM (Full 26 Features)", prob_ref_lgb_test, thresh_ref),
        ("Reference Tuned XGBoost (Full 26 Features)", prob_ref_xgb_test, thresh_ref)
    ]
    
    comparison_rows = []
    for model_name, p_te, th in models_eval:
        res = compute_metrics(y_test, p_te, threshold=th)
        res["model_name"] = model_name
        comparison_rows.append(res)
        print(f"{model_name:44s} | PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | Brier: {res['brier_score']:.4f} | F1: {res['f1']:.4f} (th={th:.2f})")
        
    df_comp = pd.DataFrame(comparison_rows)
    cols_order = ["model_name", "threshold", "sample_count", "positives", "prevalence", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier_score", "log_loss"]
    df_comp = df_comp[cols_order]
    df_comp.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    
    # Save individual GAM and EBM metrics across train/val/test splits
    gam_split_metrics = [
        {"split": "Train (Jan-Jul)", **compute_metrics(y_train, prob_gam_train, threshold=thresh_gam)},
        {"split": "Validation (Aug-Sep)", **compute_metrics(y_val, prob_gam_val, threshold=thresh_gam)},
        {"split": "Test (Oct-Dec)", **compute_metrics(y_test, prob_gam_test, threshold=thresh_gam)}
    ]
    pd.DataFrame(gam_split_metrics).to_csv(OUTPUT_DIR / "gam_model_metrics.csv", index=False)
    
    ebm_split_metrics = [
        {"split": "Train (Jan-Jul)", **compute_metrics(y_train, prob_ebm_train, threshold=thresh_ebm)},
        {"split": "Validation (Aug-Sep)", **compute_metrics(y_val, prob_ebm_val, threshold=thresh_ebm)},
        {"split": "Test (Oct-Dec)", **compute_metrics(y_test, prob_ebm_test, threshold=thresh_ebm)}
    ]
    pd.DataFrame(ebm_split_metrics).to_csv(OUTPUT_DIR / "ebm_model_metrics.csv", index=False)
    
    # 7. Interpretability: Extract Feature Response Curves and Importance
    print("\n--- Generating Feature Response Curves and Interpretability Artifacts ---")
    
    # EBM Feature Importance (Gain & Split counts)
    ebm_imp_gain = best_ebm_model.booster_.feature_importance(importance_type="gain")
    ebm_imp_split = best_ebm_model.booster_.feature_importance(importance_type="split")
    ebm_imp_df = pd.DataFrame({
        "feature": COMPATIBLE_FEATURES,
        "feature_label": [FEATURE_LABELS[f] for f in COMPATIBLE_FEATURES],
        "units": [FEATURE_UNITS[f] for f in COMPATIBLE_FEATURES],
        "importance_gain": ebm_imp_gain,
        "importance_gain_pct": np.round(ebm_imp_gain / np.sum(ebm_imp_gain) * 100.0, 2),
        "split_count": ebm_imp_split
    }).sort_values("importance_gain", ascending=False)
    ebm_imp_df.to_csv(OUTPUT_DIR / "ebm_feature_importance.csv", index=False)
    
    # EBM Interactions Summary
    ebm_int_gain = ebm_interactions_model.booster_.feature_importance(importance_type="gain")
    ebm_int_df = pd.DataFrame({
        "feature": COMPATIBLE_FEATURES,
        "importance_gain": ebm_int_gain,
        "importance_gain_pct": np.round(ebm_int_gain / np.sum(ebm_int_gain) * 100.0, 2)
    }).sort_values("importance_gain", ascending=False)
    ebm_int_df.to_csv(OUTPUT_DIR / "ebm_interactions.csv", index=False)
    
    # Compute Partial Dependence / Response Curves across Feature Ranges
    # Using baseline sample: medians of all features
    median_row = X_train.median()
    
    curve_data = []
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    for i, feat in enumerate(COMPATIBLE_FEATURES):
        f_min = float(X_train[feat].quantile(0.01))
        f_max = float(X_train[feat].quantile(0.99))
        f_grid = np.linspace(f_min, f_max, 100)
        
        # Synthetic evaluation matrix
        synth_df = pd.DataFrame([median_row] * 100, columns=COMPATIBLE_FEATURES)
        synth_df[feat] = f_grid
        
        # Predict with GAM
        gam_probs = best_gam_pipe.predict_proba(synth_df)[:, 1]
        gam_logits = np.log(gam_probs / (1.0 - gam_probs + 1e-15))
        # Center logits relative to median
        gam_rel_effect = gam_logits - np.median(gam_logits)
        
        # Predict with EBM
        ebm_probs = best_ebm_model.predict_proba(synth_df)[:, 1]
        ebm_logits = np.log(ebm_probs / (1.0 - ebm_probs + 1e-15))
        ebm_rel_effect = ebm_logits - np.median(ebm_logits)
        
        for val, g_p, g_eff, e_p, e_eff in zip(f_grid, gam_probs, gam_rel_effect, ebm_probs, ebm_rel_effect):
            curve_data.append({
                "feature": feat,
                "feature_value": val,
                "gam_predicted_probability": round(g_p, 4),
                "gam_relative_logit_effect": round(g_eff, 4),
                "ebm_predicted_probability": round(e_p, 4),
                "ebm_relative_logit_effect": round(e_eff, 4)
            })
            
        # Plot comparison
        plt.subplot(2, 3, i + 1)
        plt.plot(f_grid, gam_rel_effect, label="GAM (Cubic Splines)", color="royalblue", lw=2)
        plt.plot(f_grid, ebm_rel_effect, label="EBM (Additive Trees)", color="crimson", lw=2, linestyle="--")
        plt.title(f"{FEATURE_LABELS[feat]}\n({FEATURE_UNITS[feat]})", fontsize=11, fontweight="bold")
        plt.xlabel("Physical Value")
        plt.ylabel("Relative Logit Effect")
        if i == 0:
            plt.legend(frameon=True, fontsize=9)
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gam_ebm_response_curves_comparison.png", dpi=300)
    plt.close()
    
    pd.DataFrame(curve_data).to_csv(OUTPUT_DIR / "gam_feature_effects.csv", index=False)
    print("Saved gam_ebm_response_curves_comparison.png and gam_feature_effects.csv")
    
    # 8. Calibration Curves Plot
    print("\n--- Generating Calibration Curves and Confusion Matrices ---")
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    
    for model_name, p_te, _ in [
        ("GAM (6 Features)", prob_gam_test, None),
        ("EBM (6 Features)", prob_ebm_test, None),
        ("Reference LightGBM", prob_ref_lgb_test, None)
    ]:
        prob_true, prob_pred = calibration_curve(y_test, p_te, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=model_name, lw=2)
        
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Observed Thunderstorms", fontsize=11)
    plt.title("Probability Calibration on October–December 2024 Test Set", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curves_comparison.png", dpi=300)
    plt.close()
    
    # 9. Global Feature Importance Bar Plot (EBM)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=ebm_imp_df, x="importance_gain_pct", y="feature_label", palette="viridis")
    plt.xlabel("Relative Gain Importance (%)", fontsize=11)
    plt.ylabel("Meteorological Predictor", fontsize=11)
    plt.title("Explainable Boosting Machine (EBM) Global Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ebm_global_feature_importance.png", dpi=300)
    plt.close()
    
    # 10. Confusion Matrix Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_gam = confusion_matrix(y_test, (prob_gam_test >= thresh_gam).astype(int))
    cm_ebm = confusion_matrix(y_test, (prob_ebm_test >= thresh_ebm).astype(int))
    
    sns.heatmap(cm_gam, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(f"GAM Confusion Matrix (th={thresh_gam:.2f})", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(["No TS", "TS"])
    axes[0].set_yticklabels(["No TS", "TS"])
    
    sns.heatmap(cm_ebm, annot=True, fmt="d", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title(f"EBM Confusion Matrix (th={thresh_ebm:.2f})", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticklabels(["No TS", "TS"])
    axes[1].set_yticklabels(["No TS", "TS"])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_gam_ebm.png", dpi=300)
    plt.close()
    
    # 11. Configuration Metadata
    config_meta = {
        "project": "Florida Thunderstorm 2024 Interpretable Modeling (GAM & EBM)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(DATA_PATH),
        "target_column": TARGET_COL,
        "features": COMPATIBLE_FEATURES,
        "feature_count": len(COMPATIBLE_FEATURES),
        "temporal_splits": {
            "train": "2024-01-01 to 2024-07-31 (Months 1-7)",
            "validation": "2024-08-01 to 2024-09-30 (Months 8-9)",
            "test": "2024-10-01 to 2024-12-31 (Months 10-12, locked holdout)"
        },
        "gam_configuration": {
            "architecture": "SplineTransformer(degree=3) + StandardScaler + LogisticRegression(L2)",
            "n_knots": best_gam_params["n_knots"],
            "C_regularization": best_gam_params["C_regularization"],
            "optimal_threshold": thresh_gam
        },
        "ebm_configuration": {
            "architecture": "LightGBM Pure Additive Trees (max_depth=1, boosting_type=gbdt)",
            "learning_rate": best_ebm_params["learning_rate"],
            "n_estimators": best_ebm_params["n_estimators"],
            "reg_lambda": best_ebm_params["reg_lambda"],
            "subsample": 0.8,
            "optimal_threshold": thresh_ebm
        },
        "test_metrics_summary": {
            "gam_pr_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("GAM"), "pr_auc"].values[0]),
            "gam_roc_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("GAM"), "roc_auc"].values[0]),
            "gam_brier": float(df_comp.loc[df_comp["model_name"].str.startswith("GAM"), "brier_score"].values[0]),
            "ebm_pr_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("EBM (Additive"), "pr_auc"].values[0]),
            "ebm_roc_auc": float(df_comp.loc[df_comp["model_name"].str.startswith("EBM (Additive"), "roc_auc"].values[0]),
            "ebm_brier": float(df_comp.loc[df_comp["model_name"].str.startswith("EBM (Additive"), "brier_score"].values[0]),
        }
    }
    with open(OUTPUT_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config_meta, f, indent=2)
        
    print("\n" + "=" * 80)
    print("ALL MODEL ARTIFACTS, PLOTS, AND METRICS GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
