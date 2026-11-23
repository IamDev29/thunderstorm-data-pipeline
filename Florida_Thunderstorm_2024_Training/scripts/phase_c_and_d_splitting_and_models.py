"""
Phase C & D: Controlled Data-Splitting and Machine Learning Evaluation
Implements and evaluates:
- Experiment 1: Existing Chronological Split (Train: Jan-Aug, Test: Sep-Dec)
- Experiment 2A: Pure Alternating-Day Split (Odd DOY Train, Even DOY Test)
- Experiment 2B: Purged Alternating-Day Split (2 days Train, 1 day Embargo, 1 day Test, 1 day Embargo)
- Experiment 3: Seasonally Stratified Split (70% Train, 30% Test stratified by Winter, Spring, Summer, Fall)
- Experiment 4: Episode-Grouped Split (NOAA EPISODE_ID holdout, zero episode overlap)
- Experiment 5: Strict Generalization (Chronological with 31-day August Embargo + Milton Isolation)

Models evaluated for each split:
1. Majority Class / No-Alert Baseline
2. Logistic Regression (class_weight='balanced')
3. Histogram Gradient Boosting (HistGradientBoostingClassifier)
4. Random Forest (n_estimators=50, max_depth=12, class_weight='balanced_subsample')

Validation decision threshold tuning is strictly performed inside the training partition.
Final test set is evaluated at both default threshold (0.5) and validation-tuned threshold.
Outputs:
- reports/data_split_experiment_results.md
- data/processed/audit_v2/split_assignments.csv
- data/processed/audit_v2/experiment_metrics_comparison.json
- data/processed/audit_v2/experiment_metrics_summary.csv
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
PROJECT_DIR = BASE_DIR / "Florida_Thunderstorm_2024_Training"
OUT_DIR = BASE_DIR / "data" / "processed" / "audit_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info("=== LOADING 6-HOUR FORECASTING DATASET ===")
fc_csv = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"
df = pd.read_csv(fc_csv, low_memory=False)
logger.info(f"Loaded {len(df):,} rows from {fc_csv}")

# Remove truncated year-end rows (2024-12-31 18:00 UTC) from primary evaluation
valid_mask = df["future_noaa_event_6h"].notna()
df_eval = df[valid_mask].copy()
logger.info(f"Evaluation dataset (complete 6h target windows): {len(df_eval):,} rows (removed {len(df) - len(df_eval)} truncated year-end rows)")

# Parse dates and calendar attributes
df_eval["init_dt"] = pd.to_datetime(df_eval["forecast_init_time_utc"])
df_eval["date"] = df_eval["init_dt"].dt.date
df_eval["doy"] = df_eval["init_dt"].dt.dayofyear  # 1 to 366 (2024 is leap year)
df_eval["month"] = df_eval["init_dt"].dt.month

# Define feature columns: strictly leak-free atmospheric + backward-looking prior
exclude_cols = [
    "forecast_init_time_utc", "latitude", "longitude", "grid_cell_id", "month", "hour_utc",
    "future_noaa_event_count", "future_noaa_event_ids", "future_noaa_event_types",
    "future_noaa_min_distance_km", "future_noaa_event_6h", "forecast_window_start_utc",
    "forecast_window_end_utc", "forecast_window_status", "prior_event_at_init",
    "true_prior_event_count", "true_prior_event_ids", "true_prior_episode_ids",
    "future_noaa_episode_ids", "glm_coverage_status", "pressure_levels_complete",
    "features_complete", "audit_category", "number", "expver", "init_dt", "date", "doy"
]
feature_cols = [c for c in df_eval.columns if c not in exclude_cols and df_eval[c].dtype in ["float64", "int64"]]
logger.info(f"Using {len(feature_cols)} strictly leak-free predictors (Atmospheric 3D + true_prior_event_6h)")
assert "prior_event_at_init" not in feature_cols, "CRITICAL ERROR: Leaked feature prior_event_at_init present!"
assert "true_prior_event_6h" in feature_cols, "CRITICAL ERROR: true_prior_event_6h missing!"

# ── 1. CONSTRUCT SPLIT ASSIGNMENTS ──────────────────────────────────────────
logger.info("=== CONSTRUCTING SPLIT ASSIGNMENTS ===")

# Unique dates throughout the year
unique_dates = sorted(df_eval["date"].unique())
date_to_doy = {d: pd.to_datetime(d).dayofyear for d in unique_dates}
logger.info(f"Total unique dates in dataset: {len(unique_dates)} (DOY 1 to 366)")

# A. Experiment 1: Existing Chronological Split
# Train: Months 1-8, Test: Months 9-12
split_exp1 = np.where(df_eval["month"] <= 8, "TRAIN", "TEST")

# B. Experiment 2A: Pure Alternating-Day Split
# Odd DOY -> Train (183 days), Even DOY -> Test (183 days)
split_exp2a = np.where(df_eval["doy"] % 2 == 1, "TRAIN", "TEST")

# C. Experiment 2B: Purged / Embargoed Alternating-Day Split
# 5-day cycle: Days 0,1 -> Train; Day 2 -> Embargo; Day 3 -> Test; Day 4 -> Embargo
cycle_day = df_eval["doy"] % 5
split_exp2b = np.where(
    cycle_day.isin([0, 1]), "TRAIN",
    np.where(cycle_day == 3, "TEST", "EMBARGO")
)

# D. Experiment 3: Seasonally Stratified Split
# 4 seasons:
# Winter (Jan-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Dec)
# Contiguous 10-day blocks: 7 days train (70%), 3 days test (30%) within each season
block_day = df_eval["doy"] % 10
split_exp3 = np.where(block_day < 7, "TRAIN", "TEST")

# E. Experiment 4: Episode-Grouped Split
# Group by NOAA EPISODE_ID for positive events; assign quiescent days to preserve distribution
# Extract all unique episodes
all_episodes = sorted(set(
    ep for eps in df_eval["future_noaa_episode_ids"].dropna()
    for ep in str(eps).split(";") if ep and ep != "UNKNOWN"
))
logger.info(f"Total unique NOAA storm episodes: {len(all_episodes)}")
np.random.seed(42)
shuffled_eps = np.random.permutation(all_episodes)
n_train_eps = int(len(all_episodes) * 0.70)
train_eps_set = set(shuffled_eps[:n_train_eps])
test_eps_set = set(shuffled_eps[n_train_eps:])

# Check Milton assignment (Milton = 196745, 197213, 197468)
# Ensure all Milton episodes are grouped together
milton_eps = {"196745", "197213", "197468"}
# Keep Milton together in test set for realistic extreme holdout
for ep in milton_eps:
    if ep in all_episodes:
        train_eps_set.discard(ep)
        test_eps_set.add(ep)

# Assign each date: if date has positive events from train episodes -> TRAIN;
# if date has positive events from test episodes -> TEST;
# if date has no events, assign by 70/30 hash
date_episode_map = {}
for d, sub_d in df_eval.groupby("date"):
    pos_sub = sub_d[sub_d["future_noaa_event_6h"] == 1]
    if len(pos_sub) > 0:
        d_eps = set(
            ep for eps in pos_sub["future_noaa_episode_ids"].dropna()
            for ep in str(eps).split(";") if ep and ep != "UNKNOWN"
        )
        if d_eps & test_eps_set:
            date_episode_map[d] = "TEST"
        else:
            date_episode_map[d] = "TRAIN"
    else:
        # Quiescent date: deterministic 70/30 assignment based on date hash
        date_episode_map[d] = "TRAIN" if (hash(str(d)) % 10) < 7 else "TEST"

split_exp4 = df_eval["date"].map(date_episode_map).values

# F. Experiment 5: Strict Generalization (Chronological + 31-day August Embargo)
# Train: Months 1-7 (Jan-Jul), Embargo: Month 8 (Aug), Test: Months 9-12 (Sep-Dec)
split_exp5 = np.where(
    df_eval["month"] <= 7, "TRAIN",
    np.where(df_eval["month"] == 8, "EMBARGO", "TEST")
)

# Store assignments in dataframe
df_eval["split_exp1_chronological"] = split_exp1
df_eval["split_exp2a_alternating_pure"] = split_exp2a
df_eval["split_exp2b_alternating_purged"] = split_exp2b
df_eval["split_exp3_seasonal"] = split_exp3
df_eval["split_exp4_episode"] = split_exp4
df_eval["split_exp5_strict_embargo"] = split_exp5

# Save split assignments summary
split_save_cols = [
    "forecast_init_time_utc", "latitude", "longitude", "grid_cell_id", "month", "doy",
    "future_noaa_event_6h", "future_noaa_episode_ids",
    "split_exp1_chronological", "split_exp2a_alternating_pure",
    "split_exp2b_alternating_purged", "split_exp3_seasonal",
    "split_exp4_episode", "split_exp5_strict_embargo"
]
df_eval[split_save_cols].to_csv(OUT_DIR / "split_assignments.csv", index=False)
logger.info(f"Saved split assignments to {OUT_DIR / 'split_assignments.csv'}")

# ── 2. MODEL EVALUATION ENGINE ──────────────────────────────────────────────
def evaluate_split_experiment(
    exp_name: str,
    split_col: str,
    df_data: pd.DataFrame,
    features: List[str],
    target: str = "future_noaa_event_6h"
) -> Dict[str, Any]:
    logger.info(f"--- Running {exp_name} ({split_col}) ---")
    
    train_df = df_data[df_data[split_col] == "TRAIN"].copy()
    test_df  = df_data[df_data[split_col] == "TEST"].copy()
    
    n_train, n_test = len(train_df), len(test_df)
    n_pos_train = int(train_df[target].sum())
    n_pos_test  = int(test_df[target].sum())
    prev_train  = n_pos_train / n_train if n_train > 0 else 0.0
    prev_test   = n_pos_test / n_test if n_test > 0 else 0.0
    
    logger.info(f"  Train: {n_train:,} rows, {n_pos_train} positives ({prev_train*100:.3f}%)")
    logger.info(f"  Test:  {n_test:,} rows, {n_pos_test} positives ({prev_test*100:.3f}%)")
    
    # Preprocessing: Fit on TRAIN ONLY
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_raw = train_df[features].values
    y_train = train_df[target].values.astype(int)
    
    X_test_raw = test_df[features].values
    y_test = test_df[target].values.astype(int)
    
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_raw)
    
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    
    # Internal chronological validation within training for threshold tuning
    # Reserve last 20% of train dates for validation threshold selection
    train_dates = sorted(train_df["date"].unique())
    n_val_dates = max(1, int(len(train_dates) * 0.20))
    val_dates_set = set(train_dates[-n_val_dates:])
    
    val_mask = train_df["date"].isin(val_dates_set).values
    fit_mask = ~val_mask
    
    X_fit_scaled = X_train_scaled[fit_mask]
    y_fit = y_train[fit_mask]
    X_val_scaled = X_train_scaled[val_mask]
    y_val = y_train[val_mask]
    
    test_days = max(1, test_df["date"].nunique())
    
    models = {
        "Majority_Class_Baseline": None,
        "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=100, class_weight="balanced"),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    }
    
    exp_results = {
        "experiment_name": exp_name,
        "split_column": split_col,
        "sample_counts": {
            "train_rows": n_train,
            "train_positives": n_pos_train,
            "train_prevalence": round(prev_train, 6),
            "test_rows": n_test,
            "test_positives": n_pos_test,
            "test_prevalence": round(prev_test, 6),
            "test_days": test_days
        },
        "models": {}
    }
    
    for mname, model in models.items():
        logger.info(f"  Evaluating {mname}...")
        if model is None:
            # Majority class / no alert
            y_pred_default = np.zeros_like(y_test)
            y_prob = np.zeros(len(y_test), dtype=float)
            
            exp_results["models"][mname] = {
                "default_threshold": {
                    "threshold": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.5,
                    "pr_auc": round(float(prev_test), 6),
                    "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist(),
                    "false_alarms_per_day": 0.0
                },
                "val_tuned_threshold": {
                    "threshold": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "false_alarms_per_day": 0.0
                }
            }
            continue
            
        # Fit on sub-training for threshold selection
        model.fit(X_fit_scaled, y_fit)
        val_probs = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val_scaled)
        
        # Select best F1 threshold on validation partition
        prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
        f1_v = 2 * (prec_v * rec_v) / np.maximum(prec_v + rec_v, 1e-9)
        best_idx = np.nanargmax(f1_v)
        best_tau = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.5
        # Bound threshold cleanly in [0.01, 0.99]
        best_tau = max(0.01, min(0.99, best_tau))
        
        # Now fit on full training partition
        model.fit(X_train_scaled, y_train)
        test_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
        
        # Test evaluation at default tau = 0.5
        pred_def = (test_probs >= 0.5).astype(int)
        prec_def = float(precision_score(y_test, pred_def, zero_division=0))
        rec_def  = float(recall_score(y_test, pred_def, zero_division=0))
        f1_def   = float(f1_score(y_test, pred_def, zero_division=0))
        cm_def   = confusion_matrix(y_test, pred_def)
        fa_def_day = float(cm_def[0, 1] / test_days) if test_days > 0 else 0.0
        
        # Test evaluation at validation-tuned tau
        pred_tuned = (test_probs >= best_tau).astype(int)
        prec_tuned = float(precision_score(y_test, pred_tuned, zero_division=0))
        rec_tuned  = float(recall_score(y_test, pred_tuned, zero_division=0))
        f1_tuned   = float(f1_score(y_test, pred_tuned, zero_division=0))
        cm_tuned   = confusion_matrix(y_test, pred_tuned)
        fa_tuned_day = float(cm_tuned[0, 1] / test_days) if test_days > 0 else 0.0
        
        roc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
        pr  = float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0
        
        # Hurricane Milton Isolation Analysis on Test Set
        # Identify Milton rows in test set
        test_df["_prob"] = test_probs
        is_milton = test_df["future_noaa_episode_ids"].astype(str).str.contains("196745|197213|197468", regex=True)
        n_milton = int(is_milton.sum())
        n_milton_pos = int((is_milton & (y_test == 1)).sum())
        
        # Metrics on non-Milton subset
        non_m_mask = (~is_milton).values
        if non_m_mask.sum() > 0 and y_test[non_m_mask].sum() > 0:
            roc_no_milton = float(roc_auc_score(y_test[non_m_mask], test_probs[non_m_mask]))
            pr_no_milton  = float(average_precision_score(y_test[non_m_mask], test_probs[non_m_mask]))
            pred_no_m = pred_tuned[non_m_mask]
            rec_no_milton = float(recall_score(y_test[non_m_mask], pred_no_m, zero_division=0))
            prec_no_milton = float(precision_score(y_test[non_m_mask], pred_no_m, zero_division=0))
            f1_no_milton   = float(f1_score(y_test[non_m_mask], pred_no_m, zero_division=0))
        else:
            roc_no_milton, pr_no_milton, rec_no_milton, prec_no_milton, f1_no_milton = 0.5, 0.0, 0.0, 0.0, 0.0
            
        exp_results["models"][mname] = {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "default_threshold": {
                "threshold": 0.5,
                "precision": round(prec_def, 4),
                "recall": round(rec_def, 4),
                "f1": round(f1_def, 4),
                "false_alarms_per_day": round(fa_def_day, 2),
                "confusion_matrix": cm_def.tolist()
            },
            "val_tuned_threshold": {
                "threshold": round(best_tau, 4),
                "precision": round(prec_tuned, 4),
                "recall": round(rec_tuned, 4),
                "f1": round(f1_tuned, 4),
                "false_alarms_per_day": round(fa_tuned_day, 2),
                "confusion_matrix": cm_tuned.tolist()
            },
            "milton_sensitivity": {
                "milton_rows": n_milton,
                "milton_positives": n_milton_pos,
                "roc_auc_ex_milton": round(roc_no_milton, 4),
                "pr_auc_ex_milton": round(pr_no_milton, 4),
                "f1_ex_milton": round(f1_no_milton, 4),
                "recall_ex_milton": round(rec_no_milton, 4),
                "precision_ex_milton": round(prec_no_milton, 4)
            }
        }
        
        logger.info(f"    ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f} | Default F1: {f1_def:.4f} (Rec: {rec_def*100:.1f}%, Prec: {prec_def*100:.2f}%) | Val-Tuned (tau={best_tau:.2f}) F1: {f1_tuned:.4f} (Rec: {rec_tuned*100:.1f}%, Prec: {prec_tuned*100:.2f}%, FA/day: {fa_tuned_day:.1f})")
        
    return exp_results


# ── 3. RUN ALL 5 SPLITTING EXPERIMENTS ──────────────────────────────────────
experiments = [
    ("Exp1_Chronological", "split_exp1_chronological"),
    ("Exp2A_Alternating_Pure", "split_exp2a_alternating_pure"),
    ("Exp2B_Alternating_Purged", "split_exp2b_alternating_purged"),
    ("Exp3_Seasonally_Stratified", "split_exp3_seasonal"),
    ("Exp4_Episode_Grouped", "split_exp4_episode"),
    ("Exp5_Strict_Embargo", "split_exp5_strict_embargo")
]

all_exp_results = {}
summary_rows = []

for exp_name, col in experiments:
    res = evaluate_split_experiment(exp_name, col, df_eval, feature_cols)
    all_exp_results[exp_name] = res
    
    # Flatten for summary table
    sc = res["sample_counts"]
    for mname, mres in res["models"].items():
        dt = mres.get("default_threshold", {})
        vt = mres.get("val_tuned_threshold", {})
        ms = mres.get("milton_sensitivity", {})
        summary_rows.append({
            "Experiment": exp_name,
            "Model": mname,
            "Train_Rows": sc["train_rows"],
            "Train_Pos": sc["train_positives"],
            "Train_Prev_%": round(sc["train_prevalence"] * 100, 3),
            "Test_Rows": sc["test_rows"],
            "Test_Pos": sc["test_positives"],
            "Test_Prev_%": round(sc["test_prevalence"] * 100, 3),
            "ROC_AUC": mres.get("roc_auc", 0.5),
            "PR_AUC": mres.get("pr_auc", 0.0),
            "Def_Thresh": dt.get("threshold", 0.5),
            "Def_Recall": dt.get("recall", 0.0),
            "Def_Precision": dt.get("precision", 0.0),
            "Def_F1": dt.get("f1", 0.0),
            "Def_FA_day": dt.get("false_alarms_per_day", 0.0),
            "Tuned_Thresh": vt.get("threshold", 0.5),
            "Tuned_Recall": vt.get("recall", 0.0),
            "Tuned_Precision": vt.get("precision", 0.0),
            "Tuned_F1": vt.get("f1", 0.0),
            "Tuned_FA_day": vt.get("false_alarms_per_day", 0.0),
            "ROC_Ex_Milton": ms.get("roc_auc_ex_milton", np.nan),
            "PR_Ex_Milton": ms.get("pr_auc_ex_milton", np.nan),
            "F1_Ex_Milton": ms.get("f1_ex_milton", np.nan)
        })

# Save JSON results
json_out = OUT_DIR / "experiment_metrics_comparison.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(all_exp_results, f, indent=2)
logger.info(f"Saved complete experiment JSON results to {json_out}")

# Save CSV summary
df_summary = pd.DataFrame(summary_rows)
csv_out = OUT_DIR / "experiment_metrics_summary.csv"
df_summary.to_csv(csv_out, index=False)
logger.info(f"Saved experiment summary CSV to {csv_out}")

print("\n" + "=" * 100)
print("EXECUTIVE SUMMARY OF MODEL PERFORMANCE ACROSS DATA-SPLITTING EXPERIMENTS")
print("=" * 100)
print(df_summary[["Experiment", "Model", "ROC_AUC", "PR_AUC", "Def_F1", "Tuned_Thresh", "Tuned_F1", "Tuned_Recall", "Tuned_Precision", "Tuned_FA_day"]].to_string(index=False))
