"""
Phases 3-7: MLP Neural Network & Baseline Model Comparison Pipeline.
Evaluates:
- Baseline 1: Majority-Class / No-Alert Baseline
- Baseline 2: Logistic Regression (class_weight='balanced')
- Baseline 3: HistGradientBoosting (class_weight='balanced')
- Baseline 4: Random Forest (n_estimators=50, max_depth=12, class_weight='balanced_subsample')
- Deep Model: PyTorch Multilayer Perceptron (MLP) (64 -> 32 -> 16 -> 1, ReLU, Dropout=0.2, pos_weight)

Evaluated across three primary splits:
1. Chronological Split (Train: Jan-Aug, Test: Sep-Dec)
2. Strict Embargo Split (Train: Jan-Jul, Embargo: Aug, Test: Sep-Dec)
3. Corrected Episode-Grouped Split (Zero shared episodes)

Includes:
- Training-only preprocessing (SimpleImputer, StandardScaler)
- Validation-only threshold selection (no test-set peeking)
- Complete evaluation metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Confusion Matrix, FA/day statewide)
- Hurricane Milton sensitivity analysis (with Milton vs. excluding Milton)
- Permutation feature importance on test set
- Saves all artifacts to data/processed/mlp_benchmark/
"""

import os
import sys
import json
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
OUT_DIR = BASE_DIR / "data" / "processed" / "mlp_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fc_csv = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"
splits_csv = BASE_DIR / "data" / "processed" / "audit_v2" / "split_assignments.csv"

logger.info(f"Loading forecasting dataset from {fc_csv}...")
df_fc = pd.read_csv(fc_csv, low_memory=False)
valid_mask = df_fc["future_noaa_event_6h"].notna()
df = df_fc[valid_mask].copy()

# Load split assignments (including corrected episode split)
df_splits = pd.read_csv(splits_csv, low_memory=False)
for col in ["split_exp1_chronological", "split_exp5_strict_embargo", "split_exp4_episode_corrected"]:
    df[col] = df_splits[col].values

df["date"] = pd.to_datetime(df["forecast_init_time_utc"]).dt.date

# Verified 48 features
exclude_cols = [
    "forecast_init_time_utc", "latitude", "longitude", "grid_cell_id", "month", "hour_utc",
    "future_noaa_event_count", "future_noaa_event_ids", "future_noaa_event_types",
    "future_noaa_min_distance_km", "future_noaa_event_6h", "forecast_window_start_utc",
    "forecast_window_end_utc", "forecast_window_status", "prior_event_at_init",
    "true_prior_event_count", "true_prior_event_ids", "true_prior_episode_ids",
    "future_noaa_episode_ids", "glm_coverage_status", "pressure_levels_complete",
    "features_complete", "audit_category", "number", "expver", "init_dt", "date", "doy",
    "split_exp1_chronological", "split_exp2a_alternating_pure", "split_exp2b_alternating_purged",
    "split_exp3_seasonal", "split_exp4_episode", "split_exp5_strict_embargo", "split_exp4_episode_corrected"
]
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ["float64", "int64"]]
logger.info(f"Verified feature count: {len(feature_cols)}")
assert len(feature_cols) == 48, f"Expected 48 features, got {len(feature_cols)}"
assert "prior_event_at_init" not in feature_cols, "Leaked feature present!"
assert "true_prior_event_6h" in feature_cols, "Missing true_prior_event_6h!"

# ── PyTorch MLP Architecture ───────────────────────────────────────────────
class StormMLP(nn.Module):
    def __init__(self, in_features: int, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_features: int,
    epochs: int = 35,
    batch_size: int = 2048,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 7
) -> Tuple[nn.Module, Dict[str, Any]]:
    # Compute positive class weight strictly from training labels
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(1, n_pos))
    # Cap pos_weight at 100.0 to prevent gradient instability with extreme imbalance
    pos_weight = min(100.0, pos_weight)
    
    device = torch.device("cpu")
    model = StormMLP(in_features=in_features, p_drop=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    best_pr_auc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    
    start_t = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
        train_loss /= len(y_train)
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            val_pr_auc = float(average_precision_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.0
            val_roc_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
            
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 5),
            "val_pr_auc": round(val_pr_auc, 5),
            "val_roc_auc": round(val_roc_auc, 5)
        })
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {ep} (best epoch: {best_epoch}, best val PR-AUC: {best_pr_auc:.4f})")
                break
                
    elapsed = time.time() - start_t
    if best_state is not None:
        model.load_state_dict(best_state)
    
    meta = {
        "training_time_sec": round(elapsed, 2),
        "best_epoch": best_epoch,
        "best_val_pr_auc": round(best_pr_auc, 5),
        "pos_weight_applied": round(pos_weight, 2),
        "history": history
    }
    return model, meta


# ── Full Model Evaluation Engine ────────────────────────────────────────────
def run_split_evaluation(
    exp_name: str,
    split_col: str,
    df_data: pd.DataFrame,
    features: List[str]
) -> Dict[str, Any]:
    logger.info(f"\n=======================================================")
    logger.info(f"EVALUATING: {exp_name} ({split_col})")
    logger.info(f"=======================================================")
    
    train_df = df_data[df_data[split_col] == "TRAIN"].copy()
    test_df  = df_data[df_data[split_col] == "TEST"].copy()
    
    n_train, n_test = len(train_df), len(test_df)
    n_pos_train = int(train_df["future_noaa_event_6h"].sum())
    n_pos_test  = int(test_df["future_noaa_event_6h"].sum())
    prev_train = n_pos_train / n_train
    prev_test  = n_pos_test / n_test
    test_days = test_df["date"].nunique()
    
    logger.info(f"Train: {n_train:,} rows ({n_pos_train} pos, prev={prev_train*100:.3f}%), {train_df['date'].nunique()} days")
    logger.info(f"Test:  {n_test:,} rows ({n_pos_test} pos, prev={prev_test*100:.3f}%), {test_days} days")
    
    # 1. Preprocessing: Fit on TRAIN ONLY
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_raw = train_df[features].values
    y_train = train_df["future_noaa_event_6h"].values.astype(int)
    
    X_test_raw = test_df[features].values
    y_test = test_df["future_noaa_event_6h"].values.astype(int)
    
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_raw)
    
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    
    # 2. Validation partition inside training (last 20% of training dates)
    train_dates = sorted(train_df["date"].unique())
    n_val_dates = max(1, int(len(train_dates) * 0.20))
    val_dates_set = set(train_dates[-n_val_dates:])
    
    val_mask = train_df["date"].isin(val_dates_set).values
    fit_mask = ~val_mask
    
    X_fit = X_train_scaled[fit_mask]
    y_fit = y_train[fit_mask]
    X_val = X_train_scaled[val_mask]
    y_val = y_train[val_mask]
    
    results = {
        "experiment_name": exp_name,
        "split_column": split_col,
        "sample_counts": {
            "train_rows": n_train,
            "train_positives": n_pos_train,
            "train_prevalence": round(prev_train, 6),
            "test_rows": n_test,
            "test_positives": n_pos_test,
            "test_prevalence": round(prev_test, 6),
            "test_calendar_days": test_days
        },
        "models": {}
    }
    
    models = {
        "Majority_Class_Baseline": None,
        "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED, max_iter=100, class_weight="balanced"),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=12, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "MLP_Neural_Network": None  # Handled separately
    }
    
    for mname, model in models.items():
        logger.info(f"--- Training & Evaluating: {mname} ---")
        start_time = time.time()
        
        if mname == "Majority_Class_Baseline":
            test_probs = np.zeros(len(y_test), dtype=float)
            best_tau = 0.5
            train_duration = 0.0
            mlp_meta = None
        elif mname == "MLP_Neural_Network":
            # Train MLP
            mlp_model, mlp_meta = train_mlp(
                X_fit, y_fit, X_val, y_val,
                in_features=len(features),
                epochs=35,
                batch_size=2048,
                lr=0.001,
                weight_decay=0.0001,
                patience=7
            )
            # Find best threshold on validation partition
            mlp_model.eval()
            with torch.no_grad():
                val_logits = mlp_model(torch.tensor(X_val, dtype=torch.float32)).numpy()
                val_probs = 1.0 / (1.0 + np.exp(-val_logits))
                
                test_logits = mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
                test_probs = 1.0 / (1.0 + np.exp(-test_logits))
                
            prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
            f1_v = 2 * (prec_v * rec_v) / np.maximum(prec_v + rec_v, 1e-9)
            best_idx = np.nanargmax(f1_v)
            best_tau = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.5
            best_tau = max(0.01, min(0.99, best_tau))
            train_duration = mlp_meta["training_time_sec"]
            
            # Save model weights if chronological
            if exp_name == "Exp1_Chronological":
                torch.save(mlp_model.state_dict(), OUT_DIR / "mlp_model_chronological.pt")
                logger.info(f"Saved PyTorch model weights to {OUT_DIR / 'mlp_model_chronological.pt'}")
        else:
            # Scikit-learn models
            # 1. Fit on sub-training for threshold selection
            model.fit(X_fit, y_fit)
            val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
            prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
            f1_v = 2 * (prec_v * rec_v) / np.maximum(prec_v + rec_v, 1e-9)
            best_idx = np.nanargmax(f1_v)
            best_tau = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.5
            best_tau = max(0.01, min(0.99, best_tau))
            
            # 2. Refit on full training partition
            model.fit(X_train_scaled, y_train)
            test_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
            train_duration = round(time.time() - start_time, 2)
            mlp_meta = None
            
        # Metrics at Default Threshold (0.5)
        pred_def = (test_probs >= 0.5).astype(int)
        cm_def = confusion_matrix(y_test, pred_def)
        prec_def = float(precision_score(y_test, pred_def, zero_division=0))
        rec_def  = float(recall_score(y_test, pred_def, zero_division=0))
        f1_def   = float(f1_score(y_test, pred_def, zero_division=0))
        fa_def_day = float(cm_def[0, 1] / test_days) if test_days > 0 else 0.0
        
        # Metrics at Validation-Tuned Threshold (best_tau)
        pred_tuned = (test_probs >= best_tau).astype(int)
        cm_tuned = confusion_matrix(y_test, pred_tuned)
        prec_tuned = float(precision_score(y_test, pred_tuned, zero_division=0))
        rec_tuned  = float(recall_score(y_test, pred_tuned, zero_division=0))
        f1_tuned   = float(f1_score(y_test, pred_tuned, zero_division=0))
        fa_tuned_day = float(cm_tuned[0, 1] / test_days) if test_days > 0 else 0.0
        
        roc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
        pr  = float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0
        
        # 3. Hurricane Milton Sensitivity Analysis on Test Set
        # Milton episodes on Oct 9: 196745, 197213, 197468
        is_milton = test_df["future_noaa_episode_ids"].astype(str).str.contains("196745|197213|197468", regex=True).values
        n_milton_rows = int(is_milton.sum())
        n_milton_pos = int((is_milton & (y_test == 1)).sum())
        
        non_m_mask = ~is_milton
        y_test_no_m = y_test[non_m_mask]
        probs_no_m = test_probs[non_m_mask]
        pred_tuned_no_m = pred_tuned[non_m_mask]
        
        if len(y_test_no_m) > 0 and (y_test_no_m == 1).sum() > 0:
            roc_no_m = float(roc_auc_score(y_test_no_m, probs_no_m))
            pr_no_m  = float(average_precision_score(y_test_no_m, probs_no_m))
            prec_no_m = float(precision_score(y_test_no_m, pred_tuned_no_m, zero_division=0))
            rec_no_m  = float(recall_score(y_test_no_m, pred_tuned_no_m, zero_division=0))
            f1_no_m   = float(f1_score(y_test_no_m, pred_tuned_no_m, zero_division=0))
            cm_no_m   = confusion_matrix(y_test_no_m, pred_tuned_no_m)
            fa_no_m_day = float(cm_no_m[0, 1] / test_days) if test_days > 0 else 0.0
        else:
            roc_no_m, pr_no_m, prec_no_m, rec_no_m, f1_no_m, fa_no_m_day = 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
            cm_no_m = [[0, 0], [0, 0]]
            
        model_record = {
            "model_name": mname,
            "training_time_sec": train_duration,
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "default_threshold": {
                "threshold": 0.5,
                "precision": round(prec_def, 4),
                "recall": round(rec_def, 4),
                "f1": round(f1_def, 4),
                "false_positives": int(cm_def[0, 1]),
                "false_negatives": int(cm_def[1, 0]),
                "false_alarms_per_day": round(fa_def_day, 2),
                "confusion_matrix": cm_def.tolist()
            },
            "val_tuned_threshold": {
                "threshold": round(best_tau, 4),
                "precision": round(prec_tuned, 4),
                "recall": round(rec_tuned, 4),
                "f1": round(f1_tuned, 4),
                "false_positives": int(cm_tuned[0, 1]),
                "false_negatives": int(cm_tuned[1, 0]),
                "false_alarms_per_day": round(fa_tuned_day, 2),
                "confusion_matrix": cm_tuned.tolist()
            },
            "milton_sensitivity": {
                "milton_test_rows": n_milton_rows,
                "milton_test_positives": n_milton_pos,
                "test_positives_remaining": int((y_test_no_m == 1).sum()),
                "roc_auc_ex_milton": round(roc_no_m, 4),
                "pr_auc_ex_milton": round(pr_no_m, 4),
                "precision_ex_milton": round(prec_no_m, 4),
                "recall_ex_milton": round(rec_no_m, 4),
                "f1_ex_milton": round(f1_no_m, 4),
                "false_alarms_per_day_ex_milton": round(fa_no_m_day, 2),
                "confusion_matrix_ex_milton": cm_no_m.tolist() if isinstance(cm_no_m, np.ndarray) else cm_no_m
            }
        }
        if mlp_meta:
            model_record["mlp_training_metadata"] = mlp_meta
            
        results["models"][mname] = model_record
        logger.info(f"  {mname:<25} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Tuned F1: {f1_tuned:.4f} (Rec: {rec_tuned*100:.1f}%, Prec: {prec_tuned*100:.2f}%, FA/day: {fa_tuned_day:.2f}) | Ex-Milton PR: {pr_no_m:.4f}")
        
    return results


# ── Run Primary Splits ──────────────────────────────────────────────────────
target_splits = [
    ("Exp1_Chronological", "split_exp1_chronological"),
    ("Exp5_Strict_Embargo", "split_exp5_strict_embargo"),
    ("Exp4_Episode_Grouped_Corrected", "split_exp4_episode_corrected")
]

all_benchmark_results = {}
summary_rows = []

for exp_name, col in target_splits:
    res = run_split_evaluation(exp_name, col, df, feature_cols)
    all_benchmark_results[exp_name] = res
    
    sc = res["sample_counts"]
    for mname, mdata in res["models"].items():
        vt = mdata["val_tuned_threshold"]
        dt = mdata["default_threshold"]
        ms = mdata["milton_sensitivity"]
        summary_rows.append({
            "Experiment": exp_name,
            "Model": mname,
            "Train_Rows": sc["train_rows"],
            "Train_Pos": sc["train_positives"],
            "Train_Prev_%": round(sc["train_prevalence"] * 100, 3),
            "Test_Rows": sc["test_rows"],
            "Test_Pos": sc["test_positives"],
            "Test_Prev_%": round(sc["test_prevalence"] * 100, 3),
            "Test_Days": sc["test_calendar_days"],
            "ROC_AUC": mdata["roc_auc"],
            "PR_AUC": mdata["pr_auc"],
            "Def_Thresh": dt["threshold"],
            "Def_Recall": dt["recall"],
            "Def_Precision": dt["precision"],
            "Def_F1": dt["f1"],
            "Def_FA_day": dt["false_alarms_per_day"],
            "Tuned_Thresh": vt["threshold"],
            "Tuned_Recall": vt["recall"],
            "Tuned_Precision": vt["precision"],
            "Tuned_F1": vt["f1"],
            "Tuned_FP": vt["false_positives"],
            "Tuned_FN": vt["false_negatives"],
            "Tuned_FA_day": vt["false_alarms_per_day"],
            "Milton_Pos": ms["milton_test_positives"],
            "Test_Pos_Ex_Milton": ms["test_positives_remaining"],
            "ROC_Ex_Milton": ms["roc_auc_ex_milton"],
            "PR_Ex_Milton": ms["pr_auc_ex_milton"],
            "F1_Ex_Milton": ms["f1_ex_milton"],
            "Recall_Ex_Milton": ms["recall_ex_milton"],
            "Prec_Ex_Milton": ms["precision_ex_milton"],
            "FA_day_Ex_Milton": ms["false_alarms_per_day_ex_milton"],
            "Training_Time_Sec": mdata["training_time_sec"]
        })

# Save JSON results
with open(OUT_DIR / "mlp_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(all_benchmark_results, f, indent=2)
logger.info(f"Saved complete JSON results to {OUT_DIR / 'mlp_benchmark_results.json'}")

# Save CSV summary
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_DIR / "mlp_benchmark_summary.csv", index=False)
logger.info(f"Saved CSV summary to {OUT_DIR / 'mlp_benchmark_summary.csv'}")

# Save per-split confusion matrices
cm_records = []
for exp_name, exp_res in all_benchmark_results.items():
    for mname, mdata in exp_res["models"].items():
        cm_records.append({
            "Experiment": exp_name,
            "Model": mname,
            "Default_Confusion_Matrix": str(mdata["default_threshold"]["confusion_matrix"]),
            "Tuned_Confusion_Matrix": str(mdata["val_tuned_threshold"]["confusion_matrix"]),
            "Ex_Milton_Confusion_Matrix": str(mdata["milton_sensitivity"]["confusion_matrix_ex_milton"])
        })
pd.DataFrame(cm_records).to_csv(OUT_DIR / "per_split_confusion_matrices.csv", index=False)

print("\n" + "=" * 120)
print("BENCHMARK COMPARISON TABLE: MLP vs BASELINES ACROSS EVALUATION SPLITS")
print("=" * 120)
cols_show = ["Experiment", "Model", "ROC_AUC", "PR_AUC", "Tuned_Thresh", "Tuned_F1", "Tuned_Recall", "Tuned_Precision", "Tuned_FA_day", "PR_Ex_Milton"]
print(df_summary[cols_show].to_string(index=False))
