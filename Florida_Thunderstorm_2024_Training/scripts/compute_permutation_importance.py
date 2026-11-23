"""
Computes Permutation Feature Importance on the Chronological Test Set (Months 09-12)
for both Logistic Regression and the PyTorch Multilayer Perceptron (MLP).

Evaluates:
- Baseline test PR-AUC and ROC-AUC
- Feature permutation with N_REPEATS = 3
- Drop in PR-AUC (Delta_PR_AUC) and Drop in ROC-AUC (Delta_ROC_AUC)
- Ranks features by importance and saves output to data/processed/mlp_benchmark/
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(r"c:\Users\KIIT0001\Desktop\Data Collection")
OUT_DIR = BASE_DIR / "data" / "processed" / "mlp_benchmark"
FORECAST_FILE = BASE_DIR / "data" / "processed" / "florida_thunderstorm_2024_forecasting_6h.csv"
FEAT_META_FILE = OUT_DIR / "feature_metadata.json"
MLP_WEIGHTS_FILE = OUT_DIR / "mlp_model_chronological.pt"


class StormMLP(nn.Module):
    def __init__(self, in_features: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_mlp_probs(model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
            logits = model(batch).numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            probs.append(p)
    return np.concatenate(probs)


def main():
    logger.info("Starting Permutation Feature Importance analysis...")
    if not MLP_WEIGHTS_FILE.exists():
        logger.error(f"MLP weights file {MLP_WEIGHTS_FILE} not found. Run benchmark first!")
        return

    # 1. Load feature metadata
    with open(FEAT_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    logger.info(f"Loaded {len(feature_cols)} features.")

    # 2. Load dataset and split assignments
    splits_csv = BASE_DIR / "data" / "processed" / "audit_v2" / "split_assignments.csv"
    cols_to_load = ["forecast_init_time_utc", "future_noaa_event_6h"] + feature_cols
    df_fc = pd.read_csv(FORECAST_FILE, usecols=cols_to_load, low_memory=False)
    valid_mask = df_fc["future_noaa_event_6h"].notna()
    df = df_fc[valid_mask].copy()

    df_splits = pd.read_csv(splits_csv, low_memory=False)
    df["split_exp1_chronological"] = df_splits["split_exp1_chronological"].values
    y = df["future_noaa_event_6h"].astype(int).values

    train_mask = (df["split_exp1_chronological"] == "TRAIN").values
    test_mask = (df["split_exp1_chronological"] == "TEST").values

    X_train_raw = df.loc[train_mask, feature_cols].values
    y_train = y[train_mask]
    X_test_raw = df.loc[test_mask, feature_cols].values
    y_test = y[test_mask]

    logger.info(f"Train samples: {len(X_train_raw)} (pos: {(y_train == 1).sum()})")
    logger.info(f"Test samples:  {len(X_test_raw)} (pos: {(y_test == 1).sum()})")

    # 3. Fit preprocessing on train only
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw))

    # 4. Train Logistic Regression
    logger.info("Fitting Logistic Regression model...")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr_model.fit(X_train_scaled, y_train)

    lr_base_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_base_pr = average_precision_score(y_test, lr_base_probs)
    lr_base_roc = roc_auc_score(y_test, lr_base_probs)
    logger.info(f"Logistic Regression Baseline: PR-AUC = {lr_base_pr:.4f}, ROC-AUC = {lr_base_roc:.4f}")

    # 5. Load MLP model
    logger.info("Loading PyTorch MLP model...")
    mlp_model = StormMLP(in_features=len(feature_cols))
    mlp_model.load_state_dict(torch.load(MLP_WEIGHTS_FILE, map_location="cpu"))
    mlp_model.eval()

    mlp_base_probs = get_mlp_probs(mlp_model, X_test_scaled)
    mlp_base_pr = average_precision_score(y_test, mlp_base_probs)
    mlp_base_roc = roc_auc_score(y_test, mlp_base_probs)
    logger.info(f"MLP Baseline: PR-AUC = {mlp_base_pr:.4f}, ROC-AUC = {mlp_base_roc:.4f}")

    # 6. Permutation Importance
    N_REPEATS = 3
    logger.info(f"Computing permutation importance ({N_REPEATS} repeats per feature)...")

    lr_results = []
    mlp_results = []

    for idx, fcol in enumerate(feature_cols):
        lr_pr_drops, lr_roc_drops = [], []
        mlp_pr_drops, mlp_roc_drops = [], []

        for r in range(N_REPEATS):
            rng = np.random.RandomState(SEED + idx * 10 + r)
            perm_idx = rng.permutation(len(X_test_scaled))
            
            # Copy test matrix and permute column idx
            X_perm = X_test_scaled.copy()
            X_perm[:, idx] = X_perm[perm_idx, idx]

            # LR eval
            lr_perm_probs = lr_model.predict_proba(X_perm)[:, 1]
            lr_pr = average_precision_score(y_test, lr_perm_probs)
            lr_roc = roc_auc_score(y_test, lr_perm_probs)
            lr_pr_drops.append(lr_base_pr - lr_pr)
            lr_roc_drops.append(lr_base_roc - lr_roc)

            # MLP eval
            mlp_perm_probs = get_mlp_probs(mlp_model, X_perm)
            mlp_pr = average_precision_score(y_test, mlp_perm_probs)
            mlp_roc = roc_auc_score(y_test, mlp_perm_probs)
            mlp_pr_drops.append(mlp_base_pr - mlp_pr)
            mlp_roc_drops.append(mlp_base_roc - mlp_roc)

        lr_results.append({
            "feature": fcol,
            "lr_baseline_pr_auc": round(lr_base_pr, 5),
            "lr_baseline_roc_auc": round(lr_base_roc, 5),
            "lr_delta_pr_auc": round(float(np.mean(lr_pr_drops)), 5),
            "lr_delta_pr_auc_std": round(float(np.std(lr_pr_drops)), 5),
            "lr_delta_roc_auc": round(float(np.mean(lr_roc_drops)), 5),
        })

        mlp_results.append({
            "feature": fcol,
            "mlp_baseline_pr_auc": round(mlp_base_pr, 5),
            "mlp_baseline_roc_auc": round(mlp_base_roc, 5),
            "mlp_delta_pr_auc": round(float(np.mean(mlp_pr_drops)), 5),
            "mlp_delta_pr_auc_std": round(float(np.std(mlp_pr_drops)), 5),
            "mlp_delta_roc_auc": round(float(np.mean(mlp_roc_drops)), 5),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(feature_cols):
            logger.info(f"Processed {idx + 1}/{len(feature_cols)} features...")

    # DataFrames
    df_lr = pd.DataFrame(lr_results).sort_values(by="lr_delta_pr_auc", ascending=False)
    df_mlp = pd.DataFrame(mlp_results).sort_values(by="mlp_delta_pr_auc", ascending=False)

    df_lr.to_csv(OUT_DIR / "permutation_importance_logistic_regression.csv", index=False)
    df_mlp.to_csv(OUT_DIR / "permutation_importance_mlp.csv", index=False)

    df_combined = pd.merge(df_lr, df_mlp, on="feature")
    df_combined.to_csv(OUT_DIR / "feature_importances_combined.csv", index=False)
    logger.info(f"Saved feature importances to {OUT_DIR / 'feature_importances_combined.csv'}")

    print("\n" + "=" * 90)
    print("TOP 10 FEATURES BY PR-AUC DROP: LOGISTIC REGRESSION vs MLP")
    print("=" * 90)
    top_lr = df_lr.head(10)[["feature", "lr_delta_pr_auc", "lr_delta_roc_auc"]]
    top_mlp = df_mlp.head(10)[["feature", "mlp_delta_pr_auc", "mlp_delta_roc_auc"]]
    print("\n[Logistic Regression - Top 10 by PR-AUC Drop]")
    print(top_lr.to_string(index=False))
    print("\n[PyTorch MLP - Top 10 by PR-AUC Drop]")
    print(top_mlp.to_string(index=False))


if __name__ == "__main__":
    main()
