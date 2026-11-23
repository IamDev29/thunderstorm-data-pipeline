"""
run_step_06_importance_weighting.py
===================================
Step 6 of Thunderstorm Domain Adaptation Framework (TDAF):
Importance Weighting and Density-Ratio Estimation.

Investigates:
1. Method A: Domain Classifier Density Ratio (Logistic Regression, Random Forest, LightGBM)
2. Method B: Kernel Mean Matching (KMM) with fast Projected Gradient Descent (PGD)
3. Method C: Weight Stabilization and Clipping Policies (None, p95, p99, Max 5, Max 10)
4. Method D: Season-Aware Importance Weighting (DJF, MAM, JJAS, ON Fallback)
5. Weighted GAM (pyGAM) and EBM (InterpretML) Training
6. Evaluation on Florida Chronological Holdout Test Set and Chandipur Target Set (Raw & CAPE Clipped)
7. Extreme CAPE (81 cases) Behavior Analysis
8. Full CSV metrics and 8 publication-grade 300 DPI figures
"""

import os
import sys
import time
from pathlib import Path

# Force unbuffered real-time stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve
)
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import cdist
import lightgbm as lgb

import pygam
from pygam import LogisticGAM, s
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

# ---------------------------------------------------------
# 1. Paths & Configurations
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_06"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_06"
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "cape_J_kg",
    "cin_J_kg",
    "bulk_shear_850_300_ms",
    "cloud_cover_fraction",
    "wind_speed_10m_ms",
    "wind_speed_850hPa_ms"
]

CAPE_CLIP_THRESHOLD = 7240.0
TAU_GAM = 0.33
TAU_EBM = 0.34
RANDOM_STATE = 42

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    eps = 1e-15
    y_prob_c = np.clip(y_prob, eps, 1.0 - eps)
    ll = float(log_loss(y_true, y_prob_c))
    acc = float((tp + tn) / (tp + tn + fp + fn))
    
    return {
        "threshold": round(threshold, 4),
        "sample_count": len(y_true),
        "positives": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        "cohen_kappa": round(kappa, 6), "mcc": round(mcc, 6),
        "pr_auc": round(pr_auc, 6), "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6), "log_loss": round(ll, 6)
    }

def get_season(month):
    if month in [12, 1, 2]:
        return "DJF (Winter)"
    elif month in [3, 4, 5]:
        return "MAM (Pre-monsoon)"
    elif month in [6, 7, 8, 9]:
        return "JJAS (Monsoon)"
    else:
        return "ON (Post-monsoon)"

def compute_ess(weights):
    """Effective Sample Size: (sum(w))^2 / sum(w^2)."""
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    return float((sum_w ** 2) / (sum_w2 + 1e-12))

def project_capped_simplex(y, s, u):
    """
    Exact projection of vector y onto {x | sum(x) = s, 0 <= x_i <= u}.
    Uses root-finding via bisection on the dual variable theta:
    sum(clip(y - theta, 0, u)) = s.
    Guaranteed monotonic and converges in <= 40 iterations (machine precision).
    """
    l = np.min(y) - u
    r = np.max(y)
    for _ in range(40):
        m = (l + r) / 2.0
        val = np.sum(np.clip(y - m, 0.0, u))
        if val > s:
            l = m
        else:
            r = m
    return np.clip(y - (l + r) / 2.0, 0.0, u)

def main():
    start_time = time.time()
    print("=" * 80, flush=True)
    print("STEP 6: IMPORTANCE WEIGHTING AND DENSITY-RATIO ESTIMATION", flush=True)
    print("=" * 80, flush=True)

    # ---------------------------------------------------------
    # 2. Reconstruct Source and Target Datasets
    # ---------------------------------------------------------
    print(f"\n[1/8] Loading datasets...", flush=True)
    fl_df = pd.read_parquet(FLORIDA_PATH)
    fl_valid = fl_df[fl_df["target_asos_strict_crit_c"].notna()].copy()
    fl_valid["target"] = fl_valid["target_asos_strict_crit_c"].astype(int)

    # Chronological Splits
    train_mask = fl_valid["month"].between(1, 7)
    val_mask = fl_valid["month"].between(8, 9)
    test_mask = fl_valid["month"].between(10, 12)

    fl_train = fl_valid[train_mask].copy().reset_index(drop=True)
    fl_val = fl_valid[val_mask].copy().reset_index(drop=True)
    fl_test = fl_valid[test_mask].copy().reset_index(drop=True)

    ch_df = pd.read_csv(CHANDIPUR_PATH)
    CHANDIPUR_TO_FLORIDA = {
        "cape_Jkg": "cape_J_kg",
        "cin_Jkg": "cin_J_kg",
        "bulk_shear_850_300": "bulk_shear_850_300_ms",
        "total_cloud_cover": "cloud_cover_fraction",
        "wind_speed_10m": "wind_speed_10m_ms",
        "wind_speed_850hPa": "wind_speed_850hPa_ms"
    }
    ch_df = ch_df.rename(columns=CHANDIPUR_TO_FLORIDA)
    ch_df["valid_time_dt"] = pd.to_datetime(ch_df["valid_time"], format="mixed")
    ch_df["month"] = ch_df["valid_time_dt"].dt.month
    ch_df["season"] = ch_df["month"].apply(get_season)

    N_S = len(fl_train)
    N_T = len(ch_df)
    print(f"Source Training Set (Florida Jan-Jul): N = {N_S}, Positives = {fl_train['target'].sum()} ({fl_train['target'].mean():.4f})", flush=True)
    print(f"Florida Validation Set (Aug-Sep):      N = {len(fl_val)}, Positives = {fl_val['target'].sum()} ({fl_val['target'].mean():.4f})", flush=True)
    print(f"Florida Holdout Test Set (Oct-Dec):    N = {len(fl_test)}, Positives = {fl_test['target'].sum()} ({fl_test['target'].mean():.4f})", flush=True)
    print(f"Target Set (Chandipur 2022-2025):       N = {N_T}, Positives = {ch_df['storm_label_score'].sum()} ({ch_df['storm_label_score'].mean():.4f})", flush=True)

    X_fl_train = fl_train[FEATURES].values
    y_fl_train = fl_train["target"].values

    X_fl_test = fl_test[FEATURES].values
    y_fl_test = fl_test["target"].values

    X_ch_raw = ch_df[FEATURES].values
    y_ch = ch_df["storm_label_score"].values

    # CAPE-clipped target features
    X_ch_clipped = X_ch_raw.copy()
    X_ch_clipped[:, 0] = np.minimum(X_ch_clipped[:, 0], CAPE_CLIP_THRESHOLD)

    # ---------------------------------------------------------
    # 3. Method A: Domain Classifiers & Density-Ratio Estimation
    # ---------------------------------------------------------
    print(f"\n[2/8] Method A: Training Domain Classifiers (5-Fold Stratified CV)...", flush=True)
    X_domain = np.vstack([X_fl_train, X_ch_raw])
    y_domain = np.concatenate([np.zeros(N_S, dtype=int), np.ones(N_T, dtype=int)])
    source_mask_domain = (y_domain == 0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    prob_oof_lr = np.zeros(len(y_domain))
    prob_oof_rf = np.zeros(len(y_domain))
    prob_oof_lgb = np.zeros(len(y_domain))

    scaler = StandardScaler()
    X_domain_scaled = scaler.fit_transform(X_domain)

    print("  Running 5-fold CV across 89,631 combined domain observations...", flush=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_domain, y_domain)):
        t_fold = time.time()
        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_domain_scaled[train_idx], y_domain[train_idx])
        prob_oof_lr[val_idx] = lr.predict_proba(X_domain_scaled[val_idx])[:, 1]

        # 2. Random Forest (regularized)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(X_domain[train_idx], y_domain[train_idx])
        prob_oof_rf[val_idx] = rf.predict_proba(X_domain[val_idx])[:, 1]

        # 3. LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
        lgbm.fit(X_domain[train_idx], y_domain[train_idx])
        prob_oof_lgb[val_idx] = lgbm.predict_proba(X_domain[val_idx])[:, 1]
        print(f"    Fold {fold+1}/5 completed in {time.time()-t_fold:.1f}s", flush=True)

    # Evaluate domain classifiers
    dc_metrics = []
    for name, p_oof in [("Logistic Regression", prob_oof_lr), ("Random Forest", prob_oof_rf), ("LightGBM", prob_oof_lgb)]:
        auc = roc_auc_score(y_domain, p_oof)
        brier = brier_score_loss(y_domain, p_oof)
        ll = log_loss(y_domain, np.clip(p_oof, 1e-15, 1 - 1e-15))
        dc_metrics.append({"classifier": name, "roc_auc": round(auc, 6), "brier_score": round(brier, 6), "log_loss": round(ll, 6)})
        print(f"  {name:20s} -> Out-of-Fold Domain ROC-AUC: {auc:.4f}, Brier: {brier:.4f}, Log Loss: {ll:.4f}", flush=True)

    p_src_lr = prob_oof_lr[source_mask_domain]
    p_src_rf = prob_oof_rf[source_mask_domain]
    p_src_lgb = prob_oof_lgb[source_mask_domain]

    eps_p = 1e-4
    ratio_prior = float(N_S / N_T)

    w_raw_lr = (np.clip(p_src_lr, eps_p, 1 - eps_p) / (1.0 - np.clip(p_src_lr, eps_p, 1 - eps_p))) * ratio_prior
    w_raw_rf = (np.clip(p_src_rf, eps_p, 1 - eps_p) / (1.0 - np.clip(p_src_rf, eps_p, 1 - eps_p))) * ratio_prior
    w_raw_lgb = (np.clip(p_src_lgb, eps_p, 1 - eps_p) / (1.0 - np.clip(p_src_lgb, eps_p, 1 - eps_p))) * ratio_prior

    w_raw_lr /= np.mean(w_raw_lr)
    w_raw_rf /= np.mean(w_raw_rf)
    w_raw_lgb /= np.mean(w_raw_lgb)

    # ---------------------------------------------------------
    # 4. Method B: Kernel Mean Matching (KMM via Fast PGD)
    # ---------------------------------------------------------
    print(f"\n[3/8] Method B: Kernel Mean Matching (Representative Subsample via Fast PGD)...", flush=True)
    n_kmm = 2000
    rng = np.random.RandomState(RANDOM_STATE)
    idx_s_kmm = rng.choice(N_S, size=n_kmm, replace=False)
    idx_t_kmm = rng.choice(N_T, size=n_kmm, replace=False)

    scaler_kmm = StandardScaler().fit(X_fl_train)
    X_s_sub = scaler_kmm.transform(X_fl_train[idx_s_kmm])
    X_t_sub = scaler_kmm.transform(X_ch_raw[idx_t_kmm])

    pdist_s = cdist(X_s_sub[:500], X_s_sub[:500], metric="sqeuclidean")
    med_dist = np.median(pdist_s[np.triu_indices_from(pdist_s, k=1)])
    gamma_kmm = 1.0 / (med_dist + 1e-5)
    print(f"  KMM RBF Gamma (median heuristic): {gamma_kmm:.6f}", flush=True)

    K_s = np.exp(-gamma_kmm * cdist(X_s_sub, X_s_sub, metric="sqeuclidean"))
    K_st = np.exp(-gamma_kmm * cdist(X_s_sub, X_t_sub, metric="sqeuclidean"))
    kappa = (float(n_kmm) / float(n_kmm)) * np.sum(K_st, axis=1)

    reg_kmm = 1e-3
    H = K_s + reg_kmm * np.eye(n_kmm)

    print("  Solving KMM Quadratic Program via Projected Gradient Descent (PGD)...", flush=True)
    t_pgd_start = time.time()
    beta_kmm = np.ones(n_kmm)
    L_H = float(np.linalg.norm(H, 2))
    eta_pgd = 1.0 / L_H
    B_cap = 10.0

    for iter_idx in range(150):
        grad = H @ beta_kmm - kappa
        beta_kmm = project_capped_simplex(beta_kmm - eta_pgd * grad, float(n_kmm), B_cap)

    print(f"  PGD KMM Solved in {time.time() - t_pgd_start:.2f}s! Sum(beta) = {np.sum(beta_kmm):.2f}, Max(beta) = {np.max(beta_kmm):.2f}", flush=True)

    print("  Computing inductive out-of-sample KMM weights for all 66,255 Florida samples...", flush=True)
    X_fl_train_scaled = scaler_kmm.transform(X_fl_train)
    w_kmm = np.zeros(N_S)
    chunk_size = 5000
    for c_start in range(0, N_S, chunk_size):
        c_end = min(c_start + chunk_size, N_S)
        K_chunk = np.exp(-gamma_kmm * cdist(X_fl_train_scaled[c_start:c_end], X_s_sub, metric="sqeuclidean"))
        num = np.dot(K_chunk, beta_kmm)
        den = np.sum(K_chunk, axis=1) + 1e-12
        w_kmm[c_start:c_end] = num / den

    w_kmm /= np.mean(w_kmm)
    print(f"  KMM weights computed: ESS = {compute_ess(w_kmm):.1f} ({compute_ess(w_kmm)/N_S*100:.2f}%)", flush=True)

    # ---------------------------------------------------------
    # 5. Method C: Weight Stabilization and Clipping Policies
    # ---------------------------------------------------------
    print(f"\n[4/8] Method C: Evaluating Weight Clipping Policies...", flush=True)
    def apply_clipping_policies(w_raw, base_name):
        records = []
        weights_dict = {}

        w_none = w_raw.copy() / np.mean(w_raw)
        weights_dict[f"{base_name}_no_clip"] = w_none

        p95 = np.percentile(w_raw, 95)
        w_p95 = np.minimum(w_raw, p95)
        w_p95 /= np.mean(w_p95)
        weights_dict[f"{base_name}_clip_p95"] = w_p95

        p99 = np.percentile(w_raw, 99)
        w_p99 = np.minimum(w_raw, p99)
        w_p99 /= np.mean(w_p99)
        weights_dict[f"{base_name}_clip_p99"] = w_p99

        w_max5 = np.minimum(w_raw, 5.0)
        w_max5 /= np.mean(w_max5)
        weights_dict[f"{base_name}_max_5"] = w_max5

        w_max10 = np.minimum(w_raw, 10.0)
        w_max10 /= np.mean(w_max10)
        weights_dict[f"{base_name}_max_10"] = w_max10

        policies = [
            ("No Clipping", w_none, np.nan),
            ("95th Percentile Clipping", w_p95, p95),
            ("99th Percentile Clipping", w_p99, p99),
            ("Fixed Maximum Weight = 5", w_max5, 5.0),
            ("Fixed Maximum Weight = 10", w_max10, 10.0),
        ]

        for p_name, w_vec, clip_val in policies:
            ess = compute_ess(w_vec)
            p99_val = np.percentile(w_vec, 99)
            top1_conc = np.sum(np.sort(w_vec)[-int(0.01 * len(w_vec)):]) / np.sum(w_vec)
            top5_conc = np.sum(np.sort(w_vec)[-int(0.05 * len(w_vec)):]) / np.sum(w_vec)
            records.append({
                "weighting_method": base_name,
                "clipping_policy": p_name,
                "clipping_threshold": round(clip_val, 4) if not np.isnan(clip_val) else "None",
                "sample_count": len(w_vec),
                "ess": round(ess, 2),
                "ess_ratio": round(ess / len(w_vec), 6),
                "min_weight": round(float(np.min(w_vec)), 6),
                "max_weight": round(float(np.max(w_vec)), 6),
                "mean_weight": round(float(np.mean(w_vec)), 6),
                "std_weight": round(float(np.std(w_vec)), 6),
                "median_weight": round(float(np.median(w_vec)), 6),
                "p95_weight": round(float(np.percentile(w_vec, 95)), 6),
                "p99_weight": round(float(p99_val), 6),
                "top_1pct_concentration": round(float(top1_conc), 6),
                "top_5pct_concentration": round(float(top5_conc), 6),
            })
        return records, weights_dict

    ess_records = []
    all_weights = {}

    for w_vec, name in [(w_raw_lr, "logistic_regression"), (w_raw_rf, "random_forest"), (w_raw_lgb, "lightgbm"), (w_kmm, "kmm")]:
        recs, w_dict = apply_clipping_policies(w_vec, name)
        ess_records.extend(recs)
        all_weights.update(w_dict)

    ess_df = pd.DataFrame(ess_records)
    ess_df.to_csv(OUTPUT_DIR / "effective_sample_size_analysis.csv", index=False)
    print(f"  ESS table saved to {OUTPUT_DIR / 'effective_sample_size_analysis.csv'}", flush=True)

    # ---------------------------------------------------------
    # 6. Method D: Season-Aware Importance Weighting
    # ---------------------------------------------------------
    print(f"\n[5/8] Method D: Season-Aware Importance Weighting...", flush=True)
    fl_train["season"] = fl_train["month"].apply(get_season)
    seasonal_records = []
    w_seasonal = np.zeros(N_S)

    for season in ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)"]:
        s_src_mask = (fl_train["season"] == season).values
        s_tgt_mask = (ch_df["season"] == season).values

        n_s_sea = int(np.sum(s_src_mask))
        n_t_sea = int(np.sum(s_tgt_mask))

        X_s_sea = X_fl_train[s_src_mask]
        X_t_sea = X_ch_raw[s_tgt_mask]

        X_sea_comb = np.vstack([X_s_sea, X_t_sea])
        y_sea_comb = np.concatenate([np.zeros(n_s_sea, dtype=int), np.ones(n_t_sea, dtype=int)])

        lgb_sea = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, num_leaves=15, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
        skf_sea = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        p_oof_sea = np.zeros(len(y_sea_comb))
        for t_idx, v_idx in skf_sea.split(X_sea_comb, y_sea_comb):
            lgb_sea.fit(X_sea_comb[t_idx], y_sea_comb[t_idx])
            p_oof_sea[v_idx] = lgb_sea.predict_proba(X_sea_comb[v_idx])[:, 1]

        p_src_sea = p_oof_sea[:n_s_sea]
        w_sea_raw = (np.clip(p_src_sea, eps_p, 1 - eps_p) / (1.0 - np.clip(p_src_sea, eps_p, 1 - eps_p))) * (float(n_s_sea) / float(n_t_sea))
        w_sea_clipped = np.minimum(w_sea_raw, np.percentile(w_sea_raw, 99))
        w_sea_clipped /= np.mean(w_sea_clipped)
        w_seasonal[s_src_mask] = w_sea_clipped

        auc_sea = roc_auc_score(y_sea_comb, p_oof_sea)
        ess_sea = compute_ess(w_sea_clipped)
        seasonal_records.append({
            "season": season,
            "source_samples": n_s_sea,
            "target_samples": n_t_sea,
            "domain_classifier_auc": round(auc_sea, 4),
            "ess": round(ess_sea, 2),
            "ess_ratio": round(ess_sea / n_s_sea, 4),
            "min_weight": round(float(np.min(w_sea_clipped)), 4),
            "max_weight": round(float(np.max(w_sea_clipped)), 4),
            "std_weight": round(float(np.std(w_sea_clipped)), 4),
            "fallback_used": False
        })
        print(f"  Season {season:20s}: N_S={n_s_sea:5d}, N_T={n_t_sea:5d}, AUC={auc_sea:.4f}, ESS={ess_sea:.1f} ({ess_sea/n_s_sea*100:.2f}%)", flush=True)

    n_t_on = int(np.sum(ch_df["season"] == "ON (Post-monsoon)"))
    seasonal_records.append({
        "season": "ON (Post-monsoon)",
        "source_samples": 0,
        "target_samples": n_t_on,
        "domain_classifier_auc": np.nan,
        "ess": np.nan,
        "ess_ratio": np.nan,
        "min_weight": np.nan,
        "max_weight": np.nan,
        "std_weight": np.nan,
        "fallback_used": True
    })
    print(f"  Season ON (Post-monsoon)  : N_S=    0, N_T={n_t_on:5d} -> Florida training dataset has 0 samples; Global Fallback Applied.", flush=True)

    w_seasonal /= np.mean(w_seasonal)
    all_weights["seasonal_lgb_clip_p99"] = w_seasonal

    seasonal_df = pd.DataFrame(seasonal_records)
    seasonal_df.to_csv(OUTPUT_DIR / "seasonal_weighting_comparison.csv", index=False)

    # ---------------------------------------------------------
    # 7. Train Weighted pyGAM and InterpretML EBM Models
    # ---------------------------------------------------------
    print(f"\n[6/8] Training Weighted GAM and EBM Models...", flush=True)
    orig_gam = joblib.load(ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_gam_model.pkl")
    orig_ebm = joblib.load(ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_ebm_model.pkl")

    weight_configs = [
        ("Baseline (Unweighted)", "baseline_unweighted", np.ones(N_S)),
        ("Logistic Regression (p99)", "logistic_regression_clip_p99", all_weights["logistic_regression_clip_p99"]),
        ("Logistic Regression (Max 5)", "logistic_regression_max_5", all_weights["logistic_regression_max_5"]),
        ("LightGBM (p99)", "lightgbm_clip_p99", all_weights["lightgbm_clip_p99"]),
        ("LightGBM (Max 5)", "lightgbm_max_5", all_weights["lightgbm_max_5"]),
        ("Random Forest (p99)", "random_forest_clip_p99", all_weights["random_forest_clip_p99"]),
        ("KMM (Kernel Mean Matching)", "kmm_no_clip", all_weights["kmm_no_clip"]),
        ("Seasonal LightGBM (p99)", "seasonal_lgb_clip_p99", all_weights["seasonal_lgb_clip_p99"]),
    ]

    trained_models = {}
    model_eval_records = []
    extreme_case_eval = []

    extreme_mask = (ch_df["cape_J_kg"] > CAPE_CLIP_THRESHOLD).values
    print(f"  Identified {np.sum(extreme_mask)} Chandipur cases with CAPE > {CAPE_CLIP_THRESHOLD} J/kg.", flush=True)

    X_train_df = fl_train[FEATURES].copy()
    X_fl_test_df = fl_test[FEATURES].copy()
    X_ch_raw_df = ch_df[FEATURES].copy()
    X_ch_clipped_df = pd.DataFrame(X_ch_clipped, columns=FEATURES)

    for cfg_label, cfg_key, w_train in weight_configs:
        t_mod = time.time()
        print(f"\n  --- Fitting Models for: {cfg_label} ---", flush=True)
        gam_file = OUTPUT_DIR / f"gam_{cfg_key}.pkl"
        ebm_file = OUTPUT_DIR / f"ebm_{cfg_key}.pkl"
        if cfg_key == "baseline_unweighted":
            gam_model = orig_gam
            ebm_model = orig_ebm
        elif gam_file.exists() and ebm_file.exists():
            print(f"    Loading pre-trained models from {OUTPUT_DIR.name}...", flush=True)
            gam_model = joblib.load(gam_file)
            ebm_model = joblib.load(ebm_file)
        else:
            print(f"    Fitting pyGAM with weights (ESS = {compute_ess(w_train):.1f})...", flush=True)
            gam_model = LogisticGAM(
                s(0, n_splines=15, lam=1.0) +
                s(1, n_splines=15, lam=1.0) +
                s(2, n_splines=15, lam=1.0) +
                s(3, n_splines=15, lam=1.0) +
                s(4, n_splines=15, lam=1.0) +
                s(5, n_splines=15, lam=1.0),
                max_iter=100, tol=0.0001
            )
            gam_model.fit(X_fl_train, y_fl_train, weights=w_train)

            print(f"    Fitting InterpretML EBM with sample_weight...", flush=True)
            ebm_model = ExplainableBoostingClassifier(
                feature_names=FEATURES,
                learning_rate=0.01,
                max_bins=128,
                outer_bags=16,
                interactions=0,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            ebm_model.fit(X_train_df, y_fl_train, sample_weight=w_train)

            joblib.dump(gam_model, OUTPUT_DIR / f"gam_{cfg_key}.pkl")
            joblib.dump(ebm_model, OUTPUT_DIR / f"ebm_{cfg_key}.pkl")

        trained_models[f"gam_{cfg_key}"] = gam_model
        trained_models[f"ebm_{cfg_key}"] = ebm_model

        # Predictions
        p_fl_test_gam = gam_model.predict_proba(X_fl_test)
        p_fl_test_ebm = ebm_model.predict_proba(X_fl_test_df)[:, 1]

        for m_name, p_vec, fl_tau in [("pyGAM", p_fl_test_gam, TAU_GAM), ("InterpretML EBM", p_fl_test_ebm, TAU_EBM)]:
            m_res_opt = compute_metrics(y_fl_test, p_vec, threshold=fl_tau)
            m_res_opt.update({"weight_config": cfg_label, "model": m_name, "dataset": "Florida Holdout Test (Oct-Dec)", "inference_condition": "Standard", "threshold_type": f"FL Optimal ({fl_tau})"})
            model_eval_records.append(m_res_opt)

            m_res_std = compute_metrics(y_fl_test, p_vec, threshold=0.50)
            m_res_std.update({"weight_config": cfg_label, "model": m_name, "dataset": "Florida Holdout Test (Oct-Dec)", "inference_condition": "Standard", "threshold_type": "Standard (0.50)"})
            model_eval_records.append(m_res_std)

        p_ch_raw_gam = gam_model.predict_proba(X_ch_raw)
        p_ch_raw_ebm = ebm_model.predict_proba(X_ch_raw_df)[:, 1]

        for m_name, p_vec, fl_tau in [("pyGAM", p_ch_raw_gam, TAU_GAM), ("InterpretML EBM", p_ch_raw_ebm, TAU_EBM)]:
            m_res_opt = compute_metrics(y_ch, p_vec, threshold=fl_tau)
            m_res_opt.update({"weight_config": cfg_label, "model": m_name, "dataset": "Chandipur Target (2022-2025)", "inference_condition": "Cond_A (Raw Features)", "threshold_type": f"FL Optimal ({fl_tau})"})
            model_eval_records.append(m_res_opt)

            m_res_std = compute_metrics(y_ch, p_vec, threshold=0.50)
            m_res_std.update({"weight_config": cfg_label, "model": m_name, "dataset": "Chandipur Target (2022-2025)", "inference_condition": "Cond_A (Raw Features)", "threshold_type": "Standard (0.50)"})
            model_eval_records.append(m_res_std)

        p_ch_clip_gam = gam_model.predict_proba(X_ch_clipped)
        p_ch_clip_ebm = ebm_model.predict_proba(X_ch_clipped_df)[:, 1]

        for m_name, p_vec, fl_tau in [("pyGAM", p_ch_clip_gam, TAU_GAM), ("InterpretML EBM", p_ch_clip_ebm, TAU_EBM)]:
            m_res_opt = compute_metrics(y_ch, p_vec, threshold=fl_tau)
            m_res_opt.update({"weight_config": cfg_label, "model": m_name, "dataset": "Chandipur Target (2022-2025)", "inference_condition": "Cond_B (CAPE Clipped 7,240 J/kg)", "threshold_type": f"FL Optimal ({fl_tau})"})
            model_eval_records.append(m_res_opt)

            m_res_std = compute_metrics(y_ch, p_vec, threshold=0.50)
            m_res_std.update({"weight_config": cfg_label, "model": m_name, "dataset": "Chandipur Target (2022-2025)", "inference_condition": "Cond_B (CAPE Clipped 7,240 J/kg)", "threshold_type": "Standard (0.50)"})
            model_eval_records.append(m_res_std)

        extreme_case_eval.append({
            "weight_config": cfg_label,
            "gam_prob_raw_mean": float(np.mean(p_ch_raw_gam[extreme_mask])),
            "gam_prob_raw_min": float(np.min(p_ch_raw_gam[extreme_mask])),
            "gam_prob_raw_max": float(np.max(p_ch_raw_gam[extreme_mask])),
            "gam_prob_clip_mean": float(np.mean(p_ch_clip_gam[extreme_mask])),
            "gam_prob_clip_min": float(np.min(p_ch_clip_gam[extreme_mask])),
            "gam_prob_clip_max": float(np.max(p_ch_clip_gam[extreme_mask])),
            "ebm_prob_raw_mean": float(np.mean(p_ch_raw_ebm[extreme_mask])),
            "ebm_prob_raw_min": float(np.min(p_ch_raw_ebm[extreme_mask])),
            "ebm_prob_raw_max": float(np.max(p_ch_raw_ebm[extreme_mask])),
            "ebm_prob_clip_mean": float(np.mean(p_ch_clip_ebm[extreme_mask])),
            "ebm_prob_clip_min": float(np.min(p_ch_clip_ebm[extreme_mask])),
            "ebm_prob_clip_max": float(np.max(p_ch_clip_ebm[extreme_mask])),
        })
        print(f"    Completed evaluation for {cfg_label} in {time.time()-t_mod:.1f}s", flush=True)

    model_eval_df = pd.DataFrame(model_eval_records)
    model_eval_df.to_csv(OUTPUT_DIR / "weighted_model_comparison.csv", index=False)

    extreme_df = pd.DataFrame(extreme_case_eval)
    extreme_df.to_csv(OUTPUT_DIR / "extreme_cape_weighting_analysis.csv", index=False)
    print(f"  Model evaluation metrics saved to {OUTPUT_DIR / 'weighted_model_comparison.csv'}", flush=True)

    # ---------------------------------------------------------
    # 8. Method Comparison CSV (Summary)
    # ---------------------------------------------------------
    print(f"\n[7/8] Generating Importance Weight Comparison CSV...", flush=True)
    weight_comp_records = []
    for w_name, w_vec in all_weights.items():
        ess = compute_ess(w_vec)
        weight_comp_records.append({
            "weight_identifier": w_name,
            "sample_count": len(w_vec),
            "ess": round(ess, 2),
            "ess_ratio": round(ess / len(w_vec), 6),
            "min": round(float(np.min(w_vec)), 6),
            "max": round(float(np.max(w_vec)), 6),
            "mean": round(float(np.mean(w_vec)), 6),
            "std": round(float(np.std(w_vec)), 6),
            "median": round(float(np.median(w_vec)), 6),
            "p95": round(float(np.percentile(w_vec, 95)), 6),
            "p99": round(float(np.percentile(w_vec, 99)), 6),
        })
    pd.DataFrame(weight_comp_records).to_csv(OUTPUT_DIR / "importance_weight_comparison.csv", index=False)

    # ---------------------------------------------------------
    # 9. Publication-Grade Diagnostic Visualizations (300 DPI)
    # ---------------------------------------------------------
    print(f"\n[8/8] Rendering 8 publication-grade diagnostic figures (300 DPI)...", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Figure 1: Importance Weight Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    for ax, (w_key, title, color) in zip(axes.ravel(), [
        ("logistic_regression_clip_p99", "Logistic Regression (p99 Clipped)", "#1f77b4"),
        ("lightgbm_clip_p99", "LightGBM (p99 Clipped)", "#2ca02c"),
        ("random_forest_clip_p99", "Random Forest (p99 Clipped)", "#ff7f0e"),
        ("kmm_no_clip", "Kernel Mean Matching (KMM)", "#d62728")
    ]):
        w_vals = all_weights[w_key]
        sns.histplot(w_vals, bins=50, kde=True, ax=ax, color=color, stat="density")
        ax.set_title(f"{title}\nESS = {compute_ess(w_vals):.0f} ({compute_ess(w_vals)/N_S*100:.1f}%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Weight $w(x)$")
        ax.set_ylabel("Density")
        ax.axvline(1.0, color="black", linestyle="--", alpha=0.7, label="Neutral Weight (1.0)")
        ax.axvline(np.mean(w_vals), color="red", linestyle=":", label=f"Mean ({np.mean(w_vals):.2f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importance_weight_distributions.png", dpi=300)
    plt.close()

    # Figure 2: Source-Target Feature Comparison: Unweighted vs Weighted
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    w_best = all_weights["lightgbm_clip_p99"]
    for ax, feat in zip(axes.ravel(), FEATURES):
        sns.kdeplot(x=fl_train[feat].values, ax=ax, label="Florida Source (Unweighted)", color="blue", linewidth=1.5, alpha=0.7)
        sns.kdeplot(x=fl_train[feat].values, weights=w_best, ax=ax, label="Florida Source (LightGBM Weighted)", color="green", linewidth=2.0)
        sns.kdeplot(x=ch_df[feat].values, ax=ax, label="Chandipur Target", color="red", linestyle="--", linewidth=1.8)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "source_target_weighted_comparison.png", dpi=300)
    plt.close()

    # Figure 3: Effective Sample Size Comparison Across Policies
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ess_pivot = ess_df.pivot(index="clipping_policy", columns="weighting_method", values="ess_ratio")
    ess_pivot.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Effective Sample Size Ratio ($ESS / N_{source}$) Across Clipping Policies", fontsize=13, fontweight="bold")
    ax.set_ylabel("ESS / $N_{source}$")
    ax.set_xlabel("Clipping Policy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.axhline(0.5, color="red", linestyle="--", label="50% Sample Retention")
    ax.legend(title="Density-Ratio Method")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "effective_sample_size_comparison.png", dpi=300)
    plt.close()

    # Figure 4: Domain Classifier ROC Curves
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    for name, p_oof, col in [("Logistic Regression", prob_oof_lr, "#1f77b4"), ("Random Forest", prob_oof_rf, "#ff7f0e"), ("LightGBM", prob_oof_lgb, "#2ca02c")]:
        fpr, tpr, _ = roc_curve(y_domain, p_oof)
        auc = roc_auc_score(y_domain, p_oof)
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc:.4f})", color=col, linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax.set_title("Out-of-Fold Domain Classifier ROC Curves\n(Distinguishing Florida Source from Chandipur Target)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weighted_domain_classifier_roc.png", dpi=300)
    plt.close()

    # Figure 5: Weighted GAM & EBM Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ch_res = model_eval_df[(model_eval_df["dataset"] == "Chandipur Target (2022-2025)") & (model_eval_df["inference_condition"] == "Cond_B (CAPE Clipped 7,240 J/kg)") & (model_eval_df["threshold_type"].str.contains("FL Optimal"))]
    
    sns.barplot(data=ch_res, x="weight_config", y="f1", hue="model", ax=axes[0], palette=["#3274a1", "#e1812c"])
    axes[0].set_title("F1-Score on Chandipur Target (CAPE Clipped)", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0.4, 0.7)

    sns.barplot(data=ch_res, x="weight_config", y="pr_auc", hue="model", ax=axes[1], palette=["#3274a1", "#e1812c"])
    axes[1].set_title("PR-AUC on Chandipur Target (CAPE Clipped)", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0.4, 0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weighted_gam_ebm_performance.png", dpi=300)
    plt.close()

    # Figure 6: Seasonal Weight Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    for ax, sea in zip(axes, ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)"]):
        s_mask = (fl_train["season"] == sea).values
        sns.histplot(w_seasonal[s_mask], bins=40, kde=True, ax=ax, color="#9467bd")
        ax.set_title(f"{sea}\nESS = {compute_ess(w_seasonal[s_mask]):.0f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Seasonal Weight $w(x)$")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_weight_distribution.png", dpi=300)
    plt.close()

    # Figure 7: Extreme CAPE Case Analysis
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(data=extreme_df, x="weight_config", y="gam_prob_raw_mean", ax=ax, color="#1f77b4", alpha=0.8, label="pyGAM (Raw Chandipur CAPE)")
    sns.barplot(data=extreme_df, x="weight_config", y="gam_prob_clip_mean", ax=ax, color="#2ca02c", alpha=0.5, label="pyGAM (CAPE Clipped at 7,240 J/kg)")
    ax.set_title("Mean pyGAM Predicted Probability on 81 Extreme CAPE Cases (> 7,240 J/kg)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Predicted Thunderstorm Probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "extreme_cape_weight_analysis.png", dpi=300)
    plt.close()

    # Figure 8: Calibration Comparison (Reliability Curves)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    prob_base_ebm = orig_ebm.predict_proba(X_fl_test_df)[:, 1]
    prob_w_ebm = trained_models["ebm_lightgbm_clip_p99"].predict_proba(X_fl_test_df)[:, 1]

    fop_base, mpv_base = calibration_curve(y_fl_test, prob_base_ebm, n_bins=10)
    fop_w, mpv_w = calibration_curve(y_fl_test, prob_w_ebm, n_bins=10)

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(mpv_base, fop_base, "s-", label="EBM Baseline (Unweighted)", color="#1f77b4")
    axes[0].plot(mpv_w, fop_w, "o-", label="EBM LightGBM Weighted", color="#2ca02c")
    axes[0].set_title("Reliability Curve: Florida Holdout Test Set", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Observed Fraction of Positives")
    axes[0].legend()

    prob_base_ch = orig_ebm.predict_proba(X_ch_clipped_df)[:, 1]
    prob_w_ch = trained_models["ebm_lightgbm_clip_p99"].predict_proba(X_ch_clipped_df)[:, 1]

    fop_ch_base, mpv_ch_base = calibration_curve(y_ch, prob_base_ch, n_bins=10)
    fop_ch_w, mpv_ch_w = calibration_curve(y_ch, prob_w_ch, n_bins=10)

    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[1].plot(mpv_ch_base, fop_ch_base, "s-", label="EBM Baseline (Unweighted)", color="#1f77b4")
    axes[1].plot(mpv_ch_w, fop_ch_w, "o-", label="EBM LightGBM Weighted", color="#2ca02c")
    axes[1].set_title("Reliability Curve: Chandipur Target (CAPE Clipped)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Observed Fraction of Positives (Proxy)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_comparison.png", dpi=300)
    plt.close()

    print(f"\nAll 8 figures generated successfully in {OUTPUT_DIR}", flush=True)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.", flush=True)
    print("STEP 6 SCRIPT EXECUTION COMPLETE.", flush=True)

if __name__ == "__main__":
    main()
