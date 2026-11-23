"""
run_step_07_physically_constrained_adaptation.py
================================================
Step 7 of Thunderstorm Domain Adaptation Framework (TDAF):
Physically Constrained Feature Adaptation.

Experiments:
- Experiment A: Marginal Distribution Diagnostics (Wasserstein, KS-test, percentiles, zeros, physical violations)
- Experiment B: Physically Constrained Marginal Transformations:
    B1: Identity Transformation (Raw Control)
    B2: Robust Quantile Scaling (RQS) with physical clamping
    B3: Bounded Monotonic Transformation (BMT - Log1p smooth tail compression)
    B4: Selective Transformation (ST - BMT on CAPE/CIN only; kinematics & clouds untouched)
    B5: CAPE Boundary Clipping (7,240 J/kg - Step 4 Reference Condition)
- Physical Validity & Rank/Linear Correlation Preservation Audits
- Inference evaluation on frozen Florida GAM and EBM models
- Seasonal Diagnostics across DJF, MAM, JJAS, and ON
- Interpretability & Partial Dependence Response Curve Analysis
- 5 comprehensive CSV tables and 11 publication-grade 300 DPI figures
"""

import os
import sys
import time
from pathlib import Path

# Force unbuffered stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance, ks_2samp, spearmanr, pearsonr
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
    matthews_corrcoef
)

# ---------------------------------------------------------
# 1. Paths & Configurations
# ---------------------------------------------------------
ROOT_DIR = Path(".").resolve()
FLORIDA_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CHANDIPUR_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

OUTPUT_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "step_07"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIRROR_DIR = ROOT_DIR / "Florida_Thunderstorm_2024_Training" / "reports" / "domain_adaptation" / "step_07"
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_GAM_PATH = ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_gam_model.pkl"
NATIVE_EBM_PATH = ROOT_DIR / "reports" / "ebm_gam_florida_native" / "native_ebm_model.pkl"

FEATURES = [
    "cape_J_kg",
    "cin_J_kg",
    "bulk_shear_850_300_ms",
    "cloud_cover_fraction",
    "wind_speed_10m_ms",
    "wind_speed_850hPa_ms"
]

FEATURE_PRETTY = {
    "cape_J_kg": "CAPE (J/kg)",
    "cin_J_kg": "CIN (J/kg)",
    "bulk_shear_850_300_ms": "Bulk Shear 850-300 hPa (m/s)",
    "cloud_cover_fraction": "Cloud Cover Fraction",
    "wind_speed_10m_ms": "10 m Wind Speed (m/s)",
    "wind_speed_850hPa_ms": "850 hPa Wind Speed (m/s)"
}

CAPE_CLIP_THRESHOLD = 7240.0
TAU_GAM = 0.33
TAU_EBM = 0.34

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

def main():
    start_time = time.time()
    print("=" * 80, flush=True)
    print("STEP 7: PHYSICALLY CONSTRAINED FEATURE ADAPTATION", flush=True)
    print("=" * 80, flush=True)

    # ---------------------------------------------------------
    # 2. Ingest Source and Target Datasets
    # ---------------------------------------------------------
    print("\n[1/7] Loading and aligning source and target datasets...", flush=True)
    fl_df = pd.read_parquet(FLORIDA_PATH)
    fl_valid = fl_df[fl_df["target_asos_strict_crit_c"].notna()].copy()
    fl_train = fl_valid[fl_valid["month"].between(1, 7)].copy().reset_index(drop=True)
    fl_test = fl_valid[fl_valid["month"].between(10, 12)].copy().reset_index(drop=True)

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
    print(f"Source Domain (Florida Train Jan-Jul 2024): N = {N_S}", flush=True)
    print(f"Target Domain (Chandipur ERA5 2022-2025):   N = {N_T}, Positive proxy rate = {ch_df['storm_label_score'].mean():.4f}", flush=True)

    y_ch = ch_df["storm_label_score"].values

    # Load frozen native models
    orig_gam = joblib.load(NATIVE_GAM_PATH)
    orig_ebm = joblib.load(NATIVE_EBM_PATH)
    print(f"Frozen Florida Native Models loaded successfully: pyGAM ({type(orig_gam).__name__}), EBM ({type(orig_ebm).__name__})", flush=True)

    # ---------------------------------------------------------
    # 3. Experiment A: Marginal Distribution Diagnostics
    # ---------------------------------------------------------
    print("\n[2/7] Experiment A: Computing Marginal Distribution Diagnostics...", flush=True)
    diag_records = []
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Render 6 individual distribution plots and gather metrics
    for feat in FEATURES:
        s_vals = fl_train[feat].values
        t_vals = ch_df[feat].values

        # Statistical Moments
        s_mean, s_std, s_med = np.mean(s_vals), np.std(s_vals), np.median(s_vals)
        t_mean, t_std, t_med = np.mean(t_vals), np.std(t_vals), np.median(t_vals)

        # Percentiles
        s_p05, s_p25, s_p75, s_p95, s_p99 = np.percentile(s_vals, [5, 25, 75, 95, 99])
        t_p05, t_p25, t_p75, t_p95, t_p99 = np.percentile(t_vals, [5, 25, 75, 95, 99])

        s_min, s_max = np.min(s_vals), np.max(s_vals)
        t_min, t_max = np.min(t_vals), np.max(t_vals)

        s_zeros = np.mean(s_vals == 0.0) * 100.0
        t_zeros = np.mean(t_vals == 0.0) * 100.0

        # Physical violations in raw data
        if "cloud" in feat:
            s_viol = np.sum((s_vals < 0.0) | (s_vals > 1.0))
            t_viol = np.sum((t_vals < 0.0) | (t_vals > 1.0))
        else:
            s_viol = np.sum(s_vals < 0.0)
            t_viol = np.sum(t_vals < 0.0)

        # Distance Metrics
        wass_dist = wasserstein_distance(s_vals, t_vals)
        ks_stat, ks_pval = ks_2samp(s_vals, t_vals)

        # Meteorological Interpretation
        if feat == "cape_J_kg":
            met_type = "Genuine Meteorological: Tropical maritime monsoon thermodynamic instability vs. subtropical maritime"
        elif feat == "cin_J_kg":
            met_type = "Genuine Meteorological: Intense pre-monsoon dryline inversion lid vs. subtropical maritime boundary layer"
        elif feat == "bulk_shear_850_300_ms":
            met_type = "Genuine Meteorological: Absence of mid-latitude jet stream forcing over tropical Bay of Bengal"
        elif feat == "cloud_cover_fraction":
            met_type = "Measurement/Processing: ERA5 reanalysis diagnostic grid mean vs Florida ASOS ceilometer/satellite composite"
        elif feat == "wind_speed_10m_ms":
            met_type = "Processing/Exposure: Coastal/marine ERA5 grid-cell mean vs. terrestrial ASOS mast anemometer surface roughness"
        else:
            met_type = "Meteorological & Processing: Synoptic monsoon low-level jet vs. subtropical trade/frontal winds"

        diag_records.append({
            "feature": feat,
            "feature_label": FEATURE_PRETTY[feat],
            "source_mean": round(float(s_mean), 2),
            "target_mean": round(float(t_mean), 2),
            "source_std": round(float(s_std), 2),
            "target_std": round(float(t_std), 2),
            "source_median": round(float(s_med), 2),
            "target_median": round(float(t_med), 2),
            "source_iqr": round(float(s_p75 - s_p25), 2),
            "target_iqr": round(float(t_p75 - t_p25), 2),
            "source_p95": round(float(s_p95), 2),
            "target_p95": round(float(t_p95), 2),
            "source_max": round(float(s_max), 2),
            "target_max": round(float(t_max), 2),
            "source_pct_zeros": round(float(s_zeros), 2),
            "target_pct_zeros": round(float(t_zeros), 2),
            "target_raw_violations": int(t_viol),
            "wasserstein_distance": round(float(wass_dist), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "meteorological_classification": met_type
        })

        # Individual Distribution Plot (300 DPI)
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5), dpi=300)
        sns.kdeplot(s_vals, ax=ax_ind, label=f"Florida Source (Median={s_med:.1f})", color="#1f77b4", linewidth=2.0)
        sns.kdeplot(t_vals, ax=ax_ind, label=f"Chandipur Target (Median={t_med:.1f})", color="#d62728", linewidth=2.0, linestyle="--")
        ax_ind.set_title(f"Marginal Distribution: {FEATURE_PRETTY[feat]}\nWasserstein Dist = {wass_dist:.2f}, KS Stat = {ks_stat:.3f}", fontsize=11, fontweight="bold")
        ax_ind.set_xlabel(FEATURE_PRETTY[feat])
        ax_ind.set_ylabel("Density")
        ax_ind.legend()
        plt.tight_layout()
        safe_fname = feat.replace("_850_300_ms", "").replace("_fraction", "").replace("_ms", "")
        plt.savefig(OUTPUT_DIR / f"dist_{safe_fname}.png", dpi=300)
        plt.close()

    # Consolidated 6-panel Figure
    fig_cons, axes_cons = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    for ax, feat in zip(axes_cons.ravel(), FEATURES):
        s_vals = fl_train[feat].values
        t_vals = ch_df[feat].values
        sns.kdeplot(s_vals, ax=ax, label="Florida Source", color="#1f77b4", linewidth=1.8)
        sns.kdeplot(t_vals, ax=ax, label="Chandipur Target", color="#d62728", linewidth=1.8, linestyle="--")
        ax.set_title(FEATURE_PRETTY[feat], fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)
    fig_cons.suptitle("Consolidated Source vs. Target Marginal Feature Distributions", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "consolidated_feature_distributions.png", dpi=300)
    plt.close()

    diag_df = pd.DataFrame(diag_records)
    diag_df.to_csv(OUTPUT_DIR / "feature_transformation_comparison.csv", index=False)
    print(f"  Diagnostics computed and saved to {OUTPUT_DIR / 'feature_transformation_comparison.csv'}", flush=True)

    # ---------------------------------------------------------
    # 4. Experiment B: Implement Physically Constrained Transformations
    # ---------------------------------------------------------
    print("\n[3/7] Experiment B: Constructing Physically Constrained Transformed Datasets...", flush=True)
    X_raw = ch_df[FEATURES].values

    # B1: Identity Transformation (Raw Control)
    X_B1 = X_raw.copy()

    # B2: Robust Quantile Scaling (RQS) with physical clamping
    # x_new = median_S + (IQR_S / IQR_T) * (x - median_T), clamped to physical ranges
    X_B2 = np.zeros_like(X_raw)
    for j, feat in enumerate(FEATURES):
        s_med = fl_train[feat].median()
        s_iqr = fl_train[feat].quantile(0.75) - fl_train[feat].quantile(0.25)
        t_med = ch_df[feat].median()
        t_iqr = ch_df[feat].quantile(0.75) - ch_df[feat].quantile(0.25)
        scale_ratio = s_iqr / (t_iqr + 1e-6)
        x_scaled = s_med + scale_ratio * (X_raw[:, j] - t_med)
        
        # Enforce physical validity
        if feat in ["cape_J_kg", "cin_J_kg", "bulk_shear_850_300_ms", "wind_speed_10m_ms", "wind_speed_850hPa_ms"]:
            x_scaled = np.maximum(0.0, x_scaled)
        elif feat == "cloud_cover_fraction":
            x_scaled = np.clip(x_scaled, 0.0, 1.0)
        X_B2[:, j] = x_scaled

    # B3: Bounded Monotonic Transformation (BMT - Log1p smooth tail compression)
    # Compresses extreme energetic positive tails while mapping 0 -> 0 and preserving rank ordering exactly
    X_B3 = X_raw.copy()
    # CAPE: smooth log1p compression mapping Chandipur p95 to Florida p95
    s_cape_p95 = fl_train["cape_J_kg"].quantile(0.95)
    t_cape_p95 = ch_df["cape_J_kg"].quantile(0.95)
    alpha_cape = s_cape_p95 / np.log1p(t_cape_p95 / 1000.0)
    X_B3[:, 0] = alpha_cape * np.log1p(X_raw[:, 0] / 1000.0)

    # CIN: smooth log1p compression mapping Chandipur p95 to Florida p95
    s_cin_p95 = fl_train["cin_J_kg"].quantile(0.95)
    t_cin_p95 = ch_df["cin_J_kg"].quantile(0.95)
    alpha_cin = s_cin_p95 / np.log1p(t_cin_p95 / 100.0)
    X_B3[:, 1] = alpha_cin * np.log1p(X_raw[:, 1] / 100.0)

    # Shear & Winds: robust quantile shrinkage
    for j in [2, 4, 5]:
        s_med = fl_train[FEATURES[j]].median()
        s_p95 = fl_train[FEATURES[j]].quantile(0.95)
        t_med = ch_df[FEATURES[j]].median()
        t_p95 = ch_df[FEATURES[j]].quantile(0.95)
        scale_j = (s_p95 - s_med) / (t_p95 - t_med + 1e-6)
        X_B3[:, j] = np.maximum(0.0, s_med + scale_j * (X_raw[:, j] - t_med))
    # Cloud cover: clipped
    X_B3[:, 3] = np.clip(X_raw[:, 3], 0.0, 1.0)

    # B4: Selective Transformation (ST)
    # Transform ONLY CAPE & CIN (thermodynamic extremes); leave kinematics (shear, winds) & clouds 100% UNTOUCHED
    X_B4 = X_raw.copy()
    X_B4[:, 0] = alpha_cape * np.log1p(X_raw[:, 0] / 1000.0)  # BMT on CAPE
    X_B4[:, 1] = alpha_cin * np.log1p(X_raw[:, 1] / 100.0)    # BMT on CIN
    # Shear (col 2), Cloud (col 3), Wind10m (col 4), Wind850 (col 5) strictly untouched!

    # B5: CAPE Boundary Clipping Only (7,240 J/kg - Step 4 Reference Benchmark)
    X_B5 = X_raw.copy()
    X_B5[:, 0] = np.minimum(X_B5[:, 0], CAPE_CLIP_THRESHOLD)

    transformations = [
        ("B1_Identity_Raw", "B1: Identity (Raw Control)", X_B1),
        ("B2_Robust_Quantile_Scaling", "B2: Robust Quantile Scaling (RQS)", X_B2),
        ("B3_Bounded_Monotonic", "B3: Bounded Monotonic Transformation (BMT)", X_B3),
        ("B4_Selective_CAPE_CIN", "B4: Selective Transformation (CAPE & CIN only)", X_B4),
        ("B5_CAPE_Clipping_7240", "B5: CAPE Boundary Clipping (7,240 J/kg Reference)", X_B5)
    ]

    # ---------------------------------------------------------
    # 5. Physical Validity and Correlation Preservation Audits
    # ---------------------------------------------------------
    print("\n[4/7] Auditing Physical Validity and Correlation Preservation...", flush=True)
    validity_records = []
    corr_records = []

    # Compute raw target correlation matrices
    raw_df = pd.DataFrame(X_raw, columns=FEATURES)
    R_raw_pearson = raw_df.corr(method="pearson").values
    R_raw_spearman = raw_df.corr(method="spearman").values

    for exp_id, exp_name, X_t in transformations:
        # Physical validity checks
        t_df = pd.DataFrame(X_t, columns=FEATURES)
        neg_cape = int(np.sum(X_t[:, 0] < 0.0))
        neg_cin = int(np.sum(X_t[:, 1] < 0.0))
        neg_shear = int(np.sum(X_t[:, 2] < 0.0))
        cloud_oob = int(np.sum((X_t[:, 3] < 0.0) | (X_t[:, 3] > 1.0)))
        neg_w10 = int(np.sum(X_t[:, 4] < 0.0))
        neg_w850 = int(np.sum(X_t[:, 5] < 0.0))
        nan_count = int(np.sum(np.isnan(X_t)))
        inf_count = int(np.sum(np.isinf(X_t)))

        total_violations = neg_cape + neg_cin + neg_shear + cloud_oob + neg_w10 + neg_w850 + nan_count + inf_count

        validity_records.append({
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "negative_cape_count": neg_cape,
            "negative_cin_count": neg_cin,
            "negative_bulk_shear_count": neg_shear,
            "cloud_cover_out_of_bounds": cloud_oob,
            "negative_wind_10m_count": neg_w10,
            "negative_wind_850hPa_count": neg_w850,
            "nan_or_inf_count": nan_count + inf_count,
            "total_physical_violations": total_violations,
            "physical_validity_verdict": "PASS (Zero Violations)" if total_violations == 0 else "FAIL"
        })

        # Correlation preservation
        R_t_pearson = t_df.corr(method="pearson").values
        R_t_spearman = t_df.corr(method="spearman").values

        diff_pearson_fro = float(np.linalg.norm(R_t_pearson - R_raw_pearson, ord="fro"))
        diff_spearman_fro = float(np.linalg.norm(R_t_spearman - R_raw_spearman, ord="fro"))
        max_pearson_diff = float(np.max(np.abs(R_t_pearson - R_raw_pearson)))
        max_spearman_diff = float(np.max(np.abs(R_t_spearman - R_raw_spearman)))

        corr_records.append({
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "frobenius_diff_pearson": round(diff_pearson_fro, 6),
            "max_abs_diff_pearson": round(max_pearson_diff, 6),
            "frobenius_diff_spearman": round(diff_spearman_fro, 6),
            "max_abs_diff_spearman": round(max_spearman_diff, 6),
            "spearman_rank_preserved": "EXACT (Identical)" if diff_spearman_fro < 1e-4 else "APPROXIMATE",
            "correlation_audit_verdict": "PASS" if diff_spearman_fro < 0.10 else "FAIL"
        })

    pd.DataFrame(validity_records).to_csv(OUTPUT_DIR / "physical_validity_audit.csv", index=False)
    pd.DataFrame(corr_records).to_csv(OUTPUT_DIR / "correlation_preservation_audit.csv", index=False)
    print("  Physical validity and correlation audits saved.", flush=True)

    # ---------------------------------------------------------
    # 6. Model Evaluation on Frozen Florida Models
    # ---------------------------------------------------------
    print("\n[5/7] Evaluating Frozen Florida Models across all Transformations...", flush=True)
    perf_records = []
    seasonal_records = []

    for exp_id, exp_name, X_t in transformations:
        t_df = pd.DataFrame(X_t, columns=FEATURES)

        # pyGAM Predictions
        p_gam = orig_gam.predict_proba(X_t)
        # EBM Predictions
        p_ebm = orig_ebm.predict_proba(t_df)[:, 1]

        for m_name, p_vec, fl_tau in [("pyGAM", p_gam, TAU_GAM), ("InterpretML EBM", p_ebm, TAU_EBM)]:
            # FL Optimal Threshold
            res_opt = compute_metrics(y_ch, p_vec, threshold=fl_tau)
            res_opt.update({
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "model": m_name,
                "threshold_type": f"FL Optimal ({fl_tau})"
            })
            perf_records.append(res_opt)

            # Standard Threshold (0.50)
            res_std = compute_metrics(y_ch, p_vec, threshold=0.50)
            res_std.update({
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "model": m_name,
                "threshold_type": "Standard (0.50)"
            })
            perf_records.append(res_std)

            # Seasonal performance at FL Optimal Threshold
            for season in ["DJF (Winter)", "MAM (Pre-monsoon)", "JJAS (Monsoon)", "ON (Post-monsoon)"]:
                sea_mask = (ch_df["season"] == season).values
                y_sea = y_ch[sea_mask]
                p_sea = p_vec[sea_mask]
                res_sea = compute_metrics(y_sea, p_sea, threshold=fl_tau)
                seasonal_records.append({
                    "experiment_id": exp_id,
                    "experiment_name": exp_name,
                    "model": m_name,
                    "season": season,
                    "sample_count": len(y_sea),
                    "positives": int(np.sum(y_sea)),
                    "prevalence": round(float(np.mean(y_sea)), 4),
                    "f1": res_sea["f1"],
                    "precision": res_sea["precision"],
                    "recall": res_sea["recall"],
                    "pr_auc": res_sea["pr_auc"],
                    "roc_auc": res_sea["roc_auc"],
                    "brier_score": res_sea["brier_score"]
                })

    perf_df = pd.DataFrame(perf_records)
    perf_df.to_csv(OUTPUT_DIR / "step_07_model_performance_comparison.csv", index=False)

    seasonal_df = pd.DataFrame(seasonal_records)
    seasonal_df.to_csv(OUTPUT_DIR / "seasonal_transformation_analysis.csv", index=False)
    print("  Performance metrics and seasonal breakdown saved.", flush=True)

    # ---------------------------------------------------------
    # 7. Model Interpretability & Response Curves
    # ---------------------------------------------------------
    print("\n[6/7] Inspecting Model Interpretability and Partial Dependence...", flush=True)
    # Generate synthetic response curves across physical range of CAPE, CIN, and Bulk Shear
    fig_resp, axes_resp = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

    # Median background profile in Florida
    x_bg = np.array([fl_train[f].median() for f in FEATURES])

    # 1. CAPE sweep: 0 to 9,906 J/kg
    cape_grid = np.linspace(0, 9906, 200)
    X_sweep_cape = np.tile(x_bg, (len(cape_grid), 1))
    X_sweep_cape[:, 0] = cape_grid

    # Transform CAPE sweep under B1, B2, B3, B4, B5
    # Raw B1
    p_gam_cape_raw = orig_gam.predict_proba(X_sweep_cape)
    p_ebm_cape_raw = orig_ebm.predict_proba(pd.DataFrame(X_sweep_cape, columns=FEATURES))[:, 1]

    # B5 Clipped
    X_sweep_cape_clip = X_sweep_cape.copy()
    X_sweep_cape_clip[:, 0] = np.minimum(X_sweep_cape_clip[:, 0], CAPE_CLIP_THRESHOLD)
    p_gam_cape_clip = orig_gam.predict_proba(X_sweep_cape_clip)
    p_ebm_cape_clip = orig_ebm.predict_proba(pd.DataFrame(X_sweep_cape_clip, columns=FEATURES))[:, 1]

    # B4 Selective BMT
    X_sweep_cape_bmt = X_sweep_cape.copy()
    X_sweep_cape_bmt[:, 0] = alpha_cape * np.log1p(X_sweep_cape_bmt[:, 0] / 1000.0)
    p_gam_cape_bmt = orig_gam.predict_proba(X_sweep_cape_bmt)
    p_ebm_cape_bmt = orig_ebm.predict_proba(pd.DataFrame(X_sweep_cape_bmt, columns=FEATURES))[:, 1]

    # Row 1: pyGAM Partial Dependence
    axes_resp[0, 0].plot(cape_grid, p_gam_cape_raw, "r-", label="B1: Raw Inference", linewidth=2.0)
    axes_resp[0, 0].plot(cape_grid, p_gam_cape_clip, "g--", label="B5: CAPE Clipped (7,240)", linewidth=2.0)
    axes_resp[0, 0].plot(cape_grid, p_gam_cape_bmt, "b-.", label="B4: Selective BMT", linewidth=2.0)
    axes_resp[0, 0].set_title("pyGAM: CAPE Response Profile", fontsize=11, fontweight="bold")
    axes_resp[0, 0].set_xlabel("Input CAPE (J/kg)")
    axes_resp[0, 0].set_ylabel("Predicted Probability")
    axes_resp[0, 0].axvline(7240, color="gray", linestyle=":", label="FL Max Train (7,240)")
    axes_resp[0, 0].legend(fontsize=8)

    # 2. CIN sweep: 0 to 1,000 J/kg
    cin_grid = np.linspace(0, 1000, 200)
    X_sweep_cin = np.tile(x_bg, (len(cin_grid), 1))
    X_sweep_cin[:, 0] = 2000.0  # fixed high CAPE
    X_sweep_cin[:, 1] = cin_grid
    p_gam_cin_raw = orig_gam.predict_proba(X_sweep_cin)
    X_sweep_cin_bmt = X_sweep_cin.copy()
    X_sweep_cin_bmt[:, 1] = alpha_cin * np.log1p(X_sweep_cin_bmt[:, 1] / 100.0)
    p_gam_cin_bmt = orig_gam.predict_proba(X_sweep_cin_bmt)

    axes_resp[0, 1].plot(cin_grid, p_gam_cin_raw, "r-", label="Raw CIN", linewidth=2.0)
    axes_resp[0, 1].plot(cin_grid, p_gam_cin_bmt, "b-.", label="BMT Transformed CIN", linewidth=2.0)
    axes_resp[0, 1].set_title("pyGAM: CIN Inhibition Response", fontsize=11, fontweight="bold")
    axes_resp[0, 1].set_xlabel("Input CIN (J/kg)")
    axes_resp[0, 1].set_ylabel("Predicted Probability")
    axes_resp[0, 1].legend(fontsize=8)

    # 3. Bulk Shear sweep: 0 to 60 m/s
    shear_grid = np.linspace(0, 60, 200)
    X_sweep_shear = np.tile(x_bg, (len(shear_grid), 1))
    X_sweep_shear[:, 0] = 2000.0
    X_sweep_shear[:, 2] = shear_grid
    p_gam_shear_raw = orig_gam.predict_proba(X_sweep_shear)

    axes_resp[0, 2].plot(shear_grid, p_gam_shear_raw, "purple", linewidth=2.0, label="pyGAM Bulk Shear Shape")
    axes_resp[0, 2].set_title("pyGAM: Bulk Shear Response", fontsize=11, fontweight="bold")
    axes_resp[0, 2].set_xlabel("Bulk Shear 850-300 hPa (m/s)")
    axes_resp[0, 2].set_ylabel("Predicted Probability")
    axes_resp[0, 2].legend(fontsize=8)

    # Row 2: EBM Partial Dependence
    axes_resp[1, 0].plot(cape_grid, p_ebm_cape_raw, "r-", label="B1: Raw Inference", linewidth=2.0)
    axes_resp[1, 0].plot(cape_grid, p_ebm_cape_clip, "g--", label="B5: CAPE Clipped (7,240)", linewidth=2.0)
    axes_resp[1, 0].plot(cape_grid, p_ebm_cape_bmt, "b-.", label="B4: Selective BMT", linewidth=2.0)
    axes_resp[1, 0].set_title("InterpretML EBM: CAPE Response Profile", fontsize=11, fontweight="bold")
    axes_resp[1, 0].set_xlabel("Input CAPE (J/kg)")
    axes_resp[1, 0].set_ylabel("Predicted Probability")
    axes_resp[1, 0].axvline(7240, color="gray", linestyle=":", label="FL Max Train (7,240)")
    axes_resp[1, 0].legend(fontsize=8)

    p_ebm_cin_raw = orig_ebm.predict_proba(pd.DataFrame(X_sweep_cin, columns=FEATURES))[:, 1]
    p_ebm_cin_bmt = orig_ebm.predict_proba(pd.DataFrame(X_sweep_cin_bmt, columns=FEATURES))[:, 1]
    axes_resp[1, 1].plot(cin_grid, p_ebm_cin_raw, "r-", label="Raw CIN", linewidth=2.0)
    axes_resp[1, 1].plot(cin_grid, p_ebm_cin_bmt, "b-.", label="BMT Transformed CIN", linewidth=2.0)
    axes_resp[1, 1].set_title("InterpretML EBM: CIN Inhibition Response", fontsize=11, fontweight="bold")
    axes_resp[1, 1].set_xlabel("Input CIN (J/kg)")
    axes_resp[1, 1].set_ylabel("Predicted Probability")
    axes_resp[1, 1].legend(fontsize=8)

    p_ebm_shear_raw = orig_ebm.predict_proba(pd.DataFrame(X_sweep_shear, columns=FEATURES))[:, 1]
    axes_resp[1, 2].plot(shear_grid, p_ebm_shear_raw, "purple", linewidth=2.0, label="EBM Bulk Shear Shape")
    axes_resp[1, 2].set_title("InterpretML EBM: Bulk Shear Response", fontsize=11, fontweight="bold")
    axes_resp[1, 2].set_xlabel("Bulk Shear 850-300 hPa (m/s)")
    axes_resp[1, 2].set_ylabel("Predicted Probability")
    axes_resp[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interpretability_response_curves.png", dpi=300)
    plt.close()
    print("  Interpretability curves generated and saved.", flush=True)

    # ---------------------------------------------------------
    # 8. Diagnostic Comparison Visualizations
    # ---------------------------------------------------------
    print("\n[7/7] Generating Comparative Visualizations...", flush=True)
    # Figure: Model Performance Comparison (F1 and PR-AUC at FL Optimal)
    fig_perf, axes_p = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    opt_df = perf_df[perf_df["threshold_type"].str.contains("FL Optimal")]

    sns.barplot(data=opt_df, x="experiment_name", y="f1", hue="model", ax=axes_p[0], palette=["#3274a1", "#e1812c"])
    axes_p[0].set_title("F1-Score across Transformations on Chandipur Target", fontsize=12, fontweight="bold")
    axes_p[0].set_xticklabels(axes_p[0].get_xticklabels(), rotation=25, ha="right")
    axes_p[0].set_ylabel("F1 Score")
    axes_p[0].set_ylim(0.4, 0.7)

    sns.barplot(data=opt_df, x="experiment_name", y="pr_auc", hue="model", ax=axes_p[1], palette=["#3274a1", "#e1812c"])
    axes_p[1].set_title("PR-AUC across Transformations on Chandipur Target", fontsize=12, fontweight="bold")
    axes_p[1].set_xticklabels(axes_p[1].get_xticklabels(), rotation=25, ha="right")
    axes_p[1].set_ylabel("PR-AUC")
    axes_p[1].set_ylim(0.4, 0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_07_model_performance_comparison.png", dpi=300)
    plt.close()

    # Figure: Seasonal Performance Comparison
    fig_sea, ax_sea = plt.subplots(figsize=(14, 6), dpi=300)
    ebm_sea_df = seasonal_df[seasonal_df["model"] == "InterpretML EBM"]
    sns.barplot(data=ebm_sea_df, x="season", y="f1", hue="experiment_name", ax=ax_sea, palette="Set2")
    ax_sea.set_title("Seasonal F1 Score for InterpretML EBM across Transformations", fontsize=12, fontweight="bold")
    ax_sea.set_ylabel("F1 Score")
    ax_sea.legend(title="Transformation", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_performance_comparison.png", dpi=300)
    plt.close()

    # Figure: Correlation Heatmaps (Raw vs B4 Selective BMT)
    fig_corr, axes_c = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.heatmap(R_raw_spearman, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=axes_c[0], xticklabels=[f.split('_')[0] for f in FEATURES], yticklabels=[f.split('_')[0] for f in FEATURES])
    axes_c[0].set_title("Raw Chandipur Spearman Correlation", fontsize=11, fontweight="bold")

    R_b4_spearman = pd.DataFrame(X_B4, columns=FEATURES).corr(method="spearman").values
    sns.heatmap(R_b4_spearman, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes_c[1], xticklabels=[f.split('_')[0] for f in FEATURES], yticklabels=[f.split('_')[0] for f in FEATURES])
    axes_c[1].set_title("B4 Selective BMT Spearman Correlation\n(Rank Invariance Confirmed)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmaps_comparison.png", dpi=300)
    plt.close()

    print(f"\nAll 11 publication figures rendered successfully in {OUTPUT_DIR}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")
    print("STEP 7 EXPERIMENTAL SCRIPT EXECUTION COMPLETE.", flush=True)

if __name__ == "__main__":
    main()
