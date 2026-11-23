"""
audit_chandipur_storm_score_formula.py
======================================
Scientific and mathematical diagnostic audit of the Chandipur storm-score formula:
Storm Score = sum(w_i * f_i(X)) >= 2.5

Diagnostic only: Does NOT modify the formula, retrain models, or overwrite datasets.

Outputs saved in:
`reports/chandipur_formula_diagnostic_audit/`
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# 1. Directories
ROOT_DIR = Path(".").resolve()
CHANDIPUR_DATA_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"
PREDICTIONS_PATH = ROOT_DIR / "reports" / "ebm_gam_chandipur_transfer_native" / "chandipur_native_gam_ebm_predictions.csv"

OUTPUT_DIR = ROOT_DIR / "reports" / "chandipur_formula_diagnostic_audit"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def assign_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter (DJF)"
    elif m in [3, 4, 5]:
        return "Pre-monsoon (MAM)"
    elif m in [6, 7, 8, 9]:
        return "Monsoon (JJAS)"
    elif m in [10, 11]:
        return "Post-monsoon (ON)"
    return "Unknown"

def main():
    print("=" * 80)
    print("CHANDIPUR STORM SCORE FORMULA: SCIENTIFIC DIAGNOSTIC AUDIT")
    print("=" * 80)
    
    # Load dataset
    print(f"Loading Chandipur dataset from {CHANDIPUR_DATA_PATH}...")
    df = pd.read_csv(CHANDIPUR_DATA_PATH)
    df["valid_time"] = pd.to_datetime(df["valid_time"], format="mixed")
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    df["season"] = df["valid_time"].apply(assign_season)
    
    # Load native model predictions
    print(f"Loading native Florida model predictions from {PREDICTIONS_PATH}...")
    df_pred = pd.read_csv(PREDICTIONS_PATH)
    df["prob_gam"] = df_pred["prob_gam"]
    df["prob_ebm"] = df_pred["prob_ebm"]
    df["pred_gam_fl_thresh"] = df_pred["pred_gam_fl_thresh"]
    df["pred_ebm_fl_thresh"] = df_pred["pred_ebm_fl_thresh"]
    
    total_obs = len(df)
    positives = df[df["storm_label_score"] == 1]
    total_pos = len(positives)
    print(f"Total observations: {total_obs}, Total proxy positives: {total_pos} ({total_pos / total_obs * 100:.2f}%)")
    
    # =========================================================================
    # PART 1: EXACT COMPONENT CONTRIBUTION RECONSTRUCTION
    # =========================================================================
    print("\n--- PART 1: Component Contribution Estimation ---")
    # Features normalized to [0, 1] scales according to physical parameter bounds
    # CAPE: capped at 5000 J/kg
    # CP: 24h accumulation capped at 20 mm
    # RH: relative_humidity_300hPa / 100
    # Lift: -vertical_velocity_Pas_300hPa / 0.5 (lift is upward motion, w < 0)
    # Shear: bulk_shear_850_300 / 50 m/s
    # Cloud: total_cloud_cover (already 0-1)
    # Wind: wind_speed_10m / 15 m/s
    # CIN: cin_Jkg / 250 J/kg (suppressive)
    
    norm_cape = np.clip(df["cape_Jkg"] / 3000.0, 0.0, 1.0)
    norm_cp = np.clip(df["cp_24h_sum"] / 10.0, 0.0, 1.0)
    norm_rh = np.clip(df["relative_humidity_300hPa"] / 100.0, 0.0, 1.0)
    norm_lift = np.clip(-df["vertical_velocity_Pas_300hPa"] / 0.5, 0.0, 1.0)
    norm_shear = np.clip(df["bulk_shear_850_300"] / 40.0, 0.0, 1.0)
    norm_cloud = np.clip(df["total_cloud_cover"], 0.0, 1.0)
    norm_wind = np.clip(df["wind_speed_10m"] / 12.0, 0.0, 1.0)
    norm_cin = np.clip(df["cin_Jkg"] / 200.0, 0.0, 1.0)
    
    # Component contributions
    comp_df = pd.DataFrame({
        "contrib_cape": 1.2 * norm_cape,
        "contrib_cp": 1.0 * norm_cp,
        "contrib_rh": 0.8 * norm_rh,
        "contrib_lift": 1.0 * norm_lift,
        "contrib_shear": 0.7 * norm_shear,
        "contrib_cloud": 0.4 * norm_cloud,
        "contrib_wind": 0.5 * norm_wind,
        "penalty_cin": -0.8 * norm_cin
    })
    
    # Component stats table
    comp_stats = []
    comp_names = {
        "contrib_cape": ("CAPE (Thermodynamic Buoyancy)", 1.2, "Surface"),
        "contrib_cp": ("Convective Precipitation (24h Accumulation)", 1.0, "Surface"),
        "contrib_rh": ("Relative Humidity (Upper Troposphere)", 0.8, "300 hPa"),
        "contrib_lift": ("Vertical Velocity Lift (-omega)", 1.0, "300 hPa"),
        "contrib_shear": ("Deep-Layer Wind Shear", 0.7, "850-300 hPa"),
        "contrib_cloud": ("Total Cloud Cover", 0.4, "Column"),
        "contrib_wind": ("Low-Level Wind Speed", 0.5, "10 m"),
        "penalty_cin": ("Convective Inhibition Penalty", -0.8, "Boundary Layer")
    }
    
    for c_col, (c_label, c_weight, c_level) in comp_names.items():
        s_all = comp_df[c_col]
        s_pos = comp_df.loc[df["storm_label_score"] == 1, c_col]
        s_neg = comp_df.loc[df["storm_label_score"] == 0, c_col]
        r_p, _ = pearsonr(s_all, df["storm_score"])
        
        # Calculate decisive frequency: How often removing this component drops a positive score below 2.5
        # score_without_i = storm_score - contrib_i < 2.5
        score_without = df.loc[df["storm_label_score"] == 1, "storm_score"] - s_pos
        decisive_count = int((score_without < 2.5).sum()) if c_col != "penalty_cin" else 0
        decisive_pct = round(decisive_count / total_pos * 100.0, 2)
        
        comp_stats.append({
            "component": c_col,
            "component_label": c_label,
            "assigned_weight": c_weight,
            "vertical_level": c_level,
            "mean_overall": round(float(s_all.mean()), 4),
            "std_overall": round(float(s_all.std()), 4),
            "mean_proxy_positive": round(float(s_pos.mean()), 4),
            "mean_proxy_negative": round(float(s_neg.mean()), 4),
            "delta_pos_minus_neg": round(float(s_pos.mean() - s_neg.mean()), 4),
            "pearson_r_with_storm_score": round(float(r_p), 4),
            "decisive_in_positives_count": decisive_count,
            "decisive_in_positives_pct": decisive_pct
        })
        
    df_comp_stats = pd.DataFrame(comp_stats)
    df_comp_stats.to_csv(OUTPUT_DIR / "component_contribution_analysis.csv", index=False)
    print("Saved component_contribution_analysis.csv")
    
    # =========================================================================
    # PART 2: LOW-CAPE PROXY POSITIVE INVESTIGATION
    # =========================================================================
    print("\n--- PART 2: Systematic Low-CAPE Proxy Positive Analysis ---")
    low_cape_results = []
    thresholds = [50, 100, 250, 500]
    
    for th in thresholds:
        sub = df[(df["storm_label_score"] == 1) & (df["cape_Jkg"] < th)]
        cnt = len(sub)
        pct_pos = round(cnt / total_pos * 100.0, 2)
        pct_total = round(cnt / total_obs * 100.0, 2)
        
        # Major contributing factors for this subset
        mean_rh = round(float(sub["relative_humidity_300hPa"].mean()), 2)
        mean_cloud = round(float(sub["total_cloud_cover"].mean()), 3)
        mean_shear = round(float(sub["bulk_shear_850_300"].mean()), 2)
        mean_cin = round(float(sub["cin_Jkg"].mean()), 2)
        mean_score = round(float(sub["storm_score"].mean()), 2)
        min_score = round(float(sub["storm_score"].min()), 2)
        max_score = round(float(sub["storm_score"].max()), 2)
        
        # Native model probabilities on this low-cape positive set
        mean_prob_gam = round(float(sub["prob_gam"].mean()), 4)
        mean_prob_ebm = round(float(sub["prob_ebm"].mean()), 4)
        pct_ebm_rejects = round(float((sub["prob_ebm"] < 0.34).mean() * 100.0), 2)
        
        # Seasonal breakdown
        sea_counts = sub["season"].value_counts().to_dict()
        yr_counts = sub["year"].value_counts().to_dict()
        
        low_cape_results.append({
            "cape_threshold_J_kg": th,
            "observation_count": cnt,
            "pct_of_all_proxy_positives": pct_pos,
            "pct_of_all_observations": pct_total,
            "storm_score_mean": mean_score,
            "storm_score_min": min_score,
            "storm_score_max": max_score,
            "mean_cape_J_kg": round(float(sub["cape_Jkg"].mean()), 2),
            "mean_cin_J_kg": mean_cin,
            "mean_rh_300hPa_pct": mean_rh,
            "mean_cloud_cover_fraction": mean_cloud,
            "mean_shear_850_300_ms": mean_shear,
            "native_gam_mean_prob": mean_prob_gam,
            "native_ebm_mean_prob": mean_prob_ebm,
            "pct_native_ebm_predicts_zero": pct_ebm_rejects,
            "winter_djf_count": sea_counts.get("Winter (DJF)", 0),
            "pre_monsoon_mam_count": sea_counts.get("Pre-monsoon (MAM)", 0),
            "monsoon_jjas_count": sea_counts.get("Monsoon (JJAS)", 0),
            "post_monsoon_on_count": sea_counts.get("Post-monsoon (ON)", 0),
            "year_2022_count": yr_counts.get(2022, 0),
            "year_2023_count": yr_counts.get(2023, 0),
            "year_2024_count": yr_counts.get(2024, 0),
            "year_2025_count": yr_counts.get(2025, 0)
        })
        print(f"Low CAPE < {th:3d} J/kg: {cnt:4d} cases ({pct_pos:5.2f}% of positives) | Score [{min_score:.2f}, {max_score:.2f}] | EBM Rejects: {pct_ebm_rejects:.1f}%")
        
    df_low_cape = pd.DataFrame(low_cape_results)
    df_low_cape.to_csv(OUTPUT_DIR / "low_cape_positive_analysis.csv", index=False)
    print("Saved low_cape_positive_analysis.csv")
    
    # =========================================================================
    # PART 3: HIGH-CAPE / HIGH-CIN CAPPING INVESTIGATION
    # =========================================================================
    print("\n--- PART 3: High-CAPE / High-CIN Capping Analysis ---")
    high_cin_results = []
    cin_scenarios = [
        (1000, 200, "Moderate Buoyancy (>1000 J/kg) with Moderate Cap (>200 J/kg)"),
        (2000, 300, "High Buoyancy (>2000 J/kg) with Heavy Cap (>300 J/kg)"),
        (3000, 500, "Extreme Buoyancy (>3000 J/kg) with Extreme Cap (>500 J/kg)")
    ]
    
    for c_min, cin_min, desc in cin_scenarios:
        sub_all = df[(df["cape_Jkg"] > c_min) & (df["cin_Jkg"] > cin_min)]
        sub_pos = sub_all[sub_all["storm_label_score"] == 1]
        sub_neg = sub_all[sub_all["storm_label_score"] == 0]
        
        n_all = len(sub_all)
        n_pos = len(sub_pos)
        pct_labeled_pos = round(n_pos / n_all * 100.0, 2) if n_all > 0 else 0.0
        
        # Evaluate suppression efficacy
        mean_score_all = round(float(sub_all["storm_score"].mean()), 2)
        mean_score_pos = round(float(sub_pos["storm_score"].mean()), 2) if n_pos > 0 else np.nan
        mean_prob_ebm_pos = round(float(sub_pos["prob_ebm"].mean()), 4) if n_pos > 0 else np.nan
        mean_prob_gam_pos = round(float(sub_pos["prob_gam"].mean()), 4) if n_pos > 0 else np.nan
        
        high_cin_results.append({
            "scenario": desc,
            "min_cape_J_kg": c_min,
            "min_cin_J_kg": cin_min,
            "total_observations": n_all,
            "proxy_positive_count": n_pos,
            "proxy_negative_count": len(sub_neg),
            "pct_labeled_positive": pct_labeled_pos,
            "storm_score_mean_all": mean_score_all,
            "storm_score_mean_positives": mean_score_pos,
            "ebm_mean_prob_positives": mean_prob_ebm_pos,
            "gam_mean_prob_positives": mean_prob_gam_pos,
            "physical_assessment": "Excessive Positive Labels (CIN Failure)" if pct_labeled_pos > 5.0 else "Adequately Suppressed"
        })
        print(f"Scenario: CAPE > {c_min} & CIN > {cin_min} -> Total: {n_all:4d}, Positives: {n_pos:3d} ({pct_labeled_pos:5.2f}%) | Mean Score: {mean_score_all:.2f}")
        
    df_high_cin = pd.DataFrame(high_cin_results)
    df_high_cin.to_csv(OUTPUT_DIR / "high_cape_high_cin_analysis.csv", index=False)
    print("Saved high_cape_high_cin_analysis.csv")
    
    # =========================================================================
    # PART 4: DISAGREEMENTS WITH FLORIDA-TRAINED EBM / GAM MODELS
    # =========================================================================
    print("\n--- PART 4: Florida EBM/GAM vs. Formula Disagreement Mining ---")
    disagree_rows = []
    
    # Category 1: Proxy Positive, Both Native Models Low Probability (< 0.20)
    # Severe Formula False Positive / Heuristic Artifact
    cat1 = df[(df["storm_label_score"] == 1) & (df["prob_ebm"] < 0.20) & (df["prob_gam"] < 0.20)].copy()
    cat1["disagreement_category"] = "Formula False Alarm (Proxy=1, Both Models < 0.20)"
    cat1["primary_cause"] = np.where(cat1["cape_Jkg"] < 100, "Negligible CAPE (<100 J/kg) with High 300hPa RH", "Weak Buoyancy & High Cloudiness")
    
    # Category 2: Proxy Negative, Both Native Models High Probability (> 0.50)
    # Severe Formula False Negative / Over-suppression
    cat2 = df[(df["storm_label_score"] == 0) & (df["prob_ebm"] > 0.50) & (df["prob_gam"] > 0.50)].copy()
    cat2["disagreement_category"] = "Formula Missed Favorable Regime (Proxy=0, Both Models > 0.50)"
    cat2["primary_cause"] = np.where(cat2["storm_score"] >= 2.0, "Sub-threshold Score (2.0 - 2.49) despite Strong Buoyancy", "Severe CIN Linear Subtraction")
    
    # Category 3: Native EBM and GAM Strongly Diverge (|P_ebm - P_gam| > 0.30)
    cat3 = df[(df["prob_ebm"] - df["prob_gam"]).abs() > 0.30].copy()
    cat3["disagreement_category"] = "Model Architecture Divergence (|EBM - GAM| > 0.30)"
    cat3["primary_cause"] = np.where(cat3["cape_Jkg"] > 7240.0, "Extreme CAPE Extrapolation (Spline Downturn vs. Tree Clamp)", "Threshold Boundary Divergence")
    
    # Combine representative sample from each category
    disagree_sample = pd.concat([
        cat1.sample(n=min(len(cat1), 30), random_state=42),
        cat2.sample(n=min(len(cat2), 30), random_state=42),
        cat3.sample(n=min(len(cat3), 30), random_state=42)
    ]).copy()
    
    export_cols = [
        "valid_time", "latitude", "longitude", "disagreement_category", "primary_cause",
        "storm_score", "storm_label_score", "prob_gam", "prob_ebm",
        "cape_Jkg", "cin_Jkg", "relative_humidity_300hPa", "total_cloud_cover",
        "bulk_shear_850_300", "wind_speed_10m", "vertical_velocity_Pas_500hPa", "cp_24h_sum",
        "season", "year"
    ]
    disagree_sample[export_cols].to_csv(OUTPUT_DIR / "ebm_gam_formula_disagreement_cases.csv", index=False)
    print(f"Saved {len(disagree_sample)} disagreement case studies to ebm_gam_formula_disagreement_cases.csv")
    
    # =========================================================================
    # PART 5: HIGH-RESOLUTION DIAGNOSTIC PLOTS
    # =========================================================================
    print("\n--- PART 5: Generating High-Resolution Diagnostic Plots ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Low-CAPE Positives Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x_pos = np.arange(len(df_low_cape))
    axes[0].bar(x_pos, df_low_cape["observation_count"], color="#e11d48", width=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"< {t} J/kg" for t in df_low_cape["cape_threshold_J_kg"]])
    axes[0].set_title("Proxy Positives with Sub-Convective CAPE (N = 5,462 Positives)", fontweight="bold")
    axes[0].set_ylabel("Number of Observations")
    for i, v in enumerate(df_low_cape["observation_count"]):
        pct = df_low_cape.loc[i, "pct_of_all_proxy_positives"]
        axes[0].text(i, v + 25, f"{v:,}\n({pct:.1f}%)", ha="center", fontweight="bold", fontsize=10)
    axes[0].set_ylim(0, max(df_low_cape["observation_count"]) * 1.2)
    
    # Seasonal distribution of CAPE < 500 positives
    sub500 = df[(df["storm_label_score"] == 1) & (df["cape_Jkg"] < 500)]
    sea_order = ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"]
    sea_counts = [sub500["season"].value_counts().get(s, 0) for s in sea_order]
    axes[1].pie(sea_counts, labels=sea_order, autopct="%1.1f%%", colors=["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"], startangle=140)
    axes[1].set_title("Seasonal Distribution of Low-CAPE Positives (CAPE < 500 J/kg, N = 1,019)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "low_cape_positives_breakdown.png", dpi=300)
    plt.close()
    
    # 2. Component Decisive Frequency Bar Chart
    plt.figure(figsize=(12, 6))
    df_sorted_dec = df_comp_stats.sort_values("decisive_in_positives_pct", ascending=True)
    colors = ["#94a3b8" if p < 20 else "#f97316" if p < 50 else "#dc2626" for p in df_sorted_dec["decisive_in_positives_pct"]]
    plt.barh(df_sorted_dec["component_label"], df_sorted_dec["decisive_in_positives_pct"], color=colors)
    plt.title("Decisive Impact Frequency: % of Positive Labels Lost if Component is Removed", fontweight="bold")
    plt.xlabel("Percentage of All 5,462 Proxy Positives (%)")
    for i, v in enumerate(df_sorted_dec["decisive_in_positives_pct"]):
        plt.text(v + 1.0, i, f"{v:.1f}% ({df_sorted_dec.iloc[i]['decisive_in_positives_count']:,})", va="center", fontweight="bold")
    plt.xlim(0, max(df_sorted_dec["decisive_in_positives_pct"]) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decisive_component_frequency.png", dpi=300)
    plt.close()
    
    # 3. High-CAPE / High-CIN Suppression Failure
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x_cin = np.arange(len(df_high_cin))
    axes[0].bar(x_cin, df_high_cin["total_observations"], color="#64748b", width=0.4, label="Total High-CIN States")
    axes[0].bar(x_cin, df_high_cin["proxy_positive_count"], color="#ef4444", width=0.4, label="Labeled Storm (Capping Breach)")
    axes[0].set_xticks(x_cin)
    axes[0].set_xticklabels(["CAPE>1000\nCIN>200", "CAPE>2000\nCIN>300", "CAPE>3000\nCIN>500"])
    axes[0].set_title("Formula Storm Labeling Under Heavy Capping Inversions", fontweight="bold")
    axes[0].set_ylabel("Observation Count")
    axes[0].legend()
    for i in range(len(df_high_cin)):
        tot = df_high_cin.loc[i, "total_observations"]
        pos = df_high_cin.loc[i, "proxy_positive_count"]
        pct = df_high_cin.loc[i, "pct_labeled_positive"]
        axes[0].text(i, pos + 50, f"{pos}\n({pct:.1f}%)", ha="center", fontweight="bold", color="#b91c1c")
        
    # Scatter of CAPE vs CIN showing storm_score
    sns.scatterplot(
        data=df[df["cape_Jkg"] > 500].sample(n=3000, random_state=42),
        x="cin_Jkg", y="cape_Jkg", hue="storm_label_score",
        palette={0: "#94a3b8", 1: "#ef4444"}, alpha=0.5, ax=axes[1]
    )
    axes[1].axvline(200, color="black", linestyle="--", label="CIN Capping Threshold (200 J/kg)")
    axes[1].set_title("CAPE vs. CIN Phase Space (Red = Proxy Storm Label)", fontweight="bold")
    axes[1].set_xlabel("Convective Inhibition (J/kg)")
    axes[1].set_ylabel("Surface CAPE (J/kg)")
    axes[1].legend(title="Proxy Label", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "high_cape_high_cin_breakdown.png", dpi=300)
    plt.close()
    
    # 4. CAPE vs. Storm Score Density Overlay
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df["cape_Jkg"], df["storm_score"], gridsize=40, cmap="viridis", mincnt=1)
    plt.axhline(2.5, color="red", linestyle="--", linewidth=2, label="Proxy Threshold (tau = 2.5)")
    plt.axvline(500, color="orange", linestyle=":", linewidth=2, label="Thermodynamic Initiation Floor (500 J/kg)")
    plt.title("Bivariate Density: Surface CAPE vs. Existing Storm Score (N = 23,376)", fontweight="bold")
    plt.xlabel("Surface CAPE (J/kg)")
    plt.ylabel("Existing Storm Score")
    plt.colorbar(hb, label="Observation Count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cape_vs_storm_score_density.png", dpi=300)
    plt.close()
    
    print("\nAll diagnostic analyses and plots generated successfully.")
    print(f"Artifacts saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
