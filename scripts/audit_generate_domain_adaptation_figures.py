"""
audit_generate_domain_adaptation_figures.py
===========================================
Generates diagnostic figures for Step 01: Domain Adaptation Audit.
Outputs:
1. source_vs_target_feature_distributions.png
2. feature_correlation_heatmaps.png
3. cape_cin_distribution_comparison.png
4. seasonal_feature_distribution_comparison.png

Saved in reports/domain_adaptation/figures/step_01/
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(".").resolve()
FIGURES_DIR = ROOT_DIR / "reports" / "domain_adaptation" / "figures" / "step_01"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FL_PATH = ROOT_DIR / "data" / "processed" / "asos_parser_fixed" / "florida_asos_modeling_ready_fixed_2024.parquet"
CH_PATH = ROOT_DIR / "MET_DATA_with_storm_score_label.csv"

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
    print("Loading datasets...")
    df_fl = pd.read_parquet(FL_PATH)
    df_ch = pd.read_csv(CH_PATH)
    
    target_fl = "target_asos_strict_crit_c"
    df_fl_valid = df_fl[df_fl[target_fl].notna()].copy()
    df_fl_valid["target"] = df_fl_valid[target_fl].astype(int)
    
    df_fl_valid["valid_time_dt"] = pd.to_datetime(df_fl_valid["forecast_init_time_utc"], utc=True)
    df_fl_valid["season"] = df_fl_valid["valid_time_dt"].apply(assign_season)
    
    df_ch["valid_time_dt"] = pd.to_datetime(df_ch["valid_time"], format="mixed")
    df_ch["season"] = df_ch["valid_time_dt"].apply(assign_season)
    
    fl_features = ["cape_J_kg", "cin_J_kg", "bulk_shear_850_300_ms", "cloud_cover_fraction", "wind_speed_10m_ms", "wind_speed_850hPa_ms"]
    ch_features = ["cape_Jkg", "cin_Jkg", "bulk_shear_850_300", "total_cloud_cover", "wind_speed_10m", "wind_speed_850hPa"]
    
    feature_labels = {
        "cape_J_kg": "Surface CAPE (J/kg)",
        "cin_J_kg": "Convective Inhibition CIN (J/kg)",
        "bulk_shear_850_300_ms": "Bulk Shear 850-300 hPa (m/s)",
        "cloud_cover_fraction": "Total Cloud Cover (0-1)",
        "wind_speed_10m_ms": "10 m Wind Speed (m/s)",
        "wind_speed_850hPa_ms": "850 hPa Wind Speed (m/s)"
    }
    
    # -------------------------------------------------------------
    # Figure 1: Source vs Target Feature Distributions
    # -------------------------------------------------------------
    print("Generating Figure 1: source_vs_target_feature_distributions.png...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    axes = axes.flatten()
    
    for idx, (f_fl, f_ch) in enumerate(zip(fl_features, ch_features)):
        ax = axes[idx]
        s_fl = df_fl_valid[f_fl].dropna()
        s_ch = df_ch[f_ch].dropna()
        
        # Plot histograms/KDE
        sns.kdeplot(s_fl, ax=ax, label="Source: Florida 2024", color="#1f77b4", linewidth=2.2, fill=True, alpha=0.25)
        sns.kdeplot(s_ch, ax=ax, label="Target: Chandipur 2022-25", color="#d62728", linewidth=2.2, fill=True, alpha=0.25)
        
        # Annotate medians
        ax.axvline(s_fl.median(), color="#1f77b4", linestyle="--", alpha=0.7, label=f"FL Median: {s_fl.median():.1f}")
        ax.axvline(s_ch.median(), color="#d62728", linestyle="--", alpha=0.7, label=f"CH Median: {s_ch.median():.1f}")
        
        ax.set_title(feature_labels[f_fl], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Empirical Density", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if "cape" in f_fl:
            ax.set_xlim(0, 8000)
        elif "cin" in f_fl:
            ax.set_xlim(0, 800)
            
    plt.suptitle("Domain Adaptation Audit: Source (Florida) vs. Target (Chandipur) Feature Marginal Distributions", 
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "source_vs_target_feature_distributions.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Feature Correlation Heatmaps & Difference
    # -------------------------------------------------------------
    print("Generating Figure 2: feature_correlation_heatmaps.png...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)
    
    corr_fl = df_fl_valid[fl_features].corr()
    corr_ch = df_ch[ch_features].corr()
    corr_ch.index = fl_features
    corr_ch.columns = fl_features
    corr_diff = corr_ch - corr_fl
    
    short_labels = ["CAPE", "CIN", "Shear", "Cloud", "Wind10m", "Wind850"]
    
    sns.heatmap(corr_fl, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0],
                xticklabels=short_labels, yticklabels=short_labels, cbar_kws={'label': 'Pearson r'})
    axes[0].set_title("(a) Source Domain: Florida 2024 Correlation Matrix", fontsize=12, fontweight="bold")
    
    sns.heatmap(corr_ch, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[1],
                xticklabels=short_labels, yticklabels=short_labels, cbar_kws={'label': 'Pearson r'})
    axes[1].set_title("(b) Target Domain: Chandipur 2022-2025 Correlation Matrix", fontsize=12, fontweight="bold")
    
    sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=0.6, ax=axes[2],
                xticklabels=short_labels, yticklabels=short_labels, cbar_kws={'label': 'Delta r (CH - FL)'})
    axes[2].set_title("(c) Covariate Shift: Correlation Difference (CH - FL)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "feature_correlation_heatmaps.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: CAPE and CIN Joint Distribution Comparison
    # -------------------------------------------------------------
    print("Generating Figure 3: cape_cin_distribution_comparison.png...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)
    
    # Florida CAPE vs CIN
    ax = axes[0]
    fl_sub = df_fl_valid.sample(n=min(len(df_fl_valid), 10000), random_state=42)
    sns.scatterplot(data=fl_sub, x="cape_J_kg", y="cin_J_kg", hue="target", alpha=0.35, s=20,
                    palette={0: "#1f77b4", 1: "#ff7f0e"}, ax=ax)
    ax.axvline(7240.0, color="red", linestyle="--", linewidth=1.5, label="FL Max CAPE (7,240 J/kg)")
    ax.axhline(500.0, color="purple", linestyle=":", linewidth=1.5, label="Heavy Cap Lid (500 J/kg)")
    ax.set_title("Source Domain: Florida 2024 (ASOS Observed Target)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    ax.set_ylabel("Convective Inhibition CIN (J/kg)", fontsize=11)
    ax.set_xlim(-100, 8000)
    ax.set_ylim(-20, 1050)
    ax.legend(title="Target ASOS", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Chandipur CAPE vs CIN
    ax = axes[1]
    ch_sub = df_ch.sample(n=min(len(df_ch), 10000), random_state=42)
    sns.scatterplot(data=ch_sub, x="cape_Jkg", y="cin_Jkg", hue="storm_label_score", alpha=0.35, s=20,
                    palette={0: "#2ca02c", 1: "#d62728"}, ax=ax)
    ax.axvline(7240.0, color="red", linestyle="--", linewidth=1.5, label="FL Max Extrapolation Boundary")
    ax.axhline(500.0, color="purple", linestyle=":", linewidth=1.5, label="Heavy Cap Lid (500 J/kg)")
    ax.set_title("Target Domain: Chandipur 2022-2025 (Proxy Label)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Surface CAPE (J/kg)", fontsize=11)
    ax.set_ylabel("Convective Inhibition CIN (J/kg)", fontsize=11)
    ax.set_xlim(-100, 10500)
    ax.set_ylim(-20, 1050)
    ax.legend(title="Storm Proxy", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Thermodynamic Domain Shift: Joint CAPE-CIN Covariate Structure and Extrapolation Frontier",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "cape_cin_distribution_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: Seasonal Feature Distribution Comparisons
    # -------------------------------------------------------------
    print("Generating Figure 4: seasonal_feature_distribution_comparison.png...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    seasons_order = ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"]
    
    # Prepare combined data for boxplots
    df_fl_comb = pd.DataFrame({
        "CAPE": df_fl_valid["cape_J_kg"],
        "CIN": df_fl_valid["cin_J_kg"],
        "Shear": df_fl_valid["bulk_shear_850_300_ms"],
        "Wind10m": df_fl_valid["wind_speed_10m_ms"],
        "Season": df_fl_valid["season"],
        "Domain": "Florida (Source)"
    })
    
    df_ch_comb = pd.DataFrame({
        "CAPE": df_ch["cape_Jkg"],
        "CIN": df_ch["cin_Jkg"],
        "Shear": df_ch["bulk_shear_850_300"],
        "Wind10m": df_ch["wind_speed_10m"],
        "Season": df_ch["season"],
        "Domain": "Chandipur (Target)"
    })
    
    df_plot = pd.concat([df_fl_comb, df_ch_comb], ignore_index=True)
    df_plot = df_plot[df_plot["Season"].isin(seasons_order)]
    
    metrics_to_plot = [
        ("CAPE", "Surface CAPE (J/kg)", (0, 6000)),
        ("CIN", "Convective Inhibition CIN (J/kg)", (0, 600)),
        ("Shear", "Bulk Shear 850-300 hPa (m/s)", (0, 60)),
        ("Wind10m", "10 m Wind Speed (m/s)", (0, 15))
    ]
    
    for idx, (m_col, m_label, y_lim) in enumerate(metrics_to_plot):
        ax = axes[idx]
        sns.boxplot(data=df_plot, x="Season", y=m_col, hue="Domain", order=seasons_order,
                    palette={"Florida (Source)": "#4C72B0", "Chandipur (Target)": "#C44E52"},
                    showfliers=False, ax=ax)
        ax.set_title(f"Seasonal Comparison: {m_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(m_label, fontsize=10)
        ax.set_ylim(y_lim)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle("Seasonal Domain Divergence: Synoptic Regimes across Four Annual Quarters",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "seasonal_feature_distribution_comparison.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")
    
    print("\nAll 4 audit figures generated successfully!")

if __name__ == "__main__":
    main()
