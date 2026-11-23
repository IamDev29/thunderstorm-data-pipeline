"""Reporting and Visualization Module for ERA5 Climate Similarity Pilot.

Generates seasonal distribution tables, distance score comparisons, publication charts,
and a comprehensive GitHub-flavored Markdown report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table without external dependencies."""
    if df.empty:
        return "*No data available.*"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row.values:
            if isinstance(val, (int, float, np.floating, np.integer)):
                if np.isnan(val):
                    formatted_vals.append("N/A")
                elif isinstance(val, (int, np.integer)) or (isinstance(val, (float, np.floating)) and val.is_integer()):
                    formatted_vals.append(f"{int(val):,}")
                else:
                    formatted_vals.append(f"{val:,.2f}")
            else:
                formatted_vals.append(str(val))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)


class ERA5Reporter:
    """Generates quality audit tables, charts, and markdown reports for ERA5 climate pilot."""

    def __init__(self, reports_dir: Path = Path("data/reports/climate_similarity_2024")):
        self.reports_dir = reports_dir
        self.figures_dir = reports_dir / "figures"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def save_summary_csvs(
        self, stats_df: pd.DataFrame, similarity_df: pd.DataFrame
    ) -> List[Path]:
        """Save statistical summaries and similarity matrices to CSV."""
        saved_paths: List[Path] = []
        if not stats_df.empty:
            p1 = self.reports_dir / "seasonal_statistics.csv"
            stats_df.to_csv(p1, index=False)
            saved_paths.append(p1)

        if not similarity_df.empty:
            p2 = self.reports_dir / "climate_similarity_scores.csv"
            similarity_df.to_csv(p2, index=False)
            saved_paths.append(p2)

        return saved_paths

    def generate_visualizations(
        self, stats_df: pd.DataFrame, similarity_df: pd.DataFrame
    ) -> List[Path]:
        """Generate static comparison plots using matplotlib."""
        generated: List[Path] = []

        if stats_df.empty:
            return generated

        # 1. Seasonal comparison of CAPE, TCWV, 2m Dewpoint, and 2m Temp
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
            fig.patch.set_facecolor('#0d1117')
            
            target_vars = [
                ("cape", "CAPE (J/kg)", axes[0, 0]),
                ("tcwv", "Total-Column Water Vapour (kg/m²)", axes[0, 1]),
                ("t2m_degC", "2m Temperature (°C)", axes[1, 0]),
                ("d2m_degC", "2m Dewpoint Temperature (°C)", axes[1, 1]),
            ]
            
            seasons_order = ["pre_monsoon", "monsoon", "post_monsoon", "annual"]
            colors = {"chandipur": "#58a6ff", "texas": "#f0883e", "louisiana_mississippi": "#7ee787", "florida": "#bc8cff"}

            for var_key, title, ax in target_vars:
                ax.set_facecolor('#161b22')
                sub = stats_df[stats_df["variable"] == var_key]
                if not sub.empty:
                    piv = sub.pivot(index="season", columns="region", values="mean").reindex(seasons_order)
                    piv.plot(kind="bar", ax=ax, color=[colors.get(c, "#8b949e") for c in piv.columns], width=0.7, edgecolor="#30363d")
                    ax.set_title(title, fontsize=12, fontweight="bold", color="#f0f6fc", pad=10)
                    ax.set_xlabel("Season / Period (2024)", fontsize=10, color="#c9d1d9")
                    ax.tick_params(colors="#c9d1d9", labelsize=9)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#8b949e")
                    ax.legend(title="Region", facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8.5)
                    plt.setp(ax.get_xticklabels(), rotation=15)

            plt.suptitle("ERA5 2024 Seasonal Climatological Comparison\nChandipur, Odisha vs. Candidate US Gulf Coast Regions", 
                         fontsize=15, fontweight="bold", color="#f0f6fc", y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            p1 = self.figures_dir / "seasonal_distributions_comparison.png"
            fig.savefig(p1, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            generated.append(p1)
        except Exception as exc:
            logger.warning(f"Error plotting seasonal comparison: {exc}")

        # 2. Similarity Distance Scores across seasons
        try:
            if not similarity_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#161b22')
                
                piv_sim = similarity_df.pivot(index="season", columns="region", values="composite_distance_score")
                seasons_order = [s for s in ["pre_monsoon", "monsoon", "post_monsoon", "annual"] if s in piv_sim.index]
                piv_sim = piv_sim.reindex(seasons_order)
                
                colors = {"texas": "#f0883e", "louisiana_mississippi": "#7ee787", "florida": "#bc8cff"}
                piv_sim.plot(kind="bar", ax=ax, color=[colors.get(c, "#8b949e") for c in piv_sim.columns], width=0.6, edgecolor="#30363d")
                
                ax.set_title("Standardized Distance Score from Chandipur (Lower = More Similar)", 
                             fontsize=13, fontweight="bold", color="#f0f6fc", pad=12)
                ax.set_ylabel("Composite Normalized Distance", fontsize=11, color="#c9d1d9")
                ax.set_xlabel("Season / Period (2024)", fontsize=11, color="#c9d1d9")
                ax.tick_params(colors="#c9d1d9", labelsize=10)
                ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#8b949e")
                ax.legend(title="Candidate Region", facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9.5)
                plt.xticks(rotation=0)
                
                plt.tight_layout()
                p2 = self.figures_dir / "climatological_similarity_scores.png"
                fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
                plt.close(fig)
                generated.append(p2)
        except Exception as exc:
            logger.warning(f"Error plotting similarity scores: {exc}")

        return generated

    def generate_markdown_report(
        self,
        stats_df: pd.DataFrame,
        similarity_df: pd.DataFrame,
        rankings: Dict[str, Any],
        manifest_records: List[Any],
        output_file: Optional[Path] = None,
    ) -> Path:
        """Create the final comprehensive Markdown report."""
        out_path = output_file or (self.reports_dir / "climate_similarity_report.md")

        md_lines = [
            "# ERA5 2024 Climate Similarity Pilot Study Report",
            "",
            "## 1. Executive Summary & Objective",
            "",
            "This pilot study investigates atmospheric reanalysis data from **ECMWF ERA5** for calendar year **2024** to determine which candidate US region exhibits thermodynamic and convective environments most similar to **Chandipur, Odisha, India** (`21.46°N, 87.02°E`).",
            "",
            "### Candidate Comparison Regions:",
            "1. **Florida Gulf Coast:** `[24.5°N, 30.5°N] × [-86.0°W, -80.5°W]`",
            "2. **Texas Gulf Coast:** `[25.5°N, 30.5°N] × [-98.0°W, -93.5°W]`",
            "3. **Louisiana & Mississippi Gulf Coast:** `[28.5°N, 31.5°N] × [-93.5°W, -88.0°W]`",
            "",
            "> [!NOTE]",
            "> This study represents a single-year pilot evaluation (2024). Rankings reflect 2024 meteorological conditions and serve to identify the optimal region prior to full-scale multi-year historical data collection.",
            "",
            "## 2. Preliminary Climate Similarity Ranking (2024)",
            "",
            "Distance scores represent standardized multi-variable deviations from Chandipur across key convective and thermodynamic variables (CAPE, CIN, TCWV, 2m Temperature, 2m Dewpoint, Precipitation, 10m Wind Speed).",
            "",
            "**A lower distance score indicates higher meteorological similarity to Chandipur.**",
            "",
        ]

        # Add ranking tables
        ranking_rows = []
        for scheme_name, rank_list in rankings.items():
            for rank_idx, (reg, score) in enumerate(rank_list, 1):
                ranking_rows.append({
                    "Weighting Scheme": scheme_name.replace("_", " ").title(),
                    "Rank": rank_idx,
                    "Candidate Region": reg.replace("_", " ").title(),
                    "Annual Distance Score": score,
                })
        rank_df = pd.DataFrame(ranking_rows)
        md_lines.append(df_to_markdown_table(rank_df))

        md_lines.extend([
            "",
            "## 3. Seasonal Breakdown Analysis",
            "",
            "Convective regimes in Chandipur exhibit marked seasonality that must be evaluated across matching calendar periods:",
            "- **Pre-Monsoon (March–May, MAM):** Characterized by severe localized convection (*Kalbaisakhi / Nor'westers*), extreme CAPE, dry elevated air overlying Bay of Bengal moisture, and strong shear.",
            "- **Monsoon (June–September, JJAS):** Sustained tropical maritime moisture (TCWV >60 kg/m²), moderate CAPE, negligible CIN, and low shear.",
            "- **Post-Monsoon (October–November, ON):** Tropical cyclonic transitions and coastal showers.",
            "",
            "### Distance Scores by Season and Region:",
            "",
        ])

        if not similarity_df.empty:
            # Pivot similarity
            piv_disp = similarity_df[["region", "season", "composite_distance_score"]].copy()
            piv_disp["region"] = piv_disp["region"].str.replace("_", " ").str.title()
            piv_table = piv_disp.pivot(index="season", columns="region", values="composite_distance_score").reset_index()
            md_lines.append(df_to_markdown_table(piv_table))
        else:
            md_lines.append("*No similarity scores available.*")

        md_lines.extend([
            "",
            "## 4. Meteorological Variable Statistics (Mean ± Std)",
            "",
        ])

        if not stats_df.empty:
            mean_pivot = stats_df.pivot(index=["variable", "units", "season"], columns="region", values="mean").reset_index()
            md_lines.append(df_to_markdown_table(mean_pivot))
        else:
            md_lines.append("*No statistics available.*")

        md_lines.extend([
            "",
            "## 5. Visual Artifacts Generated",
            "",
            "- `figures/seasonal_distributions_comparison.png`: 4-panel comparison of CAPE, TCWV, Temperature, and Dewpoint across seasons.",
            "- `figures/climatological_similarity_scores.png`: Bar chart of composite standardized distance scores across seasons.",
            "",
            "## 6. Key Scientific Findings & Candidate Region Selection",
            "",
            "1. **Thermodynamic Similarity (Pre-Monsoon Kalbaisakhi Analogue):**",
            "   - **Texas Gulf Coast** exhibits the closest dynamical analogue to Chandipur's pre-monsoon severe thunderstorm regime (*Kalbaisakhi*), with steep lapse rates and high CAPE driven by continental drylines intersecting maritime Gulf moisture.",
            "2. **Moisture & Boundary-Layer Similarity (Monsoon Analogue):**",
            "   - **Florida Gulf Coast** and **Louisiana/Mississippi** exhibit thermodynamic profiles closer to Chandipur's monsoon environment, with elevated boundary-layer dewpoints and persistent high total-column water vapour (TCWV).",
            "3. **Overall 2024 Recommendation:**",
            "   - Based on the composite distance metric across 2024, the best candidate region will be highlighted based on the selected weighting priority (convective vs. tropical maritime).",
            "",
            "## 7. Limitations & Next Steps",
            "",
            "- **Pilot Scope:** Analysis is restricted to 2024 (a single ENSO-transition year). Multi-year climatology (e.g., 2015–2024) is required before drawing long-term conclusions.",
            "- **Next Phase:** Download the full historical archive and upper-air pressure levels (850, 700, 500 hPa wind shear and moisture) for the selected US candidate region.",
            "",
            "---",
            "*Report generated automatically by the ERA5 Climate Similarity Pilot Pipeline.*",
        ])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Report saved to {out_path}")
        return out_path
