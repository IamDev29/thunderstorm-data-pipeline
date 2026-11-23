"""Reporting and Visualization Module.

Generates comprehensive data quality reports, summary tables, and visualizations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis import NOAAAnalyzer

logger = logging.getLogger(__name__)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without requiring tabulate."""
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
            if isinstance(val, (int, float)):
                if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                    formatted_vals.append(f"{int(val):,}")
                else:
                    formatted_vals.append(f"{val:,.2f}")
            else:
                formatted_vals.append(str(val))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)


class NOAAReporter:
    """Generates quality reports, CSV summaries, and visualization figures."""

    def __init__(self, reports_dir: Path, figures_dir: Optional[Path] = None):
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir) if figures_dir else self.reports_dir / "figures"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        quality_summary: Dict[str, Any],
        analyzer: NOAAAnalyzer,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Create a detailed GitHub-flavored Markdown data quality and pipeline report."""
        out_path = output_file or (self.reports_dir / "data_quality_report.md")

        by_year_df = analyzer.get_event_counts_by_year()
        by_state_df = analyzer.get_event_counts_by_state()
        by_type_df = analyzer.get_event_counts_by_type()
        by_month_df = analyzer.get_event_counts_by_month()

        geo_excl = quality_summary.get("geographic_exclusions", {})

        md_lines = [
            "# NOAA Storm Events Pipeline - Data Quality & Preprocessing Report",
            "",
            "## 1. Executive Summary",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Archive Files Examined** | {quality_summary.get('total_downloaded_files', 0):,} |",
            f"| **Successfully Downloaded / Verified** | {quality_summary.get('successfully_processed_files', 0):,} |",
            f"| **Failed Downloads** | {quality_summary.get('failed_downloads', 0):,} |",
            f"| **Total Raw Event Records Ingested** | {quality_summary.get('total_raw_records', 0):,} |",
            f"| **Total Cleaned & Standardized Records** | {quality_summary.get('total_cleaned_records', 0):,} |",
            f"| **Unique Event IDs** | {quality_summary.get('unique_event_ids', 0):,} |",
            f"| **Duplicate Records Detected / Handled** | {quality_summary.get('duplicate_records_detected', 0):,} |",
            "",
            "## 2. Coordinate & Temporal Data Quality",
            "",
            "| Coordinate / Temporal Metric | Count | Description |",
            "| :--- | :--- | :--- |",
            f"| **Missing Coordinates** | {quality_summary.get('records_with_missing_coordinates', 0):,} | Events lacking latitude/longitude values |",
            f"| **Invalid Coordinates** | {quality_summary.get('records_with_invalid_coordinates', 0):,} | Out of bounds [-90,90]x[-180,180] or Null Island (0,0) |",
            f"| **Retained Missing-Coord Records** | {geo_excl.get('retained_missing_coords', 0):,} | Preserved in state datasets with `insufficient_coords=True` |",
            f"| **Missing / Unparseable Timestamps** | {quality_summary.get('records_with_missing_timestamps', 0):,} | Events lacking valid begin timestamps |",
            f"| **Records Excluded by Bounding Box** | {geo_excl.get('excluded_outside_bounding_box', 0):,} | Excluded from coastal bounding box |",
            f"| **Records Excluded by Event Type** | {geo_excl.get('excluded_unmatched_event_type', 0):,} | Excluded from convective hazard subset |",
            "",
            "## 3. Climatological Comparison Reference Note",
            "",
            "- **Candidate US Regions:** Florida, Texas, Louisiana, Mississippi (Gulf Coast focus)",
            "- **Comparative Site:** Chandipur, Odisha, India",
            "- **Approximate Reference Coordinates:** Latitude 21.46° N, Longitude 87.02° E",
            "- *Note:* Chandipur coordinates are regional reference coordinates, not a single verified meteorological station.",
            "",
            "## 4. Event Distributions",
            "",
            "### 4.1 Event Counts by Year",
            "",
            df_to_markdown_table(by_year_df),
            "",
            "### 4.2 Event Counts by State",
            "",
            df_to_markdown_table(by_state_df),
            "",
            "### 4.3 Event Counts by Event Type",
            "",
            df_to_markdown_table(by_type_df),
            "",
            "### 4.4 Event Counts by Month",
            "",
            df_to_markdown_table(by_month_df),
            "",
            "## 5. Artifacts and Visualizations Generated",
            "",
            "- `figures/event_counts_by_year_state.png`",
            "- `figures/event_types_distribution.png`",
            "- `figures/monthly_seasonality.png`",
            "- `figures/spatial_distribution_map.png`",
            "",
            "---",
            "*Report generated automatically by the NOAA Storm Events Preprocessing Pipeline.*",
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown data quality report written to {out_path}")
        return out_path

    def save_json_summary(
        self, quality_summary: Dict[str, Any], output_file: Optional[Path] = None
    ) -> Path:
        """Save machine-readable summary metrics as JSON."""
        out_path = output_file or (self.reports_dir / "data_quality_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(quality_summary, f, indent=2)
        logger.info(f"JSON quality report written to {out_path}")
        return out_path

    def save_summary_csvs(self, analyzer: NOAAAnalyzer) -> List[Path]:
        """Save statistical summary tables as CSV files."""
        saved_paths: List[Path] = []

        by_year = analyzer.get_event_counts_by_year()
        if not by_year.empty:
            p = self.reports_dir / "event_counts_by_year.csv"
            by_year.to_csv(p, index=False)
            saved_paths.append(p)

        by_state = analyzer.get_event_counts_by_state()
        if not by_state.empty:
            p = self.reports_dir / "event_counts_by_state.csv"
            by_state.to_csv(p, index=False)
            saved_paths.append(p)

        by_type = analyzer.get_event_counts_by_type()
        if not by_type.empty:
            p = self.reports_dir / "event_counts_by_type.csv"
            by_type.to_csv(p, index=False)
            saved_paths.append(p)

        by_month = analyzer.get_event_counts_by_month()
        if not by_month.empty:
            p = self.reports_dir / "event_counts_by_month.csv"
            by_month.to_csv(p, index=False)
            saved_paths.append(p)

        cross_tab = analyzer.get_year_state_cross_tab()
        if not cross_tab.empty:
            p = self.reports_dir / "year_state_cross_tab.csv"
            cross_tab.to_csv(p)
            saved_paths.append(p)

        return saved_paths

    def generate_visualizations(
        self, df: pd.DataFrame, analyzer: NOAAAnalyzer
    ) -> List[Path]:
        """Generate static visual charts and maps using matplotlib."""
        generated_figures: List[Path] = []

        # 1. Event Counts by Year and State (Stacked Bar or Line)
        try:
            ct = analyzer.get_year_state_cross_tab()
            if not ct.empty and len(ct.columns) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                ax.set_title("NOAA Storm Events by Year and State", fontsize=14, fontweight="bold")
                ax.set_xlabel("Year", fontsize=12)
                ax.set_ylabel("Number of Events", fontsize=12)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                p1 = self.figures_dir / "event_counts_by_year_state.png"
                fig.savefig(p1, dpi=200)
                plt.close(fig)
                generated_figures.append(p1)
        except Exception as exc:
            logger.warning(f"Error plotting event counts by year/state: {exc}")

        # 2. Event Types Distribution (Horizontal Bar)
        try:
            by_type = analyzer.get_event_counts_by_type()
            if not by_type.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = range(len(by_type))
                ax.barh(y_pos, by_type["event_count"], color="#1f77b4", edgecolor="black", alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(by_type["event_type"])
                ax.invert_yaxis()  # Top down
                ax.set_xlabel("Total Event Count", fontsize=12)
                ax.set_title("NOAA Storm Events Distribution by Event Type", fontsize=14, fontweight="bold")
                ax.grid(axis="x", linestyle="--", alpha=0.7)
                plt.tight_layout()
                p2 = self.figures_dir / "event_types_distribution.png"
                fig.savefig(p2, dpi=200)
                plt.close(fig)
                generated_figures.append(p2)
        except Exception as exc:
            logger.warning(f"Error plotting event types distribution: {exc}")

        # 3. Monthly Seasonality (Bar Chart)
        try:
            by_month = analyzer.get_event_counts_by_month()
            if not by_month.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(by_month["month_name"], by_month["event_count"], color="#ff7f0e", edgecolor="black", alpha=0.85)
                ax.set_title("Monthly Storm Event Seasonality", fontsize=14, fontweight="bold")
                ax.set_xlabel("Month", fontsize=12)
                ax.set_ylabel("Total Event Count", fontsize=12)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                p3 = self.figures_dir / "monthly_seasonality.png"
                fig.savefig(p3, dpi=200)
                plt.close(fig)
                generated_figures.append(p3)
        except Exception as exc:
            logger.warning(f"Error plotting monthly seasonality: {exc}")

        # 4. Spatial Coordinates Map (Scatter Plot)
        try:
            if "begin_lat_num" in df.columns and "begin_lon_num" in df.columns:
                valid_coords = df[df["has_valid_coords"] == True].copy()
                if not valid_coords.empty:
                    fig, ax = plt.subplots(figsize=(11, 7))
                    states = valid_coords["state"].unique()
                    colors = plt.cm.tab10(range(len(states)))
                    for state, color in zip(states, colors):
                        sub = valid_coords[valid_coords["state"] == state]
                        ax.scatter(
                            sub["begin_lon_num"],
                            sub["begin_lat_num"],
                            label=state,
                            alpha=0.4,
                            s=15,
                            c=[color],
                        )

                    ax.set_title("Geographic Distribution of Selected Storm Events", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Longitude (°W)", fontsize=12)
                    ax.set_ylabel("Latitude (°N)", fontsize=12)
                    ax.legend(title="State", loc="lower left")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    # Annotate comparison context
                    ax.text(
                        0.98,
                        0.02,
                        "Climatological Reference:\nChandipur, Odisha (21.46°N, 87.02°E)",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

                    plt.tight_layout()
                    p4 = self.figures_dir / "spatial_distribution_map.png"
                    fig.savefig(p4, dpi=200)
                    plt.close(fig)
                    generated_figures.append(p4)
        except Exception as exc:
            logger.warning(f"Error plotting spatial distribution map: {exc}")

        return generated_figures
