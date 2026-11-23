"""Statistical Analysis Module for NOAA Storm Events Pipeline.

Computes aggregations, cross-tabulations, missingness profiles, and
climatological summary statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NOAAAnalyzer:
    """Analyzer for cleaned and filtered NOAA storm event datasets."""

    def __init__(self, details_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.df = details_df.copy()
        self.config = config or {}

    def get_event_counts_by_year(self) -> pd.DataFrame:
        """Calculate event counts, total injuries, deaths, and property damage grouped by year."""
        if self.df.empty or "year" not in self.df.columns:
            return pd.DataFrame(columns=["year", "event_count"])

        agg_dict = {"event_id": "count"}
        if "injuries_direct_num" in self.df.columns:
            agg_dict["injuries_direct_num"] = "sum"
        if "deaths_direct_num" in self.df.columns:
            agg_dict["deaths_direct_num"] = "sum"
        if "damage_property_num" in self.df.columns:
            agg_dict["damage_property_num"] = "sum"

        res = (
            self.df.groupby("year", as_index=False)
            .agg(agg_dict)
            .rename(
                columns={
                    "event_id": "event_count",
                    "injuries_direct_num": "direct_injuries",
                    "deaths_direct_num": "direct_deaths",
                    "damage_property_num": "total_property_damage_usd",
                }
            )
            .sort_values("year")
        )
        return res

    def get_event_counts_by_state(self) -> pd.DataFrame:
        """Calculate event counts and impacts grouped by state."""
        if self.df.empty or "state" not in self.df.columns:
            return pd.DataFrame(columns=["state", "event_count"])

        res = (
            self.df.groupby("state", as_index=False)["event_id"]
            .count()
            .rename(columns={"event_id": "event_count"})
            .sort_values("event_count", ascending=False)
        )
        return res

    def get_event_counts_by_type(self) -> pd.DataFrame:
        """Calculate event counts grouped by event type."""
        if self.df.empty or "event_type" not in self.df.columns:
            return pd.DataFrame(columns=["event_type", "event_count"])

        res = (
            self.df.groupby("event_type", as_index=False)["event_id"]
            .count()
            .rename(columns={"event_id": "event_count"})
            .sort_values("event_count", ascending=False)
        )
        return res

    def get_event_counts_by_month(self) -> pd.DataFrame:
        """Calculate event counts grouped by calendar month."""
        if self.df.empty:
            return pd.DataFrame(columns=["month_name", "event_count"])

        # Use month_name or extract from datetime
        if "month_name" in self.df.columns and self.df["month_name"].notna().any():
            month_order = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            res = (
                self.df.groupby("month_name", as_index=False)["event_id"]
                .count()
                .rename(columns={"event_id": "event_count"})
            )
            # Sort according to month_order
            res["month_order"] = res["month_name"].map({m: i for i, m in enumerate(month_order)})
            res = res.sort_values("month_order").drop(columns=["month_order"], errors="ignore")
            return res

        return pd.DataFrame(columns=["month_name", "event_count"])

    def get_year_state_cross_tab(self) -> pd.DataFrame:
        """Generate a pivot table of event counts by year and state."""
        if self.df.empty or "year" not in self.df.columns or "state" not in self.df.columns:
            return pd.DataFrame()
        ct = pd.crosstab(self.df["year"], self.df["state"])
        return ct

    def get_comprehensive_quality_summary(
        self,
        raw_count: int,
        manifest_records: List[Any],
        preprocessor_stats: Dict[str, Any],
        geo_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile complete data quality summary dictionary."""
        total_downloads = len(manifest_records)
        successful_downloads = sum(1 for m in manifest_records if getattr(m, "status", None) in ("DOWNLOADED", "CACHED"))
        failed_downloads = sum(1 for m in manifest_records if getattr(m, "status", None) == "FAILED")

        total_cleaned = len(self.df)
        unique_ids = self.df["event_id"].nunique() if "event_id" in self.df.columns else 0

        # Coordinate status
        missing_coords = (
            (self.df["coord_status"] == "MISSING").sum()
            if "coord_status" in self.df.columns
            else 0
        )
        invalid_coords = (
            (~self.df["has_valid_coords"] & (self.df["coord_status"] != "MISSING")).sum()
            if "has_valid_coords" in self.df.columns
            else 0
        )
        insufficient_coords = (
            self.df["insufficient_coords"].sum()
            if "insufficient_coords" in self.df.columns
            else missing_coords
        )

        return {
            "total_downloaded_files": total_downloads,
            "successfully_processed_files": successful_downloads,
            "failed_downloads": failed_downloads,
            "total_raw_records": raw_count,
            "total_cleaned_records": total_cleaned,
            "unique_event_ids": int(unique_ids),
            "duplicate_records_detected": preprocessor_stats.get("duplicate_records", 0),
            "records_with_missing_coordinates": int(missing_coords),
            "records_with_invalid_coordinates": int(invalid_coords),
            "records_with_insufficient_coordinates": int(insufficient_coords),
            "records_with_missing_timestamps": preprocessor_stats.get("missing_timestamps", 0),
            "missing_schema_columns": preprocessor_stats.get("missing_schema_columns", {}),
            "geographic_exclusions": {
                "excluded_outside_bounding_box": geo_diagnostics.get("excluded_outside_bounding_box", 0),
                "excluded_unmatched_event_type": geo_diagnostics.get("excluded_unmatched_event_type", 0),
                "retained_missing_coords": geo_diagnostics.get("retained_missing_coords", 0),
            },
        }
