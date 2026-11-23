"""Geographic and Event Type Filtering Module.

Implements two-stage filtering:
1. State-level filtering (Florida, Texas, Louisiana, Mississippi)
2. Geographic bounding box and coastal focus filtering
3. Event-type filtering (Thunderstorm Wind, Hail, Lightning, Heavy Rain, Flash Flood,
   Tornado, Tropical Storm, Hurricane)
4. Preserves records with missing coordinates with clear flags and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Geographic bounding box specification."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if coordinates fall within the bounding box."""
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)


class GeographicFilter:
    """Configurable two-stage geographic and event-type filter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.geo_config = config.get("geographic_filters", {})
        self.target_states: Dict[str, str] = self.geo_config.get(
            "target_states",
            {
                "florida": "FLORIDA",
                "texas": "TEXAS",
                "louisiana": "LOUISIANA",
                "mississippi": "MISSISSIPPI",
            },
        )
        self.configured_event_types: List[str] = config.get(
            "event_types",
            [
                "Thunderstorm Wind",
                "Hail",
                "Lightning",
                "Heavy Rain",
                "Flash Flood",
                "Tornado",
                "Tropical Storm",
                "Hurricane (Typhoon)",
            ],
        )
        # Lowercase set for case-insensitive matching
        self._event_types_lower: Set[str] = {t.lower() for t in self.configured_event_types}

        # Track exclusion diagnostics
        self.filter_diagnostics: Dict[str, Any] = {
            "initial_events": 0,
            "excluded_unmatched_state": 0,
            "excluded_unmatched_event_type": 0,
            "retained_missing_coords": 0,
            "excluded_outside_bounding_box": 0,
            "state_counts": {},
            "event_type_counts": {},
        }

    def filter_by_event_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to retain configured thunderstorm-related event types.

        Preserves original event_type casing and values in the output.
        """
        if "event_type" not in df.columns or df.empty:
            return df

        mask = df["event_type"].astype(str).str.lower().isin(self._event_types_lower)
        excluded = (~mask).sum()
        self.filter_diagnostics["excluded_unmatched_event_type"] += int(excluded)
        logger.info(f"Retained {mask.sum():,} events matching {len(self._event_types_lower)} event types; excluded {excluded:,}.")
        return df[mask].copy()

    def filter_by_state(self, df: pd.DataFrame, state_name: str) -> pd.DataFrame:
        """Stage 1: Filter dataset by exact state name (case-insensitive)."""
        if "state" not in df.columns or df.empty:
            return df

        state_norm = state_name.strip().upper()
        mask = df["state"].astype(str).str.strip().str.upper() == state_norm
        filtered = df[mask].copy()
        logger.info(f"State filter [{state_norm}]: {len(filtered):,} records found.")
        return filtered

    def apply_bounding_box_filter(
        self,
        df: pd.DataFrame,
        bbox: BoundingBox,
        include_missing_coords: bool = True,
    ) -> pd.DataFrame:
        """Stage 2: Filter records based on coordinates.

        - If include_missing_coords is True, records with missing coordinates are NOT dropped;
          they are retained and flagged with insufficient_coords = True.
        - Records with valid coordinates outside the bounding box are excluded.
        """
        if df.empty or "begin_lat_num" not in df.columns or "begin_lon_num" not in df.columns:
            df["insufficient_coords"] = True
            return df

        df = df.copy()

        # Check coordinate validity
        valid_coords_mask = (
            df["begin_lat_num"].notna()
            & df["begin_lon_num"].notna()
            & (df["coord_status"] == "VALID")
        )

        in_bbox_mask = (
            valid_coords_mask
            & (df["begin_lat_num"] >= bbox.lat_min)
            & (df["begin_lat_num"] <= bbox.lat_max)
            & (df["begin_lon_num"] >= bbox.lon_min)
            & (df["begin_lon_num"] <= bbox.lon_max)
        )

        missing_coords_mask = ~valid_coords_mask

        # Assign insufficient_coords flag
        df["insufficient_coords"] = missing_coords_mask

        if include_missing_coords:
            final_mask = in_bbox_mask | missing_coords_mask
            excluded = (~final_mask).sum()
        else:
            final_mask = in_bbox_mask
            excluded = (~final_mask).sum()

        self.filter_diagnostics["excluded_outside_bounding_box"] += int(excluded)
        self.filter_diagnostics["retained_missing_coords"] += int(missing_coords_mask.sum())

        retained_df = df[final_mask].copy()
        logger.info(
            f"Bounding box filter [{bbox.lat_min},{bbox.lat_max}] x [{bbox.lon_min},{bbox.lon_max}]: "
            f"retained {len(retained_df):,} records ({missing_coords_mask.sum():,} with insufficient coords), "
            f"excluded {excluded:,} out-of-bounds records."
        )
        return retained_df

    def process_all_regions(
        self, clean_details_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Perform full multi-region filtering pipeline.

        Returns dictionary of DataFrames:
        - 'florida'
        - 'texas'
        - 'louisiana'
        - 'mississippi'
        - 'combined_gulf_coast'
        - 'thunderstorm_events_combined'
        """
        self.filter_diagnostics["initial_events"] = len(clean_details_df)
        results: Dict[str, pd.DataFrame] = {}

        region_configs = self.geo_config.get("regions", {})

        # Process individual states
        for region_key, state_name in self.target_states.items():
            # Stage 1: State Filter
            state_df = self.filter_by_state(clean_details_df, state_name)

            # Stage 2: Geographic Bounding Box if configured
            reg_cfg = region_configs.get(region_key, {})
            bbox_dict = reg_cfg.get("coastal_focus") or reg_cfg.get("bounding_box")
            if bbox_dict:
                bbox = BoundingBox(
                    lat_min=float(bbox_dict["lat_min"]),
                    lat_max=float(bbox_dict["lat_max"]),
                    lon_min=float(bbox_dict["lon_min"]),
                    lon_max=float(bbox_dict["lon_max"]),
                )
                state_filtered_df = self.apply_bounding_box_filter(
                    state_df, bbox, include_missing_coords=True
                )
            else:
                state_filtered_df = state_df.copy()
                state_filtered_df["insufficient_coords"] = state_filtered_df.get("has_valid_coords", True) == False

            results[region_key] = state_filtered_df
            self.filter_diagnostics["state_counts"][state_name] = len(state_filtered_df)

        # Combined Gulf Coast candidate regions
        combined_cfg = self.geo_config.get("combined_gulf_coast", {})
        combined_states = [s.upper() for s in combined_cfg.get("states", list(self.target_states.values()))]

        state_mask = clean_details_df["state"].astype(str).str.strip().str.upper().isin(combined_states)
        combined_state_df = clean_details_df[state_mask].copy()

        c_bbox_dict = combined_cfg.get("bounding_box")
        if c_bbox_dict:
            c_bbox = BoundingBox(
                lat_min=float(c_bbox_dict["lat_min"]),
                lat_max=float(c_bbox_dict["lat_max"]),
                lon_min=float(c_bbox_dict["lon_min"]),
                lon_max=float(c_bbox_dict["lon_max"]),
            )
            combined_gulf = self.apply_bounding_box_filter(
                combined_state_df, c_bbox, include_missing_coords=True
            )
        else:
            combined_gulf = combined_state_df.copy()
            combined_gulf["insufficient_coords"] = combined_gulf.get("has_valid_coords", True) == False

        results["combined_gulf_coast"] = combined_gulf

        # Convective / Thunderstorm-specific dataset for combined Gulf Coast
        ts_df = self.filter_by_event_types(combined_gulf)
        results["thunderstorm_events_combined"] = ts_df

        return results

    @staticmethod
    def filter_associated_locations(
        clean_locations_df: pd.DataFrame, event_ids: Set[Any]
    ) -> pd.DataFrame:
        """Filter locations table to retain only locations belonging to the selected event_ids."""
        if clean_locations_df.empty or "event_id" not in clean_locations_df.columns:
            return clean_locations_df
        # Normalize event_id comparison
        event_ids_str = {str(eid) for eid in event_ids}
        mask = clean_locations_df["event_id"].astype(str).isin(event_ids_str)
        return clean_locations_df[mask].copy()

    @staticmethod
    def filter_associated_fatalities(
        clean_fatalities_df: pd.DataFrame, event_ids: Set[Any]
    ) -> pd.DataFrame:
        """Filter fatalities table to retain only fatalities belonging to the selected event_ids."""
        if clean_fatalities_df.empty or "event_id" not in clean_fatalities_df.columns:
            return clean_fatalities_df
        event_ids_str = {str(eid) for eid in event_ids}
        mask = clean_fatalities_df["event_id"].astype(str).isin(event_ids_str)
        return clean_fatalities_df[mask].copy()
