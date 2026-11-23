"""Process, clean, and standardize NOAA Storm Events data for Florida (2024).

Performs:
1. Event-level reading of details, locations, and fatalities.
2. Coordinate consolidation without record duplication.
3. High-precision Florida state boundary polygon containment filtering.
4. UTC timestamp normalization with timezone tracking.
5. Preservation of original event identifiers, types, magnitudes, and damage values.
"""

from __future__ import annotations

from datetime import timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shapely.geometry

logger = logging.getLogger(__name__)


class NOAAProcessor:
    """Cleans and standardizes Florida NOAA Storm Events."""

    def __init__(
        self,
        raw_dir: Path = Path("data/raw/noaa"),
        boundary_file: Path = Path("data/raw/geographic/florida_boundary.geojson"),
        interim_dir: Path = Path("data/interim/noaa"),
        processed_dir: Path = Path("data/processed"),
    ):
        self.raw_dir = raw_dir
        self.boundary_file = boundary_file
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.florida_polygon = self._load_florida_boundary()

    def _load_florida_boundary(self) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Load Shapely geometry for Florida state boundary."""
        if not self.boundary_file.exists():
            logger.warning(f"Boundary file not found at {self.boundary_file}. Boundary filtering will use bounding box.")
            return None
        try:
            with open(self.boundary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            geom_data = data["features"][0]["geometry"] if "features" in data else data
            poly = shapely.geometry.shape(geom_data)
            logger.info(f"Loaded Florida state polygon boundary (bounds: {poly.bounds}).")
            return poly
        except Exception as exc:
            logger.error(f"Error loading Florida boundary polygon: {exc}")
            return None

    def load_raw_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load details, locations, and fatalities CSV files."""
        # Find raw or reused files
        p_det = self.raw_dir / "StormEvents_details_2024.csv.gz"
        p_loc = self.raw_dir / "StormEvents_locations_2024.csv.gz"
        p_fat = self.raw_dir / "StormEvents_fatalities_2024.csv.gz"

        # Fallback to parent processed if raw not available
        if not p_det.exists():
            p_det = Path("../data/processed/florida/details_florida.csv")
        if not p_loc.exists():
            p_loc = Path("../data/processed/florida/locations_florida.csv")
        if not p_fat.exists():
            p_fat = Path("../data/processed/florida/fatalities_florida.csv")

        logger.info(f"Loading NOAA details from {p_det}...")
        df_details = pd.read_csv(p_det, compression="gzip" if str(p_det).endswith(".gz") else None, low_memory=False)

        logger.info(f"Loading NOAA locations from {p_loc}...")
        df_locations = (
            pd.read_csv(p_loc, compression="gzip" if str(p_loc).endswith(".gz") else None, low_memory=False)
            if p_loc.exists()
            else pd.DataFrame()
        )

        logger.info(f"Loading NOAA fatalities from {p_fat}...")
        df_fatalities = (
            pd.read_csv(p_fat, compression="gzip" if str(p_fat).endswith(".gz") else None, low_memory=False)
            if p_fat.exists()
            else pd.DataFrame()
        )

        return df_details, df_locations, df_fatalities

    def process_florida_events(self) -> pd.DataFrame:
        """Process, filter, and standardize Florida NOAA events."""
        df_details, df_locations, df_fatalities = self.load_raw_datasets()

        # Normalize column names first
        df_details.columns = [c.strip().lower() for c in df_details.columns]

        # Filter state
        if "state" in df_details.columns:
            df_details = df_details[df_details["state"].str.upper() == "FLORIDA"].copy()
        logger.info(f"Isolate Florida events in details: {len(df_details):,} rows")

        # 1. Parse timestamps and handle timezones properly
        # NOAA format typically '01-JUL-24 15:51:00' with CZ_TIMEZONE ('EST-5', 'CST-6', 'EDT-4', etc.)
        TZ_MAP = {
            "EST-5": timezone(timedelta(hours=-5), name="EST"),
            "CST-6": timezone(timedelta(hours=-6), name="CST"),
            "EDT-4": timezone(timedelta(hours=-4), name="EDT"),
            "CDT-5": timezone(timedelta(hours=-5), name="CDT"),
        }

        # Preserve local timestamp string
        if "begin_date_time" in df_details.columns:
            df_details["begin_date_time_local"] = df_details["begin_date_time"].astype(str)
        elif "begin_date_time_parsed" in df_details.columns:
            df_details["begin_date_time_local"] = df_details["begin_date_time_parsed"].astype(str)
        else:
            df_details["begin_date_time_local"] = ""

        # Normalize CZ_TIMEZONE
        if "cz_timezone" not in df_details.columns:
            df_details["cz_timezone"] = "UNKNOWN"
        df_details["cz_timezone"] = df_details["cz_timezone"].fillna("UNKNOWN").astype(str).str.strip()

        utc_naive_list = []
        status_list = []

        for _, r in df_details.iterrows():
            loc_str = r.get("begin_date_time_local", "")
            tz_str = r.get("cz_timezone", "UNKNOWN")

            dt_local = pd.to_datetime(loc_str, format="%d-%b-%y %H:%M:%S", errors="coerce")
            if pd.isna(dt_local):
                dt_local = pd.to_datetime(loc_str, errors="coerce")

            if pd.isna(dt_local):
                utc_naive_list.append(pd.NaT)
                status_list.append("FAILED_DATE_PARSE")
            elif tz_str in TZ_MAP:
                tz_obj = TZ_MAP[tz_str]
                # Localize with designated timezone
                aware_dt = dt_local.replace(tzinfo=tz_obj)
                # Convert to UTC timezone-aware
                utc_aware = aware_dt.astimezone(timezone.utc)
                # Naive UTC for serialization
                utc_naive_list.append(utc_aware.replace(tzinfo=None))
                status_list.append(f"CONVERTED_{tz_str}")
            else:
                # Flag ambiguous or unestablished timezone
                utc_naive_list.append(pd.NaT)
                status_list.append(f"UNRECOGNIZED_TZ_{tz_str}")

        df_details["timestamp_utc"] = utc_naive_list
        df_details["timezone_conversion_status"] = status_list
        logger.info(f"Timezone conversion summary:\n{df_details['timezone_conversion_status'].value_counts()}")

        # 2. Extract best coordinates
        # Check begin_lat / begin_lon, or fallback to location table centroid
        if "begin_lat" in df_details.columns and "begin_lon" in df_details.columns:
            df_details["latitude"] = pd.to_numeric(df_details["begin_lat"], errors="coerce")
            df_details["longitude"] = pd.to_numeric(df_details["begin_lon"], errors="coerce")
        else:
            df_details["latitude"] = np.nan
            df_details["longitude"] = np.nan

        # Fill missing coordinates from locations table if available
        if not df_locations.empty and "event_id" in df_locations.columns:
            df_locations.columns = [c.strip().lower() for c in df_locations.columns]
            loc_lat_col = "latitude" if "latitude" in df_locations.columns else "lat"
            loc_lon_col = "longitude" if "longitude" in df_locations.columns else "lon"
            if loc_lat_col in df_locations.columns and loc_lon_col in df_locations.columns:
                loc_agg = (
                    df_locations.groupby("event_id")
                    .agg({
                        loc_lat_col: "first",
                        loc_lon_col: "first",
                    })
                    .reset_index()
                )
                loc_dict_lat = dict(zip(loc_agg["event_id"], loc_agg[loc_lat_col]))
                loc_dict_lon = dict(zip(loc_agg["event_id"], loc_agg[loc_lon_col]))

                mask_missing = df_details["latitude"].isna()
                df_details.loc[mask_missing, "latitude"] = df_details.loc[mask_missing, "event_id"].map(loc_dict_lat)
                df_details.loc[mask_missing, "longitude"] = df_details.loc[mask_missing, "event_id"].map(loc_dict_lon)

        # 3. Flag coordinates availability
        df_details["coordinates_available"] = df_details["latitude"].notna() & df_details["longitude"].notna()

        # 4. Filter with Florida state boundary polygon if coordinates exist
        if self.florida_polygon is not None:
            in_fl = []
            for _, row in df_details.iterrows():
                if row["coordinates_available"]:
                    pt = shapely.geometry.Point(row["longitude"], row["latitude"])
                    in_fl.append(self.florida_polygon.contains(pt))
                else:
                    # Retain events without coordinates with flag
                    in_fl.append(True)
            df_details["in_florida_polygon"] = in_fl
            df_details = df_details[df_details["in_florida_polygon"]].copy()

        # 5. Clean damage fields if present
        for col_dmg in ["damage_property", "damage_crops"]:
            if col_dmg in df_details.columns and f"{col_dmg}_num" not in df_details.columns:
                df_details[f"{col_dmg}_num"] = 0.0

        # Save interim copies
        interim_path = self.interim_dir / "florida_events_interim.csv"
        df_details.to_csv(interim_path, index=False)
        logger.info(f"Saved interim Florida events ({len(df_details):,} rows) to {interim_path}")

        # Select standardized columns for processed output
        cols_to_keep = [
            "event_id",
            "state",
            "event_type",
            "cz_name",
            "cz_timezone",
            "begin_date_time_local",
            "timestamp_utc",
            "timezone_conversion_status",
            "latitude",
            "longitude",
            "coordinates_available",
        ]
        cols_to_keep = [c for c in cols_to_keep if c in df_details.columns]
        for opt_col in ["magnitude", "magnitude_type", "damage_property_num", "damage_crops_num", "event_narrative"]:
            if opt_col in df_details.columns:
                cols_to_keep.append(opt_col)

        df_processed = df_details[cols_to_keep].copy()
        df_processed = df_processed.drop_duplicates(subset=["event_id"])

        out_path = self.processed_dir / "noaa_florida_2024.csv"
        df_processed.to_csv(out_path, index=False)
        logger.info(f"Processed Florida NOAA dataset saved to {out_path} ({len(df_processed):,} unique events)")

        return df_processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = NOAAProcessor()
    df_out = processor.process_florida_events()
    print("NOAA processed rows:", len(df_out))
