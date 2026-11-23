"""NOAA Storm Events Data Preprocessing Module.

Standardizes, cleans, and validates NOAA Storm Events datasets:
- Standardizes column names
- Safely parses dates and timestamps
- Validates geographic coordinates and flags missing/invalid coordinates
- Safely converts damage expressions ('10K', '1.5M', etc.) to numeric
- Validates injuries, deaths, and magnitude
- Detects and handles duplicate records
- Handles schema shifts across historical files gracefully
- Preserves all original columns and event identifiers
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NOAA damage multiplier regex: e.g. "10.00K", "1.5M", "2B", "500"
DAMAGE_REGEX = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmBbTt]?)\s*$")
DAMAGE_MULTIPLIERS = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
}

REQUIRED_DETAILS_FIELDS = [
    "episode_id",
    "event_id",
    "state",
    "year",
    "month_name",
    "event_type",
    "cz_type",
    "cz_name",
    "begin_date_time",
    "end_date_time",
    "begin_lat",
    "begin_lon",
    "end_lat",
    "end_lon",
    "injuries_direct",
    "injuries_indirect",
    "deaths_direct",
    "deaths_indirect",
    "damage_property",
    "damage_crops",
    "source",
    "magnitude",
    "magnitude_type",
    "episode_narrative",
    "event_narrative",
]


def parse_damage_value(val: Any) -> Optional[float]:
    """Safely convert a NOAA damage string (e.g. '10.00K', '1.5M', '0', None) to float.

    Returns float value, or None if missing or unparseable.
    Does NOT assume missing values are zero.
    """
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("null", "nan", "none", "?"):
        return None

    match = DAMAGE_REGEX.match(val_str)
    if not match:
        # Check if plain numeric
        try:
            return float(val_str)
        except ValueError:
            return None

    number_part, unit_part = match.groups()
    try:
        base_num = float(number_part)
        multiplier = DAMAGE_MULTIPLIERS.get(unit_part.upper(), 1.0)
        return base_num * multiplier
    except (ValueError, KeyError):
        return None


def validate_coordinates(
    lat: Any, lon: Any
) -> Tuple[bool, str]:
    """Validate latitude and longitude values.

    Returns (is_valid, status_code):
    - ('VALID', True)
    - ('MISSING', False)
    - ('OUT_OF_BOUNDS', False)
    - ('NULL_ISLAND', False)
    """
    if pd.isna(lat) or pd.isna(lon):
        return False, "MISSING"

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return False, "INVALID_NUMERIC"

    if np.isnan(lat_f) or np.isnan(lon_f):
        return False, "MISSING"

    # Null Island check (0.0, 0.0) is not within US territory
    if abs(lat_f) < 1e-4 and abs(lon_f) < 1e-4:
        return False, "NULL_ISLAND"

    # Range check
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return False, "OUT_OF_BOUNDS"

    return True, "VALID"


class NOAAPreprocessor:
    """Preprocessor for NOAA Storm Events bulk CSV datasets."""

    def __init__(self):
        self.quality_stats: Dict[str, Any] = {
            "total_raw_records": 0,
            "unique_event_ids": 0,
            "duplicate_records": 0,
            "missing_coords": 0,
            "invalid_coords": 0,
            "missing_timestamps": 0,
            "missing_schema_columns": {},
        }

    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase, stripped of whitespace."""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def clean_details(self, input_data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """Clean and standardize event details dataset."""
        if isinstance(input_data, (Path, str)):
            logger.info(f"Loading details file: {input_data}")
            df = pd.read_csv(input_data, compression="infer", low_memory=False, dtype=str)
        else:
            df = input_data.copy()

        df = self.normalize_column_names(df)
        self.quality_stats["total_raw_records"] += len(df)

        # Check required fields presence
        missing_cols = [c for c in REQUIRED_DETAILS_FIELDS if c not in df.columns]
        if missing_cols:
            logger.warning(f"Schema notice: Missing optional/required columns: {missing_cols}")
            for col in missing_cols:
                self.quality_stats["missing_schema_columns"][col] = (
                    self.quality_stats["missing_schema_columns"].get(col, 0) + 1
                )
                df[col] = np.nan

        # Duplicate checking
        initial_count = len(df)
        # Deduplicate on all columns or event_id if present
        if "event_id" in df.columns:
            exact_duplicates = df.duplicated().sum()
            event_id_dups = df.duplicated(subset=["event_id"]).sum()
            self.quality_stats["duplicate_records"] += int(exact_duplicates)
            # Remove exact duplicates
            df = df.drop_duplicates()
            if exact_duplicates > 0:
                logger.info(f"Removed {exact_duplicates} exact duplicate rows.")

        self.quality_stats["unique_event_ids"] = df["event_id"].nunique() if "event_id" in df.columns else 0

        # Safe datetime parsing
        for col_name in ["begin_date_time", "end_date_time"]:
            if col_name in df.columns:
                parsed_col = f"{col_name}_parsed"
                # NOAA uses formats like '30-APR-24 20:33:00' or '04/01/2012 20:48:00'
                parsed_dt = pd.to_datetime(df[col_name], errors="coerce", format="mixed")
                df[parsed_col] = parsed_dt
                if col_name == "begin_date_time":
                    missing_ts = df[col_name].isna() | df[parsed_col].isna()
                    self.quality_stats["missing_timestamps"] += int(missing_ts.sum())

        # Numeric damage fields creation (preserving original strings)
        if "damage_property" in df.columns:
            df["damage_property_num"] = df["damage_property"].apply(parse_damage_value)
        if "damage_crops" in df.columns:
            df["damage_crops_num"] = df["damage_crops"].apply(parse_damage_value)

        # Numeric conversions for injuries and deaths
        for col in ["injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect"]:
            if col in df.columns:
                df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Numeric conversion for magnitude
        if "magnitude" in df.columns:
            df["magnitude_num"] = pd.to_numeric(df["magnitude"], errors="coerce")

        # Coordinate validation
        if "begin_lat" in df.columns and "begin_lon" in df.columns:
            df["begin_lat_num"] = pd.to_numeric(df["begin_lat"], errors="coerce")
            df["begin_lon_num"] = pd.to_numeric(df["begin_lon"], errors="coerce")

            valid_flags = []
            status_codes = []
            for lat, lon in zip(df["begin_lat_num"], df["begin_lon_num"]):
                is_valid, status = validate_coordinates(lat, lon)
                valid_flags.append(is_valid)
                status_codes.append(status)

            df["has_valid_coords"] = valid_flags
            df["coord_status"] = status_codes

            missing_coords_count = (df["coord_status"] == "MISSING").sum()
            invalid_coords_count = (~df["has_valid_coords"] & (df["coord_status"] != "MISSING")).sum()
            self.quality_stats["missing_coords"] += int(missing_coords_count)
            self.quality_stats["invalid_coords"] += int(invalid_coords_count)

        if "end_lat" in df.columns and "end_lon" in df.columns:
            df["end_lat_num"] = pd.to_numeric(df["end_lat"], errors="coerce")
            df["end_lon_num"] = pd.to_numeric(df["end_lon"], errors="coerce")

        # Normalize state names to uppercase stripped strings for consistent joining/filtering
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip().str.upper()

        return df

    def clean_locations(self, input_data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """Clean and standardize event locations dataset."""
        if isinstance(input_data, (Path, str)):
            logger.info(f"Loading locations file: {input_data}")
            df = pd.read_csv(input_data, compression="infer", low_memory=False, dtype=str)
        else:
            df = input_data.copy()

        df = self.normalize_column_names(df)
        df = df.drop_duplicates()

        # Coordinate parsing
        if "latitude" in df.columns and "longitude" in df.columns:
            df["latitude_num"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude_num"] = pd.to_numeric(df["longitude"], errors="coerce")

            valid_flags = []
            status_codes = []
            for lat, lon in zip(df["latitude_num"], df["longitude_num"]):
                is_valid, status = validate_coordinates(lat, lon)
                valid_flags.append(is_valid)
                status_codes.append(status)

            df["has_valid_coords"] = valid_flags
            df["coord_status"] = status_codes

        if "location_index" in df.columns:
            df["location_index"] = pd.to_numeric(df["location_index"], errors="coerce")

        return df

    def clean_fatalities(self, input_data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """Clean and standardize event fatalities dataset."""
        if isinstance(input_data, (Path, str)):
            logger.info(f"Loading fatalities file: {input_data}")
            df = pd.read_csv(input_data, compression="infer", low_memory=False, dtype=str)
        else:
            df = input_data.copy()

        df = self.normalize_column_names(df)
        df = df.drop_duplicates()

        if "fatality_age" in df.columns:
            df["fatality_age_num"] = pd.to_numeric(df["fatality_age"], errors="coerce")

        if "fatality_date" in df.columns:
            df["fatality_date_parsed"] = pd.to_datetime(df["fatality_date"], errors="coerce", format="mixed")

        if "fatality_type" in df.columns:
            df["fatality_type"] = df["fatality_type"].astype(str).str.strip().str.upper()

        return df
