"""Statistical Analysis and Climate Similarity Module for ERA5 Pilot Study.

Computes seasonal statistics, spatial domain aggregates, reference-point extractions,
and preliminary climate similarity distance scores between Chandipur and US Gulf Coast candidate regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

SEASONS = {
    "pre_monsoon": [3, 4, 5],        # March, April, May (MAM)
    "monsoon": [6, 7, 8, 9],          # June, July, August, September (JJAS)
    "post_monsoon": [10, 11],         # October, November (ON)
    "annual": list(range(1, 13)),     # All months
}

CORE_ANALYSIS_VARS = [
    "cape",
    "cin",
    "tcwv",
    "t2m_degC",
    "d2m_degC",
    "tp_mm",
    "wind_speed_10m",
]


@dataclass
class VariableStats:
    """Statistical summary for a single meteorological variable in a given period."""

    region: str
    season: str
    variable: str
    units: str
    valid_count: int
    missing_pct: float
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float


class ERA5Analyzer:
    """Analyzer for multidimensional ERA5 datasets and climate similarity scoring."""

    def __init__(self, datasets: Dict[str, xr.Dataset], config: Optional[Dict[str, Any]] = None):
        self.datasets = datasets
        self.config = config or {}

    @staticmethod
    def get_seasonal_mask(ds: xr.Dataset, months: List[int]) -> xr.Dataset:
        """Filter dataset along time coordinate to match specified calendar months."""
        time_coord = "time" if "time" in ds.coords else "valid_time"
        month_vals = ds[time_coord].dt.month
        mask = month_vals.isin(months)
        return ds.sel({time_coord: mask})

    @staticmethod
    def compute_distribution_stats(
        values: np.ndarray, region: str, season: str, var_name: str, units: str
    ) -> VariableStats:
        """Compute robust distribution statistics over a numpy array."""
        clean_vals = values[~np.isnan(values)]
        total_len = len(values)
        valid_len = len(clean_vals)
        missing_pct = 0.0 if total_len == 0 else (total_len - valid_len) / total_len * 100.0

        if valid_len == 0:
            return VariableStats(
                region=region,
                season=season,
                variable=var_name,
                units=units,
                valid_count=0,
                missing_pct=100.0,
                mean=float("nan"),
                median=float("nan"),
                std=float("nan"),
                p10=float("nan"),
                p25=float("nan"),
                p75=float("nan"),
                p90=float("nan"),
                p95=float("nan"),
            )

        return VariableStats(
            region=region,
            season=season,
            variable=var_name,
            units=units,
            valid_count=valid_len,
            missing_pct=missing_pct,
            mean=float(np.mean(clean_vals)),
            median=float(np.median(clean_vals)),
            std=float(np.std(clean_vals)),
            p10=float(np.percentile(clean_vals, 10)),
            p25=float(np.percentile(clean_vals, 25)),
            p75=float(np.percentile(clean_vals, 75)),
            p90=float(np.percentile(clean_vals, 90)),
            p95=float(np.percentile(clean_vals, 95)),
        )

    def compute_all_regional_seasonal_stats(self) -> pd.DataFrame:
        """Calculate statistical summaries for all regions, variables, and seasons."""
        records: List[Dict[str, Any]] = []

        for region_key, ds in self.datasets.items():
            for season_name, months in SEASONS.items():
                ds_season = self.get_seasonal_mask(ds, months)
                for var in CORE_ANALYSIS_VARS:
                    if var in ds_season.data_vars:
                        arr = ds_season[var].values.flatten()
                        units = ds_season[var].attrs.get("units", "unknown")
                        st = self.compute_distribution_stats(arr, region_key, season_name, var, units)
                        records.append(st.__dict__)

        return pd.DataFrame(records)

    def extract_point_series(
        self, ds: xr.Dataset, target_lat: float, target_lon: float
    ) -> xr.Dataset:
        """Extract nearest grid cell time series for a reference point."""
        # Check coordinate standard
        lat_coord = "latitude" if "latitude" in ds.coords else "lat"
        lon_coord = "longitude" if "longitude" in ds.coords else "lon"

        # Handle longitude wrapping
        if ds[lon_coord].max() > 180 and target_lon < 0:
            target_lon = target_lon % 360

        point_ds = ds.sel(
            {lat_coord: target_lat, lon_coord: target_lon},
            method="nearest",
        )
        return point_ds

    def compute_climate_similarity(
        self,
        ref_region: str = "chandipur",
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute transparent multi-variable normalized distance scores.

        Lower distance = Higher similarity.
        Compares Chandipur with candidate US regions across seasons and annual period.
        """
        stats_df = self.compute_all_regional_seasonal_stats()
        if stats_df.empty:
            return pd.DataFrame(), {}

        candidate_regions = [r for r in self.datasets.keys() if r != ref_region]

        # Default equal weighting
        if weights is None:
            weights = {v: 1.0 / len(CORE_ANALYSIS_VARS) for v in CORE_ANALYSIS_VARS}

        similarity_records: List[Dict[str, Any]] = []

        for season_name in SEASONS.keys():
            ref_season = stats_df[(stats_df["region"] == ref_region) & (stats_df["season"] == season_name)]
            if ref_season.empty:
                continue

            ref_means = dict(zip(ref_season["variable"], ref_season["mean"]))
            ref_stds = dict(zip(ref_season["variable"], ref_season["std"]))

            for cand in candidate_regions:
                cand_season = stats_df[(stats_df["region"] == cand) & (stats_df["season"] == season_name)]
                if cand_season.empty:
                    continue

                cand_means = dict(zip(cand_season["variable"], cand_season["mean"]))

                var_distances: Dict[str, float] = {}
                weighted_dist = 0.0
                total_weight = 0.0

                for var in CORE_ANALYSIS_VARS:
                    if var in ref_means and var in cand_means and not np.isnan(ref_means[var]) and not np.isnan(cand_means[var]):
                        # Normalize difference by pooled/reference std dev (or mean if std is 0)
                        scale = ref_stds.get(var, 1.0)
                        if scale == 0 or np.isnan(scale):
                            scale = max(abs(ref_means[var]), 1.0)
                        
                        norm_diff = abs(cand_means[var] - ref_means[var]) / scale
                        var_distances[f"dist_{var}"] = float(norm_diff)
                        
                        w = weights.get(var, 1.0)
                        weighted_dist += w * norm_diff
                        total_weight += w

                composite_score = weighted_dist / total_weight if total_weight > 0 else float("nan")

                rec = {
                    "region": cand,
                    "season": season_name,
                    "composite_distance_score": composite_score,
                }
                rec.update(var_distances)
                similarity_records.append(rec)

        res_df = pd.DataFrame(similarity_records)

        # Sensitivity analysis with alternative weighting schemes
        schemes = {
            "equal_weights": weights,
            "convective_instability_focused": {
                "cape": 0.30,
                "cin": 0.20,
                "tcwv": 0.20,
                "t2m_degC": 0.10,
                "d2m_degC": 0.10,
                "tp_mm": 0.05,
                "wind_speed_10m": 0.05,
            },
            "moisture_thermal_focused": {
                "tcwv": 0.25,
                "d2m_degC": 0.25,
                "t2m_degC": 0.20,
                "tp_mm": 0.15,
                "cape": 0.05,
                "cin": 0.05,
                "wind_speed_10m": 0.05,
            },
        }

        # Calculate annual rankings across schemes
        rankings = {}
        annual_res = res_df[res_df["season"] == "annual"]
        for scheme_name, scheme_weights in schemes.items():
            scheme_scores = {}
            for cand in candidate_regions:
                ref_annual = stats_df[(stats_df["region"] == ref_region) & (stats_df["season"] == "annual")]
                cand_annual = stats_df[(stats_df["region"] == cand) & (stats_df["season"] == "annual")]
                ref_means = dict(zip(ref_annual["variable"], ref_annual["mean"]))
                ref_stds = dict(zip(ref_annual["variable"], ref_annual["std"]))
                cand_means = dict(zip(cand_annual["variable"], cand_annual["mean"]))

                w_sum = 0.0
                d_sum = 0.0
                for v in CORE_ANALYSIS_VARS:
                    if v in ref_means and v in cand_means and not np.isnan(ref_means[v]) and not np.isnan(cand_means[v]):
                        scale = ref_stds.get(v, 1.0) or 1.0
                        diff = abs(cand_means[v] - ref_means[v]) / scale
                        w = scheme_weights.get(v, 1.0)
                        d_sum += w * diff
                        w_sum += w
                scheme_scores[cand] = float(d_sum / w_sum) if w_sum > 0 else float("nan")

            rankings[scheme_name] = sorted(scheme_scores.items(), key=lambda x: x[1])

        return res_df, rankings
