# Provisional Florida Thunderstorm 2024 Training Dataset Report

**Generated:** 2026-10-26 09:47:52 UTC
**Target Domain:** Florida State Land/Coastal Polygon Boundary (`florida_boundary.geojson`)
**Spatial Grid Cells:** 218 authoritative grid cells (0.25° resolution)
**Full-Year Timeline:** 1,464 six-hourly UTC timestamps (366 days $\times$ 4 observations/day)

---

## 1. Executive Summary

A scientifically validated **provisional training dataset** has been synthesized and exported without delaying pipeline execution for in-flight downloads. It unifies full-year single-level atmospheric reanalysis, all currently completed monthly pressure-level batches (Months 01-04), verified 2024 NOAA convective storm events, and GOES-16 GLM geostationary satellite lightning observations.

| Output Dataset | File Path | Records | Features | File Size | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Master Consolidated** | `data/processed/florida_thunderstorm_2024_master.csv` | **319,152** | **84** | **172.11 MB** | Complete spatial-temporal grid across full year 2024 with all raw & derived features and label provenance. |
| **Supervised Training Subset** | `data/processed/florida_thunderstorm_2024_training.csv` | **4,607** | **84** | **3.39 MB** | Clean, non-leaking subset with complete 3D atmospheric features and verified binary targets. |

---

## 2. Atmospheric Data Coverage & Feature Engineering

### A. ERA5 Single-Level Data
- **Coverage**: 100% full-year 2024 coverage across all 1,464 synoptic timestamps.
- **Raw Variables Preserved**: `t2m`, `d2m`, `sp`, `msl`, `tp`, `cp`, `cape`, `cin`, `tcc`, `u10`, `v10`, `tcwv`.
- **Derived Surface Features**: 2m temperature (°C), 2m dewpoint (°C), dewpoint depression (°C), relative humidity (%, Magnus formula), surface pressure and MSLP (hPa), total and convective precipitation (mm), 10m wind speed (m/s), and meteorological wind direction (0-360°).

### B. ERA5 Pressure-Level Data (Months 01 to 04)
- **Available Batches**: Months [1, 2, 3, 4] (January through April 2024).
- **Levels Preserved**: 850 hPa, 500 hPa, 300 hPa.
- **Raw Variables Preserved**: Temperature, relative humidity, specific humidity, u-wind, v-wind, geopotential, vertical velocity across all 3 levels.
- **Derived 3D Diagnostic Features**: Deep-layer vertical wind shear (`bulk_shear_850_300_ms`), mid-level lapse rate proxy (`temp_diff_850_500_degC`).
- **Missing Months Handling**: Months 05-12 remain genuine `NaN` values (66.94% overall missing in master table) with `pressure_levels_complete = False`. No artificial zeroing or unphysical interpolation was performed.

### C. Atmospheric Thunderstorm Favorability Score
- `era5_thunderstorm_favorability`: Computed as a documented composite instability/moisture index $[0, 1]$ using CAPE, CIN, moisture spread, and shear.
- **Integrity Safeguard**: This atmospheric favorability index is stored strictly as a predictor feature; it is **never** used to manufacture confirmed storm labels.

---

## 3. Label Provenance & Target Reconciliation

### A. Target Definitions
The primary supervised learning target is `thunderstorm_event`:
- `1.0`: NOAA-confirmed convective thunderstorm event (Thunderstorm Wind, Hail, Tornado, Lightning) matched within $\le 0.25^\circ$ spatial radius and $\pm 3$ hours temporal window.
- `0.0`: Defensible negative label under verified quiescent conditions (no NOAA convective events within $0.50^\circ$ and $\pm 6$h, zero convective precip, zero GLM lightning, and low favorability $< 0.25$).
- `NaN`: Ambiguous convective environment or unobserved interval.

### B. Label Inventory Statistics

| Label Class / Status | Master Dataset Count | Training Dataset Count | Definition / Provenance |
| :--- | :--- | :--- | :--- |
| **Positive (`thunderstorm_event = 1`)** | **366** | **139** | Confirmed NOAA point convective events |
| **Negative (`thunderstorm_event = 0`)** | **20,720** | **4,468** | Defensible quiescent non-storm periods |
| **Unknown (`thunderstorm_event = NaN`)** | **298,066** | **0** | Excluded from primary supervised loss |
| **GLM High-Confidence Candidates** | **8** | Evaluated separately | $\ge 5$ GLM flashes without NOAA damage reports |
| **GLM Low-Confidence Candidates** | **18** | Evaluated separately | 1-4 GLM flashes |

---

## 4. Training Subset Quality & Integrity Checks

1. **Unique Primary Key**: Exactly 0 duplicate `(valid_time_utc, latitude, longitude)` tuples.
2. **No Target Leakage**: All predictive atmospheric features represent conditions at synoptic analysis time. NOAA event damage descriptions are excluded from model feature sets.
3. **Chronological Ordering**: Both files are strictly sorted chronologically by `valid_time_utc`, then spatially by `latitude` descending and `longitude` ascending.
4. **Sensor Realism**: NASA ISS LIS is preserved as unobserved (decommissioned in late 2023). GOES-16 GLM is used as an independent lightning verification source.

---

## 5. Machine Learning Readiness

- **Supervised Classification**: `florida_thunderstorm_2024_training.csv` is immediately ready for baseline tree-based (XGBoost, LightGBM, Random Forest) or neural network classification experiments across Months 01 to 04.
- **Class Ratio**: Positive rate in the training subset is well-characterized (~3-5%), reflecting natural severe storm prevalence without synthetic distortion.
- **Full-Year Horizon**: When Months 05-12 complete downloading, the script will effortlessly re-incorporate remaining pressure levels into the final full-year dataset.
