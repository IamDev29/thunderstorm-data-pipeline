# Florida Thunderstorm Training Dataset Quality Audit Report

## 1. Executive Summary

Overall Dataset Quality Status: **PASSED**

## 2. Integrity Verification Checks

| Verification Check | Status | Details |
| :--- | :---: | :--- |
| `read_back_successful` | **PASS** | Verified programmatically |
| `csv_parquet_row_match` | **PASS** | Verified programmatically |
| `is_chronologically_and_spatially_sorted` | **PASS** | Verified programmatically |
| `no_duplicate_composite_keys` | **PASS** | Verified programmatically |
| `coordinates_within_florida_bounds` | **PASS** | Verified programmatically |
| `target_leakage_prevented` | **PASS** | Verified programmatically |
| `glm_unobserved_never_imputed_as_zero` | **PASS** | Verified programmatically |
| `valid_storm_label_values` | **PASS** | Verified programmatically |

## 3. Predictor Feature Completeness & Missing Rates

| Atmospheric Predictor | Missing Value % | Quality Assessment |
| :--- | :---: | :--- |
| `temperature_2m_degC` | 0.00% | Complete (0% missing) |
| `dewpoint_2m_degC` | 0.00% | Complete (0% missing) |
| `relative_humidity_2m_pct` | 0.00% | Complete (0% missing) |
| `surface_pressure_hPa` | 1.45% | Acceptable |
| `mslp_hPa` | 1.45% | Acceptable |
| `total_precipitation_mm` | 0.00% | Complete (0% missing) |
| `convective_precipitation_mm` | 1.45% | Acceptable |
| `cape_J_kg` | 0.00% | Complete (0% missing) |
| `cin_J_kg` | 89.41% | Acceptable |
| `cloud_cover_fraction` | 1.45% | Acceptable |
| `u_wind_10m_ms` | 0.00% | Complete (0% missing) |
| `v_wind_10m_ms` | 0.00% | Complete (0% missing) |
| `wind_speed_10m_ms` | 0.00% | Complete (0% missing) |
| `wind_direction_10m_deg` | 0.00% | Complete (0% missing) |
| `tcwv_kg_m2` | 0.00% | Complete (0% missing) |

## 4. Dataset Dimensions & Serialization

- **Total Rows**: 53,092
- **Total Columns**: 53
- **CSV Path**: `data\training\florida_thunderstorm_training_2024.csv`
- **Parquet Path**: `data\training\florida_thunderstorm_training_2024.parquet`
