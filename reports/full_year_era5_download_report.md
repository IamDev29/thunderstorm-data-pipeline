# Full-Year 2024 ERA5 Download and Validation Report

**Updated:** 2026-10-26 22:46:11 UTC
**Domain:** Florida, USA & coastal waters (`[31.5, -88.0, 24.5, -79.5]`)
**Temporal Target:** Leap Year 2024 (2024-01-01 00:00 to 2024-12-31 18:00 UTC, 6-hourly, 1,464 timestamps)

---

## 1. Executive Summary

Both **Single-Level (Surface)** and **Pressure-Level (3D Atmospheric)** ERA5 datasets for Florida covering the entire calendar year 2024 are **100% complete, fully downloaded, consolidated, and strictly validated**.

| Dataset | File Path | File Size | Timestamps | Temporal Range | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Levels** | `data/raw/era5/full_year/era5_florida_2024_single_levels.nc` | `27,583,700 bytes` (26.31 MB) | 1,464 / 1,464 (100%) | 2024-01-01 00:00 to 2024-12-31 18:00 UTC | **VALIDATED** |
| **Pressure-Levels** | `data/raw/era5/full_year/era5_florida_2024_pressure_levels.nc` | `66,427,306 bytes` (63.35 MB) | 1,464 / 1,464 (100%) | 2024-01-01 00:00 to 2024-12-31 18:00 UTC | **VALIDATED** |

---

## 2. Authentication & Credential Safety

- **Status**: Verified and Authenticated successfully via Copernicus CDS API.
- **User Account ID**: Registered CDS user account resolved (`606860`).
- **Accepted Licences**: `cc-by`, `licence-to-use-copernicus-products`, `data-protection-privacy-statement`, `terms-of-use-cds`.
- **Credential Protection**: No API keys or passwords printed, logged, or saved in plain text in repository source code or reports.

---

## 3. Request 1: Single-Level Full-Year Dataset

- **Dataset**: `reanalysis-era5-single-levels`
- **CDS Request ID**: `9b419dea-7dc9-4513-bddc-09fc322da88f` (Estimated Cost: 105,408 / 121,000 limit, completed)
- **Consolidated File**: `data/raw/era5/full_year/era5_florida_2024_single_levels.nc`
- **Mirrored Path**: `Florida_Thunderstorm_2024_Training/data/raw/era5/full_year/era5_florida_2024_single_levels.nc`
- **Unique Timestamps**: `1,464` (First: `2024-01-01 00:00:00`, Last: `2024-12-31 18:00:00`)
- **Leap Day (2024-02-29)**: Verified present (4 synoptic timestamps: 00, 06, 12, 18 UTC).
- **Variables (12)**: `2m_temperature` (`t2m`), `2m_dewpoint_temperature` (`d2m`), `surface_pressure` (`sp`), `mean_sea_level_pressure` (`msl`), `total_precipitation` (`tp`), `convective_precipitation` (`cp`), `convective_available_potential_energy` (`cape`), `convective_inhibition` (`cin`), `total_cloud_cover` (`tcc`), `10m_u_component_of_wind` (`u10`), `10m_v_component_of_wind` (`v10`), `total_column_water_vapour` (`tcwv`).

---

## 4. Request 2: Pressure-Level Dataset (Cost-Aware Batch Retrieval)

### A. Initial Full-Year Rejection Analysis
The initial attempt to request all 12 months in one full-year API call was rejected prior to execution by Copernicus CDS:
```
HTTPError 403 Client Error: Forbidden (cost limits exceeded: 184,464.0 units vs limit of 60,000.0 units)
```

### B. Cost-Aware Monthly Batch Implementation
Each monthly request was pre-estimated against the 60,000-unit limit and verified (~14,600 to ~15,600 units, well within the limit). Individual monthly batches were downloaded into `data/raw/era5/full_year/pressure_level_batches/`:

| Month | Date Range | CDS Request ID | Estimated Cost | Limit | Status | Batch Size | Timestamps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | 2024-01-01 to 2024-01-31 | `d04589df-1bb6-4157-ac7b-bd2ac14ba2b0` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,181,397 bytes | 124 |
| **02** | 2024-02-01 to 2024-02-29 | `Local Cache / Verified` | N/A | 60,000 | **REUSED_VALID** | 4,920,858 bytes | 116 |
| **03** | 2024-03-01 to 2024-03-31 | `6e090ddc-48e3-4944-8bbd-cd5b7c9597fd` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,295,880 bytes | 124 |
| **04** | 2024-04-01 to 2024-04-30 | `d0482196-3f8a-4f79-b4d5-7401bf8d11b1` | 15,120.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,091,749 bytes | 120 |
| **05** | 2024-05-01 to 2024-05-31 | `bd7cd81f-c76c-453c-a8e0-d9ec185d8945` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,383,188 bytes | 124 |
| **06** | 2024-06-01 to 2024-06-30 | `87f8205b-c524-42bc-ad01-522121fc2bfa` | 15,120.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,206,045 bytes | 120 |
| **07** | 2024-07-01 to 2024-07-31 | `68c09adf-1bec-40d5-86be-559e0d3f69cd` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,354,516 bytes | 124 |
| **08** | 2024-08-01 to 2024-08-31 | `23c0cbd2-1ab1-44c9-879f-23329d43d150` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,370,357 bytes | 124 |
| **09** | 2024-09-01 to 2024-09-30 | `722f7a1b-b392-4e09-8bc7-f10431325bf0` | 15,120.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,174,949 bytes | 120 |
| **10** | 2024-10-01 to 2024-10-31 | `4b5393cd-51d9-495f-b88d-d7cee0035e65` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,221,663 bytes | 124 |
| **11** | 2024-11-01 to 2024-11-30 | `026a1b9b-f987-470c-8d5a-40d6cedf93e6` | 15,120.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,080,850 bytes | 120 |
| **12** | 2024-12-01 to 2024-12-31 | `c72566fc-e434-40be-adcf-043a35c42fbc` | 15,624.0 | 60,000.0 | **DOWNLOADED_VALID** | 5,179,416 bytes | 124 |

### C. Batch Consolidation
- All 12 monthly batches were chronologically assembled and concatenated along the time dimension.
- Overlapping timestamp check: **0 duplicates** across all batch boundaries.
- Missing timestamp check: **0 missing timestamps** (exactly 1,464 synoptic observations).
- Output saved to: `data/raw/era5/full_year/era5_florida_2024_pressure_levels.nc` and mirrored to `Florida_Thunderstorm_2024_Training/data/raw/era5/full_year/era5_florida_2024_pressure_levels.nc`.

### D. Pressure-Level Validation Results
- **Unique Timestamps**: `1,464` / 1,464 expected
- **First Timestamp**: `2024-01-01 00:00:00`
- **Last Timestamp**: `2024-12-31 18:00:00`
- **February 29, 2024 (Leap Day)**: Present (4 timestamps: 00, 06, 12, 18 UTC)
- **Pressure Levels**: `[850.0, 500.0, 300.0]` hPa (850, 500, 300 hPa)
- **Variables Present**: `['t', 'r', 'q', 'u', 'v', 'z', 'w']` (`t`, `r`, `q`, `u`, `v`, `z`, `w`)
- **Latitude Range**: `[24.5, 31.5]`
- **Longitude Range**: `[-88.0, -79.5]`

#### Variable Fill/NaN Check
| Variable | Long Name / Description | Dimensions | NaN Count (%) | Data Status |
| :--- | :--- | :--- | :--- | :--- |
| `t` | Air Temperature | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `r` | Relative Humidity | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `q` | Specific Humidity | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `u` | U component of wind | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `v` | V component of wind | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `z` | Geopotential | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |
| `w` | Vertical velocity | `(1464, 3, 29, 35)` | 0 (0.0%) | **NON-EMPTY & VALID** |

---

## 5. Downstream Integration and Processing Readiness

1. **Dual Full-Year Validation**: Both single-level and pressure-level datasets now have complete 100% full-year 2024 coverage (1,464 timestamps each) over the exact Florida domain.
2. **Pipeline Configuration**: `src/process_era5.py` and `Florida_Thunderstorm_2024_Training/src/process_era5.py` are configured to prioritize these full-year files and merge 12 surface features with 21 pressure-level features (7 variables $	imes$ 3 levels: 850, 500, 300 hPa).
3. **Source Integrity Preserved**: All existing NOAA Storm Events archives, GOES-16 GLM granules, and monthly ERA5 files were preserved without modification.
4. **Processing Readiness**: **READY FOR DOWNSTREAM PROCESSING & MERGING**.
