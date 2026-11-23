# Baseline Project Structure & Data Source Audit (2024)

**Generated:** 2026-10-26 15:15:00 UTC  
**Scope:** Comparative audit of `data/` and `Florida_Thunderstorm_2024_Training/data/` across all NOAA, ERA5, NASA LIS, and GOES-16 GLM assets.

---

## 1. Directory Structure & Mirrored Asset Comparison

The project exists across two main directory trees:
1. `data/`: The top-level workspace data directory.
2. `Florida_Thunderstorm_2024_Training/data/`: The encapsulated Florida machine-learning training pipeline data directory.

### Asset Reconciliation Table:

| Data Category | Root Directory (`data/`) | Project Directory (`Florida_Thunderstorm_2024_Training/data/`) | Status / Relationship |
| :--- | :--- | :--- | :--- |
| **NOAA Storm Events 2024** | `data/raw/details/`, `data/raw/fatalities/`, `data/raw/locations/` (12.69 MB, 13 KB, 983 KB) | `Florida_Thunderstorm_2024_Training/data/raw/noaa/` (identical checksums) | **Identical Mirror**: Raw NOAA GZ files are mirrored. Preprocessed Florida NOAA events (1,388 unique Florida events) reside in `Florida_Thunderstorm_2024_Training/data/processed/noaa_florida_2024.csv`. |
| **ERA5 Single-Level Full-Year** | `data/raw/era5/full_year/era5_florida_2024_single_levels.nc` (27,583,700 bytes) | `Florida_Thunderstorm_2024_Training/data/raw/era5/full_year/era5_florida_2024_single_levels.nc` (identical) | **Identical Mirror**: Contains all 1,464 six-hourly timestamps for leap year 2024 across Florida domain with all 12 variables (`t2m`, `d2m`, `sp`, `msl`, `tp`, `cp`, `cape`, `cin`, `tcc`, `u10`, `v10`, `tcwv`). |
| **ERA5 Pressure-Level Batches** | `data/raw/era5/full_year/pressure_level_batches/` (Months 01, 02, 03, 04 complete) | Mirrored batches in `Florida_Thunderstorm_2024_Training/data/raw/era5/full_year/pressure_level_batches/` | **Active Download Stream**: Months 01 to 04 are complete and strictly validated. Months 05-12 in queue. |
| **ERA5 Legacy Monthly** | `data/raw/era5_2024/florida/era5_florida_2024_05.nc` (128 KB) | `Florida_Thunderstorm_2024_Training/data/raw/era5/single_levels/` (Months 01, 02, 05) | **Preserved Legacy**: Monthly files kept intact; full-year consolidated files take priority. |
| **GOES-16 GLM Level 2 LCFA** | N/A | `Florida_Thunderstorm_2024_Training/data/raw/glm/` (12 NetCDF granules for Day 136, May 15 2024) | **Independent Child Asset**: Processed in `Florida_Thunderstorm_2024_Training/data/processed/glm_florida_2024.csv` (166 flashes). |
| **NASA ISS LIS** | `NASA LIS DATA/` | `Florida_Thunderstorm_2024_Training/data/processed/lis_florida_2024.csv` | **Sensor Decommissioned**: Zero 2024 observations; correctly marked unobserved without fabrication. |
| **Geographic Polygon** | N/A | `Florida_Thunderstorm_2024_Training/data/raw/geographic/florida_boundary.geojson` | **Authoritative Florida Polygon**: 218 spatial grid cells inside state boundaries. |

---

## 2. Pre-Modification Safety Backups

Before generating new processed datasets, existing outputs were preserved:
- `Florida_Thunderstorm_2024_Training/data/processed/era5_florida_2024.csv.bak` (15,190,535 bytes)
- `Florida_Thunderstorm_2024_Training/data/processed/florida_merged_2024.csv.bak` (20,346,403 bytes)
- `Florida_Thunderstorm_2024_Training/data/training/florida_thunderstorm_training_2024.csv.bak` (20,078,356 bytes)

No raw or legacy files will be overwritten or deleted.
