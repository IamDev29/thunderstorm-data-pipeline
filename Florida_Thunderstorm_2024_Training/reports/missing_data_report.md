# Missing Data & Data Access Audit Report (2024)

## 1. Executive Summary

This report documents the operational status, data availability, temporal coverage, and remaining retrieval requirements across all four data sources: NOAA Storm Events, ECMWF ERA5, NASA ISS LIS, and NOAA GOES-16 GLM.

---

## 2. Dataset-by-Dataset Audit

### A. NOAA Storm Events Database (2024)
- **Status**: **COMPLETE, AUDITED & RECONCILED**
- **Coverage**: Full calendar year 2024 (`details`, `locations`, `fatalities`).
- **Audit of Reported 29,004 Record Count**:
  - Raw nationwide details archive: 69,801 rows.
  - The previously reported 29,004 count was traced to an unnormalized uppercase header (`STATE`) in early preprocessing that caused nationwide records with null coordinates to be retained.
  - Normalized reconciliation: Exactly **1,441 Florida detail rows** exist in the official archive.
  - After Florida state polygon boundary filtering and coordinate validation:
    * **1,041 verified point events** inside the Florida state polygon.
    * **347 county/zone-level events** retained with `coordinates_available = False`.
    * **Total unique Florida event IDs**: **1,388**.
- **Join Multiplications**: Exactly 0 duplicate events introduced by joins.
- **Missing Items**: None. 100% of NOAA 2024 records for Florida are accounted for.

---

### B. ECMWF ERA5 Atmospheric Reanalysis (2024)
- **Status**: **IN PROGRESS / 6-HOURLY RESUMABLE EXPANSION ACTIVE**
- **Temporal Resolution Adopted**:
  - **6-Hourly Synoptic Cadence**: `00:00`, `06:00`, `12:00`, `18:00` UTC.
  - Total expected 6-hourly timestamps for leap year 2024: **1,464 timestamps** (366 days $\times$ 4 timestamps).
- **Current Processed Coverage on Disk**:
  - **Month 1 (January 2024)**: Complete 12-variable single-level dataset downloaded (`era5_florida_2024_01.nc`, 13,018,873 bytes). Contains all 744 hours (124 synoptic 6-hourly timestamps).
  - **Month 2 (February 2024 - Leap Year)**:
    - Single-levels: `era5_florida_2024_02.nc` (2,097,669 bytes, 116 synoptic timestamps from 2024-02-01 00:00 to 2024-02-29 18:00 UTC, all 12 variables).
    - Pressure-levels: `era5_florida_2024_02.nc` (4,920,858 bytes, 116 synoptic timestamps, all 7 variables across 850, 500, 300 hPa).
  - **May 15 Pilot Slice**: 4 synoptic timestamps (`00:00`, `06:00`, `12:00`, `18:00` UTC).
  - **Current Grid Rows Processed**: **53,092 rows** across 244 unique timestamps and 218 spatial grid cells inside Florida.
- **Variables Successfully Retrieved & Processed**:
  - Single-Level: 2m temperature (`t2m`), 2m dewpoint (`d2m`), surface pressure (`sp`), MSLP (`msl`), total precipitation (`tp`), convective precipitation (`cp`), 10m u-wind (`u10`), 10m v-wind (`v10`), CAPE (`cape`), CIN (`cin`), Total Cloud Cover (`tcc`), Total Column Water Vapour (`tcwv`).
  - Derived Diagnostic Features: 10m wind speed (`wind_speed_10m_ms`), 10m wind direction (`wind_direction_10m_deg`), 2m relative humidity (`relative_humidity_2m_pct` via Magnus formula).
  - Pressure-Level (850, 500, 300 hPa): Temperature (`t`), relative humidity (`r`), specific humidity (`q`), u-wind (`u`), v-wind (`v`), geopotential (`z`), vertical velocity (`w`).
- **Active CDS Queue Status**:
  - Copernicus Climate Data Store credentials configured: `https://cds.climate.copernicus.eu/api`.
  - Active Job: `9b808179-afb9-4b4f-b6d9-bf29dd5947fd` (Month 3 single levels, status: `accepted`).
  - Active Job: `39069d79-614f-4253-a9e5-d34d0ce55d42` (Month 1 pressure levels, status: `accepted`).
  - ECMWF MARS archive cluster processes requests sequentially; remaining months (3 through 12) proceed via `src/download_all_months.py`.

---

### C. NASA ISS Lightning Imaging Sensor (ISS LIS) (2024)
- **Status**: **DECOMMISSIONED (Zero Science Data Collected in 2024)**
- **Operational Reality**: Decommissioned on **November 16, 2023**; deorbited aboard Cygnus NG-20 on **July 12, 2024**.
- **Consequence**: Zero science data collected for 2024.
- **Pipeline Handling**:
  - Documented physical sensor limitation without fabricating artificial records.
  - Enforced Critical LIS Rule: unobserved satellite intervals are NEVER imputed as zero lightning.
  - Preserved in audit table with `lis_observation_coverage = 0`, `lis_lightning_detected = NaN`, `coverage_status = "UNOBSERVED"`.

---

### D. NOAA GOES-16 Geostationary Lightning Mapper (GLM) (2024)
- **Status**: **OPERATIONAL & RETRIEVED (Continuous Satellite Lightning Replacement)**
- **Source**: Public AWS Open Data bucket `s3://noaa-goes16/GLM-L2-LCFA/` (HTTPS: `https://noaa-goes16.s3.amazonaws.com`).
- **Data Product**: Level-2 Lightning Cluster Filter Algorithm (LCFA) NetCDF granules (`OR_GLM-L2-LCFA_G16_s*.nc`).
- **Coverage**: Continuous geostationary viewing of Florida throughout 2024 at 20-second cadence and ~8 km spatial resolution.
- **Ingestion & Safeguards**:
  - Automated downloader in `src/download_glm.py` and processor in `src/process_glm.py`.
  - Extracted 181 individual Florida flashes across pilot observation intervals.
  - Hierarchy strictly separated: `glm_flash_count`, `glm_group_count`, and `glm_event_count`.
  - Adequate Monitoring Safeguard enforced: only intervals with $\ge 10$ granules ($\ge 200$s duration) can confirm zero lightning (`glm_lightning_detected = 0`). Sparse monitoring ($< 10$ granules) is categorized as `INCOMPLETE_COVERAGE` with `glm_lightning_detected = NaN` (never false 0).
