# Source Data Inventory Report (Pre-Execution Audit)

## 1. Overview

Audit of existing and retrieved data assets in the workspace for the Florida Thunderstorm 2024 Prediction Dataset.

---

## 2. File-by-File Inspection Table

| Source | File Name | Size (MB) | Geographic Bounds | Time Range | Reusable? | Assessment Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NOAA Storm Events** | `StormEvents_details_2024.csv.gz` | 5.30 | US Nationwide / Florida | Jan 1, 2024 to Dec 31, 2024 | **YES** | Official NOAA 2024 bulk details archive (reused from parent workspace). |
| **NOAA Storm Events** | `StormEvents_locations_2024.csv.gz` | 2.51 | US Nationwide / Florida | Jan 1, 2024 to Dec 31, 2024 | **YES** | Official NOAA 2024 locations archive. Joins on `event_id`. |
| **NOAA Storm Events** | `StormEvents_fatalities_2024.csv.gz` | 0.07 | US Nationwide / Florida | Jan 1, 2024 to Dec 31, 2024 | **YES** | Official NOAA 2024 fatalities archive. |
| **NOAA Storm Events** | `details_florida.csv` | 2.06 | 1 State (FLORIDA) | 04-FEB-24 to 11-SEP-24 | **YES** | Pre-filtered 1,441 Florida convective records. |
| **ECMWF ERA5 Reanalysis** | `era5_florida_2024_01.nc` | 12.42 | Lat: [24.50, 31.50], Lon: [-88.00, -79.50] | 2024-01-01 00:00 to 2024-01-31 23:00 UTC | **YES** | Complete 12-variable January 2024 single-level dataset (744 hours, 10 instant + 2 accum variables). |
| **ECMWF ERA5 Reanalysis** | `era5_florida_2024_02.nc` (Single-Level) | 2.00 | Lat: [24.50, 31.50], Lon: [-88.00, -79.50] | 2024-02-01 00:00 to 2024-02-29 18:00 UTC | **YES** | Complete 12-variable February 2024 leap year dataset (116 6-hourly synoptic timestamps). |
| **ECMWF ERA5 Reanalysis** | `era5_florida_2024_02.nc` (Pressure-Levels) | 4.69 | Lat: [24.50, 31.50], Lon: [-88.00, -79.50] | 2024-02-01 00:00 to 2024-02-29 18:00 UTC | **YES** | Complete 7-variable pressure-level dataset at 850, 500, 300 hPa (116 6-hourly synoptic timestamps). |
| **ECMWF ERA5 Reanalysis** | `era5_florida_2024_05.nc` | 0.12 | Lat: [24.50, 30.50], Lon: [-86.00, -80.50] | 2024-05-15 00:00 to 18:00 UTC | **YES** | Verified Florida synoptic pilot slice (4 synoptic timestamps). |
| **ECMWF ERA5 Reanalysis** | `test_cds_ping.nc` (Zip Bundle) | 0.17 | Lat: [24.50, 31.50], Lon: [-88.00, -79.50] | 2024-01-01 00:00 and 12:00 UTC | **YES** | Complete 12-variable single-level bundle (`t2m`, `d2m`, `sp`, `msl`, `tp`, `cp`, `cape`, `cin`, `tcc`, `u10`, `v10`, `tcwv`). |
| **NOAA GOES-16 GLM** | `OR_GLM-L2-LCFA_G16_s2024136*.nc` (12 files) | ~4.80 | Geostationary Full Disk / Florida | 2024-05-15 (00, 06, 12, 18 UTC) | **YES** | Level-2 LCFA NetCDF granules retrieved from AWS Open Data (`s3://noaa-goes16/GLM-L2-LCFA/`). |
| **Geographic GIS Boundary** | `florida_boundary.geojson` | 0.01 | Lat: [25.12, 31.00], Lon: [-87.63, -80.03] | Static authoritative GIS | **YES** | Florida state polygon (peninsula + panhandle) for boundary filtering. |

---

## 3. NASA ISS LIS Instrument Audit

- **Path**: `c:\Users\KIIT0001\Desktop\Data Collection\NASA LIS DATA`
- **Total Granules Found**: 8,814 files
- **Years Discovered in Archive**: 2022 and 2023 (concluding on Nov 16, 2023)
- **Covers Target Year 2024?**: **False**
- **Mission Operational Status**: Decommissioned on November 16, 2023; deorbited on July 12, 2024.
- **Finding**: Zero science data collected by ISS LIS in 2024. Replaced by continuous NOAA GOES-16 GLM geostationary observations.

---

## 4. NOAA GOES-16 GLM Satellite Audit

- **Data Bucket**: `s3://noaa-goes16/GLM-L2-LCFA/` (AWS Open Data)
- **Granule Cadence**: 20 seconds, ~8 km resolution at nadir
- **Variables**: `flash_lat`, `flash_lon`, `flash_energy`, `flash_area`, `group_lat`, `group_lon`, `event_lat`, `event_lon`
- **Relevance for Florida 2024**: Full, continuous coverage of the Florida peninsula and panhandle with zero overpass gaps.
