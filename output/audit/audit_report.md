# NASA ISS-LIS Scientific Audit and Validation Report

## Executive Summary

An independent scientific audit of the NASA ISS-LIS lightning dataset and processing pipeline was conducted across all **4,407 NetCDF granules** for calendar year 2023 over Chandipur, Odisha, India.

### Key Audit Verdicts

| Audit Domain | Initial Pipeline Report | Audited & Corrected Verdict | Status |
|---|---|---|---|
| **Chandipur Detections** | 8 flashes | **8 flashes independently verified** in raw granules | **PASS** |
| **TAI93 Timestamps** | 10.0s offset | **Exact constant 10.0s leap second offset** verified across all granules | **PASS** |
| **Satellite Overpasses** | 310 viewing windows | **295 exact Chandipur overpasses** (15 buffer-only excluded) | **WARNING / CORRECTED** |
| **Positive Labels (`1`)** | 2 overpasses | **2 overpasses** (March 21 & Sept 29, 2023) | **PASS** |
| **Negative Labels (`0`)** | 308 overpasses | **247 robust negative overpasses** (duration ≥ 30s) | **CORRECTED** |
| **Unknown Labels (`-1`)** | 0 overpasses | **61 ambiguous overpasses** (<30s duration or buffer only) | **CORRECTED** |
| **Wider Local Region** | N/A | **467 flashes across 25 dates** (21.0-22.0N, 86.5-88.0E) | **PASS** |

## 1. Independent Verification of the 8 Chandipur Flashes

Every one of the 8 Chandipur flash records was traced back directly to its source NetCDF granule:

- **Granule 1 (`ISS_LIS_SC_V3.0_20230321_100444_FIN.nc`)**: Orbit 100444
  - Orbit start: `2023-03-21T10:04:44.400000Z`
  - Flash 731: `2023-03-21T10:35:24.798388Z` (`16:05:24 IST`), Lat 21.451052°N, Lon 87.194923°E, Radiance 797,313 uJ, 23 groups, 78 events
  - Flash 734: `2023-03-21T10:35:59.314669Z` (`16:05:59 IST`), Lat 21.401718°N, Lon 87.155594°E, Radiance 140,448 uJ, 9 groups, 14 events
  - Meteorological event: Pre-monsoon Kalbaisakhi (Nor'wester) afternoon squall line storm.

- **Granule 2 (`ISS_LIS_SC_V3.0_20230929_145717_FIN.nc`)**: Orbit 145717
  - Orbit start: `2023-09-29T14:57:17.200000Z`
  - Flashes 443, 450, 451, 452, 460, 462 occurred between `15:59:42 UTC` and `16:00:54 UTC` (`21:29:42 IST` to `21:30:54 IST`).
  - Latitudes: 21.3028°N to 21.4834°N, Longitudes: 87.1166°E to 87.2324°E.
  - Radiances: 34,935 to 501,324 uJ, child groups: 1 to 11, events: 7 to 57.
  - Meteorological event: Late monsoon post-sunset coastal thunderstorm.

> [!NOTE]
> All 8 flashes fall strictly inside the primary bounding box [21.25, 21.50]°N, [87.00, 87.25]°E, and within their parent granule orbit time bounds. No duplicate records exist.

## 2. Why Are There Only 8 Chandipur Detections?

A comprehensive spatial sensitivity comparison was performed across three nested geographic domains:

| Domain | Bounding Box | Area (km²) | Flashes | Distinct Dates | Granules |
|---|---|---|---|---|---|
| **Primary Chandipur Box** | 21.25–21.50°N, 87.00–87.25°E | 729 km² | **8** | 2 | 2 |
| **Wider Local Box** | 21.00–22.00°N, 86.50–88.00°E | 17,050 km² | **467** | **25** | **26** |
| **Regional Odisha & Bay** | 17.50–23.50°N, 81.00–89.00°E | 550,000 km² | **8,854** | **140** | **181** |

### Scientific Reasons for the Low Count in the Primary Box:
1. **Spatial Scale Mismatch**: The primary box is tiny (27 km × 27 km = 729 km²). Mesoscale thunderstorm systems span 50–200 km.
2. **Orbital Sampling Intermittency**: The ISS passed over Chandipur only 310 times in 2023, each overpass lasting ~90 seconds. Total annual observation time over Chandipur was ~6.5 hours out of 8,760 hours in the year (<0.08% temporal duty cycle).
3. **Atmospheric Climatology**: In coastal Odisha, thunderstorms occur almost exclusively during pre-monsoon (March–May) and late monsoon (September–October). Dry winter months have zero lightning activity.
4. **Wider Local Significance**: Expanding the study region slightly to a 1.0° × 1.5° box (which aligns with 4×6 grid cells of ERA5 0.25° reanalysis) yields **467 flashes on 25 distinct thunderstorm dates**, with an afternoon convective peak at **16:00 IST** (214 flashes).

## 3. TAI93 to UTC & IST Timestamp Audit

- **TAI93 Definition**: Seconds since `1993-01-01 00:00:00 TAI`.
- **Leap Second Verification**: Between 1993-01-01 and 2023-12-31, exactly 10 leap seconds were inserted (the last in Dec 2016). No leap seconds occurred in 2023.
- **Offset Stability**: `orbit_summary_TAI93_start - (orbit_summary_UTC_start - 1993-01-01 UTC) = 10.000000s` identically across all 4,407 granules.
- **Conversion Formula Verified**: `UTC = datetime(1993, 1, 1, tz=UTC) + timedelta(seconds=(TAI93 - 10.0))`.
- **IST Conversion**: `UTC + 05:30` using `Asia/Kolkata` with full microsecond preservation.

## 4. Satellite Coverage & Scientifically Defensible Labeling

The initial pipeline classified all 308 overpasses with 0 flashes as verified negative labels. The audit revealed that this was overly optimistic:

1. **15 overpasses were buffer-only**: The satellite swath covered adjacent 0.5° cells (e.g. 20.75° or 21.75°) but never entered the Chandipur cell (21.25N, 87.25E). These are now classified as **Unknown (-1)**.
2. **46 overpasses had effective duration < 30.0s**: Observation intervals of 1 to 20 seconds are too brief to rule out an active thunderstorm, as lightning flashes can occur several minutes apart. These are reclassified as **Unknown (-1)**.
3. **247 overpasses had sustained coverage (≥ 30.0s, avg 94.6s)**: These represent robust, defensible **Negative labels (0)**.
4. **2 overpasses contained detected lightning**: **Positive labels (1)**.

### Corrected Overpass Label Distribution

| Label Value | Meaning | Overpass Count | Scientific Justification |
|---|---|---|---|
| **`1` (Positive)** | Confirmed Thunderstorm | **2** | Flash detections verified in Chandipur during overpass |
| **`0` (Negative)** | Verified Non-Thunderstorm | **247** | Sustained viewing (≥30s, avg 94.6s), clean telemetry, zero lightning |
| **`-1` (Unknown)** | Ambiguous / Insufficient | **61** | Grazing overpasses (<30s) or spatial buffer-only exclusions |

## 5. ERA5 Integration Readiness

The dataset is **READY** for integration with ECMWF ERA5 reanalysis data:

1. Verified Lightning Events: [`output/audit/corrected/chandipur_lightning_events.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/output/audit/corrected/chandipur_lightning_events.csv)
2. Validated Observation Windows (with 1, 0, -1 labels): [`output/audit/corrected/chandipur_observation_windows.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/output/audit/corrected/chandipur_observation_windows.csv)
3. Detailed Overpass Coverage Audit: [`output/audit/coverage_audit.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/output/audit/coverage_audit.csv)
4. Timestamp Verification Audit: [`output/audit/timestamp_audit.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/output/audit/timestamp_audit.csv)
5. Regional Sensitivity Dataset: [`output/audit/regional_comparison.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/output/audit/regional_comparison.csv)

> [!TIP]
> **Recommendation for ML Model Training**:
> Train the ERA5 machine-learning model on the **249 high-confidence observation windows** (`is_usable_for_ml_training == 1`: 2 positive + 247 robust negative). Exclude the 61 ambiguous windows (`assigned_label == -1`) from the primary loss function to avoid false negative label noise.
