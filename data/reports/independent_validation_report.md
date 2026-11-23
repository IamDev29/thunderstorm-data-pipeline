# Independent Validation Report: NOAA Storm Events 2024 Pipeline & Dataset Audit

**Document Version:** 1.0  
**Audit Date:** 2026-10-24  
**Project:** Automated NOAA Storm Events Data Collection and Preprocessing Pipeline (Thunderstorm Project)  
**Auditor:** Independent Automated Audit Subsystem  
**Reference Site:** Chandipur, Odisha, India (21.46°N, 87.02°E)

---

## 1. Executive Validation Summary

An independent, end-to-end verification of all raw downloads, intermediate artifacts, filtered subsets, and report outputs produced by the NOAA Storm Events pipeline for calendar year 2024 was conducted.

### Core Audit Verdict: **VALIDATED & VERIFIED**
- **Raw File Integrity:** All 3 downloaded bulk CSV archive files (`details`, `locations`, `fatalities`) are 100% intact, non-empty, and successfully pass full gzip stream decompression tests without bit corruption or truncation.
- **Record Reconciliation:** Raw and cleaned details datasets contain exactly **69,801 rows** with **69,801 unique event IDs** (zero duplicates, zero missing event IDs).
- **Relational Integrity:** Clean details and locations datasets remain strictly decoupled. If a naive 1-to-many join had been executed, record counts would have falsely inflated from 69,801 to 85,896 (+23.06%). The pipeline correctly maintains separate relational tables linked by `event_id`.
- **Event-Type Filtering:** The 3,351-record convective subset (`thunderstorm_events_gulf_coast.csv`) was audited record-by-record and matches all 8 configured severe convective categories.
- **Geographic Filtering:** Two-stage filtering successfully isolates candidate Gulf Coast events while retaining records lacking point coordinates with `insufficient_coords = True`.

---

## 2. Quantitative Comparison: Actual vs. Previously Reported Outputs

| Dataset / Metric | File Path | Reported Count | Actual Verified Count | Discrepancy & Root Cause Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Details Archive** | `data/raw/details/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz` | ~12.7 MB | 12,693,243 bytes (66.63 MB decomp) | 0 bytes; fully verified gzip archive. |
| **Raw Locations Archive** | `data/raw/locations/StormEvents_locations-ftp_v1.0_d2024_c20260728.csv.gz` | ~983 KB | 983,723 bytes (3.06 MB decomp) | 0 bytes; fully verified gzip archive. |
| **Raw Fatalities Archive**| `data/raw/fatalities/StormEvents_fatalities-ftp_v1.0_d2024_c20260728.csv.gz` | ~13 KB | 13,057 bytes (0.08 MB decomp) | 0 bytes; fully verified gzip archive. |
| **Cleaned Details** | `data/processed/clean_details.csv` | 69,801 rows | 69,801 rows (66 columns) | 0; exactly 69,801 unique `event_id` records. |
| **Cleaned Locations** | `data/processed/clean_locations.csv` | 41,574 rows | 41,574 rows (15 columns) | 0; 25,479 unique events (1 to 8 locations/event). |
| **Cleaned Fatalities** | `data/processed/clean_fatalities.csv` | 1,141 rows | 1,141 rows (12 columns) | 0; 602 unique events. |
| **Florida Details** | `data/processed/florida/details_florida.csv` | 1,441 events | 1,441 events | 0; 1,094 valid coords, 347 zone-based records. |
| **Texas Details** | `data/processed/texas/details_texas.csv` | 3,283 events | 3,283 events | 0; 947 valid coords, 2,336 zone-based records. |
| **Louisiana Details** | `data/processed/louisiana/details_louisiana.csv` | 937 events | 937 events | 0; 544 valid coords, 393 zone-based records. |
| **Mississippi Details** | `data/processed/mississippi/details_mississippi.csv`| 717 events | 717 events | 0; 109 valid coords, 608 zone-based records. |
| **Sum of Regional Files** | Sum of above 4 state files | 6,378 events | 6,378 events | 0; sum of individual state extractions. |
| **Combined Gulf Coast** | `data/processed/combined/details_gulf_coast.csv` | 6,899 events | 6,899 events | **+521 events relative to sum of regional files.** (See Section 4). |
| **Convective Subset** | `data/processed/combined/thunderstorm_events_gulf_coast.csv` | 3,351 events | 3,351 events | 0; exactly matches the 8 target event types. |

---

## 3. Audit of Event Labels & Meteorological Categorization

The combined Gulf Coast dataset contains 6,899 total records spanning 31 distinct NOAA event types. The 3,351-record convective subset isolates the severe convective and associated hazards.

### Breakdown of the 3,351 Convective Subset Records:

| Event Type (NOAA Label) | Record Count | Meteorological Classification | Point Coordinates | Physical Relevance to Thunderstorm Research |
| :--- | :--- | :--- | :--- | :--- |
| **Thunderstorm Wind** | 1,325 | Direct Convective Observation | 1,325 / 1,325 (100% Valid) | Direct severe downdraft / microburst observation. Crucial ground-truth label for high-wind convective cells. |
| **Flash Flood** | 697 | Associated Convective Hazard | 697 / 697 (100% Valid) | Rapid-onset hydrometeorological impact resulting from localized extreme convective rainfall rates. |
| **Hail** | 579 | Direct Convective Observation | 579 / 579 (100% Valid) | Direct consequence of intense convective updrafts reaching above the freezing level. Unambiguous thunderstorm signature. |
| **Tornado** | 320 | Direct Convective Observation | 320 / 320 (100% Valid) | Severe rotating convective updraft (supercell dynamics). |
| **Tropical Storm** | 242 | Distinct Synoptic Phenomenon | 0 / 242 (0% Valid, Zone-Based) | Synoptic-scale tropical cyclone system. Contains embedded deep convection, but dynamics differ from isolated diurnal thunderstorms. |
| **Heavy Rain** | 101 | Associated Convective Hazard | 101 / 101 (100% Valid) | High-precipitation convective rain event without severe wind/hail criteria. |
| **Hurricane (Typhoon)** | 47 | Distinct Synoptic Phenomenon | 0 / 47 (0% Valid, Zone-Based) | Severe tropical cyclone. High wind/surge impacts driven by core pressure gradients, not standard diurnal instability. |
| **Lightning** | 40 | Direct Convective Observation | 40 / 40 (100% Valid) | Definitive physical signature of lightning-producing thunderclouds. |
| **TOTAL** | **3,351** | — | **3,062 Valid / 289 Zone** | All 8 target classes accounted for. |

### Critical Meteorological Audit Findings:
1. **Severe Damage Reporting Bias (The Lightning Anomaly):**
   - In 2024, only **40 Lightning events** are recorded across the entire US Gulf Coast in NOAA Storm Events.
   - *Explanation:* Under NWS Directive 10-1605, lightning is **only entered into Storm Events if it causes a fatality, direct injury, or significant structural damage**. Ordinary lightning flashes are not recorded. Therefore, NOAA Storm Events is a **damage impact catalog**, NOT an exhaustive meteorological flash count network. (Satellite sensors such as NASA ISS-LIS or ground networks like WWLLN/ENTLN must be used for physical flash counts).
2. **Convective vs. Synoptic Separation:**
   - **Tropical Storms (242)** and **Hurricanes (47)** represent synoptic-scale warm-core cyclones rather than localized diurnal thunderstorms. While they contain massive convective bands, they should be evaluated as a distinct subcategory when comparing with Chandipur's pre-monsoon convective regime.

---

## 4. Geographic Coverage & Bounding Box Assessment

### Analysis of the Bounding Boxes in `config.yaml`:
- **Florida Coastal Focus:** `[24.5°N, 31.0°N] × [-88.0°W, -80.0°W]`
- **Texas Coastal Focus:** `[25.8°N, 31.5°N] × [-98.5°W, -93.5°W]`
- **Louisiana Coastal Focus:** `[28.8°N, 31.0°N] × [-94.1°W, -89.0°W]`
- **Mississippi Coastal Focus:** `[30.1°N, 31.5°N] × [-90.0°W, -88.0°W]`
- **Combined Gulf Coast Bounding Box:** `[24.5°N, 32.0°N] × [-98.5°W, -80.0°W]`

### Explanation of the 521-Record Discrepancy:
- The regional state files use tighter, custom coastal focus boxes (e.g. Louisiana lat max 31.0°N, Mississippi lat max 31.5°N).
- The `combined_gulf_coast` bounding box extends up to **32.0°N**, capturing an additional **521 records** located in Central/Northern Louisiana, Central Mississippi, and Central Texas.
- **Recommendation:** Align the combined bounding box with the union of regional coastal focus polygons, or parameterize an explicit coastal distance threshold (e.g., within 100 km of Gulf shoreline).

### Point Coordinates vs. Zone-Based Records:
- **County-Level Events (`cz_type == 'C'`):** **100% of events (3,215 of 3,215) possess valid, verified geographic coordinates.**
- **NWS Forecast Zone Events (`cz_type == 'Z'`):** **100% of zone events (3,684 of 3,684) lack point coordinates** because NWS issues them for entire forecast zones (e.g., heat waves, droughts, hurricanes).
- **Audit Compliance:** In accordance with project requirements, **zero records were silently dropped**. All 3,684 zone-level events are retained and explicitly flagged with `insufficient_coords = True`.

---

## 5. Visual Artifacts Produced

The following validation figures have been generated and saved:
1. `data/reports/figures/geographic_coverage_validation_map.png`: Publication-grade map showing 3,062 individual convective event coordinates, regional bounding boxes, and the Chandipur, Odisha reference inset.
2. `data/reports/figures/convective_events_by_state.png`: Stacked bar chart showing the composition of severe convective hazards across Florida, Texas, Louisiana, and Mississippi.
3. `data/reports/figures/event_counts_by_year_state.png`: Historical comparison figure.
4. `data/reports/figures/event_types_distribution.png`: Frequency distribution of convective event types.
5. `data/reports/figures/monthly_seasonality.png`: Monthly event seasonality.

---

## 6. Proposed ERA5 Data Collection & Climate Similarity Plan

To determine which US Gulf Coast candidate region is meteorologically most similar to **Chandipur, Odisha, India** (`21.46°N, 87.02°E`), atmospheric reanalysis data from ECMWF ERA5 must be utilized as the common physical ground truth.

### Required Meteorological Variables (ERA5 Single Levels & Pressure Levels):
1. **Thermodynamic Instability:**
   - CAPE (Convective Available Potential Energy, J/kg)
   - CIN (Convective Inhibition, J/kg)
   - K-Index & Total Totals Index (derived from pressure levels)
2. **Moisture Availability:**
   - Total Column Water Vapour (TCWV, kg/m²)
   - 2-metre Dewpoint Temperature ($T_d$, K)
   - Specific / Relative Humidity at 850 hPa, 700 hPa, and 500 hPa
3. **Kinematic & Dynamic Forcing:**
   - 10-metre U and V wind components (m/s)
   - Deep-layer vertical wind shear (0–6 km bulk shear calculated between surface/10m and 500 hPa)
   - Low-level jet (LLJ) wind speed at 850 hPa
4. **Hydrometeorological Surface Diagnostics:**
   - Total Precipitation & Convective Precipitation (m)
   - Surface sensible and latent heat fluxes ($W/m^2$)
   - 2-metre Temperature ($T_{2m}$, K) and Sea Surface Temperature (SST, K)

### Spatial & Temporal Configuration:
- **Spatial Resolution:** Native ERA5 grid at 0.25° × 0.25° (~28–31 km).
- **Target Spatial Domains:**
  - **Chandipur Domain:** `[20.0°N, 23.0°N] × [85.5°E, 88.5°E]` (Centred on Chandipur coast).
  - **Texas Gulf Coast Domain:** `[25.5°N, 30.5°N] × [-98.0°W, -93.5°W]`.
  - **Louisiana / Mississippi Domain:** `[28.5°N, 31.5°N] × [-93.5°W, -88.0°W]`.
  - **Florida Gulf Coast Domain:** `[24.5°N, 30.5°N] × [-86.0°W, -80.5°W]`.
- **Temporal Horizon:** 10 complete calendar years (e.g. 2014–2023 or 2016–2025) sampled at 1-hourly or 3-hourly intervals.

### Seasonal Comparison Strategy:
Convective regimes in Chandipur follow a distinct seasonal cycle that must be matched with dynamical analogues in the US:
1. **Pre-Monsoon Season (March – May, MAM):**
   - *Chandipur Phenomenon:* Characterized by intense, severe localized storms called **Kalbaisakhi** (*Nor'westers*). Driven by extremely high CAPE (>2,500 J/kg), dry continental air from the Chota Nagpur plateau overlying maritime Bay of Bengal moisture, and strong vertical wind shear.
   - *US Analogue:* Spring severe thunderstorm season in the Western Gulf Coast / Texas / Louisiana coastal plain (continental dryline / frontal boundary interaction with Gulf moisture).
2. **Monsoon Season (June – September, JJAS):**
   - *Chandipur Phenomenon:* Sustained high moisture (TCWV >60 kg/m²), moderate CAPE, near-zero CIN, low vertical shear, high cloud cover, and frequent tropical monsoon depressions.
   - *US Analogue:* Subtropical summer sea-breeze convection in the Florida Peninsula and Gulf Coast (maritime tropical airmass, diurnal pulse storms, low shear).
3. **Post-Monsoon / Cyclonic Transition (October – November, ON):**
   - *Chandipur Phenomenon:* Severe cyclonic storms in the Bay of Bengal.
   - *US Analogue:* Late-season Gulf of Mexico tropical cyclones / hurricanes.

### Retrieval Infrastructure Status:
- The `cdsapi` client is **installed and verified** in the local Python environment.
- A valid Copernicus CDS configuration file (`C:\Users\KIIT0001\.cdsapirc`) is **already configured and authenticated** against `https://cds.climate.copernicus.eu/api`.

---

## 7. Unresolved Issues & Recommended Next Steps

1. **Polygon-Based Coastal Clipping:**
   - Rectangular bounding boxes in Texas capture some inland continental areas (e.g., up to Austin/College Station).
   - *Next Step:* Implement an optional distance-to-coast filter (e.g., NOAA shoreline shapefile or Haversine distance $\le 100\text{ km}$) to isolate strictly coastal plain events.
2. **Separate Convective vs. Tropical Cyclone Datasets:**
   - In the convective subset, 289 events are Tropical Storms and Hurricanes (`cz_type == 'Z'`), while 3,062 events are localized convective storms (`cz_type == 'C'`).
   - *Next Step:* Create an isolated `pure_thunderstorm_events_gulf_coast.csv` (excluding TC records) alongside the combined convective dataset.
3. **Automated Multi-Year Bulk Fetching:**
   - Once the 2024 baseline is approved, execute batch downloads for the remaining 9 complete years (2015–2023) using the built-in HTTP Range resume downloader.
