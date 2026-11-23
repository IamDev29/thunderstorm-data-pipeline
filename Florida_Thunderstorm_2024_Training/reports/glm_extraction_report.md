# GOES-16 GLM Flash Extraction Pipeline & Audit Report
**Florida Thunderstorm 2024 Research Project**  
**Date:** October 27, 2026  
**Authoritative Dataset:** `data/processed/glm_pilot_extracted/florida_glm_flash_observations_2024.parquet`  
**Granule Audit Log:** `data/processed/glm_pilot_extracted/glm_extraction_audit.csv`  
**Pipeline Script:** `Florida_Thunderstorm_2024_Training/src/extract_glm_observations.py`  
**Verification Suite:** `Florida_Thunderstorm_2024_Training/tests/test_glm_extraction_reproducibility.py`

---

## 1. Executive Summary

This report documents the implementation, audit, and empirical verification of the scientifically grounded GOES-16 Geostationary Lightning Mapper (GLM) Level-2 Lightning Cluster Filter Algorithm (LCFA) extraction pipeline for the Florida Thunderstorm 2024 forecasting research repository.

The pipeline replaces the initial pilot extraction (`process_glm.py`), which suffered from three major scientific limitations:
1. **Timestamp Collapse:** Flash timestamps were previously truncated to coarse hourly or synoptic synch points (`floor("h")`), discarding physical microsecond-level timing and making strict half-open interval evaluation $(t, t+6\text{h}]$ impossible.
2. **Synthetic Multipliers:** Physical groups and events were never extracted from the NetCDF arrays; instead, synthetic multipliers (`flash * 12` groups, `flash * 28` events) were artificially applied.
3. **Offshore Discard & Grid Misalignment:** 15 coastal/terrestrial flashes were dropped during merging due to a historical 193-cell grid mask that omitted 25 active Florida terrestrial cells (notably cell `FL_26.5_-80.25`, which contained 7 flashes).

### Summary of Extraction Results across the 12 Pilot Granules (May 15, 2024)

| Metric | Raw Observations | Filtered / Accepted | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Total GLM Granules Evaluated** | 12 files | 12 files | 100% processed successfully (0% uncorrectable L0 errors) |
| **Total Raw Flash Records** | 2,658 | — | Evaluated from raw NetCDF arrays |
| **Rejected: Out of Domain Bounding Box** | 1,970 | — | Outside $[24.5, 31.5]^\circ\text{N}, [-88.0, -79.5]^\circ\text{W}$ |
| **Rejected: Outside Florida Polygon** | 507 | — | Gulf of Mexico, Atlantic Ocean, Georgia, Alabama |
| **Accepted Florida Polygon Flashes** | **181** | **181** | **100% spatially contained in Florida state boundary** |
| — *Active Terrestrial Grid Cells (218)* | — | **173** | Snapped to active Florida terrestrial forecast cells |
| — *Coastal Marine Cells (Offshore)* | — | **8** | Preserved with `in_active_terrestrial_grid=False` |
| **Measured Constituent Groups** | 71,940 raw | **4,534** hierarchy<br>*(4,480 spatial)* | **Measured directly** (mean: **25.05** groups/flash vs 12.0 synthetic) |
| **Measured Constituent Events** | 148,879 raw | **9,243** hierarchy<br>*(9,073 spatial)* | **Measured directly** (mean: **51.07** events/flash vs 28.0 synthetic) |
| **Duplicate Flash Records** | 0 | 0 | 0 within-granule; 0 cross-granule duplicates |
| **Quality Flags (Acceptance: 100%)** | 181 valid | 172 good, 9 degraded | 9 degraded due to high event count threshold in deep convection |

---

## 2. Sensor Parameters, Granule Inventory & Provenance

The 12 raw pilot granules represent GOES-16 GLM-L2-LCFA products observing the full disk with high sensitivity to total lightning (in-cloud and cloud-to-ground):
- **Platform:** GOES-16 (`G16`), Flight Model 1 (`FM1`), Geostationary East ($75.2^\circ\text{W}$).
- **Instrument:** Geostationary Lightning Mapper (GLM).
- **Nominal Granule Cadence:** Exactly 20.0 seconds per granule.
- **Sampling Bursts:** 3 consecutive 20-second granules sampled at each 6-hour synoptic cycle:
  - Burst 00:00 UTC (Granules s20241360000000 to s20241360001000): 60 seconds coverage.
  - Burst 06:00 UTC (Granules s20241360600000 to s20241360601000): 60 seconds coverage.
  - Burst 12:00 UTC (Granules s20241361200000 to s20241361201000): 60 seconds coverage.
  - Burst 18:00 UTC (Granules s20241361800000 to s20241361801000): 60 seconds coverage.
  - **Total Monitored Duration:** 240.0 seconds (4.0 minutes) across May 15, 2024.

### Granule-by-Granule Extraction Audit

| Granule File Name | Coverage Start (UTC) | Raw Flashes | BBox Flashes | FL Poly Flashes | Active Land Flashes | Coastal Marine Flashes | Measured Groups | Measured Events | L0 Uncorr. Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `OR_GLM-L2-LCFA_G16_s20241360000000_...nc` | 2024-05-15 00:00:00 | 280 | 17 | 6 | 5 | 1 | 70 | 125 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241360000200_...nc` | 2024-05-15 00:00:20 | 286 | 21 | 6 | 6 | 0 | 83 | 176 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241360000400_...nc` | 2024-05-15 00:00:40 | 291 | 24 | 8 | 8 | 0 | 76 | 174 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600000_...nc` | 2024-05-15 06:00:00 | 166 | 10 | 3 | 3 | 0 | 101 | 192 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600200_...nc` | 2024-05-15 06:00:20 | 183 | 18 | 6 | 6 | 0 | 166 | 340 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600400_...nc` | 2024-05-15 06:00:40 | 163 | 17 | 6 | 6 | 0 | 251 | 553 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200000_...nc` | 2024-05-15 12:00:00 | 185 | 114 | 31 | 31 | 0 | 900 | 1,719 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200200_...nc` | 2024-05-15 12:00:20 | 181 | 125 | 31 | 31 | 0 | 955 | 2,130 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200400_...nc` | 2024-05-15 12:00:40 | 186 | 125 | 32 | 31 | 1 | 1,038 | 2,169 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800000_...nc` | 2024-05-15 18:00:00 | 245 | 73 | 16 | 14 | 2 | 314 | 584 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800200_...nc` | 2024-05-15 18:00:20 | 249 | 71 | 19 | 17 | 2 | 226 | 400 | 0.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800400_...nc` | 2024-05-15 18:00:40 | 243 | 73 | 17 | 15 | 2 | 354 | 681 | 0.0% |
| **Total / Summary** | — | **2,658** | **688** | **181** | **173** | **8** | **4,534** | **9,243** | **0.0%** |

---

## 3. Timestamp Reconstruction & Microsecond UTC Precision

### The Previous Defect
The previous `process_glm.py` pipeline assigned all flashes to `hour_dt = dt_start.floor("h")`, meaning that:
- Every flash detected at 00:00:15.342 UTC was collapsed to `2024-05-15 00:00:00`.
- Flashes detected during the 00:00:40 granule were indistinguishable in time from flashes detected at 00:00:02.
- Flashes occurring exactly at interval boundary $t$ could not be rigorously separated according to the strict $(t, t+6\text{h}]$ interval rule required by `lightning_ground_truth_spec.md`.

### The Reconstructed Microsecond Formula
In the GLM LCFA product, the time variables have the following properties:
- `flash_time_offset_of_first_event` has the NetCDF attribute `units: 'seconds since YYYY-MM-DD HH:MM:SS.000'`.
- The `netCDF4` library automatically applies the scale factor (`0.0003814756`) and add offset (`-5.0`), yielding the exact offset in floating-point seconds relative to the base reference time.
- The true physical observation timestamp is derived as:
  $$\text{flash\_timestamp\_utc} = \text{pd.to\_datetime}(\text{base\_time}) + \text{pd.to\_timedelta}(\text{offset\_seconds}, \text{unit}='\text{s}')$$
- Similarly, `flash_time_offset_of_last_event` provides the completion time of the flash, enabling exact measurement of physical flash duration:
  $$\text{flash\_duration\_ms} = (\text{flash\_timestamp\_last\_utc} - \text{flash\_timestamp\_utc}) \times 1000.0$$

### Verified Empirical Timestamp Properties
- **Earliest Flash Timestamp:** `2024-05-14 23:59:58.837644+00:00` (flash initiated 1.16 seconds prior to nominal 00:00:00 boundary).
- **Latest Flash Timestamp:** `2024-05-15 18:00:59.327081+00:00`.
- **Physical Flash Durations:**
  - Minimum: $0.0\text{ ms}$ (single-event transient).
  - Median: $270.5\text{ ms}$.
  - Maximum: $3,133.1\text{ ms}$ (extended multi-group propagating discharge).
- **Sub-Second Dispersion:** Over 170 distinct sub-second fractional timestamps across the 181 flashes.
- **Zero Synoptic Collapse:** Exactly 0 of 181 flashes equal the exact synoptic boundary times.

---

## 4. Physical Group and Event Hierarchies vs Synthetic Multipliers

### Elimination of Heuristic Multipliers
The legacy code utilized hardcoded constants:
```python
agg["glm_group_count"] = agg["glm_flash_count"] * 12
agg["glm_event_count"] = agg["glm_flash_count"] * 28
```
This violated the fundamental Lightning Observation Constraint that prohibits conflating or scaling observation levels.

### Physical NetCDF Tree Hierarchy
The LCFA algorithm builds a strict three-tier hierarchy:
$$\text{Event} \xrightarrow{\text{event\_parent\_group\_id}} \text{Group} \xrightarrow{\text{group\_parent\_flash\_id}} \text{Flash}$$
Our pipeline traverses this tree directly from the raw NetCDF arrays:
1. Every group record is mapped to its parent `flash_id`.
2. Every event record is mapped to its parent `group_id`.
3. For each flash, `group_count_measured` and `event_count_measured` are computed as the exact number of constituent child groups and events.

### Comparative Hierarchy Statistics

| Metric | Legacy Synthetic Formula | Measured Hierarchy (Flash Parentage) | Measured Spatial Centroid |
| :--- | :--- | :--- | :--- |
| **Total Groups in FL Flashes** | $181 \times 12 = \mathbf{2,172}$ | **4,534** | 4,480 |
| **Total Events in FL Flashes** | $181 \times 28 = \mathbf{5,068}$ | **9,243** | 9,073 |
| **Mean Groups per Flash** | 12.00 (fixed) | **25.05** ($\pm 14.8$) | — |
| **Mean Events per Flash** | 28.00 (fixed) | **51.07** ($\pm 38.2$) | — |
| **Mean Events per Group** | 2.33 (implied) | **2.04** | — |
| **Maximum Groups in a Single Flash** | — | **84 groups** | — |
| **Maximum Events in a Single Flash** | — | **217 events** | — |

The legacy multipliers underestimated physical lightning group counts by **52.1%** and event counts by **45.2%**. The improved pipeline records exact physical measurements for every observation.

---

## 5. Spatial Partition & Reconciliation of Reported Counts

### The 181 vs 173 vs 166 Flash Discrepancy Resolved
Previous audit reports noted a discrepancy between 181 Florida polygon flashes and 166 flashes in the processed dataset:
1. **181 Florida Polygon Flashes:** 181 flash centroids fall strictly within the official Florida state polygon boundary (`florida_boundary.geojson`).
2. **The 193-Cell Historical Grid Bug (166 flashes):** In the early pilot, `target_grid` on May 15 only contained 193 grid cells. When `df_flashes` was joined to `target_grid`, 25 active Florida terrestrial cells were omitted. One of those omitted cells—`FL_26.5_-80.25` in Palm Beach County—contained **7 flashes** at 18:00 UTC. Omitting this active land cell reduced the count from 173 to 166 ($173 - 7 = 166$).
3. **The True 218 Active Terrestrial Grid (173 flashes):** Matching the 181 flashes against the authoritative 218 active Florida terrestrial grid cells confirms that **173 flashes** belong to active terrestrial cells.
4. **Coastal Marine Flashes (8 flashes):** The remaining 8 flashes have centroids inside the state polygon (barrier islands, sounds, coastal waters) but snap to 0.25-degree grid centers located over water outside the 218 terrestrial land mask.

### Ledger of Coastal Marine Flashes

| Flash ID | Timestamp (UTC) | Latitude | Longitude | Grid Cell ID | Location Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1024` | 2024-05-15 00:00:15.823 | $28.1848^\circ\text{N}$ | $-80.5956^\circ\text{W}$ | `COASTAL_MARINE_28.25_-80.5` | Cape Canaveral / Melbourne offshore |
| `988` | 2024-05-15 12:00:53.642 | $28.8898^\circ\text{N}$ | $-80.8572^\circ\text{W}$ | `COASTAL_MARINE_29.0_-80.75` | New Smyrna Beach coastal waters |
| `1115` | 2024-05-15 18:00:13.487 | $26.9818^\circ\text{N}$ | $-80.0898^\circ\text{W}$ | `COASTAL_MARINE_27.0_-80.0` | Jupiter Inlet coastal zone |
| `1121` | 2024-05-15 18:00:18.914 | $27.5276^\circ\text{N}$ | $-80.3122^\circ\text{W}$ | `COASTAL_MARINE_27.5_-80.25` | Hutchinson Island / Fort Pierce coast |
| `1134` | 2024-05-15 18:00:23.704 | $27.5108^\circ\text{N}$ | $-80.3132^\circ\text{W}$ | `COASTAL_MARINE_27.5_-80.25` | Hutchinson Island / Fort Pierce coast |
| `1142` | 2024-05-15 18:00:28.118 | $27.5030^\circ\text{N}$ | $-80.3477^\circ\text{W}$ | `COASTAL_MARINE_27.5_-80.25` | Hutchinson Island / Fort Pierce coast |
| `1158` | 2024-05-15 18:00:46.035 | $27.5084^\circ\text{N}$ | $-80.2988^\circ\text{W}$ | `COASTAL_MARINE_27.5_-80.25` | Hutchinson Island / Fort Pierce coast |
| `1167` | 2024-05-15 18:00:54.120 | $27.5151^\circ\text{N}$ | $-80.3224^\circ\text{W}$ | `COASTAL_MARINE_27.5_-80.25` | Hutchinson Island / Fort Pierce coast |

### Handling Rule
Rather than discarding offshore flashes or forcing unphysical snapping to distant terrestrial cells:
- All 181 flashes are preserved in `florida_glm_flash_observations_2024.parquet`.
- `in_florida_polygon = True` for all 181 records.
- `in_active_terrestrial_grid = True` for the 173 terrestrial records.
- `in_active_terrestrial_grid = False` for the 8 coastal marine records.
- Terrestrial forecasting models can filter on `in_active_terrestrial_grid == True` without data corruption.

---

## 6. Duplicate Investigation & Quality Flags

### Duplicate Analysis
1. **Within-Granule Duplication:** Every granule was evaluated for duplicate `flash_id` entries. Exactly **0 within-granule duplicates** were detected across all 2,658 raw flashes.
2. **Cross-Granule Duplication:** Flashes occurring near the 20-second boundary of consecutive granules were examined for duplicate coordinates `(latitude, longitude)` and identical timestamps `flash_timestamp_utc`. Exactly **0 duplicate flashes** were found across adjacent granules.
3. **Conclusion:** GLM LCFA granules do not replicate flash entities across 20-second granule boundaries. Each flash record in a granule represents a unique physical clustering entity.

### Flash Quality Flags
Across all 181 extracted Florida flashes:
- **Flag 0 (`good_quality_qf`):** 172 flashes (95.03%). High-confidence flash clustering meeting all nominal timing and event-count thresholds.
- **Flag 3 (`degraded_due_to_flash_constituent_event_count_exceeds_threshold_qf`):** 9 flashes (4.97%). Highly energetic convective flashes where the number of constituent optical events exceeded the standard clustering cap.
- **Flag 1 & Flag 5:** 0 flashes.
- **Handling:** All 181 flashes represent verified physical lightning. Flag 3 records are retained and flagged so models or ablation studies can evaluate quality-filtered vs total lightning subsets.

---

## 7. Deliverables & Data Schema

The following authoritative files have been generated:
1. `data/processed/glm_pilot_extracted/florida_glm_flash_observations_2024.parquet` (181 rows, 23 columns)
2. `data/processed/glm_pilot_extracted/florida_glm_flash_observations_2024.csv` (181 rows, 23 columns)
3. `data/processed/glm_pilot_extracted/glm_extraction_audit.csv` (12 rows, 18 columns)
4. `data/processed/glm_pilot_extracted/glm_rejection_audit.csv` (2,477 rows, 5 columns)

### Schema of `florida_glm_flash_observations_2024.parquet`

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `source_granule_id` | `string` | NetCDF granule file name (e.g., `OR_GLM-L2-LCFA_G16_s2024136...nc`) |
| `flash_id` | `int64` | Native integer identifier from NetCDF `flash_id` array |
| `flash_timestamp_utc` | `timestamp[ns, UTC]` | Exact UTC observation start timestamp (sub-second precision) |
| `flash_timestamp_last_utc` | `timestamp[ns, UTC]` | Exact UTC observation end timestamp of final constituent event |
| `flash_duration_ms` | `float64` | Measured physical duration of flash in milliseconds |
| `latitude` | `float64` | Flash centroid latitude in decimal degrees |
| `longitude` | `float64` | Flash centroid longitude in decimal degrees |
| `grid_lat` | `float64` | Nearest 0.25-degree grid latitude |
| `grid_lon` | `float64` | Nearest 0.25-degree grid longitude |
| `in_florida_bbox` | `bool` | True if inside $[24.5, 31.5]^\circ\text{N}, [-88.0, -79.5]^\circ\text{W}$ |
| `in_florida_polygon` | `bool` | True if strictly inside Florida state polygon |
| `in_active_terrestrial_grid` | `bool` | True if grid coordinate is in the 218 Florida terrestrial cells |
| `grid_cell_id` | `string` | e.g. `FL_28.0_-82.0` or `COASTAL_MARINE_28.25_-80.5` |
| `flash_energy_joules` | `float64` | Calibrated optical radiant energy in Joules ($J$) |
| `flash_area_km2` | `float64` | Footprint area of flash in square kilometers ($\text{km}^2$) |
| `flash_quality_flag` | `int64` | GLM quality flag integer (0 = good, 3 = degraded event count) |
| `flash_quality_meaning` | `string` | Human-readable quality description |
| `group_count_measured` | `int64` | Measured count of constituent groups in this flash |
| `event_count_measured` | `int64` | Measured count of constituent optical events in this flash |
| `platform_id` | `string` | Satellite platform (`G16`) |
| `instrument_id` | `string` | Instrument identifier (`FM1`) |
| `dataset_name` | `string` | NetCDF dataset name attribute |
| `audit_status` | `string` | Verification audit status (`VALID_OBSERVATION`) |

---

## 8. Reproducibility Test Verification

All unit and integration tests have been implemented and passed:

```
collected 8 items
tests/test_glm_extraction_reproducibility.py::test_01_exact_florida_flash_counts PASSED [ 12%]
tests/test_glm_extraction_reproducibility.py::test_02_rejection_accounting_integrity PASSED [ 25%]
tests/test_glm_extraction_reproducibility.py::test_03_timestamp_subsecond_precision PASSED [ 37%]
tests/test_glm_extraction_reproducibility.py::test_04_measured_group_and_event_hierarchy PASSED [ 50%]
tests/test_glm_extraction_reproducibility.py::test_05_quality_flags_audit PASSED [ 62%]
tests/test_glm_extraction_reproducibility.py::test_06_zero_cross_granule_duplicates PASSED [ 75%]
tests/test_glm_extraction_reproducibility.py::test_07_parquet_and_csv_parity PASSED [ 87%]
tests/test_glm_extraction_reproducibility.py::test_08_granule_audit_integrity PASSED [100%]

============================== 8 passed in 1.67s ==============================
```

Additionally, all 15 specification and contract tests in `test_lightning_ground_truth_spec.py` and `test_glm.py` passed with 100% success rate.

---

## 9. Recommendations for Future Full-Year Ingestion

When downloading and processing additional GOES-16 GLM data in future project phases:
1. **Streaming Processing:** Use the pipeline's per-granule streaming processing pattern (`process_granule`), which consumes $< 15\text{ MB}$ RAM per file and releases file handles immediately. Do not accumulate open NetCDF datasets in memory.
2. **Partitioned Storage:** Save full-year flash observations in daily or monthly Parquet partitions (`year=2024/month=XX/day=YY.parquet`) with standard zstd compression.
3. **Temporal Evaluation:** When evaluating $(t, t+6\text{h}]$ forecasting windows, join flashes directly on `flash_timestamp_utc` using half-open intervals ($t < \text{flash\_timestamp\_utc} \le t + 6\text{h}$) as verified in Test 7 and Test 8 of the ground truth spec.
4. **Coverage Masking:** For each 6-hour forecast window, apply the coverage rules defined in `lightning_ground_truth_spec.md` (minimum 95% granule completeness and $< 15\text{ min}$ maximum temporal gap) before assigning verified zero-lightning labels.
