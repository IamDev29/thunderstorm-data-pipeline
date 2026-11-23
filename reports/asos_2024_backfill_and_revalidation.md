# Florida ASOS/METAR 2024 Data Repair, Backfill & Revalidation Report

**Project**: Florida Thunderstorm 2024 Forecasting Research  
**Author**: ASOS Data Repair, Backfill & Quality Assurance Pipeline  
**Execution Date**: 2026-10-27  
**Scope**: Targeted Remediation of 73 Truncated Calendar Dates across Full Leap Year 2024  
**Repaired Artifacts Preserved In**: [`data/processed/asos_repair/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/)  
**Final Production Verdict**: **FULL GO FOR STATION-LEVEL MODEL TRAINING**

---

## 1. Executive Summary & Root Cause Confirmation

The ASOS model-readiness audit discovered that exactly 73 calendar days (19.95% of leap year 2024) were missing from the observations dataset. This report documents the targeted data repair, complete retrieval of the 73 missing dates, deduplication, downstream dataset reconstruction, baseline re-benchmarking, and final validation.

### Root Cause Confirmation:
- The Iowa Environmental Mesonet (IEM) ASOS CGI endpoint (`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`) interprets date-only parameters `year2, month2, day2` as midnight `00:00:00 UTC` (an exclusive upper bound for that day) when `hour2` and `minute2` are omitted.
- Consequently, Batch 1 requested `day1=1, day2=5`, which returned January 1, 2, 3, and 4 only (`2024-01-01 00:00` to `2024-01-04 23:58`). Batch 2 then requested `day1=6, day2=10`, skipping January 5 entirely. Every 5th day of the year was systematically omitted across Batches 1 through 73.
- All original raw batch files and original datasets were **preserved unmodified**.

### Headline Remediation Outcomes:
1. **100% Backfill Success**: All **73 missing calendar dates** were programmatically retrieved using verified inclusive query parameters (`hour2=23, minute2=59`) with zero drops, zero timeouts, and 100% status `SUCCESS`.
2. **Observations Dataset Growth**: Acquired **1,145,486 backfilled observations**, expanding the repaired observations dataset from 4,582,946 to **5,728,432 observations** across all 104 active Florida stations with **zero duplicate records**.
3. **Continuous Calendar Coverage**: Repaired dataset covers **366 / 366 unique calendar dates** (100.0% full leap year coverage).
4. **Terrestrial Modeling Expansion**: Terrestrial station-window rows merged with ERA5 predictors increased from 98,353 to **119,663 rows** (+21,310 rows, +21.67%).
5. **Enhanced Baseline Performance**: Restoring calendar continuity eliminated artificial gaps in the time-series. Persistence baseline test performance improved to **$F_1 = 0.6860$** (up from 0.6687), **Recall = 0.6836** (up from 0.6431), and **PR-AUC = 0.5002** (up from 0.4479).
6. **Automated Test Suite**: 26 / 26 automated unit and integration tests passing in pytest.

---

## 2. Phase 1: Missing Dates Identification & Repair Manifest

The missing calendar dates were independently calculated by comparing all 366 dates of 2024 against the unique observation dates in `florida_asos_observations_2024.parquet`:
- **Total Calendar Days in 2024**: 366
- **Originally Present Dates**: 293
- **Missing Dates**: Exactly **73 dates** (`2024-01-05`, `2024-01-10`, `2024-01-15`, ..., `2024-12-25`, `2024-12-30`).
- **Mathematical Reconciliation**: Every missing date coincided with the `end_date` of Batches 1 through 73 in `asos_download_manifest.csv` (Batch 74 was `2024-12-31` to `2024-12-31`, which had succeeded).
- **Repair Manifest**: Persisted to [`data/processed/asos_repair/asos_missing_dates_manifest.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/asos_missing_dates_manifest.csv), recording date, status, row count, active station count, duration, and timestamp range for each date.

---

## 3. Phase 2: Targeted Backfill Execution & Single-Date Validation

### 3.1 Parameter Verification on Single-Date Test (`2024-01-05`)
Before launching the complete batch backfill, the inclusive parameter configuration was verified on `2024-01-05`:
- Query: `year1=2024, month1=1, day1=5, year2=2024, month2=1, day2=5, hour2=23, minute2=59`
- Result: **HTTP 200**, duration **9.46s**, payload **2.54 MB**, **15,489 observations**, **100 stations**.
- Timestamp range: `2024-01-05 00:00:00 UTC` to `2024-01-05 23:58:00 UTC`.
- Verified that `hour2=23, minute2=59` fully and accurately captures the entire 24-hour UTC day.

### 3.2 Full Backfill Execution
- **Concurrency**: 2 parallel request workers with 0.5s polite delay and exponential backoff retry.
- **Atomic Storage**: Each date was saved immediately upon retrieval to [`data/processed/asos_repair/raw_backfill_batches/backfill_YYYY-MM-DD.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/raw_backfill_batches/).
- **Execution Outcome**: 73 / 73 dates completed with status `SUCCESS` in ~7.5 minutes.
- **Total Backfilled Observations**: **1,145,486 rows** (average ~15,690 observations per day).

---

## 4. Phase 3: Rebuilt & Validated Downstream Datasets

All downstream datasets were reconstructed from scratch combining original observations and the 73 backfilled dates. Original files were preserved untouched:

| Dataset Artifact | Storage Path | Row Count (Repaired) | Previous Row Count | Net Change |
| :--- | :--- | :---: | :---: | :---: |
| **Repaired Observations** | `data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet` | **5,728,432** | 4,582,946 | **+1,145,486 (+25.0%)** |
| **Repaired Forecast Windows** | `data/processed/asos_repair/florida_asos_forecast_windows_repaired_2024.parquet` | **143,687** | 118,027 | **+25,660 (+21.7%)** |
| **Repaired Modeling Table** | `data/processed/asos_repair/florida_asos_modeling_ready_repaired_2024.parquet` | **119,663** | 98,353 | **+21,310 (+21.7%)** |

### 4.1 Observations Quality & Deduplication
- **Total Raw Combined Observations**: 5,728,432
- **Duplicate Records Dropped**: **0** (exact station-timestamp uniqueness maintained).
- **Calendar Days Covered**: **366 / 366 (100.0%)**.
- **Active Station Outage Consistency**: Exactly **104 active stations** observed. Station **`AGR`** (MacGill AFB) was confirmed as the sole non-reporting station across all 366 days of 2024.

### 4.2 Forecast Window Aggregation under Criterion C
Every observation was snapped to half-open $(t, t+6\text{h}]$ forecast cycles:
- **`COVERED_THUNDERSTORM` ($y = 1$)**: **30,064 windows** (20.92%)
- **`COVERED_NO_THUNDERSTORM` ($y = 0$)**: **101,368 windows** (70.55%)
- **`INCOMPLETE_MONITORING` ($y = \text{NaN}$)**: **12,255 windows** (8.53%)
- **Total Observed Windows**: **143,687**
- **Unobserved Station-Windows (Full Cartesian Product $105 \times 1,464 = 153,720$)**: **10,033 station-windows** (6.53%). (Down from 35,693 prior to backfill).

### 4.3 Disaggregated Target Breakdown (Terrestrial Modeling Windows: 119,663)
```
+-------------------------------------------------------------+-----------+------------+
| Target Variable                                             | Positives | Prevalence |
+-------------------------------------------------------------+-----------+------------+
| target_asos_strict_crit_c (Strict TS | VCTS | Genuine LTG)  | 24,975    | 22.16%     |
| target_asos_aerodrome_ts_vcts (In-Station or Vicinity TS)   | 12,185    | 12.29%     |
| target_asos_explicit_ts (Explicit In-Station Thunderstorm)  | 8,602     | 9.07%      |
| target_noaa_severe_6h (NOAA Severe Convective Storm)        | 257       | 0.21%      |
+-------------------------------------------------------------+-----------+------------+
```

---

## 5. Phase 4: Model-Readiness Checks & Prevalence Analysis

### 5.1 Investigation of Seasonal Prevalence Differences Across Splits
A prominent feature of Florida convective climatology is the sharp variation in thunderstorm prevalence across the year:
- **Train Set (Jan 1 – Jul 31, 2024)**: 66,314 monitored windows, 14,673 positives (**22.13% prevalence**).
- **Validation Set (Aug 1 – Sep 30, 2024)**: 18,204 monitored windows, 7,669 positives (**42.13% prevalence**).
- **Test Set (Oct 1 – Dec 31, 2024)**: 28,159 monitored windows, 2,633 positives (**9.35% prevalence**).

#### Meteorological Climatology Explanation:
1. **Peak Summer Convection (Aug–Sep, 42.1%)**: The Florida peninsula experiences peak subtropical sea-breeze convergence, extreme insolation, and copious tropical moisture (PWAT $\ge 50\text{ mm}$), producing daily pulse thunderstorms across almost all inland and coastal convergence zones. In late September, Hurricane Helene and early tropical troughs also contributed extensive convective bands.
2. **Spring Convective Transition (Jan–Jul, 22.1%)**: Combines quiet, dry winter months (Jan–Feb prevalence $< 10\%$) with rising spring and early-summer convective initiation (May–Jul prevalence $\sim 30\text{--}35\%$), averaging to 22.1%.
3. **Autumn/Winter Transition & Tropical Landfall (Oct–Dec, 9.35%)**: Convection in October is punctuated by violent tropical systems (notably Hurricane Milton, Oct 9–10), followed by the seasonal collapse of the Bermuda High and arrival of dry, stable continental polar air masses in November and December (monthly prevalence drops below 5%).

This seasonal variation is a genuine physical feature of the Florida climate and reinforces why **chronological evaluation** is essential: evaluating on Oct–Dec rigorously tests whether a model trained on summer convection can generalize to seasonal decline and synoptic frontal regimes.

---

### 5.2 Updated Baseline Benchmark Performance on Repaired Test Set

Evaluated on the complete, unbroken **Chronological Test Set (October 1 – December 31, 2024)** across **28,159 monitored station-windows** (Test class prevalence = **9.35%**):

| Baseline Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology (Constant Prevalence)** | 0.2213 | **1.0000** | 0.3624 | 0.0935 | 0.5000 | 0.1011 |
| **Persistence (Thunderstorm at $t-6\text{h}$)** | **0.6883** | 0.6836 | **0.6860** | **0.5002** | **0.8259** | **0.0585** |
| **CAPE Threshold ($\ge 1000\text{ J/kg}$)** | 0.5022 | 0.1728 | 0.2571 | 0.2943 | 0.7280 | 0.0934 |
| **Logistic Regression (7 ERA5 Vars)** | 0.4397 | 0.2229 | 0.2959 | 0.2738 | 0.7280 | 0.0825 |

*Metrics archived in [`data/processed/asos_repair/asos_repaired_baseline_metrics.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/asos_repaired_baseline_metrics.csv).*

### Key Takeaway from Repaired Persistence:
- **Restoring Temporal Continuity Boosts Persistence**: On the previously truncated dataset, 73 calendar breaks disrupted persistence chains. With all 366 days present, persistence Recall rose from 0.6431 to **0.6836**, F1 rose from 0.6687 to **0.6860**, and PR-AUC rose from 0.4479 to **0.5002**.
- Persistence establishes an exceptionally rigorous standard: an ML model must exceed an F1 of **0.6860** and ROC-AUC of **0.8259** on the test set to demonstrate predictive utility beyond lag-1 persistence.

---

## 6. Automated Test Suite Execution

All 26 automated unit and integration tests across the complete testing suite pass with 100% success:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-9.0.3, pluggy-1.6.0
rootdir: Florida_Thunderstorm_2024_Training
collected 26 items

test_asos_acquisition.py::test_01_station_inventory_completeness PASSED [  3%]
test_asos_acquisition.py::test_02_metar_parsing_rules PASSED            [  7%]
test_asos_acquisition.py::test_03_half_open_window_snapping PASSED     [ 11%]
test_asos_acquisition.py::test_04_false_zero_safeguard PASSED          [ 15%]
test_asos_acquisition.py::test_05_master_observations_integrity PASSED [ 19%]
test_asos_acquisition.py::test_06_forecast_windows_integrity PASSED    [ 23%]
test_asos_acquisition.py::test_07_download_manifest_completeness PASSED [ 26%]
test_asos_audit.py::test_01_metar_semantic_separation PASSED           [ 30%]
test_asos_audit.py::test_02_cb_alone_reduction_impact PASSED           [ 34%]
test_asos_audit.py::test_03_reporting_gap_handling PASSED              [ 38%]
test_asos_audit.py::test_04_half_open_boundary_snapping PASSED         [ 42%]
test_asos_audit.py::test_05_spatial_representation_and_distances PASSED [ 46%]
test_asos_audit.py::test_06_manifest_completeness PASSED               [ 50%]
test_asos_audit.py::test_07_audited_windows_integrity PASSED           [ 53%]
test_asos_model_readiness.py::test_01_modeling_ready_parquet_integrity PASSED [ 57%]
test_asos_model_readiness.py::test_02_leakage_safeguards PASSED         [ 61%]
test_asos_model_readiness.py::test_03_target_disaggregation PASSED      [ 65%]
test_asos_model_readiness.py::test_04_chronological_split_integrity PASSED [ 69%]
test_asos_model_readiness.py::test_05_baseline_metrics_integrity PASSED [ 73%]
test_asos_model_readiness.py::test_06_verify_73_missing_days_audit_finding PASSED [ 76%]
test_asos_repair.py::test_01_manifest_completeness PASSED              [ 80%]
test_asos_repair.py::test_02_repaired_observations_completeness PASSED  [ 84%]
test_asos_repair.py::test_03_repaired_forecast_windows_integrity PASSED [ 88%]
test_asos_repair.py::test_04_repaired_modeling_dataset_integrity PASSED [ 92%]
test_asos_repair.py::test_05_repaired_baselines_integrity PASSED       [ 96%]
test_asos_repair.py::test_06_unresolved_station_date_gaps PASSED       [100%]

============================= 26 passed in 25.35s =============================
```

---

## 7. Deliverables & Artifact Inventory

The following versioned deliverables have been generated and validated:

1. **Missing-Date Repair Manifest**:  
   [`data/processed/asos_repair/asos_missing_dates_manifest.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/asos_missing_dates_manifest.csv) (73 dates, 100% `SUCCESS`).
2. **Repaired Master Observations Parquet**:  
   [`data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/florida_asos_observations_repaired_2024.parquet) (5,728,432 rows, 366 dates, zero duplicates).
3. **Repaired Criterion C Forecast Windows**:  
   [`data/processed/asos_repair/florida_asos_forecast_windows_repaired_2024.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/florida_asos_forecast_windows_repaired_2024.parquet) (143,687 window records).
4. **Repaired Modeling-Ready Dataset**:  
   [`data/processed/asos_repair/florida_asos_modeling_ready_repaired_2024.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/florida_asos_modeling_ready_repaired_2024.parquet) (119,663 terrestrial rows with 29 ERA5 features and disaggregated targets).
5. **Repaired Baseline Benchmark Metrics**:  
   [`data/processed/asos_repair/asos_repaired_baseline_metrics.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_repair/asos_repaired_baseline_metrics.csv).
6. **Automated Pytest Repair Suite**:  
   [`Florida_Thunderstorm_2024_Training/tests/test_asos_repair.py`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/Florida_Thunderstorm_2024_Training/tests/test_asos_repair.py) (6/6 passing).

---

## 8. Final Decision: FULL GO FOR MODEL TRAINING

### Production Verdict: **FULL GO FOR STATION-LEVEL MODEL TRAINING**

### Rationale:
1. **Complete Calendar Coverage**: Zero missing dates remain; all 366 days of 2024 are fully represented with continuous, unbroken time-series.
2. **Zero Leakage**: All 29 predictors are strictly indexed at forecast initialization time $t$. Prior persistence utilizes strictly $(t-6\text{h}, t]$.
3. **Rigorous Criterion C Ground Truth**: Eliminates gappy windows from negative targets and isolates Cumulonimbus-alone false positives.
4. **Benchmark Targets Established**: The modeling pipeline has established rigorous baseline targets across all metrics ($F_1 = 0.6860$, $\text{ROC-AUC} = 0.8259$, $\text{PR-AUC} = 0.5002$). Machine learning models (Gradient Boosted Trees, MLP Neural Networks) can now be trained and fairly evaluated.
