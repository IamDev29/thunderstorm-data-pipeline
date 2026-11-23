# Florida ASOS/METAR 2024 Model-Readiness Audit Report

**Project**: Florida Thunderstorm 2024 Forecasting Research  
**Author**: Machine Learning Model-Readiness & Data Integrity Audit Pipeline  
**Execution Date**: 2026-10-27  
**Scope**: Final Pre-Training Validation of Audited Station Labels, Feature Pipeline, Chronological Splits, and Baselines  
**Audited Tables Preserved In**: [`data/processed/asos_modeling/`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_modeling/)  
**Final Verdict**: **NO-GO FOR FINAL MODEL TRAINING** (Pending 73-Day Acquisition Backfill); **GO FOR MODELING PIPELINE ARCHITECTURE & BASELINE HARNESS**

---

## 1. Executive Summary & Audit Mandate

Prior to training any machine learning or neural network model on Florida ASOS/METAR 2024 data, an exhaustive, independent model-readiness audit was performed. This audit inspected the actual audited Parquet dataset ([`florida_asos_forecast_windows_audited.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_audit/florida_asos_forecast_windows_audited.parquet)), verified feature-generation pipelines for data leakage, established the precise modeling objective, implemented a strict chronological evaluation split, disaggregated target semantics, and benchmarked four simple baseline models.

### Key Audit Findings:
1. **Critical Acquisition Defect Discovered (The 73 Missing Days)**:
   - An independent check of observation dates revealed that **exactly 73 calendar days (19.95% of leap year 2024) are completely absent from the observations dataset**.
   - **Root Cause**: The Iowa Environmental Mesonet (IEM) ASOS API (`asos.py`) treats `year2, month2, day2` as midnight `00:00:00 UTC` (an exclusive upper bound for that day) when hours/minutes are omitted. Consequently, requesting `day1=1, day2=5` returned January 1, 2, 3, and 4 only; Batch 2 then requested `day1=6, day2=10`, skipping January 5 entirely. Every 5th day of the year was systematically omitted across all 74 batches.
   - **Mandatory Consequence**: **Final model training CANNOT proceed** on a dataset with 73 missing days spread uniformly across the year. The missing 73 days must be retrieved via a targeted backfill before the final production model is fit.
2. **Target Semantic Disaggregation Verified**:
   - `target_crit_c` strictly excludes Cumulonimbus-alone (`CB`) reports, preventing 11,850 non-electrified convective reports from creating false-positive targets.
   - Four separate target definitions were extracted and preserved in the modeling table: (a) Strict Thunderstorm (`TS | VCTS | Genuine LTG`), (b) Explicit In-Station Thunderstorm (`TS`), (c) Aerodrome/Vicinity Thunderstorm (`TS | VCTS`), and (d) NOAA Severe Convective Event.
3. **Leakage-Free Feature Alignment**:
   - Merged 98,353 terrestrial station-windows with ERA5 thermodynamic and kinematic predictors available at forecast initialization time $t$. Zero target-window $(t, t+6\text{h}]$ observations leak into predictors.
4. **Strong Persistence Baseline Established**:
   - Prior 6-hour ASOS thunderstorm persistence ($t-6\text{h}$) achieves **$F_1 = 0.6687$**, **Precision = 0.6964**, **Recall = 0.6431**, and **ROC-AUC = 0.8342** on the chronological test set (Oct–Dec 2024), setting a high benchmark for complex ML models.

---

## 2. Dataset Verification & The 73 Missing Days Discovery

### 2.1 Audited Parquet Inspection
The audited forecast window Parquet file was verified directly from disk:
- **File**: `data/processed/asos_audit/florida_asos_forecast_windows_audited.parquet`
- **Total Rows**: Exactly 118,027 window-station records.
- **Columns**: 22 columns including gap metrics (`max_gap_minutes`, `obs_span_hours`), separated convective counts (`ts_count`, `vcts_count`, `ltg_count`, `cb_alone_count`), and 4 evaluated criteria targets (`target_crit_a` through `target_crit_d`).
- **Station Coverage**: 105 stations cataloged (88 terrestrial, 17 coastal marine). 104 active stations observed; station `AGR` (MacGill AFB) confirmed offline for all of 2024.

### 2.2 Discovery of the 73 Missing Calendar Days
While previous reports confirmed that timestamps spanned from `2024-01-01 00:00:00 UTC` to `2024-12-31 23:58:00 UTC`, an independent audit of unique dates revealed:
- **Expected Calendar Days in Leap Year 2024**: 366
- **Actual Calendar Days Present**: **293**
- **Missing Days**: **Exactly 73 days**

```
+--------------------------------------------------------------------------------------------------------+
| THE IEM ASOS API BATCH BOUNDARY DEFECT                                                                |
+--------------------------------------------------------------------------------------------------------+
| Batch 01: Asked for 2024-01-01 to 2024-01-05 --> Returned Jan 1, Jan 2, Jan 3, Jan 4 (Jan 5 missing!)  |
| Batch 02: Asked for 2024-01-06 to 2024-01-10 --> Returned Jan 6, Jan 7, Jan 8, Jan 9 (Jan 10 missing!) |
| Batch 03: Asked for 2024-01-11 to 2024-01-15 --> Returned Jan 11, Jan 12, Jan 13, Jan 14 (Jan 15 miss!) |
| ... Repeated across all 73 5-day intervals across the entire year!                                     |
+--------------------------------------------------------------------------------------------------------+
```

### Explanation of Missing Forecast Init Times:
- Because every 5th day was truncated at `00:00:00 UTC`, the 6-hour forecast cycles initialized at `00:00:00`, `06:00:00`, and `12:00:00` on those days were omitted.
- Expected 6-hour forecast cycles: 1,464.
- Actual cycles present in audited ASOS dataset: **1,246** (218 cycles omitted).
- **Impact on Modeling**: Because the missing days are distributed evenly (every 5 days), existing data provides an accurate statistical sample of seasonal weather, but creates periodic breaks in continuous multi-day time-series and persistence chains.

---

## 3. Modelling Objective & Task Definition

The modeling objective is formally defined to prevent domain confusion:

> **Primary Modeling Task**:  
> Given numerical weather prediction (ERA5) thermodynamic and kinematic atmospheric predictors available at forecast initialization time $t$, predict the occurrence of a verified thunderstorm report at an ASOS surface station over the subsequent six-hour window $(t, t+6\text{h}]$.

### Critical Boundaries:
- **Point Aerodrome Thunderstorm Occurrence, NOT Statewide Lightning**: This task predicts localized convective initiation and transit over a specific airport sensor ($\sim 10\text{ mile}$ radius). It must not be conflated with predicting lightning everywhere across Florida.
- **Station-Specific Prediction**: The model predicts $P(Y_{s, t} = 1 \mid X_{g(s), t})$, where $s$ is the station and $g(s)$ is the ERA5 grid cell containing station $s$.

---

## 4. Target Disaggregation & Multi-Task Taxonomy

To avoid silently combining disparate convective phenomena, four distinct binary targets were generated and preserved in the modeling table [`data/processed/asos_modeling/florida_asos_modeling_ready_2024.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_modeling/florida_asos_modeling_ready_2024.parquet):

```mermaid
flowchart TD
    A["Raw METAR Convective Phenomena"] --> B["Strict ASOS Convection<br/>(target_asos_strict_crit_c)<br/>20,678 Pos (22.83%)"]
    A --> C["In-Station Thunderstorm<br/>(target_asos_explicit_ts)<br/>7,100 Pos (9.22%)"]
    A --> D["Aerodrome / Vicinity<br/>(target_asos_aerodrome_ts_vcts)<br/>10,021 Pos (12.54%)"]
    E["NOAA Storm Events Data"] --> F["NOAA Severe Convective Event<br/>(target_noaa_severe_6h)<br/>211 Pos (0.21%)"]
```

### Target Distribution on Active Terrestrial Monitored Windows (98,353 Total):
| Target Variable | Target Description | Positive Windows ($y=1$) | Negative Windows ($y=0$) | Incomplete / NaN | Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `target_asos_strict_crit_c` | In-station TS, vicinity VCTS, or genuine LTG remarks (Criterion C) | **20,678** | **69,910** | **7,765** | **22.83%** |
| `target_asos_explicit_ts` | Explicit in-station thunder (`TS`, `+TSRA`, `-TSRA`) | **7,100** | **69,910** | **21,343** | **9.22%** |
| `target_asos_aerodrome_ts_vcts` | In-station or vicinity thunderstorm (`TS | VCTS`) | **10,021** | **69,910** | **18,422** | **12.54%** |
| `target_noaa_severe_6h` | Severe convective storm (damaging wind, hail $\ge 1"$, tornado) | **211** | **98,058** | **84** | **0.21%** |

*Note: Incomplete/NaN windows are strictly withheld from training sets to prevent false-negative contamination.*

---

## 5. Feature Engineering & Leakage Safeguards

All predictors are strictly evaluated at forecast initialization time $t$:

### 5.1 Permitted Predictor Variables (at time $t$):
1. **Thermodynamic Predictors**: `cape_J_kg`, `cin_J_kg`, `temperature_2m_degC`, `dewpoint_2m_degC`, `dewpoint_depression_degC`, `tcwv_kg_m2`, `total_precipitation_mm`, `convective_precipitation_mm`.
2. **Kinematic & Dynamic Predictors**: `bulk_shear_850_300_ms`, `wind_speed_10m_ms`, `wind_direction_10m_deg`, `u10`, `v10`, `wind_speed_850hPa_ms`.
3. **Mid-Tropospheric Stability**: `temp_diff_850_500_degC`, `temperature_850hPa_degC`, `temperature_500hPa_degC`, `relative_humidity_850hPa_pct`, `relative_humidity_500hPa_pct`.
4. **Surface Pressure**: `surface_pressure_hPa`, `mslp_hPa`.
5. **Cyclical & Solar Time**: `month`, `hour_utc`.
6. **Prior Station Persistence**: Thunderstorm occurrence at the station during the *prior* cycle $(t-6\text{h}, t]$.

### 5.2 Strict Leakage Barriers:
- **No Future ASOS Observations**: Zero observations from $(t, t+6\text{h}]$ are permitted in the predictor matrix.
- **No Future NOAA Storm Events**: Future event IDs, event counts, or distances from $(t, t+6\text{h}]$ are strictly targets/benchmarks.
- **No Target Window GLM Flashes**: GLM flashes occurring during the forecast window are isolated.

---

## 6. Chronological Evaluation Split

Random row splitting is **prohibited** due to intense temporal autocorrelation between successive 6-hour cycles at the same station. A strict chronological partition was implemented:

```
+----------------------------------------------------------------------------------------------------+
| 2024 FULL-YEAR CHRONOLOGICAL SPLIT TIMELINE                                                        |
+----------------------------------------------------------------------------------------------------+
| [========== TRAIN SET (59.1%) ==========] [=== VAL SET (16.2%) ===] [==== TEST SET (24.7%) ====]   |
| Jan 01 ------------------------- Jul 31   Aug 01 ----------- Sep 30   Oct 01 ------------ Dec 31   |
| 53,494 Monitored Station-Windows          14,711 Windows              22,383 Windows               |
| Positive Prevalence: 22.79%               Positive Prevalence: 42.59% Positive Prevalence: 9.93%   |
+----------------------------------------------------------------------------------------------------+
```

### Partition Properties:
- **Train Set (Jan 1 – Jul 31, 2024)**: Captures winter fronts, spring convective transition, and early summer sea-breeze convection.
- **Validation Set (Aug 1 – Sep 30, 2024)**: Captures the peak subtropical convective season and tropical cyclone outer bands.
- **Test Set (Oct 1 – Dec 31, 2024)**: Evaluates seasonal decline, late-season major hurricane landfalls (Hurricane Milton, Oct 9–10), and dry winter transition.
- **Station Representation**: All 88 active terrestrial stations are present in Train, Val, and Test, ensuring model evaluation reflects genuine temporal out-of-sample forecasting rather than spatial interpolation.

---

## 7. Baseline Model Benchmark Results

Four simple baseline models were evaluated on the **Chronological Test Set (October 1 – December 31, 2024)** using `target_asos_strict_crit_c`. 

Test set sample size: **22,383 monitored station-windows**; **2,222 positive thunderstorm windows** (Prevalence = **9.93%**).

### Comprehensive Baseline Performance Matrix:
| Model / Baseline | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Climatology Baseline** | 0.2279 | **1.0000** | 0.3712 | 0.0993 | 0.5000 | 0.1060 | Predicts constant training set prevalence ($\hat{p} = 0.2279$). |
| **Persistence Baseline** | **0.6964** | 0.6431 | **0.6687** | **0.4479** | **0.8342** | **0.0633** | Predicts thunderstorm if station had thunderstorm in $(t-6\text{h}, t]$. |
| **CAPE Threshold ($\ge 1000\text{ J/kg}$)** | 0.5412 | 0.1598 | 0.2467 | 0.3192 | 0.7342 | 0.0969 | Predicts thunderstorm if ERA5 CAPE $\ge 1000\text{ J/kg}$ at init. |
| **Logistic Regression (7 ERA5 Vars)** | 0.4595 | 0.2246 | 0.3017 | 0.2899 | 0.7331 | 0.0857 | Linear combination of CAPE, CIN, depression, shear, temp, rain. |

*Full machine-readable baseline metrics saved to [`data/processed/asos_modeling/asos_baseline_metrics.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_modeling/asos_baseline_metrics.csv).*

### Key Baseline Takeaway:
- **The Dominance of Persistence**: In Florida, multi-cycle convective regimes and synoptic frontal passages cause convective activity to persist across consecutive 6-hour windows. The simple persistence baseline achieves an **F1-score of 0.6687** and **ROC-AUC of 0.8342**, outperforming simple static thermodynamic thresholding. Any machine learning model (MLP, GBDT) must beat **$F_1 = 0.6687$** on the test set to justify its complexity.

---

## 8. Verified Results vs Recommendations

| Item | Status | Verified Finding / Actionable Recommendation |
| :--- | :---: | :--- |
| **CB Isolation** | **VERIFIED** | Exactly 11,850 observations contain CB alone. Removing them prevented 11,822 false positives. |
| **Criterion C Gap Safeguard** | **VERIFIED** | Enforcing $N \ge 6, \text{gap} \le 75\text{m}, \text{span} \ge 5.0\text{h}$ reclassified 4,970 gappy windows from negative to incomplete. |
| **73 Missing Days Defect** | **VERIFIED** | Exactly 73 days (218 forecast cycles) were omitted due to IEM `day2` midnight cutoff. |
| **Feature Leakage Prevention** | **VERIFIED** | All 29 merged features are indexed at forecast init time $t$. Zero future observations leak into inputs. |
| **Baseline Benchmark** | **VERIFIED** | Persistence baseline establishes a test F1 benchmark of 0.6687 and ROC-AUC of 0.8342. |
| **Acquisition Backfill** | **RECOMMENDED** | Execute a targeted 73-day retrieval with `hour2=23&minute2=59` to restore full 366-day continuity. |
| **Multi-Target Training** | **RECOMMENDED** | Train separate heads or models for in-station TS ($9.2\%$ prevalence) vs vicinity/lightning ($22.8\%$). |

---

## 9. Final Decision: NO-GO FOR FINAL MODEL TRAINING

### Verdict: **NO-GO FOR FINAL PRODUCTION MODEL TRAINING**
### Sub-Verdict: **GO FOR MODELING PIPELINE ARCHITECTURE & BASELINES**

### Justification:
1. **The 73 Missing Days**: Training a production forecasting model on a dataset with 20% of days missing at regular 5-day intervals compromises temporal continuity, disrupts multi-day sequence modeling, and creates gaps in test evaluation.
2. **Remediation Roadmap (Path to GO)**:
   - **Step 1**: Run a targeted acquisition script for the 73 missing dates (e.g. `2024-01-05`, `2024-01-10`, `...`, `2024-12-30`) specifying `hour2=23&minute2=59`.
   - **Step 2**: Merge the backfilled observations into `florida_asos_observations_2024.parquet` and re-run window aggregation.
   - **Step 3**: Re-execute `evaluate_asos_baselines.py` on the complete 366-day dataset.
   - **Step 4**: Full **GO** to train machine learning (LightGBM/XGBoost) and Multilayer Perceptron (MLP) neural network models.
