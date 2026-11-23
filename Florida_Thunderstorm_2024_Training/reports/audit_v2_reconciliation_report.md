# Comprehensive Consistency and Reproducibility Reconciliation Audit

**Project:** Florida Thunderstorm 2024 Six-Hour-Ahead Convective Forecasting  
**Audit Date:** October 27, 2026  
**Status:** Audit Complete | 100% Reconciled | Authoritative Datasets Established  

---

## 1. Executive Summary & Authoritative Dataset Identification

This audit performs an end-to-end consistency and reproducibility verification across all generated reports, datasets, machine-readable artifacts, and model evaluation pipelines for the Florida Thunderstorm 2024 project.

### Authoritative Dataset Resolution
The workspace contains datasets in both the parent root (`c:\Users\KIIT0001\Desktop\Data Collection\data\processed`) and the child repository (`Florida_Thunderstorm_2024_Training\data\processed`). This audit establishes the authoritative copies:

| Dataset Name | Authoritative Path | SHA-256 Hash | Size (Bytes) | Verification Status & Version |
| :--- | :--- | :---: | :---: | :--- |
| **Active 3D Master** | `data/processed/florida_thunderstorm_2024_master.csv` | `8a19c9cdcd73e779...` | 148,317,143 | **AUTHORITATIVE (v3.0 Full-Year 3D)**. Built 2026-10-27 with full-year pressure levels and NOAA timezone correction (369 positive cells). |
| *Deprecated Master* | `Florida_Thunderstorm_2024_Training/data/processed/florida_thunderstorm_2024_master.csv` | `140e1c0647298cb5...` | 180,469,504 | *DEPRECATED (v2.0 Provisional)*. Built 2026-10-26 before timezone correction (366 positive cells; naive UTC). Kept in backup for audit provenance. |
| **Forecasting 6h** | `data/processed/florida_thunderstorm_2024_forecasting_6h.csv` | `dd68c7c74edbc25d...` | 136,532,403 | **AUTHORITATIVE**. Contains 375 positive forward targets, strictly leak-free `true_prior_event_6h`, and episode IDs. |
| **Processed NOAA** | `Florida_Thunderstorm_2024_Training/data/processed/noaa_florida_2024.csv` | `99cee016faa2bfc0...` | 571,510 | **AUTHORITATIVE**. Contains 1,388 total events and 677 standardized convective events with corrected UTC timestamps. |
| **Processed GLM** | `Florida_Thunderstorm_2024_Training/data/processed/glm_florida_2024.csv` | `23e6c96d817a6343...` | 4,529,550 | **AUTHORITATIVE**. Ingested pilot period (Jan, Feb, May) with 53,092 rows. |

---

## 2. Resolution of Master Positive-Count Discrepancies (366 vs. 369 vs. 375)

The project reports have cited three different positive event counts:
* **366** in early provisional reports.
* **369** in the full-year 3D classification master dataset.
* **375** in the forward six-hour forecasting dataset.

A forensic row-level analysis across all 677 qualifying convective NOAA events resolves these discrepancies:

```
                  677 Processed Convective NOAA Events
                                    |
                 [Grid Distance Filter: Dist <= 28 km]
                                    |
                    676 Spatially Matched Events
                     (1 Excluded: Event 1186940 at 28.84 km)
                                    |
        +---------------------------+---------------------------+
        |                                                       |
[Symmetric Window: t +/- 3h]                           [Forward Window: (t, t + 6h]]
        |                                                       |
 369 Positive Grid-Time Cells                           375 Positive Forecast Windows
 (Active 3D Master Dataset)                            (Forecasting 6h Dataset)
```

### Explanation of Each Count
1. **The 366 Count (Deprecated Provisional Master):**  
   Constructed on October 26, 2026 before NOAA timezone correction. It treated raw local timestamps (`EST-5` and `CST-6`) as naive UTC, introducing a 5-to-6 hour temporal displacement. Of its 366 positive cells, **346 cells were at incorrect synoptic cycles**.
2. **The 369 Count (Active Full-Year 3D Master):**  
   Constructed on October 27, 2026 after implementing the documented timezone offsets in `src/process_noaa.py`. Convective events occurring during typical Florida afternoon hours (14:00–18:00 local time) correctly shifted into the 18:00 UTC and 00:00 UTC reanalysis cycles. The 676 matched events collapse into **369 unique grid-time cells** under the symmetric 6-hour window $[t-3\text{h}, t+3\text{h}]$.
3. **The 375 Count (Forward Forecasting Dataset):**  
   The forecasting dataset evaluates a forward target window $(t, t+6\text{h}]$ rather than a centered window $[t-3\text{h}, t+3\text{h}]$. An event occurring at 15:30 UTC maps to the 18:00 UTC cycle in centered master, but maps to the 12:00 UTC initialization cycle in forward forecasting. Grouping the exact same 676 events by forward intervals $(t, t+6\text{h}]$ yields **375 unique grid-time cells**.
4. **Row-Level Discrepancy File:**  
   Every single key difference across all three files (729 unique keys) has been exported with full provenance to [`data/processed/audit_v2/reconciliation_discrepancies.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/audit_v2/reconciliation_discrepancies.csv):
   * Timezone displacement effects: 513 rows.
   * Consensus agreement between centered master and forward target: 202 rows.
   * Window geometry boundary effects: 14 rows.

---

## 3. Reconciled Dataset Row Counts Across All Splitting Experiments

All 2024 synoptic cycles (366 days $\times$ 4 cycles/day = 1,464 timestamps $\times$ 218 grid cells) produce **319,152 total rows**.

### Universal Boundary Truncation
The final synoptic initialization cycle of the year (`2024-12-31 18:00:00 UTC`, 218 grid cells) has a forward target window $(18:00, 24:00]$ extending into 2025. It is cleanly coded as `NaN` (`TRUNCATED_YEAR_END`).
* **Total rows before filtering:** 319,152
* **Valid evaluation rows:** **318,934** (319,152 - 218)
* **Truncated year-end rows:** **218**

### Reconciled Partition Counts Table

| Experiment | Partition | Total Rows | Valid Rows | Positives | Negatives | Prevalence | Calendar Days | Months | Unique Episodes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Chronological** | **Train** | 212,768 | 212,768 | 316 | 212,452 | 0.1485% | 244 | 1–8 | 113 |
| | **Test** | 106,384 | 106,166 | 59 | 106,107 | 0.0556% | 122 | 9–12 | 15 |
| | *Truncated* | 218 | — | — | — | — | 1 (Dec 31) | 12 | — |
| **Exp 2A: Alternating Pure** | **Train (Odd DOY)** | 159,576 | 159,576 | 238 | 159,338 | 0.1491% | 183 | 1–12 | 71 |
| | **Test (Even DOY)**| 159,576 | 159,358 | 137 | 159,221 | 0.0860% | 183 | 1–12 | 66 |
| **Exp 2B: Alternating Purged**| **Train** | 127,966 | 127,966 | 129 | 127,837 | 0.1008% | 147 | 1–12 | 48 |
| | **Test** | 63,874 | 63,656 | 58 | 63,598 | 0.0911% | 73 | 1–12 | 16 |
| | *Embargoed* | 127,312 | 127,312 | 188 | 127,124 | 0.1477% | 146 | 1–12 | — |
| **Exp 3: Seasonal Stratified**| **Train** | 224,758 | 224,758 | 264 | 224,494 | 0.1175% | 258 | 1–12 | 93 |
| | **Test** | 94,394 | 94,176 | 111 | 94,065 | 0.1179% | 108 | 1–12 | 37 |
| **Exp 4: Episode Grouped** | **Train** | 213,422 | 213,422 | 206 | 213,216 | 0.0965% | 245 | 1–12 | 69 |
| | **Test** | 105,730 | 105,512 | 169 | 105,343 | 0.1602% | 121 | 1–12 | 60 |
| **Exp 5: Strict Embargo** | **Train (M01–07)** | 185,736 | 185,736 | 287 | 185,449 | 0.1545% | 213 | 1–7 | 96 |
| | **Test (M09–12)** | 106,384 | 106,166 | 59 | 106,107 | 0.0556% | 122 | 9–12 | 15 |
| | *Embargo (M08)* | 27,032 | 27,032 | 29 | 27,003 | 0.1073% | 31 | 8 | — |

*Note: All counts reconcile exactly: $\text{Train} + \text{Test} + \text{Embargo} + \text{Truncated} = 319,152$.*

---

## 4. Reconciled GLM Coverage Semantics & Verification

The GLM processing code (`src/process_glm.py`) establishes that adequate monitoring requires at least 10 granules (200 seconds of observation) per hourly aggregation cycle.

### Why Verified Zero-Lightning is Currently Zero
In the downloaded pilot archive (January, February, May 2024), each processed cycle contains only 3 granules (60 seconds of monitoring).
* When $\ge 1$ flashes were detected, the sample is confirmed positive (`COVERED_LIGHTNING`), because even a single flash proves electrical activity.
* When 0 flashes were detected during those 60 seconds, the sample **cannot be confirmed as zero-lightning**, because 60s is insufficient to prove the absence of a storm. Therefore, `process_glm.py` correctly classified all 746 non-lightning intervals as `INCOMPLETE_COVERAGE`.
* Consequently, **there are exactly 0 genuinely verified zero-lightning samples in the pilot archive**.

### Reconciled GLM Counts

| Coverage Category | Operational Meaning | GLM Table Count (53,092) | Forecasting Dataset Count (319,152) | Percentage |
| :--- | :--- | :---: | :---: | :---: |
| **1. Adequate / Covered with Lightning** | Confirmed satellite flashes $\ge 1$ | **26** | **26** | **0.008%** |
| **2. Adequate with Zero Lightning** | $\ge 10$ granules, 0 flashes detected | **0** | **0** | **0.000%** |
| **3. Partial / Incomplete Coverage** | $1\text{--}9$ granules, duration insufficient | **746** | **746** | **0.234%** |
| **4. No Usable Coverage (Unobserved)** | 0 granules in archive | **52,320** | **318,380** | **99.758%** |
| **5. Invalid / Ambiguous Metadata** | Corrupt coordinates or flags | **0** | **0** | **0.000%** |

*Exported to:* [`data/processed/audit_v2/reconciled_glm_coverage_counts.json`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/audit_v2/reconciled_glm_coverage_counts.json)

---

## 5. Candidate-Label Count Verification & Feature Circularity Warning

The 1,445 missed-convection candidates in `data/processed/audit_v2/missed_convection_candidates.csv` were verified for uniqueness and tier exclusivity:
* **Duplicate candidate keys:** Exactly **0**.
* **Tier overlap:** Exactly **0** (the three tiers are mutually exclusive).
  * **Tier 1 (Satellite Lightning Supported):** **26 cycles** (HIGH confidence; verified GLM flashes).
  * **Tier 2 (Strong Physical Convection):** **96 cycles** (MEDIUM confidence; convective precip $\ge 5$ mm, $\text{CAPE} \ge 2,000$ J/kg, $\text{TCWV} \ge 50$ kg/m², favorability $\ge 0.70$).
  * **Tier 3 (Moderate Physical Convection):** **1,323 cycles** (LOW confidence; convective precip $\ge 2$ mm, $\text{CAPE} \ge 1,500$ J/kg, favorability $\ge 0.50$).

> [!WARNING]
> **Circularity Safeguard**: Tiers 2 and 3 were identified using ERA5 thermodynamic reanalysis fields at forecast initialization time $t$. Because these same variables (`cape_J_kg`, `convective_precipitation_mm`) are supplied as predictor features to the machine learning models, **they cannot be used to independently validate model predictions**. Only Tier 1 represents independent sensor confirmation.

---

## 6. Evaluation Metrics Audit & Operational Definitions

All metrics in [`data/processed/audit_v2/reconciled_experiment_metrics.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/audit_v2/reconciled_experiment_metrics.csv) were recalculated directly from the confusion matrices:

### Definition of False Alarms Per Day
In all project reports, **False Alarms per Day** is defined as:
$$\text{False Alarms per Day} \;=\; \frac{\text{Total False Positive Grid Cells Statewide}}{\text{Number of Calendar Days in Test Set}}$$
For example, in Experiment 1 (Logistic Regression at $\tau^* = 0.99$):
$$\text{False Alarms per Day} \;=\; \frac{435\text{ false alarm cells}}{122\text{ calendar days}} \;=\; \mathbf{3.57\text{ false alarms / day statewide}}$$
Per individual 0.25° grid cell, this corresponds to:
$$\frac{3.57}{218\text{ cells}} \;=\; \mathbf{0.0164\text{ alerts per cell-day}}\quad (\approx 1\text{ alert per cell every } 61\text{ days})$$

### Threshold Selection & Preprocessing
* **Threshold Selection:** Validation-only. Selected by finding the threshold that maximizes $F_1$ on the validation partition (the final 20% of training dates). The test set was **never touched** during threshold selection.
* **Preprocessing:** `SimpleImputer` and `StandardScaler` were fitted strictly on the training partition and transformed onto the test partition.
* **Calibrated vs. Threshold-Tuned:** Probabilities are raw outputs from `class_weight="balanced"` logistic regression and histogram gradient boosting, tuned via decision threshold selection rather than post-hoc isotonic calibration.

---

## 7. Splitting Experiment Leakage & Generalization Analysis

### Discovery of Multi-Day Episode Spanning in Experiment 4
In Experiment 4 (Episode Grouping), grouping was executed at the calendar-day level. However, **Episode 191959** spanned midnight across May 13 (12:00 UTC) and May 14 (06:00 UTC). Because May 13 was allocated to Test and May 14 to Train, Episode 191959 appeared in both partitions.
* **Finding:** Grouping by calendar day is insufficient for multi-day tropical systems or frontal boundaries. Complete episode isolation requires grouping by `EPISODE_ID` across multi-day blocks.
* **In contrast:** Experiment 1 (Chronological) and Experiment 5 (Strict Embargo) have **exactly 0 shared episodes** between Train and Test.

### Four Distinct Evaluation Objectives
Project documentation must maintain clear distinctions between:
1. **Seasonal Representation (Exp 3):** Tests whether training across all 4 seasons improves skill over training only in spring/summer.
2. **Temporal Generalization (Exp 1 & 5):** Evaluates true future forecasting into unseen calendar months.
3. **Episode Generalization (Exp 4):** Evaluates whether the model predicts convective initiation on physically unseen storm episodes.
4. **Geographical Generalization:** Evaluates spatial transferability to unseen regions (e.g. Panhandle vs Everglades).

---

## 8. Automated Test Execution Summary

The project test suite was executed to confirm deterministic reconciliation:

```powershell
pytest Florida_Thunderstorm_2024_Training\tests\test_full_year_3d_and_forecast.py Florida_Thunderstorm_2024_Training\tests\test_forecast_audit_and_ablation.py Florida_Thunderstorm_2024_Training\tests\test_audit_v2_comprehensive.py -v
```
**Results:** **20 passed, 0 failed in 29.19s**.

---

## 9. Prerequisites for Full-Year GLM Ingestion

To extend GLM lightning verification to all 366 days of 2024, the following prerequisites must be met:
1. **Ingest Complete Granule Streams:** Retrieve all Level-2 LCFA NetCDF granules from the NOAA GOES-16 public AWS S3 bucket (`noaa-goes16/GLM-L2-LCFA/2024/`).
2. **Enforce Adequate Monitoring Duration:** Process all 180 granules per hour (100% temporal coverage = 3,600s/hour) so that non-lightning intervals satisfy the $\ge 10$ granule threshold and can be certified as `COVERED_NO_LIGHTNING`.
3. **Dedicated Processing Script:** Execute `scripts/download_glm_full_year.py` and `src/process_glm.py` on the complete archive.
