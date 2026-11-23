# Full-Year 3D Florida Thunderstorm Model Readiness & Evaluation Report

**Generated:** 2026-10-27 05:07:40 UTC  
**Domain:** Florida Geographic Bounding Box (218 cells, 0.25° resolution)  
**Atmospheric Data:** ERA5 Reanalysis Full Year 2024 (Single-level + Complete 850/500/300 hPa Pressure Levels)  
**Evaluation Protocol:** Strict Chronological Train-Test Split (Train: Months 01–08, Test: Months 09–12)  

---

## 1. Executive Dataset Summary

With the download and validation of all 12 pressure-level monthly batches complete, the full-year 3D dataset is **100% complete** across all 1,464 synoptic observation cycles in 2024.

* **Total Grid Observations:** **319,152 rows** (218 cells $\times$ 1,464 timestamps).
* **Pressure-Level Completeness:** **100.0%** (850, 500, and 300 hPa features present for every observation).
* **Confirmed NOAA Severe Convective Events:** **369 unique grid-cell/time positive keys** (all 369 now retain full 3D upper-air features).
* **Primary Key Uniqueness:** `(valid_time_utc, latitude, longitude)` strictly unique (0 duplicates).

---

## 2. Chronological Baseline Experiment Results (Full-Year 3D Classification)

- **Training Period:** Months 01–08 (January 1 to August 31, 2024) $\to$ **310 Positives / 212,458 Negatives**
- **Test Period:** Months 09–12 (September 1 to December 31, 2024, 122 days) $\to$ **59 Positives / 106,325 Negatives**
- **Test Positive Prevalence:** **0.055%**

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | False Alarms / Day | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | 0.0 | 0.0 | 0.0 | 0.5 | 0.0006 | 0.0 | `[[106325, 0], [59, 0]]` |
| **Climatology Baseline** | 0.0 | 0.0 | 0.0 | 0.5 | 0.0006 | 0.0 | `[[106325, 0], [59, 0]]` |
| **Logistic Regression (Balanced)** | 0.0077 | 0.8644 | 0.0152 | 0.9664 | 0.1437 | 54.13 | `[[99721, 6604], [8, 51]]` |
| **Random Forest (Balanced)** | 0.0192 | 0.0339 | 0.0245 | 0.8948 | 0.004 | 0.84 | `[[106223, 102], [57, 2]]` |
| **HistGradientBoosting (Balanced)** | 0.0026 | 0.1186 | 0.0052 | 0.9029 | 0.0052 | 21.61 | `[[103689, 2636], [52, 7]]` |

### Top Predictive 3D Features (Random Forest):
- `total_precipitation_mm`: 0.1134
- `convective_precipitation_mm`: 0.0799
- `mslp_hPa`: 0.0785
- `cape_J_kg`: 0.0653
- `surface_pressure_hPa`: 0.0466
- `vertical_velocity_500hPa_Pa_s`: 0.0412
- `cin_J_kg`: 0.0411
- `vertical_velocity_300hPa_Pa_s`: 0.0389

### Monthly Performance Breakdown on Test Split (Logistic Regression):
| Month | True NOAA Positive Keys | Model Predicted Detections | True Positives (Hits) | False Positives (Alarms) | Detection Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Month 09** | 12 | 4489 | 7 | 4482 | **58.3%** |
| **Month 10** | 36 | 1109 | 36 | 1073 | **100.0%** |
| **Month 11** | 0 | 413 | 0 | 413 | **0.0%** |
| **Month 12** | 11 | 644 | 8 | 636 | **72.7%** |

---

## 3. Spatial and Temporal Autocorrelation Discussion

A critical methodological requirement is that machine-learning models in atmospheric science must **never be evaluated using random k-fold cross validation across spatiotemporally adjacent observations**. Convective systems span multiple adjacent 0.25° grid cells and can persist across 6-to-12 hour observation cycles. Random cross-validation leaks storm systems across folds, producing artificially elevated verification scores. The strict chronological split employed here guarantees that models are tested exclusively on future, unseen meteorological regimes.
