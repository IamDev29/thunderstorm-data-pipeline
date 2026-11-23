# Florida 2024 Thunderstorm Forecasting: GOES-16 GLM Feature Integration Evaluation
**Experiment Directory:** `experiments/glm_integration_2024/`  
**Evaluation Date:** October 28, 2026  
**Primary Reference Model:** Tuned LightGBM (`experiments/advanced_tuning_2024/final_lightgbm_tuned.joblib`)  
**Target Label:** `target_asos_strict_crit_c` (Strict ASOS Thunderstorm / In-Cloud Lightning in $(t, t+6\text{h}]$ under Criterion C)  
**Primary Dataset:** `data/processed/asos_parser_fixed/florida_asos_modeling_ready_fixed_2024.parquet`  
**Dataset SHA256:** `ebaa442727daf506af3889a4a96f1a411dc68f03d11b2efeb7abbc6851d147e4`  

---

## 1. Executive Summary & Final Scientific Decision

This report provides a scientifically controlled, leak-free empirical evaluation of whether adding satellite-observed lightning observations from the **GOES-16 Geostationary Lightning Mapper (GLM)** improves 6-hour-ahead thunderstorm prediction beyond ERA5 atmospheric predictors, station observation history, spatial coordinates, and diurnal cycles across Florida.

### Final Decisions on Core Research Questions:

1. **Whether the available GLM data support a valid experiment:**  
   **CONDITIONAL / PILOT ONLY.** The local repository contains exactly **12 raw GLM NetCDF granules** covering four 60-second bursts (240 seconds total) across a single calendar day (**May 15, 2024**). Across the 366 calendar days in 2024, **365 days (99.73%) are missing on disk**. While this severe coverage gap prevents full-year operational deployment or multi-fold expanding-window cross-validation, it permits a rigorous, leak-free **common-cohort pilot evaluation** on May 15, 2024 ($N = 320$ monitored forecast windows).
2. **Whether GLM adds measurable predictive value beyond the reference model:**  
   **NO.** On the May 15 common evaluation cohort, the baseline physics + persistence model (**Model C**) achieved **PR-AUC 0.9151** and **ROC-AUC 0.9171** (Brier score 0.1208). Adding GLM features (**Model D**) yielded **PR-AUC 0.8081** and **ROC-AUC 0.8475** (Brier score 0.1555). Paired date-block bootstrap analysis (1,000 resamples) confirms a statistically significant degradation: $\Delta\text{PR-AUC} = -0.1063$ (95% CI: $[-0.1725, -0.0490]$, $p = 0.0010$).
3. **Whether the improvement is consistent across temporal folds:**  
   **NO.** Because GLM data is entirely absent for 11 out of 12 months (including the entire October–December test period), temporal consistency across expanding-window folds could not be established.
4. **Whether the experiment is limited by satellite coverage:**  
   **YES, SEVERELY.** 112,189 out of 112,509 monitored station-windows (**99.72%**) were excluded due to satellite data unavailability on disk. Storing the full year of raw GLM NetCDF files (~497 GB) would exceed the host drive capacity (~53 GB available).
5. **Whether the satellite-enhanced model should proceed to prototype integration:**  
   **DECISION: NO-GO.** The satellite-enhanced model must **not** proceed to prototype integration at this time. Integration must wait until a continuous streaming GLM pipeline extracts flash aggregations directly from NOAA AWS S3 without storing uncompressed raw NetCDF archives locally.

---

## 2. Sensor Parameters, Granule Inventory & Provenance

- **Platform:** GOES-16 (`G16`), Flight Model 1 (`FM1`), positioned at Geostationary East ($75.2^\circ\text{W}$).
- **Instrument:** Geostationary Lightning Mapper (GLM).
- **Product:** Level-2 Lightning Cluster Filter Algorithm (`OR_GLM-L2-LCFA_G16`), Version 2.
- **Granule Inventory:** Exactly 12 files on May 15, 2024 (DOY 136) sampled at 00:00, 06:00, 12:00, and 18:00 UTC (3 granules $\times$ 20s = 60s per burst). Total observation time: 240 seconds (4.0 minutes).
- **Rejection Accounting:**
  - Raw Flashes Evaluated: 2,658
  - Rejected (Out of domain bbox $[24.5^\circ, 31.5^\circ]\text{N}, [-88.0^\circ, -79.5^\circ]\text{W}$): 1,970 (74.12%)
  - Rejected (Outside Florida state boundary polygon): 507 (19.07%)
  - Accepted Florida Flashes: **181** (6.81%)
    - *Active Terrestrial ASOS / ERA5 Grid Cells:* **173** flashes
    - *Coastal Marine Grid Cells:* **8** flashes
- **Physical Hierarchy:** 4,534 groups measured (mean: 25.05/flash) and 9,243 events measured (mean: 51.07/flash). Legacy synthetic multipliers (12.0 and 28.0) were eliminated.
- **Quality & Navigation:** 0.0% uncorrectable L0 errors, 100% navigated L1b events, 0 duplicate flashes.

---

## 3. Data Quality & Spatial/Temporal Coverage Summary

| Metric | Total in 2024 | Present on Disk | Coverage % | Excluded / Missing | Excluded % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calendar Days** | 366 | 1 (May 15) | 0.27% | 365 days | 99.73% |
| **6-Hour Synoptic Cycles** | 1,464 | 4 cycles | 0.27% | 1,460 cycles | 99.73% |
| **Monitored Station-Windows** | 112,509 | 320 windows | 0.28% | 112,189 windows | 99.72% |
| **Oct–Dec Test Holdout Windows** | 28,073 | 0 windows | 0.00% | 28,073 windows | 100.00% |

> [!WARNING]
> Missing satellite granules must **never** be silently filled with zeros as confirmed lightning-free periods. Doing so conflates "no satellite observations" with "confirmed zero convection," causing catastrophic predictor bias. In this pipeline, missing coverage is explicitly marked with `glm_coverage_available == 0`.

---

## 4. Causally Valid GLM Feature Engineering & Leakage Controls

All GLM features for forecast cycle $t$ look back strictly into $(t - \Delta t, t]$:
- Flashes with $t_{\text{flash}} > t$ are strictly barred. (For instance, flashes in the 00:00:00 to 00:01:00 UTC burst occurred *after* $t = 00\text{Z}$, so they are not used at $00\text{Z}$, but are valid prior observations for $t = 06\text{Z}$).
- Features generated:
  - `glm_flash_count_25km_1h`, `glm_flash_count_50km_1h`: GLM flashes in $(t-1\text{h}, t]$ within 25 km and 50 km.
  - `glm_flash_count_25km_3h`, `glm_flash_count_50km_3h`: GLM flashes in $(t-3\text{h}, t]$ within 25 km and 50 km.
  - `glm_flash_count_25km_6h`, `glm_flash_count_50km_6h`: GLM flashes in $(t-6\text{h}, t]$ within 25 km and 50 km.
  - `glm_flash_density_50km_6h`: `glm_flash_count_50km_6h` / $(\pi \times 50^2\text{ km}^2)$ (flashes/$\text{km}^2$).
  - `glm_time_since_last_flash_min`: Minutes from most recent flash in 50 km buffer to $t$ (sentinel: 360.0 min).
  - `glm_statewide_flash_count_6h`: Statewide total accepted flashes.
  - `has_prior_flash_50km_6h`: Binary flag indicating $\ge 1$ flash in 50 km buffer.
  - `glm_coverage_available`: Binary flag (1 if valid GLM observations were retrieved on disk, 0 otherwise).

---

## 5. Model Comparison on Common Evaluation Cohort

All 5 configurations were evaluated on **exactly the same 320 monitored forecast windows** on May 15, 2024 (166 positives, 154 negatives; prevalence: 51.88%).

| Model Configuration | Feature Set | PR-AUC | ROC-AUC | Brier Score | Log Loss | F1 (th=0.5) | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Reference (Full Tuned)** | 21 ERA5 + 4 Spat/Diurn + ASOS Hist | **0.9630** | **0.9650** | **0.0819** | **0.2863** | **0.8989** | **0.8421** | **0.9639** |
| **Model B: ERA5 + Spatial/Diurnal** | 21 ERA5 + 4 Spat/Diurn | 0.8134 | 0.8565 | 0.1519 | 0.4700 | 0.8086 | 0.7317 | 0.9036 |
| **Model C: Baseline C (+ ASOS Hist)** | Model B + `prior_strict_ts_6h` | 0.9151 | 0.9171 | 0.1208 | 0.3876 | 0.8579 | 0.7850 | 0.9458 |
| **Model D: GLM-Enhanced** | Model C + 5 GLM Features (OOF) | 0.8081 | 0.8475 | 0.1555 | 0.5579 | 0.8046 | 0.7692 | 0.8434 |
| **Model E: Calibrated GLM-Enhanced** | Model D + Platt Scaling | 0.7802 | 0.8065 | 0.1779 | 0.5578 | 0.7435 | 0.7127 | 0.7771 |

---

## 6. Incremental Value & Paired Date-Block Bootstrap Uncertainty

A 1,000-iteration station/date-block bootstrap was performed on the common cohort to calculate paired differences ($\text{Model D} - \text{Model C}$) while preserving spatial correlations:

| Paired Difference Metric | Mean Difference | 95% Confidence Interval | p-value | Statistically Significant? | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\Delta\text{PR-AUC}$ (D vs C) | **-0.1063** | $[-0.1725, -0.0490]$ | **0.0010** | **YES** | Statistically significant degradation |
| $\Delta\text{ROC-AUC}$ (D vs C) | **-0.0696** | $[-0.1063, -0.0391]$ | **0.0010** | **YES** | Statistically significant degradation |
| $\Delta\text{Brier Score}$ (D vs C) | **+0.0347** | $[+0.0182, +0.0528]$ | **0.0010** | **YES** | Error increased (worse calibration) |
| $\Delta\text{F1 Score}$ (D vs C) | **-0.0537** | $[-0.0876, -0.0216]$ | **0.0010** | **YES** | Decision accuracy degraded |
| $\Delta\text{PR-AUC}$ (D vs A) | **-0.1551** | $[-0.2294, -0.0901]$ | **0.0010** | **YES** | Statistically significant degradation |
| $\Delta\text{ROC-AUC}$ (D vs A) | **-0.1182** | $[-0.1601, -0.0812]$ | **0.0010** | **YES** | Statistically significant degradation |

### Physical & Methodological Root Cause:
Why did GLM features degrade performance?
1. **Severe Temporal Under-sampling:** 1 minute of observation every 6 hours misses 99.72% of convective evolution. A storm that develops 2 hours before initialization is completely missed by a 60-second burst at $t-6\text{h}$.
2. **False Sense of Inactivity:** In the absence of continuous lightning tracking, a station with 0 flashes in a 60-second window is frequently misclassified as suppressed, even when high CAPE and moisture trigger severe convection within the upcoming 6 hours.
3. **Overfitting on Small Pilot Cohort:** In small sample regimes ($N=320$), tree/linear models over-weight the sparse lightning signals at the expense of robust thermodynamic predictors (`cape_J_kg`, `tcwv_kg_m2`, `bulk_shear_850_300_ms`).

---

## 7. Seasonal Threshold Investigation

Using pre-test validation folds (January–September 2024, $N = 84,436$ monitored windows), we evaluated whether season-conditioned operational decision thresholds improve precision-recall trade-offs over a global threshold:

| Strategy | Threshold Configuration | F1 Score | Precision | Recall | Missed Positives (FN) | False Alarms (FP) | Miss Rate % | False Alarm % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Optimal Threshold** | Fixed $\theta = 0.64$ | 0.8243 | 0.7858 | 0.8667 | **2,740** | 4,858 | **13.33%** | 8.23% |
| **Predefined Seasonal Thresholds** | Jan–Apr: 0.59, May–Aug: 0.66, Sep: 0.64 | **0.8253** | 0.7897 | 0.8642 | 2,791 | **4,733** | 13.58% | **8.00%** |
| **Platt Calibrated Probabilities** | Fixed $\theta = 0.50$ | 0.5889 | **0.9521** | 0.4262 | 11,796 | **441** | 57.38% | **0.70%** |
| **Uncalibrated Baseline** | Fixed $\theta = 0.50$ | 0.8048 | 0.7056 | **0.9363** | **1,309** | 8,031 | **6.37%** | 14.38% |

### Trade-Off Findings:
- **Global Threshold (0.64):** Maximizes balanced $F_1 = 0.8243$, capturing 86.67% of thunderstorm events with an 8.23% false alarm rate.
- **Seasonal Thresholds:** Tuning thresholds by season ($\theta = 0.59$ in winter, $\theta = 0.66$ in summer) achieves the highest overall $F_1 = 0.8253$ while reducing false alarms by 125 events over the pre-test period.
- **Operational Recommendation:** In safety-critical aviation applications, the **Seasonal Thresholds** provide the best risk management profile.

---

## 8. Final Recommendations & Roadmap to Full-Year GLM

1. **Retain Model C / Model A as Active Baseline:** Do not replace the existing tuned LightGBM model with the pilot GLM model.
2. **Implement Cloud-Streaming GLM Ingestion:** Full-year NetCDF downloads (497 GB) are prohibited by local storage limits. Implement direct S3 range requests or streaming xarray reads from `s3://noaa-goes16/GLM-L2-LCFA/` to extract station-buffered flash counts on-the-fly.
3. **Continuous 1-Hour Aggregation:** To test the true scientific hypothesis of satellite lightning value, features must integrate **continuous 100% temporal coverage** over $(t-1\text{h}, t]$ rather than 60-second burst snapshots.
