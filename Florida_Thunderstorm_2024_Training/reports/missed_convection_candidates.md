# Missed Convection Candidates & Label Incompleteness Audit

**Project:** Florida Thunderstorm 2024 Prediction  
**Dataset:** 2024 Full-Year Reanalysis and Satellite Ingest  
**Date:** October 27, 2026  
**Status:** Audit Complete | Candidates Separated from Ground Truth  

---

## 1. Executive Summary & Scientific Disclaimer

> [!IMPORTANT]
> **Candidate Labels Are NOT Ground-Truth Positives**: Samples identified in this queue are **candidates for meteorological review only**. They must **never** be silently merged into the primary supervised training or benchmark evaluation targets. NOAA Storm Events captures severe weather reports that meet NWS warning thresholds (winds $\ge 50$ kts, hail $\ge 1$ inch, tornadoes, or damaging lightning); absence of a NOAA report does not prove the absence of ordinary convection.

A comprehensive audit of the 318,559 NOAA-negative six-hour forecasting cycles revealed significant evidence of unrecorded convective activity:
1. **Tier 1 (Satellite Lightning Consensus):** **26 cycles** have confirmed GOES-16 GLM lightning detections (1 to 38 flashes, up to 1,064 optical events) during verified satellite monitoring periods where NOAA recorded zero severe weather reports.
2. **Tier 2 (Extreme Physical Convection):** **96 cycles** exhibited extreme thermodynamic and convective signatures ($\text{CAPE} \ge 2,000\text{ J/kg}$, convective precipitation $\ge 5.0\text{ mm}$, total column water vapour $\ge 50\text{ kg/m}^2$, and physical favorability index $\ge 0.70$) without matching NOAA severe storm reports.
3. **Tier 3 (Moderate Convective Activity):** **1,323 cycles** demonstrated active convective precipitation ($\ge 2.0\text{ mm}$) and moderate instability ($\text{CAPE} \ge 1,500\text{ J/kg}$).

Across the entire full-year dataset, **NOAA-confirmed severe events account for only 375 cycles (0.12% prevalence)**, whereas physically active convection occurs in **over 53,000 cycles (16.70%)**.

---

## 2. GOES-16 GLM Lightning Evidence (Tier 1)

During the pilot GLM observation period (May 2024), 26 cycles recorded verified lightning flashes with zero corresponding NOAA Storm Events records:

| Forecast Init Time (UTC) | Latitude | Longitude | Grid Cell ID | GLM Flashes | GLM Events | CAPE (J/kg) | Convective Precip (mm) | TCWV (kg/m²) | Favorability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2024-05-15 00:00:00 | 29.50 | -82.50 | FL_29.5_-82.5 | 38 | 1,064 | 1,482.0 | 3.23 | 50.15 | 0.85 |
| 2024-05-15 00:00:00 | 29.75 | -82.25 | FL_29.75_-82.25 | 24 | 672 | 1,620.5 | 2.85 | 49.80 | 0.82 |
| 2024-05-15 06:00:00 | 30.00 | -81.75 | FL_30.0_-81.75 | 19 | 456 | 980.2 | 1.95 | 48.90 | 0.74 |
| 2024-05-15 06:00:00 | 30.25 | -81.50 | FL_30.25_-81.5 | 12 | 320 | 850.0 | 1.40 | 48.10 | 0.71 |
| 2024-05-15 12:00:00 | 28.50 | -81.50 | FL_28.5_-81.5 | 15 | 390 | 1,890.0 | 4.10 | 51.20 | 0.88 |
| 2024-05-15 18:00:00 | 27.75 | -82.50 | FL_27.75_-82.5 | 31 | 890 | 2,150.0 | 5.40 | 52.40 | 0.91 |

*Note: The complete list of all 26 Tier 1 candidates is exported in `data/processed/audit_v2/missed_convection_candidates.csv`.*

### GLM Monitoring Tiers
- **COVERED_LIGHTNING:** 26 cycles (100% of these detected lightning, but 0% had NOAA severe reports).
- **INCOMPLETE_COVERAGE:** 746 cycles (1 to 9 granules, monitoring duration < 200s; insufficient to confirm non-convective status).
- **UNOBSERVED:** 318,380 cycles (99.76% of all full-year 6h windows had 0 GLM granules).

---

## 3. Atmospheric Feature Distributions: NOAA-Positive vs. Negative Groups

The table below contrasts thermodynamic and kinematic distributions between confirmed NOAA positives and candidate negative groups:

| Variable | Group | Mean | Median | Std Dev | 25th Pct | 75th Pct |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CAPE (J/kg)** | NOAA_POSITIVE | 1,180.5 | 890.0 | 1,024.1 | 310.0 | 1,820.0 |
| | GLM_LIGHTNING_NOAA_NEG | 1,450.2 | 1,320.0 | 650.4 | 940.0 | 1,980.0 |
| | HIGH_CONVECTION_NOAA_NEG | 2,340.5 | 2,210.0 | 510.2 | 1,920.0 | 2,680.0 |
| | ALL_NOAA_NEGATIVE | 412.3 | 45.0 | 682.1 | 0.0 | 540.0 |
| **Convective Precip (mm)** | NOAA_POSITIVE | 0.68 | 0.05 | 1.32 | 0.00 | 0.69 |
| | GLM_LIGHTNING_NOAA_NEG | 1.78 | 0.80 | 2.17 | 0.00 | 3.23 |
| | HIGH_CONVECTION_NOAA_NEG | 3.50 | 3.02 | 1.59 | 2.41 | 4.05 |
| | ALL_NOAA_NEGATIVE | 0.08 | 0.00 | 0.43 | 0.00 | 0.00 |
| **Total Water Vapour (kg/m²)** | NOAA_POSITIVE | 47.23 | 46.56 | 9.28 | 41.27 | 52.83 |
| | GLM_LIGHTNING_NOAA_NEG | 48.71 | 49.33 | 2.42 | 47.88 | 50.15 |
| | ALL_NOAA_NEGATIVE | 33.81 | 33.75 | 15.75 | 21.60 | 47.14 |
| **Favorability Index [0, 1]** | NOAA_POSITIVE | 0.52 | 0.54 | 0.15 | 0.41 | 0.63 |
| | GLM_LIGHTNING_NOAA_NEG | 0.68 | 0.77 | 0.21 | 0.49 | 0.85 |
| | ALL_NOAA_NEGATIVE | 0.40 | 0.39 | 0.12 | 0.31 | 0.46 |

### Meteorological Insights
1. **Thermodynamic Similarity**: The 26 `GLM_LIGHTNING_NOAA_NEG` cases exhibit higher mean CAPE (1,450 J/kg) and convective precipitation (1.78 mm) than the NOAA severe positives (1,180 J/kg and 0.68 mm). This is consistent with afternoon sea-breeze thunderstorms that produce lightning and heavy local rain without reaching NWS severe wind/hail criteria.
2. **False Positives in Machine Learning**: When supervised models predict a high probability of convection in summer afternoons, they are identifying genuine thermodynamic instability and updrafts. In the absence of an independent ground-truth lightning sensor, these valid physical detections are penalized as false positives.

---

## 4. Candidate Missed Convection Queue Summary

| Evidence Tier | Confidence | Count | Primary Evidence | Recommended Action |
| :--- | :--- | :---: | :--- | :--- |
| **Tier 1: Satellite Lightning** | HIGH | 26 | Verified GOES-16 GLM flashes $\ge 1$ under covered monitoring | Candidate for weak-label benchmark experiment |
| **Tier 2: Strong Physical Convection** | MEDIUM | 96 | CP $\ge 5.0\text{ mm}$, CAPE $\ge 2,000\text{ J/kg}$, TCWV $\ge 50\text{ kg/m}^2$, Fav $\ge 0.70$ | Review queue for case-study analysis |
| **Tier 3: Moderate Physical Convection** | LOW | 1,323 | CP $\ge 2.0\text{ mm}$, CAPE $\ge 1,500\text{ J/kg}$, Fav $\ge 0.50$ | Retained as unlabeled/ambiguous background |
| **Total Review Candidates** | | **1,445** | Multi-source physical or optical evidence | **Separated from primary supervised training set** |

### Monthly Distribution of Candidates
- **Winter (Jan-Feb):** 0 candidates (quiescent atmosphere).
- **Spring (Mar-May):** 138 candidates (13 in Mar, 5 in Apr, 120 in May).
- **Summer (Jun-Aug):** 1,025 candidates (238 in Jun, 518 in Jul, 269 in Aug — 70.9% of all candidates).
- **Fall (Sep-Dec):** 282 candidates (213 in Sep, 51 in Oct, 18 in Nov, 0 in Dec).

---

## 5. Label Policy & Recommendations

1. **Maintain Target Purity**: The primary forecasting target `future_noaa_event_6h` must strictly represent **NOAA-confirmed severe events**.
2. **Do Not Relabel Automatically**: Atmospheric variables must never be used to relabel NOAA negatives as positives.
3. **Secondary Weak-Label Evaluation**: If exploratory weak-label models are evaluated, they must be benchmarked on a distinct column (e.g. `weak_convection_target`) and clearly reported as predicting convective occurrence rather than severe storm hazards.
