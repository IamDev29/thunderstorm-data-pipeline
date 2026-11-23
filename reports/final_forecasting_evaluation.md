# Final Forecasting Evaluation: Synthesis and Scientific Assessment

**Project:** Florida Thunderstorm 2024 Prediction  
**Dataset:** 2024 Full-Year 3D Reanalysis and Satellite Ingest  
**Date:** October 27, 2026  
**Status:** Audit & Multi-Split Benchmark Complete | Defensible Baseline Established  

---

## 1. Executive Summary

This evaluation synthesizes the comprehensive thunderstorm label audit and the controlled multi-split machine learning experiments for 6-hour-ahead convective storm forecasting in Florida across 2024.

### Key Conclusions
1. **Model Capability (Narrow Scientific Claim)**:  
   Using strictly leak-free predictors (surface and pressure-level thermodynamic and kinematic fields at initialization time $t$, plus historical convection in the preceding $(t-6\text{h}, t]$ window), the calibrated Logistic Regression model demonstrates **defensible skill in forecasting severe thunderstorm environments six hours in advance**. In the strict chronological test set (September–December 2024), at a validation-tuned probability threshold ($\tau^* = 0.99$), the model achieves:
   - **ROC-AUC:** $0.9674$
   - **PR-AUC:** $0.0731$ (122$\times$ higher than the climatology baseline of $0.0006$)
   - **Recall:** $27.12\%$ ($16$ of $59$ positive forecast windows captured)
   - **Precision:** $3.54\%$ (1 alert in 28 corresponds to a confirmed severe storm report)
   - **Operational False Alarms:** **3.57 false alarms per day** across the entire state of Florida (218 grid cells).
2. **Persistence Baseline Disproven**:  
   Forecasting six hours in advance cannot rely on persistence. A true leak-free prior event indicator (`true_prior_event_6h`) achieves an $F_1$ score of only $0.0081$ and recall of $1.69\%$. Over $91.5\%$ of confirmed severe storm windows represent newly initiated convection.
3. **Data-Splitting Strategy Findings**:  
   - **Pure alternating-day splitting** suffers from severe day-to-day weather autocorrelation and calibration failure, collapsing PR-AUC to $0.0112$ and triggering over 200 false alarms per day.
   - **Episode-grouped holdouts** reveal that tree-based models (HistGradientBoosting and Random Forest) experience significant degradation (ROC-AUC drops from ~0.95 to ~0.78), indicating that tree ensembles partially overfit to synoptic dates in un-grouped splits.
   - **Linear models** (Logistic Regression with class balancing) generalize most robustly across all splits, maintaining ROC-AUC $> 0.91$.
4. **Dominance of Hurricane Milton**:  
   Hurricane Milton's outer-band tornado and severe wind outbreak on October 9, 2024 accounted for **44 of the 59 positive test windows (74.58%)**. When Hurricane Milton is excluded, test PR-AUC falls to $0.0041$. The model is exceptionally skilled at detecting organized tropical cyclone severe convective environments, but forecasting isolated autumn severe convective initiation remains severely data-limited.

---

## 2. Definitive Model Comparison Across Evaluation Strategies

| Evaluation Paradigm | Primary Model | ROC-AUC | PR-AUC | Tuned Recall | Tuned Precision | Tuned $F_1$ | False Alarms / Day |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chronological Benchmark (Exp 1)** | Logistic Regression ($\tau^*=0.99$) | **0.9674** | **0.0731** | 27.1% | 3.54% | **0.0626** | **3.57** |
| | HistGradientBoosting ($\tau^*=0.68$) | 0.9637 | 0.0152 | 10.2% | 3.85% | 0.0558 | **1.23** |
| **Purged Alternating-Day (Exp 2B)** | Logistic Regression ($\tau^*=0.99$) | 0.9418 | **0.1069** | 32.8% | 4.14% | **0.0735** | 6.03 |
| | HistGradientBoosting ($\tau^*=0.17$) | 0.6926 | 0.0028 | 29.3% | 0.37% | 0.0073 | 62.37 |
| **Seasonally Stratified (Exp 3)** | HistGradientBoosting ($\tau^*=0.82$) | 0.9284 | 0.0606 | 36.0% | **7.81%** | **0.1284** | 4.37 |
| | Logistic Regression ($\tau^*=0.97$) | 0.9637 | 0.1056 | 55.9% | 6.35% | 0.1141 | 8.46 |
| **Episode-Grouped Holdout (Exp 4)** | Logistic Regression ($\tau^*=0.99$) | 0.9136 | 0.0308 | 16.0% | 4.13% | 0.0656 | 5.18 |
| | Random Forest ($\tau^*=0.45$) | 0.8039 | 0.0191 | 17.2% | 3.28% | 0.0550 | 7.07 |
| **Strict Embargo (Exp 5)** | Logistic Regression ($\tau^*=0.96$) | **0.9683** | 0.0838 | 62.7% | 3.46% | **0.0656** | 8.46 |
| | HistGradientBoosting ($\tau^*=0.67$) | 0.9419 | 0.0226 | 28.8% | 3.21% | 0.0577 | 4.20 |

---

## 3. Narrow Scientific Statement of Model Performance

What the model **has demonstrated**:
1. High discrimination capability (ROC-AUC $\approx 0.967$) in distinguishing synoptic and mesoscale conditions that produce NWS-confirmed severe storm reports from quiescent background days.
2. Significant enrichment of positive probability: at $\tau^* = 0.99$, the model improves positive prevalence from the base rate of 1 in 1,800 ($0.056\%$) up to 1 in 28 ($3.54\%$), representing a **63-fold precision gain**.
3. Practical operational utility: generating only **3.5 false alarms per day** across 218 grid cells while successfully warning on more than one-quarter of severe convective episodes six hours in advance.

What the model **has NOT demonstrated**:
1. It does not predict precise storm-cell trajectory or convective morphology (cell vs line vs cluster).
2. It cannot distinguish between a severe thunderstorm that causes recorded damage and an intense thunderstorm that occurs over unpopulated pine flatwoods or coastal swamps.
3. It does not generalize equally to all meteorological regimes; skill is dominated by tropical cyclone convective outer bands and synoptic fronts.

---

## 4. Remaining Scientific Limitations

1. **Extreme Ground-Truth Imbalance**: With only 375 positive 6-hour windows in the entire year (0.12% prevalence), test metrics are dominated by a handful of high-impact storm episodes.
2. **Label Incompleteness in Summer**: Over 1,000 summer pulse thunderstorms produce heavy precipitation, cloud-to-ground lightning, and gusty winds without generating recorded property damage in NOAA Storm Events. Models predicting these events are penalized as false alarms.
3. **Sparse Satellite Lightning Archive**: Full-year GOES-16 GLM granules are currently processed only for a pilot period (January, February, May). Expanding GLM to all 12 months is essential to create a definitive, sensor-based convective ground truth.

---

## 5. Next Research Priorities

1. **Full-Year GOES-16 GLM Processing**: Ingest and process the complete 366-day GLM Level-2 archive from NOAA AWS S3 to establish an objective, sensor-based lightning ground truth independent of human reporting.
2. **Multi-Task / Auxiliary Objective Formulation**: Train models with a secondary objective predicting general convective precipitation or lightning occurrence alongside the primary severe storm target.
3. **Spatial Blocked Cross-Validation**: Evaluate regional transferability (e.g., training on North Florida / Panhandle and evaluating on South Florida / Everglades).
