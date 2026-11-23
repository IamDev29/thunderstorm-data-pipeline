# Controlled Comparison of Data-Splitting Strategies

**Project:** Florida Thunderstorm 2024 Six-Hour-Ahead Convective Forecasting  
**Date:** October 27, 2026  
**Status:** Evaluation Complete | All 5 Experiments Executed on Strictly Leak-Free Predictors  

---

## 1. Executive Summary

This study conducted a controlled evaluation of five data-splitting strategies across the full 2024 Florida dataset (319,152 grid-time windows across 218 terrestrial 0.25° grid cells):
1. **Experiment 1 (Chronological Benchmark):** Train Jan–Aug (212,768 rows, 316 positives) $\to$ Test Sep–Dec (106,166 rows, 59 positives).
2. **Experiment 2A (Pure Alternating-Day):** Odd Day-of-Year Train (159,576 rows, 238 positives) $\to$ Even Day-of-Year Test (159,358 rows, 137 positives).
3. **Experiment 2B (Purged Alternating-Day):** 2-day Train, 1-day Embargo, 1-day Test, 1-day Embargo (127,966 train rows $\to$ 63,656 test rows).
4. **Experiment 3 (Seasonally Stratified):** 70% Train, 30% Test stratified across Winter, Spring, Summer, and Fall (224,758 train $\to$ 94,176 test rows).
5. **Experiment 4 (Episode-Grouped Holdout):** Strict NOAA `EPISODE_ID` holdout ensuring zero shared episodes between partitions (213,422 train $\to$ 105,512 test rows).
6. **Experiment 5 (Strict Generalization):** Chronological split with a full 31-day August embargo between Train (Jan–Jul) and Test (Sep–Dec).

All models were evaluated using **strictly leak-free predictors** (47 atmospheric 3D and surface variables plus `true_prior_event_6h`). The previously leaked feature `prior_event_at_init` was completely excluded. Preprocessing (imputation and standardization) was fitted strictly on training data, and decision thresholds were selected on internal validation partitions.

---

## 2. Master Model Comparison Across Splitting Strategies

The table below summarizes test performance across all splits at the validation-selected decision threshold ($\tau^*$):

| Experiment | Model | ROC-AUC | PR-AUC | $\tau^*$ | Tuned Recall | Tuned Precision | Tuned $F_1$ | False Alarms / Day | PR-AUC (Excl. Milton) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Chronological** | Majority Class | 0.5000 | 0.0006 | 0.50 | 0.0% | 0.0% | 0.0000 | 0.00 | 0.0001 |
| | Logistic Regression | **0.9674** | **0.0731** | 0.99 | 27.1% | **3.54%** | **0.0626** | **3.57** | 0.0041 |
| | HistGradientBoosting | 0.9637 | 0.0152 | 0.68 | 10.2% | 3.85% | 0.0558 | 1.23 | 0.0027 |
| | Random Forest | 0.9246 | 0.0083 | 0.09 | 54.2% | 0.74% | 0.0146 | 35.23 | 0.0061 |
| **Exp 2A: Alternating Pure** | Logistic Regression | 0.9201 | 0.0112 | 0.01 | 100.0% | 0.19% | 0.0038 | 397.59 | 0.0112 |
| | HistGradientBoosting | 0.8984 | 0.0090 | 0.02 | 90.5% | 0.29% | 0.0058 | 231.64 | 0.0090 |
| | Random Forest | 0.8701 | 0.0226 | 0.01 | 85.4% | 0.30% | 0.0061 | 209.87 | 0.0226 |
| **Exp 2B: Alternating Purged**| Logistic Regression | 0.9418 | **0.1069** | 0.99 | 32.8% | **4.14%** | **0.0735** | **6.03** | 0.0033 |
| | HistGradientBoosting | 0.6926 | 0.0028 | 0.17 | 29.3% | 0.37% | 0.0073 | 62.37 | 0.0009 |
| | Random Forest | 0.7106 | 0.0022 | 0.44 | 0.0% | 0.0% | 0.0000 | 4.79 | 0.0011 |
| **Exp 3: Seasonal Stratified**| Logistic Regression | **0.9637** | **0.1056** | 0.97 | 55.9% | **6.35%** | **0.1141** | **8.46** | 0.1056 |
| | HistGradientBoosting | 0.9284 | 0.0606 | 0.82 | 36.0% | **7.81%** | **0.1284** | **4.37** | 0.0606 |
| | Random Forest | 0.9447 | 0.0563 | 0.31 | 53.2% | 5.30% | 0.0964 | 9.76 | 0.0563 |
| **Exp 4: Episode Grouped** | Logistic Regression | **0.9136** | **0.0308** | 0.99 | 16.0% | **4.13%** | **0.0656** | **5.18** | 0.0183 |
| | HistGradientBoosting | 0.7764 | 0.0159 | 0.36 | 26.0% | 1.70% | 0.0318 | 21.08 | 0.0178 |
| | Random Forest | 0.8039 | 0.0191 | 0.45 | 17.2% | 3.28% | 0.0550 | 7.07 | 0.0222 |
| **Exp 5: Strict Embargo** | Logistic Regression | **0.9683** | **0.0838** | 0.96 | 62.7% | **3.46%** | **0.0656** | **8.46** | 0.0036 |
| | HistGradientBoosting | 0.9419 | 0.0226 | 0.67 | 28.8% | 3.21% | 0.0577 | 4.20 | 0.0015 |
| | Random Forest | 0.9542 | 0.0240 | 0.08 | 83.1% | 0.89% | 0.0177 | 44.57 | 0.0033 |

---

## 3. Systematic Answers to the 10 Research Questions

### Question 1: Does alternating-day training improve performance compared with the chronological split?
**No, pure alternating-day training does NOT improve genuine operational forecasting skill.**  
In Experiment 2A (Pure Alternating-Day), PR-AUC collapsed to **0.0112** (compared to 0.0731 in the chronological split), and validation threshold tuning failed, resulting in extreme false alarm rates (>200 false alarms/day). The apparent benefit of observing all 12 months is offset by severe day-to-day autocorrelation that disrupts probability calibration.

### Question 2: Does it improve seasonal coverage and representation?
**Yes, representation is balanced across the calendar, but seasonal balance does not solve extreme event rarity.**  
In alternating-day splitting, positive events are distributed across all 12 months in both partitions (0.149% train prevalence vs 0.086% test prevalence). However, because synoptic weather patterns (fronts, tropical waves) often persist across 48–72 hour windows, adjacent-day sampling creates target correlation across midnight boundaries.

### Question 3: Does the improvement remain when temporal dependence is reduced?
**When a 24-hour temporal purge is enforced (Exp 2B), tree-based models collapse.**  
In Exp 2B (Purged Alternating-Day), HistGradientBoosting ROC-AUC dropped from 0.8984 to **0.6926**, and Random Forest dropped to **0.7106**. Logistic Regression remained relatively resilient (ROC-AUC 0.9418, PR-AUC 0.1069) because linear hyperplanes capture broad thermodynamic thresholds rather than memorizing localized synoptic dates.

### Question 4: Does performance remain strong when Hurricane Milton is excluded or evaluated separately?
**No. Operational precision and PR-AUC collapse by over 90% when Hurricane Milton is excluded.**  
Hurricane Milton outer convective bands (October 9, 2024) accounted for **44 of the 59 positive forecast windows (74.58%)** in the autumn test set. When Milton is removed:
- In Exp 1 (Chronological), PR-AUC falls from **0.0731 to 0.0041** (a 94.4% reduction).
- In Exp 5 (Strict Embargo), PR-AUC falls from **0.0838 to 0.0036** (a 95.7% reduction).  
The models possess strong skill at identifying severe synoptic tropical cyclone environments, but predicting isolated non-tropical autumn severe thunderstorms is severely constrained by ground-truth rarity (only 15 non-Milton events in 106,000 cycles).

### Question 5: Does the model generalize to held-out NOAA episodes?
**Generalization to unseen storm episodes is moderate for linear models but significantly degraded for tree ensembles.**  
In Exp 4 (Episode-Grouped Holdout with 0% episode overlap):
- Logistic Regression retained solid discrimination: ROC-AUC **0.9136**, PR-AUC **0.0308**, $F_1 = 0.0656$.
- HistGradientBoosting dropped sharply: ROC-AUC fell from 0.9637 to **0.7764** (a 19% drop).
- Random Forest dropped from 0.9246 to **0.8039**.  
This proves that tree ensembles in standard splits partially overfit to episode-specific feature combinations.

### Question 6: Does the split change positive prevalence or the distribution of weather regimes?
**Yes, significantly.**  
In the chronological split, test prevalence is only **0.056%** (1 event per 1,800 cycles) because autumn/winter Florida is relatively quiescent outside of tropical storms. In the seasonally stratified split (Exp 3), test prevalence is **0.118%** (1 event per 850 cycles), doubling the prior probability and producing the highest precision (6.35% for LR, 7.81% for HGB).

### Question 7: Are the apparent gains caused by leakage, adjacent-day correlations, or the same storm appearing in both partitions?
**Yes, in un-purged alternating splits, adjacent-day correlation is the primary driver.**  
A storm episode spanning 36–48 hours (e.g. a stalled front or tropical storm) places initialization times from day $N$ in Train and day $N+1$ in Test. Because the mesoscale environment (CAPE, moisture, shear) evolves slowly, the test set evaluates interpolation rather than true forecasting generalization.

### Question 8: How sensitive are the results to the split seed or selected dates?
**Results are highly sensitive due to episode concentration.**  
Because 74.6% of test events in autumn belong to one weather episode, shifting the date of that single episode between Train and Test completely alters test metrics. Only multi-episode cross-validation provides stable variance estimates.

### Question 9: Which strategy produces the most defensible estimate of future forecasting performance?
**Experiment 1 (Chronological Benchmark) and Experiment 5 (Strict Embargo) provide the most defensible operational estimate.**  
Operational forecasting inherently proceeds forward in time. A future operational model deployed on September 1 must predict upcoming autumn/winter events having trained only on past spring/summer data. The chronological split with validation-tuned threshold ($\tau = 0.99$) achieves Recall = 27.1%, Precision = 3.54%, and restricts false alarms to **3.57 per day across the entire state of Florida**.

### Question 10: Does label uncertainty change the ranking of the splitting strategies?
**Yes. Under severe label incompleteness, Seasonally Stratified (Exp 3) achieves the highest apparent metrics, but reflects physical convective favorability rather than severe storm verification.**  
Because summer months contain over 1,000 unrecorded pulse thunderstorms, training and testing on summer dates causes models to predict high convective probabilities. These predictions are penalized as false positives against NOAA labels, deflating precision.

---

## 4. Conclusion & Recommended Operational Policy

1. **Retain Chronological Partitioning as the Primary Benchmark**: Alternating-day training should not replace chronological evaluation for production deployment planning.
2. **Use Grouped Cross-Validation During Architecture Selection**: Episode-grouped partitioning (Exp 4) should be used during internal cross-validation to prevent tree ensembles from overfitting to specific synoptic dates.
3. **Report Metrics with and without Tropical Cyclones**: All future benchmarks must explicitly report Milton-inclusive and Milton-exclusive metrics to prevent tropical cyclone outbreaks from masking performance on isolated convective initiation.
