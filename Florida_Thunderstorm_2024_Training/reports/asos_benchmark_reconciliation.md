# Florida ASOS 2024: Benchmark Metrics Reconciliation Report

**Date**: October 27, 2026  
**Auditor**: Antigravity Autonomous Data Science & Verification Agent  
**Dataset Reference**: [`data/processed/asos_parser_fixed/florida_asos_modeling_ready_fixed_2024.parquet`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_parser_fixed/florida_asos_modeling_ready_fixed_2024.parquet)  
**Primary Report**: [`reports/asos_parser_fix_and_revalidation.md`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/reports/asos_parser_fix_and_revalidation.md)  
**Baseline Metric Records**: [`data/processed/asos_parser_fixed/baseline_metrics.csv`](file:///c:/Users/KIIT0001/Desktop/Data%20Collection/data/processed/asos_parser_fixed/baseline_metrics.csv)

---

## 1. Executive Summary & Objective

Following the discovery and fix of the callsign-substring parser bug (`KDTS` and `KTTS` false positives), this report provides an exhaustive methodological reconciliation of all baseline benchmarks for Florida thunderstorm forecasting in 2024.

Specifically, this report:
1. Reconciles all benchmark evaluation rules on an identical, strictly verified test-set row cohort ($N = 28,073$).
2. Formally fixes and decouples the definition of the **Climatology** benchmark into its proper components: a continuous probabilistic prediction ($\hat{P} = \bar{y}_{\text{train}}$) evaluated at threshold $\tau = 0.50$ (predicting non-thunderstorm for all windows) versus an **Always Positive** classification upper bound.
3. Audits the **Six-Hour Persistence** calculation to verify that lag-1 station status is derived strictly from the preceding cycle $[t-6\text{h}, t]$, that zero temporal leakage occurs, and that missing or incomplete observation windows are never bridged.
4. Explains the exact mathematical discrepancy between scikit-learn's trapezoidal `average_precision_score` ($0.3463$) and the naive step-product metric ($0.3162$) for binary persistence.
5. Rigorously reconciles the prior "Clean 85-Station Diagnostic Cohort" ($N = 27,427$, $F_1 = 0.5656$) with the newly restored "87-Station Master Cohort" ($N = 28,073$, $F_1 = 0.5623$), detailing why the two cohorts differ by $+646$ test windows and $+17$ genuine thunderstorm events.

---

## 2. Test Set Specification & Integrity Audit

All baseline evaluations in Phase 1 operate on the exact same chronological holdout window:

* **Evaluation Period**: October 1, 2024 00:00:00 UTC through December 31, 2024 18:00:00 UTC.
* **Eligible Terrestrial Windows**: 30,190 station-cycles across 87 terrestrial stations.
* **Excluded Incomplete Windows (Criterion C)**: 2,117 windows (7.01% unmonitored; `obs_count < 4`, `max_gap > 75` min, or `obs_span < 4.5` h).
* **Final Eligible Test Rows ($N$)**: **28,073** strictly monitored forecast windows.
* **Observed Thunderstorm Positives ($y = 1$)**: **1,918** windows (**6.8322%** test prevalence).
* **Observed Non-Thunderstorm Negatives ($y = 0$)**: **26,155** windows (**93.1678%** test prevalence).
* **Target Definition**: `target_asos_strict_crit_c`. A window is strictly positive if $\ge 1$ METAR report in $(t, t+6\text{h}]$ contains explicit present weather `TS`, `VCTS`, or genuine lightning remarks (`LTG...`, `TSB`, `TSE`). Standalone cumulonimbus cloud remarks (`CB alone`) are strictly excluded from generating positive labels.

---

## 3. Baseline Metrics Reconciliation Table

All metrics below are computed on the identical 28,073 test rows.

| Benchmark Model | Prediction Type | Classification Threshold ($\tau$) | TP | FP | TN | FN | Precision | Recall | $F_1$ Score | PR-AUC | ROC-AUC | Brier Score | Operational Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Class** | Binary Rule | $\hat{y} = 0$ ($\hat{P} = 0.0$) | 0 | 0 | 26,155 | 1,918 | 0.0000 | 0.0000 | 0.0000 | 0.0683 | 0.5000 | 0.0683 | Trivial lower bound; ignores all positive risk |
| **Climatology (Probabilistic)** | Continuous Prob. | $\tau = 0.50$ | 0 | 0 | 26,155 | 1,918 | 0.0000 | 0.0000 | 0.0000 | 0.0683 | 0.5000 | 0.0809 | Constant train prevalence $\hat{P} = 0.1997$; never exceeds 0.50 |
| **Always Positive** | Binary Rule | $\hat{y} = 1$ ($\hat{P} = 1.0$) | 1,918 | 26,155 | 0 | 0 | 0.0683 | 1.0000 | 0.1279 | 0.0683 | 0.5000 | 0.9317 | Trivial upper bound; alerts on every single cycle |
| **Prior 6h Persistence** | Lag-1 State | Prior Cycle strict TS $= 1$ | 1,074 | 828 | 25,327 | 844 | **0.5647** | **0.5600** | **0.5623** | **0.3463** | **0.7641** | **0.0596** | Genuine primary operational benchmark to beat |
| **CAPE Threshold** | Binary Rule | $\text{CAPE} \ge 1000\text{ J/kg}$ | 442 | 461 | 25,694 | 1,476 | 0.4895 | 0.2304 | 0.3134 | 0.3206 | 0.8064 | 0.0690 | Standard thermodynamic convective threshold |
| **Logistic Regression** | Continuous Prob. | $\tau = 0.50$ (uncalibrated) | 579 | 812 | 25,343 | 1,339 | 0.4162 | 0.3019 | 0.3499 | 0.2896 | 0.8178 | 0.0622 | 7 standard ERA5 environmental predictors |

---

## 4. Deep-Dive Methodological Reconciliations

### 4.1 Resolution of the Climatology Benchmark Definition
In earlier repair iterations, the entry labeled `Climatology` in `baseline_metrics.csv` showed:
`TP=1,918, FP=26,155, TN=0, FN=0, Recall=1.0, F1=0.1279`.

This was caused by an ambiguous implementation: the probability was defined as training prevalence ($\bar{y}_{\text{train}} = 0.1997$), but classification predictions were generated by thresholding at $\tau \le \bar{y}_{\text{train}}$, which forced every observation into the positive class ($\hat{y} = 1$). That represents an **Always Positive** classifier, not a standard calibrated climatological classifier.

**Reconciled Standard**:
1. **Probabilistic Climatology**: For any input, output $\hat{P}(y = 1) = \bar{y}_{\text{train}} = 13,228 / 66,255 = 0.199653$.
   - **Brier Score**:
     $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y}_{\text{train}})^2 = \bar{y}_{\text{test}} (1 - \bar{y}_{\text{train}})^2 + (1 - \bar{y}_{\text{test}}) (\bar{y}_{\text{train}})^2 = 0.080920$$
   - **ROC-AUC & PR-AUC**: A constant forecast provides zero discriminatory ranking, yielding $\text{ROC-AUC} = 0.5000$ and $\text{PR-AUC} = \bar{y}_{\text{test}} = 0.068322$.
2. **Classification at Standard Threshold $\tau = 0.50$**:
   - Because $0.199653 < 0.50$, the classifier predicts $\hat{y} = 0$ for all samples.
   - Confusion matrix: $\text{TP} = 0, \text{FP} = 0, \text{TN} = 26,155, \text{FN} = 1,918$. Precision, Recall, and $F_1$ are identically $0.0$.
3. **Always Positive**: Maintained as an explicit boundary benchmark ($\text{Precision} = 0.0683, \text{Recall} = 1.0, F_1 = 0.1279, \text{Brier} = 0.9317$).

### 4.2 Precision-Recall AUC for Discrete Binary Persistence
In `baseline_metrics.csv`, the PR-AUC for persistence was recorded as $0.316191$.
A direct scikit-learn call `average_precision_score(y_test, y_pred_persistence)` yields **$0.346255$**.

**Explanation of Difference**:
- A discrete binary prediction $\hat{y} \in \{0, 1\}$ produces a precision-recall curve with two operating points:
  1. Threshold $> 1$: $\text{Recall} = 0$, $\text{Precision} = 1.0$ (or undefined).
  2. Threshold $\in (0, 1]$: $\text{Recall} = 0.559958$, $\text{Precision} = 0.564669$.
  3. Threshold $\le 0$: $\text{Recall} = 1.0$, $\text{Precision} = \bar{y}_{\text{test}} = 0.068322$.
- The naive calculation evaluated solely the rectangle under the first operating point:
  $$\text{PR}_{\text{naive}} = \text{Precision} \times \text{Recall} = 0.564669 \times 0.559958 = 0.316191$$
- Scikit-learn's `average_precision_score` computes the full weighted sum across the step function:
  $$\text{PR-AUC} = (R_1 - R_0) P_1 + (R_2 - R_1) P_2 = (0.559958 \times 0.564669) + (1.0 - 0.559958) \times 0.068322 = 0.316191 + 0.030064 = \mathbf{0.346255}$$
Both definitions reflect the exact same discrete confusion matrix ($\text{TP}=1,074, \text{FP}=828, \text{TN}=25,327, \text{FN}=844$). Moving forward, scikit-learn's standard `average_precision_score` is adopted.

### 4.3 Persistence Window Continuity & Leakage Audit
We audited whether `prior_strict_ts_6h` accurately represents the preceding observation cycle:
1. **Preceding Time Window**: For a forecast initialized at $t$, persistence requires that the station reported a strict thunderstorm in $[t-6\text{h}, t]$.
2. **Missing / Incomplete Windows**: If the cycle at $t-6\text{h}$ was missing or incomplete under Criterion C, `prev_strict_target` is `NaN`. The logic:
   ```python
   (cycle_diff_hours == 6.0) & (prev_strict_target == 1.0)
   ```
   strictly evaluates to `False`. Thus, persistence **never bridges missing or unmonitored windows**. Across the 112,509 monitored rows, exactly 0 bridging errors occurred.
3. **Partition Boundary Transition**: Because lag shifting was performed over the entire time-ordered station sequence prior to test-set slicing, predictions at the partition boundary (`2024-10-01 00:00:00 UTC`) correctly reference the verified final validation window (`2024-09-30 18:00:00 UTC`).

---

## 5. Cohort Reconciliation: Clean 85-Station vs Restored 87-Station

The table below reconciles the diagnostic 85-station cohort (which removed Destin and Shuttle Landing Facility) with the restored 87-station master cohort:

| Cohort Dimension | Clean 85-Station Diagnostic | Restored 87-Station Master | Net Difference | Rationale & Scientific Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Active Stations Evaluated** | 83 reporting in Q4 | 85 reporting in Q4 | +2 stations | KDTS and KTTS were restored after fixing the parser bug |
| **Total Monitored Test Windows** | 27,427 | 28,073 | +646 windows | DTS contributed 348 windows; TTS contributed 298 windows |
| **Test Positives ($y = 1$)** | 1,901 | 1,918 | +17 windows | DTS contributed 9 genuine TS; TTS contributed 8 genuine TS |
| **Test Set Prevalence** | 6.9311% | 6.8322% | -0.0989% | Autumn/winter convective quiescence at DTS (2.59%) and TTS (2.68%) |
| **Persistence TP** | 1,071 | 1,074 | +3 | DTS had 2 persistent events; TTS had 1 persistent event |
| **Persistence FP** | 815 | 828 | +13 | DTS had 7 false alarms; TTS had 6 false alarms |
| **Persistence TN** | 24,711 | 25,327 | +616 | Correctly recognized quiescent periods at DTS and TTS |
| **Persistence FN** | 830 | 844 | +14 | Unheralded storm initiations at DTS (7) and TTS (7) |
| **Persistence Precision** | 0.5679 | 0.5647 | -0.0032 | Statistically indistinguishable ($\Delta < 0.004$) |
| **Persistence Recall** | 0.5634 | 0.5600 | -0.0034 | Statistically indistinguishable ($\Delta < 0.004$) |
| **Persistence $F_1$ Score** | **0.5656** | **0.5623** | **-0.0033** | Authentic, robust operational benchmark |
| **Persistence ROC-AUC** | 0.7657 | 0.7642 | -0.0015 | Virtually identical discrimination |
| **Persistence Brier Score** | 0.0600 | 0.0596 | -0.0004 | Slight Brier score improvement due to higher TN count |

> [!IMPORTANT]
> **Conclusion on Cohort Comparability**:  
> The 85-station and 87-station cohorts must not be conflated, but the metrics demonstrate that restoring DTS and TTS does **not** degrade benchmark integrity or introduce noise. With an $F_1$ score difference of only **0.0033**, the 87-station cohort is confirmed as the single authoritative, complete ground-truth baseline for Florida ASOS modeling.

---

## 6. Authoritative Baseline Target for Supervised Modeling

Any candidate machine-learning model (Logistic Regression, LightGBM, XGBoost) trained on the 2024 ASOS dataset must satisfy the following criteria on the holdout test set (October 1 – December 31, 2024, $N = 28,073$):

1. **Surpass the Prior 6h Persistence $F_1$ Benchmark**: $F_1 > \mathbf{0.5623}$.
2. **Surpass the Prior 6h Persistence Discrimination**: $\text{ROC-AUC} > \mathbf{0.7642}$ and $\text{PR-AUC} > \mathbf{0.3463}$.
3. **Surpass the Environmental Logistic Regression Benchmark**: $F_1 > \mathbf{0.3500}$ and $\text{ROC-AUC} > \mathbf{0.8178}$.
4. **Demonstrate Superior Probability Calibration**: $\text{Brier Score} < \mathbf{0.0596}$.
