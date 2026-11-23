# Six-Hour-Ahead Florida Thunderstorm Forecast Evaluation Report (Audited & De-Leaked)

**Generated:** 2026-10-27 06:10:34 UTC  
**Dataset:** `data/processed/florida_thunderstorm_2024_forecasting_6h.csv`  
**Forecast Initialization ($t$):** Four daily synoptic cycles: 00:00, 06:00, 12:00, 18:00 UTC  
**Target Interval:** $(t, t + 6\text{ hours}]$  
**Target Variable:** `future_noaa_event_6h` (1 if confirmed NOAA convective event occurred in window, 0 otherwise)  
**Historical Window:** $(t - 6\text{ hours}, t]$ (`true_prior_event_6h`, strictly leak-free)  

---

## 1. Forecast Target Definition & Leakage Safeguards

### Exact Timing Conventions:
1. **Forecast Target Window:** $(t, t + 6\text{h}]$: An event occurring at $t_{event}$ belongs to initialization $t$ if and only if $t < t_{event} \le t + 6\text{h}$.
2. **Prior Event Predictor Window:** $(t - 6\text{h}, t]$ (`true_prior_event_6h`): Strictly precedes initialization $t$. The intersection $(t - 6\text{h}, t] \cap (t, t + 6\text{h}] = \emptyset$, guaranteeing zero temporal leakage.
3. **Audit Finding on Original Persistence Baseline:**
   The previously reported persistence score ($F_1 = 0.6271$) was caused by a 3-hour lookahead leakage in `prior_event_at_init` (which sampled events up to $t + 3\text{h}$). In this audited report, `true_prior_event_6h` replaces the leaked feature.
4. **Year-End Truncation:** Cycle `2024-12-31 18:00:00` extends into 2025 and is excluded from test evaluation (`TRUNCATED_YEAR_END`).

---

## 2. Chronological Forecast Evaluation Results (Months 09–12 Test Set)

- **Test Period:** Months 09–12 (September 1 to December 31, 2024, 122 days) $\to$ **59 Forecast Positives / 106,107 Negatives**
- **Test Prevalence:** **0.056%** (1 event per 1,800 observations)

### Comprehensive Evaluation Table:
| Setup | Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Operational False Alarms / Day | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climatology Baseline** | Empirical Prior ($P=0.0015$) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0006 | 0.00 | `[[106107, 0], [59, 0]]` |
| **True Persistence Baseline** | `true_prior_event_6h` | **0.0053** | **0.0169** | **0.0081** | **0.5076** | **0.0006** | **1.54** | `[[105919, 188], [58, 1]]` |
| **Atmospheric Only (Setup B)** | Logistic Regression (0.5) | 0.0137 | **0.9492** | 0.0270 | **0.9850** | **0.2612** | 33.10 | `[[102069, 4038], [3, 56]]` |
| **Atmospheric Only (Setup B)** | Logistic Reg (Val-Tuned $\tau=0.74$) | **0.0381** | 0.5763 | **0.0715** | 0.9850 | 0.2612 | **7.11** | `[[105239, 868], [25, 34]]` |
| **Atmospheric Only (Setup B)** | HistGradientBoosting (0.5) | 0.0076 | 0.0508 | 0.0133 | 0.9576 | 0.0116 | 3.25 | `[[105711, 396], [56, 3]]` |
| **Atmospheric + True Prior (Setup D)** | Logistic Regression (0.5) | 0.0137 | **0.9492** | 0.0270 | **0.9850** | **0.2613** | 33.10 | `[[102069, 4038], [3, 56]]` |
| **Atmospheric + True Prior (Setup D)** | HistGradientBoosting (0.5) | 0.0076 | 0.0508 | 0.0133 | 0.9576 | 0.0116 | 3.25 | `[[105711, 396], [56, 3]]` |

---

## 3. Operational Interpretation and Defensibility

1. **Why True Persistence Fails in Forward Forecasting:**
   Convective storm cells in Florida have lifecycles of 30 to 90 minutes. Over a 6-hour forward synoptic cycle, a cell moves or dissipates. Consequently, a cell observed in $(t-6\text{h}, t]$ is rarely producing damage at the exact same 0.25° grid location 3 to 6 hours later. True persistence achieves only 1 hit out of 59 test events ($1.7\%$ recall).
2. **Defensibility of Machine Learning Forecasting:**
   Atmospheric predictors (CAPE, CIN, TCWV, mid-tropospheric vertical velocity, and sea-level pressure) demonstrate genuine physical predictive skill (ROC-AUC = 0.9850, PR-AUC = 0.2612 vs. 0.0006 climatology). The model identifies pre-convective thermodynamic instability and mesoscale convergence up to 6 hours before storm initiation.
3. **Operational Threshold Selection:**
   At default threshold 0.5, balanced logistic regression captures **94.9%** of events with 33 false alarms/day. When threshold is tuned on validation data ($\tau = 0.74$), false alarms are reduced by **78%** (down to 7.1/day) while retaining **57.6%** detection recall.
