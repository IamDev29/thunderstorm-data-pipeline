# Step 6: Importance Weighting and Density-Ratio Estimation Experiment Report

## Executive Summary

As part of the **Thunderstorm Domain Adaptation Framework (TDAF)**, this report evaluates **Importance Weighting and Density-Ratio Estimation** for cross-region transfer between Florida (Source Domain) and Chandipur, India (Target Domain).

Unlike Step 5's Correlation Alignment (CORAL), which rotated feature coordinate axes and generated unphysical artifacts (e.g. 37.6% negative CAPE), importance weighting operates purely in sample space: it adjusts the importance of Florida training observations without altering the coordinate values, scales, or physical units of Chandipur predictors.

We trained and evaluated weighted variants of native Generalized Additive Models (`pyGAM`) and Explainable Boosting Machines (`InterpretML` EBM) across 66,255 Florida training observations and evaluated them on both the Florida locked chronological holdout test set ($N = 28,073$) and the multi-year Chandipur target dataset ($N = 23,376$, 2022–2025).

### Key Empirical Findings:

1. **Complete Preservation of Physical Integrity**:
   Across all weighting schemes, **zero physical boundary violations were produced**. CAPE, CIN, wind speeds, and cloud cover remained 100% within their natural physical domains ($\text{CAPE} \ge 0$, $\text{CIN} \ge 0$, $\text{Wind} \ge 0$, $\text{Cloud} \in [0, 1]$), resolving the fundamental flaw of CORAL.
2. **Effective Sample Size (ESS) and Weight Concentration**:
   Unclipped density ratios from non-linear classifiers suffer from catastrophic weight concentration:
   - In unclipped LightGBM, the maximum sample weight reached **$177.8$**, and the top 1% of source samples absorbed **$17.8\%$ of all weight**, collapsing the Effective Sample Size (ESS) to **$9,526.8$** ($14.38\%$ sample retention).
   - Stabilization via **99th percentile clipping** or **Fixed Maximum Weight = 5** restored ESS to **$22,100$–$30,800$** ($33.4\%$–$46.6\%$ retention).
   - **Kernel Mean Matching (KMM)** yielded very smooth, conservative weights ($\text{ESS} = 64,338.9$, $97.11\%$ retention).
3. **Importance Weighting Does NOT Improve Chandipur Transfer Skill**:
   Counter to conventional domain adaptation expectations, reweighting Florida training observations to resemble Chandipur's feature distribution **consistently failed to improve transfer performance against Chandipur proxy labels**:
   - Baseline Unweighted EBM (CAPE-clipped): **F1 = $0.6059$**, PR-AUC = **$0.6134$**, ROC-AUC = **$0.8572$**.
   - Logistic Regression (Max 5) EBM: F1 = $0.5861$, PR-AUC = $0.5391$, ROC-AUC = $0.8341$.
   - LightGBM (p99) EBM: F1 = $0.5703$, PR-AUC = $0.5371$, ROC-AUC = $0.8268$.
   - KMM EBM: F1 = $0.6019$, PR-AUC = $0.6026$, ROC-AUC = $0.8539$.
   - Seasonal LightGBM EBM: F1 = $0.5819$, PR-AUC = $0.5890$, ROC-AUC = $0.8390$.
4. **The Root Cause: Violation of the Covariate Shift Assumption (Concept Shift)**:
   Importance weighting mathematically relies on the assumption that $P(Y \mid X)$ is invariant across domains. However, our empirical results confirm severe **concept shift**:
   - In Florida, days with elevated CAPE ($> 3,000\text{ J/kg}$) and high moisture have an observed ASOS thunderstorm frequency of approximately $20\%$–$30\%$ (due to localized entrainment and sea-breeze timing).
   - In Chandipur, the mathematical heuristic formula assigns positive storm labels almost deterministically whenever CAPE and wind speed are high.
   - Upweighting Florida's convective extremes forces the models to learn from a small subset of Florida storms that do not reflect the generative mechanism of Chandipur's proxy labels, inflating variance without fixing concept disparity.
5. **Spline Extrapolation Breakdown Persists on Raw Features**:
   On the 81 extreme Chandipur cases ($\text{CAPE} > 7,240\text{ J/kg}$), raw unclipped pyGAM predictions collapsed to near zero ($0.081$–$0.096$) across all weighted variants. Only **Condition B (CAPE clipping at $7,240\text{ J/kg}$)** restored stable predictions ($0.124$–$0.165$).

---

## 1. Density-Ratio Estimation & Domain Classifier Performance

To compute the density ratio $w(x) = \frac{P(z=1 \mid x)}{1 - P(z=1 \mid x)} \cdot \frac{N_S}{N_T}$, three classifiers were evaluated under 5-fold stratified cross-validation on the combined domain dataset ($N = 89,631$):

| Domain Classifier | Out-of-Fold ROC-AUC | Brier Score | Log Loss | Computational Time |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7799 | 0.1463 | 0.4577 | 3.2 s |
| **Random Forest** | 0.8847 | 0.1111 | 0.3598 | 18.4 s |
| **LightGBM** | **0.8864** | **0.1080** | **0.3480** | **10.1 s** |

LightGBM achieved the best domain discrimination (ROC-AUC = $0.8864$) and probability calibration (Brier score = $0.1080$), capturing non-linear feature interactions between thermodynamic instability and kinematic shear.

---

## 2. Weight Distribution and Effective Sample Size (ESS) Analysis

Stabilization policies were tested across all four density-ratio estimators:

| Weighting Method | Clipping Policy | Max Weight | Mean Weight | Std Weight | Median Weight | 99th Pct | Top 1% Conc. | ESS | ESS / $N_S$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | No Clipping | 45.82 | 1.00 | 1.37 | 0.60 | 6.44 | 9.75% | 23,098.6 | 34.86% |
| | 95th Pct Clipping | 3.57 | 1.00 | 0.93 | 0.68 | 3.57 | 3.57% | 35,509.6 | 53.60% |
| | **99th Pct Clipping** | 6.66 | 1.00 | 1.14 | 0.63 | 6.66 | 6.66% | **28,842.9** | **43.53%** |
| | **Fixed Max = 5** | 5.28 | 1.00 | 1.07 | 0.64 | 5.28 | 5.27% | **30,864.7** | **46.58%** |
| | Fixed Max = 10 | 10.13 | 1.00 | 1.24 | 0.61 | 6.53 | 8.55% | 26,192.2 | 39.53% |
| **LightGBM** | No Clipping | 177.82 | 1.00 | 2.44 | 0.53 | 8.42 | 17.80% | 9,526.8 | 14.38% |
| | 95th Pct Clipping | 4.04 | 1.00 | 1.04 | 0.67 | 4.04 | 4.03% | 31,739.6 | 47.91% |
| | **99th Pct Clipping** | 9.29 | 1.00 | 1.41 | 0.59 | 9.29 | 9.29% | **22,101.6** | **33.36%** |
| | **Fixed Max = 5** | 5.85 | 1.00 | 1.21 | 0.62 | 5.85 | 5.85% | **26,952.1** | **40.68%** |
| | Fixed Max = 10 | 10.87 | 1.00 | 1.49 | 0.58 | 9.16 | 10.65% | 20,656.0 | 31.18% |
| **Random Forest** | 99th Pct Clipping | 8.08 | 1.00 | 1.23 | 0.67 | 8.08 | 8.08% | 26,337.0 | 39.75% |
| **KMM** | No Clipping | 2.63 | 1.00 | 0.17 | 0.98 | 1.55 | 1.68% | **64,338.9** | **97.11%** |
| **Seasonal LightGBM** | 99th Pct Clipping | 14.47 | 1.00 | 1.89 | 0.55 | 11.23 | 12.45% | **14,469.9** | **21.84%** |

### Key Diagnostic Observations:
- **LightGBM without clipping collapses ESS to $14.38\%$**, indicating that unconstrained tree-based density estimation is highly vulnerable to extreme sample weighting.
- **Fixed Max = 5 and 99th percentile clipping successfully stabilize the sample variance**, retaining $40\%$–$47\%$ of the effective source sample.
- **KMM is exceptionally conservative**, producing weights centered tightly around $1.0$ (std $= 0.17$), resulting in $97.11\%$ effective sample retention.

---

## 3. Comprehensive Model Evaluation

### 3.1 Chandipur Target Transfer (Condition B: CAPE Clipped at 7,240 J/kg)

Evaluated across all 23,376 Chandipur observations under Florida optimal decision thresholds ($\tau_{\text{GAM}} = 0.33$, $\tau_{\text{EBM}} = 0.34$) and standard threshold ($\tau = 0.50$):

| Model Architecture | Weight Configuration | Threshold | Precision | Recall | F1 Score | Cohen's $\kappa$ | MCC | PR-AUC | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pyGAM** | **Baseline (Unweighted)** | 0.33 | 0.4858 | **0.8365** | **0.6146** | 0.4529 | 0.4889 | 0.5526 | **0.8502** | **0.1299** |
| | Logistic Regression (p99) | 0.33 | 0.4892 | 0.6888 | 0.5721 | 0.4112 | 0.4228 | 0.4943 | 0.8259 | 0.1379 |
| | Logistic Regression (Max 5) | 0.33 | 0.4910 | 0.6999 | 0.5771 | 0.4170 | 0.4298 | 0.4978 | 0.8280 | 0.1371 |
| | LightGBM (p99) | 0.33 | 0.4736 | 0.7069 | 0.5672 | 0.3990 | 0.4150 | 0.4908 | 0.8201 | 0.1398 |
| | LightGBM (Max 5) | 0.33 | 0.4767 | 0.7085 | 0.5699 | 0.4032 | 0.4189 | 0.4908 | 0.8201 | 0.1396 |
| | Random Forest (p99) | 0.33 | 0.4803 | 0.7148 | 0.5745 | 0.4095 | 0.4256 | 0.5035 | 0.8258 | 0.1374 |
| | KMM (Kernel Mean Matching) | 0.33 | 0.4914 | 0.8202 | **0.6146** | 0.4554 | 0.4870 | 0.5416 | 0.8471 | 0.1308 |
| | Seasonal LightGBM (p99) | 0.33 | 0.4978 | 0.7204 | 0.5888 | 0.4317 | 0.4462 | 0.5532 | 0.8416 | 0.1319 |
| **InterpretML EBM** | **Baseline (Unweighted)** | 0.34 | 0.4902 | **0.7931** | **0.6059** | 0.4458 | 0.4726 | **0.6134** | **0.8572** | **0.1270** |
| | Logistic Regression (p99) | 0.34 | 0.5039 | 0.6919 | 0.5831 | 0.4287 | 0.4390 | 0.5335 | 0.8320 | 0.1351 |
| | Logistic Regression (Max 5) | 0.34 | 0.5047 | 0.6988 | 0.5861 | 0.4320 | 0.4430 | 0.5391 | 0.8341 | 0.1343 |
| | LightGBM (p99) | 0.34 | 0.4888 | 0.6845 | 0.5703 | 0.4092 | 0.4205 | 0.5371 | 0.8268 | 0.1364 |
| | LightGBM (Max 5) | 0.34 | 0.4895 | 0.6816 | 0.5698 | 0.4091 | 0.4199 | 0.5386 | 0.8273 | 0.1362 |
| | Random Forest (p99) | 0.34 | 0.4902 | 0.6873 | 0.5723 | 0.4118 | 0.4232 | 0.5466 | 0.8291 | 0.1355 |
| | KMM (Kernel Mean Matching) | 0.34 | 0.4933 | 0.7719 | 0.6019 | 0.4431 | 0.4658 | 0.6026 | 0.8539 | 0.1278 |
| | Seasonal LightGBM (p99) | 0.34 | 0.5037 | 0.6888 | 0.5819 | 0.4273 | 0.4373 | 0.5890 | 0.8390 | 0.1315 |

---

### 3.2 Florida Chronological Holdout Test Performance (Sanity Check)

Evaluated on Florida's locked chronological test set ($N = 28,073$, months 10–12):

| Model Architecture | Weight Configuration | Threshold | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pyGAM** | **Baseline (Unweighted)** | 0.33 | 0.3798 | 0.4421 | 0.4086 | 0.3362 | 0.8288 | 0.0556 |
| | Logistic Regression (p99) | 0.33 | 0.3705 | 0.4510 | 0.4068 | 0.3406 | 0.8281 | 0.0559 |
| | LightGBM (p99) | 0.33 | 0.3688 | 0.4447 | 0.4032 | 0.3345 | 0.8294 | 0.0561 |
| | KMM (Kernel Mean Matching) | 0.33 | 0.3795 | 0.4442 | 0.4093 | 0.3358 | 0.8292 | 0.0556 |
| | Seasonal LightGBM (p99) | 0.33 | 0.3570 | 0.4359 | 0.3925 | 0.3309 | 0.8258 | 0.0566 |
| **InterpretML EBM** | **Baseline (Unweighted)** | 0.34 | 0.3578 | 0.4541 | 0.4003 | 0.3513 | 0.8304 | 0.0564 |
| | Logistic Regression (p99) | 0.34 | 0.3525 | 0.4734 | 0.4041 | 0.3553 | 0.8311 | 0.0565 |
| | LightGBM (p99) | 0.34 | 0.3476 | 0.4604 | 0.3961 | 0.3471 | 0.8303 | 0.0570 |
| | KMM (Kernel Mean Matching) | 0.34 | 0.3565 | 0.4572 | 0.4006 | 0.3500 | 0.8309 | 0.0565 |
| | Seasonal LightGBM (p99) | 0.34 | 0.3451 | 0.4437 | 0.3882 | 0.3323 | 0.8276 | 0.0578 |

On Florida's own test set, global weighting (Logistic Regression and KMM) preserves source domain accuracy, while **Seasonal Weighting degrades Florida test PR-AUC from $0.3513 \to 0.3323$**.

---

## 4. Seasonal Weighting Diagnostics

| Season | Florida Train ($N_S$) | Chandipur Target ($N_T$) | Domain ROC-AUC | ESS | ESS Ratio | Fallback Policy Used? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DJF (Winter)** | 18,688 | 5,776 | 0.9207 | 4,423.0 | 23.67% | No |
| **MAM (Pre-monsoon)** | 28,407 | 5,888 | 0.9616 | 6,089.9 | 21.44% | No |
| **JJAS (Monsoon)** | 19,160 | 7,808 | 0.9265 | 3,994.5 | 20.85% | No |
| **ON (Post-monsoon)** | **0** | 3,904 | N/A | N/A | N/A | **Yes (Global Fallback)** |

In season ON, Florida training data has zero observations (the dataset ends July 31). Enforcing seasonal weighting requires an artificial global fallback for $16.7\%$ of target cases, violating seasonal continuity.

---

## 5. Behavior on Extreme Convective Environments (CAPE > 7,240 J/kg)

Evaluated across the 81 extreme Chandipur cases:

| Weight Configuration | pyGAM Prob (Raw CAPE) | pyGAM Prob (CAPE Clipped) | EBM Prob (Raw CAPE) | EBM Prob (CAPE Clipped) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Unweighted)** | 0.0948 (min 0.0043) | **0.1645 (min 0.0244)** | 0.3833 (min 0.1049) | **0.3833 (min 0.1049)** |
| **Logistic Regression (p99)** | 0.0884 (min 0.0067) | 0.1243 (min 0.0147) | 0.3164 (min 0.0825) | 0.3164 (min 0.0825) |
| **Logistic Regression (Max 5)** | 0.0855 (min 0.0056) | 0.1273 (min 0.0158) | 0.3186 (min 0.0859) | 0.3186 (min 0.0859) |
| **LightGBM (p99)** | 0.0828 (min 0.0028) | 0.1547 (min 0.0134) | 0.3604 (min 0.0742) | 0.3604 (min 0.0742) |
| **KMM (Kernel Mean Matching)** | 0.0935 (min 0.0051) | 0.1530 (min 0.0223) | 0.3697 (min 0.1022) | 0.3697 (min 0.1022) |
| **Seasonal LightGBM (p99)** | 0.1085 (min 0.0127) | 0.1490 (min 0.0147) | 0.3408 (min 0.0803) | 0.3408 (min 0.0803) |

### Critical Finding:
Under raw features, **importance weighting cannot prevent pyGAM spline extrapolation collapse**. At $\text{CAPE} > 7,240\text{ J/kg}$, pyGAM probabilities dive towards zero across all weighting schemes (dropping to $0.0028$ in LightGBM-weighted models). 
Only univariate boundary clipping (Condition B) successfully bounds spline evaluations within Florida's known support.

---

## 6. Visual Diagnostics and Figures

All figures have been generated at publication resolution (300 DPI) in `reports/domain_adaptation/step_06/`:
1. `importance_weight_distributions.png`: Density histograms of weights for LR, LightGBM, RF, and KMM.
2. `source_target_weighted_comparison.png`: 6-panel marginal feature distributions showing unweighted vs. weighted Florida vs. Chandipur target.
3. `effective_sample_size_comparison.png`: Bar chart of ESS ratios across all 5 clipping policies.
4. `weighted_domain_classifier_roc.png`: Out-of-fold ROC curves distinguishing Florida from Chandipur.
5. `weighted_gam_ebm_performance.png`: Grouped bar charts comparing F1-score and PR-AUC across weighting configurations.
6. `seasonal_weight_distribution.png`: Seasonal weight histograms for DJF, MAM, and JJAS.
7. `extreme_cape_weight_analysis.png`: Probability response on the 81 extreme CAPE cases.
8. `calibration_comparison.png`: Reliability curves on Florida holdout and Chandipur target.

---

## 7. Conclusion

Importance weighting is mathematically sound and physically safe (it creates no negative energy or unphysical coordinates). However, it **fails to improve target cross-region transfer** because the underlying assumption of covariate shift ($P_S(Y \mid X) = P_T(Y \mid X)$) is violated between Florida ASOS observations and Chandipur heuristic proxy labels.
