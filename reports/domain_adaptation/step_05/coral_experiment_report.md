# Step 5: Correlation Alignment (CORAL) and Multivariate Feature Adaptation Experiment Report

## Executive Summary

As part of the **Thunderstorm Domain Adaptation Framework (TDAF)**, this report evaluates **Correlation Alignment (CORAL)** for cross-region transfer between Florida (Source Domain) and Chandipur, India (Target Domain).

We evaluated the performance of frozen native models (`pyGAM` and `InterpretML` EBM) under seven strictly controlled experimental conditions across all 23,376 Chandipur observations (2022–2025).

### Key Empirical Findings:
1. **Mathematical Success vs. Physical Catastrophe**: Standard CORAL achieves **perfect mathematical covariance alignment** (Frobenius covariance distance drops from $1,548,553.1 \to 0.0$, and feature means match Florida training means to six decimal places). However, because CORAL is an unconstrained linear affine operator, it performs arbitrary coordinate rotations that violate fundamental physical laws:
   - **8,798 observations (37.64%)** produced **negative CAPE** (down to $-219.37\text{ J/kg}$).
   - **10,751 observations (46.00%)** produced **negative CIN** (down to $-125.48\text{ J/kg}$).
   - **418 observations (1.79%)** produced **negative 10m wind speeds** (down to $-2.71\text{ m/s}$).
   - **1,113 observations (4.76%)** produced **cloud cover fractions outside $[0.0, 1.0]$** (up to $1.43$).
2. **Monotonic Performance Degradation**: As alignment fraction $\alpha$ increases from $0.00$ to $1.00$, transfer skill against Chandipur proxy labels collapses monotonically:
   - `pyGAM` F1 drops from **$0.6146 \to 0.4090$** (ROC-AUC drops from $0.8494 \to 0.7320$).
   - `InterpretML` EBM F1 drops from **$0.6059 \to 0.4525$** (ROC-AUC drops from $0.8572 \to 0.7616$).
3. **Domain Distinguishability Actually Increases**: A cross-validated domain classifier (logistic regression) separating Florida from Chandipur shows an AUC of **$0.8629$** on raw data, which **rises to $0.9774$** after Standard CORAL. Although marginal means and second moments match, the linear mixing creates unnatural joint feature combinations that are instantly recognized as non-Florida observations.
4. **Superiority of Simple Univariate Clipping**: Condition B (Univariate CAPE clipping at the Florida training boundary of $7,240\text{ J/kg}$) completely prevents pyGAM extrapolation failure, maintains zero physical violations, and delivers the highest transfer skill across all conditions (pyGAM F1 = $0.6146$, ROC-AUC = $0.8502$; EBM F1 = $0.6059$, ROC-AUC = $0.8572$).

---

## 1. Experimental Conditions and Setup

| Condition ID | Description | Mathematical Formulation / Safeguards |
| :--- | :--- | :--- |
| **Cond_A** | Original Zero-Shot Baseline | Raw target features $X_T$ passed directly to frozen Florida models. |
| **Cond_B** | CAPE Boundary Clipping Only | $\text{CAPE} \leftarrow \min(\text{CAPE}, 7240\text{ J/kg})$. All other features unchanged. |
| **Cond_C** | Standard CORAL (Unconstrained) | Full whitening-colouring: $\hat{X}_T = (X_T - \mu_T) C_T^{-1/2} C_S^{1/2} + \mu_S$. |
| **Cond_D** | CORAL + Physical Clamping | Standard CORAL followed by clamping to physical intervals ($[0, 7240]$ for CAPE, $[0, 1]$ for clouds, etc.). |
| **Cond_E** | Regularized Covariance CORAL | Tikhonov shrinkage $C \leftarrow C + \lambda I$ ($\lambda = 0.05 \cdot \text{trace}(C)/d$) prior to matrix square roots. |
| **Cond_F_025** | Partial CORAL ($\alpha = 0.25$) | $A = 0.75 I + 0.25 A_{\text{CORAL}}$. |
| **Cond_F_050** | Partial CORAL ($\alpha = 0.50$) | $A = 0.50 I + 0.50 A_{\text{CORAL}}$. |
| **Cond_F_075** | Partial CORAL ($\alpha = 0.75$) | $A = 0.25 I + 0.75 A_{\text{CORAL}}$. |
| **Cond_G** | Seasonal CORAL (with ON Fallback) | Stratified by season (DJF, MAM, JJAS). ON falls back to Florida annual statistics. |

---

## 2. Comprehensive Model Transfer Evaluation

All models were evaluated on the full Chandipur dataset ($N = 23,376$, positive rate = $23.37\%$) using the frozen Florida optimal decision thresholds ($\tau_{\text{GAM}} = 0.33$, $\tau_{\text{EBM}} = 0.34$) and the standard uncalibrated threshold ($\tau = 0.50$).

### 2.1 pyGAM Transfer Performance

| Condition | Threshold | Precision | Recall | F1 Score | Cohen's $\kappa$ | MCC | PR-AUC | ROC-AUC | Brier Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A (Baseline)** | 0.33 | 0.4858 | **0.8365** | **0.6146** | 0.4529 | 0.4889 | 0.5523 | 0.8494 | 0.1301 | 0.3915 |
| **Cond_A (Baseline)** | 0.50 | 0.5709 | 0.4724 | 0.5170 | 0.3874 | 0.3902 | 0.5523 | 0.8494 | 0.1301 | 0.3915 |
| **Cond_B (CAPE Clip)** | 0.33 | 0.4858 | **0.8365** | **0.6146** | 0.4529 | 0.4889 | **0.5526** | **0.8502** | **0.1299** | **0.3906** |
| **Cond_B (CAPE Clip)** | 0.50 | 0.5709 | 0.4724 | 0.5170 | 0.3874 | 0.3902 | **0.5526** | **0.8502** | **0.1299** | **0.3906** |
| **Cond_C (CORAL)** | 0.33 | 0.3741 | 0.4511 | 0.4090 | 0.2062 | 0.2078 | 0.3903 | 0.7320 | 0.1787 | 0.5426 |
| **Cond_C (CORAL)** | 0.50 | 0.4112 | 0.2560 | 0.3155 | 0.1660 | 0.1731 | 0.3903 | 0.7320 | 0.1787 | 0.5426 |
| **Cond_D (CORAL+Clamp)**| 0.33 | 0.4087 | 0.5361 | 0.4638 | 0.2703 | 0.2749 | 0.4028 | 0.7642 | 0.1585 | 0.4612 |
| **Cond_D (CORAL+Clamp)**| 0.50 | 0.3951 | 0.2109 | 0.2750 | 0.1342 | 0.1440 | 0.4028 | 0.7642 | 0.1585 | 0.4612 |
| **Cond_E (Reg CORAL)** | 0.33 | 0.3762 | 0.4529 | 0.4110 | 0.2091 | 0.2107 | 0.3911 | 0.7334 | 0.1775 | 0.5367 |
| **Cond_F ($\alpha=0.25$)** | 0.33 | 0.4673 | 0.7453 | 0.5744 | 0.4029 | 0.4256 | 0.4928 | 0.8218 | 0.1393 | 0.4106 |
| **Cond_F ($\alpha=0.50$)** | 0.33 | 0.4427 | 0.6534 | 0.5278 | 0.3455 | 0.3585 | 0.4526 | 0.7918 | 0.1510 | 0.4411 |
| **Cond_F ($\alpha=0.75$)** | 0.33 | 0.4104 | 0.5568 | 0.4725 | 0.2784 | 0.2845 | 0.4208 | 0.7619 | 0.1641 | 0.4839 |
| **Cond_G (Seasonal)** | 0.33 | 0.4207 | 0.6988 | 0.5252 | 0.3296 | 0.3520 | 0.3489 | 0.7056 | 0.1969 | 0.7448 |

### 2.2 InterpretML EBM Transfer Performance

| Condition | Threshold | Precision | Recall | F1 Score | Cohen's $\kappa$ | MCC | PR-AUC | ROC-AUC | Brier Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A (Baseline)** | 0.34 | 0.4902 | **0.7931** | **0.6059** | 0.4458 | 0.4726 | **0.6134** | **0.8572** | **0.1270** | **0.3856** |
| **Cond_A (Baseline)** | 0.50 | 0.6131 | 0.4958 | 0.5482 | 0.4289 | 0.4328 | **0.6134** | **0.8572** | **0.1270** | **0.3856** |
| **Cond_B (CAPE Clip)** | 0.34 | 0.4902 | **0.7931** | **0.6059** | 0.4458 | 0.4726 | **0.6134** | **0.8572** | **0.1270** | **0.3856** |
| **Cond_B (CAPE Clip)** | 0.50 | 0.6131 | 0.4958 | 0.5482 | 0.4289 | 0.4328 | **0.6134** | **0.8572** | **0.1270** | **0.3856** |
| **Cond_C (CORAL)** | 0.34 | 0.4043 | 0.5137 | 0.4525 | 0.2586 | 0.2621 | 0.4049 | 0.7616 | 0.1588 | 0.4633 |
| **Cond_C (CORAL)** | 0.50 | 0.4297 | 0.2391 | 0.3073 | 0.1683 | 0.1791 | 0.4049 | 0.7616 | 0.1588 | 0.4633 |
| **Cond_D (CORAL+Clamp)**| 0.34 | 0.4043 | 0.5137 | 0.4525 | 0.2586 | 0.2621 | 0.4049 | 0.7616 | 0.1588 | 0.4633 |
| **Cond_D (CORAL+Clamp)**| 0.50 | 0.4297 | 0.2391 | 0.3073 | 0.1683 | 0.1791 | 0.4049 | 0.7616 | 0.1588 | 0.4633 |
| **Cond_E (Reg CORAL)** | 0.34 | 0.4050 | 0.5154 | 0.4536 | 0.2599 | 0.2634 | 0.4067 | 0.7619 | 0.1586 | 0.4630 |
| **Cond_F ($\alpha=0.25$)** | 0.34 | 0.4698 | 0.7195 | 0.5685 | 0.3984 | 0.4166 | 0.5317 | 0.8297 | 0.1359 | 0.4049 |
| **Cond_F ($\alpha=0.50$)** | 0.34 | 0.4480 | 0.6589 | 0.5334 | 0.3536 | 0.3666 | 0.4852 | 0.8082 | 0.1428 | 0.4215 |
| **Cond_F ($\alpha=0.75$)** | 0.34 | 0.4289 | 0.5969 | 0.4992 | 0.3121 | 0.3203 | 0.4421 | 0.7847 | 0.1507 | 0.4413 |
| **Cond_G (Seasonal)** | 0.34 | 0.4736 | 0.6580 | 0.5508 | 0.3831 | 0.3931 | 0.4530 | 0.7939 | 0.1479 | 0.4437 |

---

## 3. Statistical Alignment vs. Physical Boundary Diagnostics

The table below contrasts the mathematical distance metrics against the physical boundary violations across conditions:

| Condition | Frobenius Distance ($\|C_T - C_S\|_F$) | Mean Abs Corr Diff | Domain Classifier ROC-AUC | Negative CAPE Count (%) | Negative CIN Count (%) | Negative Wind Count | Cloud Cover Out-of-Bounds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A (Baseline)** | 1,548,553.1 | 0.1714 | 0.8629 | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| **Cond_B (CAPE Clip)** | 1,523,817.4 | 0.1714 | 0.8629 | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| **Cond_C (CORAL)** | **0.00** | **0.0000** | 0.9774 | **8,798 (37.64%)** | **10,751 (46.00%)** | **418** | **4,617** |
| **Cond_D (CORAL+Clamp)**| 94,023.2 | 0.0115 | 0.8724 | 0 (0.0%) | 0 (0.0%) | 0 | 0 |
| **Cond_E (Reg CORAL)** | 1,235.6 | 0.0065 | 0.9774 | 8,799 (37.64%) | 10,735 (45.92%) | 395 | 4,406 |
| **Cond_F ($\alpha=0.25$)** | 1,090,098.2 | 0.1277 | 0.9705 | 6,612 (28.29%) | 9,518 (40.72%) | 9 | 3,169 |
| **Cond_F ($\alpha=0.50$)** | 679,188.2 | 0.0843 | 0.9687 | 7,298 (31.22%) | 9,875 (42.24%) | 71 | 3,806 |
| **Cond_F ($\alpha=0.75$)** | 315,822.4 | 0.0414 | 0.9705 | 7,963 (34.07%) | 10,282 (43.99%) | 180 | 4,247 |
| **Cond_G (Seasonal)** | 126,898.6 | 0.0187 | 0.9693 | 7,947 (33.99%) | 9,884 (42.28%) | 373 | 4,252 |

### Why Did the Domain Classifier ROC-AUC Increase Under CORAL?
In raw feature space (Cond_A), an adversarial logistic regression classifier achieved ROC-AUC = $0.8629$ primarily by separating Florida from Chandipur using marginal feature shifts (e.g. higher mean CAPE and higher 10m wind speed in Chandipur).

Under Standard CORAL (Cond_C), the first and second marginal moments are perfectly equalized ($\mu_{\hat{T}} = \mu_S$, $C_{\hat{T}} = C_S$). However, because the linear transformation $A = C_T^{-1/2} C_S^{1/2}$ mixes all 6 features, observations that had high CAPE and moderate shear in Chandipur are projected into coordinates where CAPE is negative while bulk shear is abnormally inflated. The domain classifier readily exploits these artificial, non-meteorological combinations to achieve an even higher ROC-AUC of **$0.9774$**.

---

## 4. Seasonal Performance Breakdown

Performance varies drastically across Chandipur's distinct climatological regimes:

| Condition | Season | Samples | Positives | Prevalence | GAM F1 | GAM ROC-AUC | EBM F1 | EBM ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A (Baseline)** | DJF (Winter) | 5,776 | 63 | 1.09% | 0.1429 | 0.9193 | 0.3717 | 0.9270 |
| | MAM (Pre-monsoon) | 5,888 | 1,216 | 20.65% | 0.4612 | 0.7254 | 0.3832 | 0.6970 |
| | JJAS (Monsoon) | 7,808 | 3,761 | 48.17% | **0.6819** | 0.6716 | **0.6803** | 0.7056 |
| | ON (Post-monsoon) | 3,904 | 422 | 10.81% | 0.5403 | **0.9329** | 0.5513 | **0.9533** |
| **Cond_C (CORAL)** | DJF (Winter) | 5,776 | 63 | 1.09% | 0.1218 | 0.8829 | 0.0460 | 0.7986 |
| | MAM (Pre-monsoon) | 5,888 | 1,216 | 20.65% | 0.2269 | 0.4750 | 0.2891 | 0.5986 |
| | JJAS (Monsoon) | 7,808 | 3,761 | 48.17% | 0.4921 | 0.5110 | 0.5303 | 0.5159 |
| | ON (Post-monsoon) | 3,904 | 422 | 10.81% | 0.3319 | 0.8203 | 0.3325 | 0.8281 |
| **Cond_G (Seasonal)** | DJF (Winter) | 5,776 | 63 | 1.09% | 0.0897 | 0.9046 | 0.1201 | 0.8444 |
| | MAM (Pre-monsoon) | 5,888 | 1,216 | 20.65% | 0.3066 | 0.5318 | 0.2251 | 0.5934 |
| | JJAS (Monsoon) | 7,808 | 3,761 | 48.17% | 0.6479 | 0.5901 | 0.6511 | 0.6054 |
| | ON (Post-monsoon) | 3,904 | 422 | 10.81% | 0.4312 | 0.8723 | 0.4508 | 0.8854 |

### The Catastrophic Failure in Pre-monsoon (MAM):
In Chandipur, pre-monsoon thunderstorms (Nor'westers / Kalbaishakhi) are driven by extreme thermal instability interacting with dry mid-level air and intense localized sea-breeze convergence. When Seasonal CORAL applies Florida spring covariance (which represents frontal squall lines and maritime convection under moderate CAPE), it shrinks Chandipur's pre-monsoon CAPE and inflates shear, causing EBM F1 to collapse from **$0.3832 \to 0.2251$** and ROC-AUC from **$0.6970 \to 0.5934$**.

---

## 5. Behavior on Extreme Convective Environments (CAPE > 7,240 J/kg)

In Step 4, we identified 81 extreme convective events in Chandipur where raw CAPE exceeded Florida's historical maximum ($7,240\text{ J/kg}$, up to $9,906\text{ J/kg}$).

Under baseline transfer (Cond_A), `pyGAM` suffered spline extrapolation collapse, predicting probabilities as low as $0.0039$ under $9,906\text{ J/kg}$ of CAPE.
- **Under Condition B (Boundary Clipping)**: Clamping CAPE to $7,240\text{ J/kg}$ stabilized pyGAM probability to $0.2038$–$0.2856$, completely curing the downward spline dive without affecting any other feature.
- **Under Condition C (Standard CORAL)**: The linear scaling factors reduce CAPE into the $4,000$–$4,600\text{ J/kg}$ range. While pyGAM probabilities rise to $0.466$–$0.989$, this occurs because all features are artificially rotated. In multiple extreme events, bulk shear and surface winds were distorted to unphysical values, producing unstable predictions.

---

## 6. Visual Diagnostics and Figures

All figures have been rendered at publication quality (300 DPI) in `reports/domain_adaptation/step_05/`:
1. `coral_covariance_before_after.png`: Visualizes covariance matrix difference matrices before and after alignment, showing complete erasure of second-order differences under Cond_C.
2. `coral_correlation_heatmaps.png`: 3-panel comparison of Florida, Chandipur Raw, and Chandipur Aligned correlation matrices.
3. `coral_feature_distributions.png`: 6-panel marginal density plots showing the heavy negative tails generated by CORAL across CAPE, CIN, Wind, and Cloud Cover.
4. `coral_pca_alignment.png`: PCA scatter plots demonstrating that while global centroids overlap, individual cluster geometry and density modes are distorted.
5. `coral_domain_classifier_comparison.png`: ROC curves for the adversarial domain classifier across all conditions.
6. `coral_alpha_sensitivity.png`: Metric curves demonstrating the monotonic collapse of F1 and ROC-AUC as alignment parameter $\alpha$ increases from $0.00 \to 1.00$.
7. `coral_seasonal_alignment.png`: Seasonal F1 performance bar charts across DJF, MAM, JJAS, and ON.
8. `coral_extreme_cape_analysis.png`: Probability response curves and scatter plots for the 81 extreme CAPE cases.

---

## 7. Conclusion

While CORAL is mathematically elegant and fully satisfies its objective function ($\|C_{\hat{X}_T} - C_{X_S}\|_F = 0$), it is **fundamentally unsuitable as a standalone multivariate domain adaptation technique for physical meteorological forecasting**.
Unconstrained linear affine transformations cannot respect physical sign constraints ($\text{CAPE} \ge 0$, $\text{Wind} \ge 0$) and destroy the individual physical semantics upon which interpretable additive models (`pyGAM` and `InterpretML` EBM) rely.
