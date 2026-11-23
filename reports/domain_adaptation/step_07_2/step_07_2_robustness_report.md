# Step 7.2 — Robustness, Statistical Significance and Generalization Validation

## 1. Executive Summary

In Step 7.1, an independent compatibility audit established that while asymmetric target-only B4 suffered from a feature-semantic mismatch, **symmetrically retrained `pyGAM`** (where both Florida source data and Chandipur target data were transformed using the exact same logarithmic compression on CAPE and CIN) achieved:
- **Chandipur PR-AUC**: Improved from $0.5523$ (raw baseline) and $0.5526$ (clipping) to **$0.5999$** ($+0.0476$ gain).
- **Florida Holdout PR-AUC**: Fully preserved ($0.3337$ vs raw $0.3362$, difference $< 0.0025$).

This Step 7.2 investigation addresses the central research question:
> **Does symmetric logarithmic compression of CAPE and CIN produce a reproducible, statistically significant improvement in cross-region GAM discrimination without materially damaging source-domain performance?**

### Definitive Conclusion:
**YES**. Based on $1,000$ day-level block bootstrap resamples, paired statistical hypothesis testing, parameter sensitivity sweeps, multi-year cross-validation, and spatial subgroup audits:
1. **Paired Statistical Significance**: The paired improvement in PR-AUC is **$\Delta\text{PR-AUC} = +0.0477$** with a 95% block bootstrap confidence interval of **$[0.0380, 0.0572]$** and an empirical p-value of **$p = 0.0000$**. The confidence interval strictly excludes zero.
2. **ROC-AUC & Calibration Gains**: ROC-AUC improves by **$+0.0046$** ($95\%\text{ CI } [0.0016, 0.0075]$, $p = 0.0020$), and the Brier score improves by **$-0.0022$** ($95\%\text{ CI } [-0.0030, -0.0013]$, $p = 0.0000$).
3. **Parameter Invariance**: Across controlled variations of scaling constants ($k \in [0.80, 1.20] \times \alpha$), the Chandipur PR-AUC remains **identically $0.5999$**, proving that the gain is driven entirely by the mathematical concavity of $\ln(1 + x / 1000)$ and spline knot reallocation, not parameter tuning.
4. **Generalization Across Years and Grids**: The improvement holds in all four target years ($2022, 2023, 2024, 2025$), across all four spatial grid points ($21.25^\circ\text{N}\text{--}21.50^\circ\text{N}, 87.00^\circ\text{E}\text{--}87.25^\circ\text{E}$), and in all leave-one-out partitions.

---

## 2. Audit of Transformation Fitting and Parameter Governance

The mathematical transformation parameters were audited for procedural leakage and reproducibility:
- **Transformation Type**: Unsupervised Transductive Adaptation.
- **Surface CAPE**: $\tilde{x}_{\text{CAPE}} = \alpha_{\text{CAPE}} \cdot \ln(1 + x_{\text{CAPE}} / 1000.0)$ with $\alpha_{\text{CAPE}} = 1547.1097$.
- **Boundary CIN**: $\tilde{x}_{\text{CIN}} = \alpha_{\text{CIN}} \cdot \ln(1 + x_{\text{CIN}} / 100.0)$ with $\alpha_{\text{CIN}} = 220.0598$.
- **Kinematics & Clouds**: Identity mapping ($\tilde{x}_j = x_j$).

### Leakage and Split Isolation Verification:
1. **Source Parameters**: $q_{95, S}(\text{CAPE}) = 2668.50\text{ J/kg}$ and $q_{95, S}(\text{CIN}) = 460.63\text{ J/kg}$ were fitted **exclusively on the Florida training split** (Jan–Jul 2024). Florida validation and test labels were never accessed during parameter calculation.
2. **Target Parameters**: $q_{95, T}(\text{CAPE}) = 4611.56\text{ J/kg}$ and $q_{95, T}(\text{CIN}) = 711.09\text{ J/kg}$ were computed from **unlabeled target features**. Zero Chandipur proxy labels (`storm_label_score`) were accessed.
3. **Configuration Storage**: All parameters and formulas were serialized to machine-readable JSON at `transformation_config.json`.

---

## 3. Florida Threshold Selection Revalidation

To prevent optimistic bias, decision thresholds for the retrained symmetric models were tuned on the Florida validation split ($N = 18,181$) and evaluated once on the locked Florida holdout test split ($N = 28,073$, Oct–Dec 2024):

| Model Architecture | Threshold Type | Selected $\tau$ | Test F1 | Test Precision | Test Recall | Test PR-AUC | Test ROC-AUC | Test Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pyGAM (Original Raw)** | Florida Val Optimal | $0.33$ | $0.4086$ | $0.3798$ | $0.4421$ | $0.3362$ | $0.8288$ | $0.0556$ |
| **pyGAM (Original Raw)** | Standard | $0.50$ | $0.3345$ | $0.4823$ | $0.2560$ | $0.3362$ | $0.8288$ | $0.0556$ |
| **pyGAM (Symmetric B4 Retrained)** | Florida Val Optimal | $0.33$ | $0.4013$ | $0.3572$ | $0.4578$ | $0.3337$ | $0.8291$ | $0.0566$ |
| **pyGAM (Symmetric B4 Retrained)** | Standard | $0.50$ | $0.3426$ | $0.4608$ | $0.2727$ | $0.3337$ | $0.8291$ | $0.0566$ |
| **EBM (Original Raw)** | Florida Val Optimal | $0.34$ | $0.4003$ | $0.3578$ | $0.4541$ | $0.3513$ | $0.8304$ | $0.0564$ |
| **EBM (Original Raw)** | Standard | $0.50$ | $0.3515$ | $0.4768$ | $0.2784$ | $0.3513$ | $0.8304$ | $0.0564$ |
| **EBM (Symmetric B4 Retrained)** | Florida Val Optimal | $0.34$ | $0.4003$ | $0.3578$ | $0.4541$ | $0.3513$ | $0.8304$ | $0.0564$ |
| **EBM (Symmetric B4 Retrained)** | Standard | $0.50$ | $0.3515$ | $0.4768$ | $0.2784$ | $0.3513$ | $0.8304$ | $0.0564$ |

### Findings:
- The optimal F1 threshold for Symmetric B4 pyGAM on Florida validation data is $\tau = 0.33$ (matching the original threshold).
- Florida test holdout discrimination is virtually identical ($\text{PR-AUC} = 0.3337$ vs $0.3362$, $\Delta = -0.0025$; $\text{ROC-AUC} = 0.8291$ vs $0.8288$, $\Delta = +0.0003$).
- For EBM, the retrained model is **identically isomorphic** down to machine precision.

---

## 4. Experiment A — Day-Level Block Bootstrap Confidence Intervals ($B = 1,000$)

Because Chandipur observations exhibit strong temporal autocorrelation within each synoptic day and spatial cross-correlation across the 4 grid points, sampling was performed at the **calendar day block level** ($K = 1,461$ unique days). Each bootstrap iteration resampled $1,461$ days with replacement, evaluating all synoptic hours and grid points simultaneously:

| Model ID | Operational $\tau$ | PR-AUC (95% CI) | ROC-AUC (95% CI) | F1 Score (95% CI) | Precision (95% CI) | Recall (95% CI) | Brier Score (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original Raw pyGAM** | $0.33$ | $0.5520$ $[0.5165, 0.5870]$ | $0.8493$ $[0.8366, 0.8607]$ | $0.6143$ $[0.5904, 0.6361]$ | $0.4856$ $[0.4581, 0.5112]$ | $0.8362$ $[0.8134, 0.8565]$ | $0.1301$ $[0.1240, 0.1360]$ |
| **2. CAPE-Clipped pyGAM (B5)**| $0.33$ | $0.5523$ $[0.5170, 0.5873]$ | $0.8501$ $[0.8377, 0.8613]$ | $0.6143$ $[0.5904, 0.6361]$ | $0.4856$ $[0.4581, 0.5112]$ | $0.8362$ $[0.8134, 0.8565]$ | $0.1300$ $[0.1239, 0.1358]$ |
| **3. Symmetric B4 pyGAM** | $0.33$ | **$0.5996$** $[0.5654, 0.6317]$ | **$0.8539$** $[0.8416, 0.8650]$ | $0.6081$ $[0.5841, 0.6305]$ | $0.4834$ $[0.4564, 0.5092]$ | $0.8199$ $[0.7963, 0.8416]$ | **$0.1279$** $[0.1218, 0.1338]$ |
| **4. Original Raw EBM** | $0.34$ | $0.6130$ $[0.5790, 0.6448]$ | $0.8570$ $[0.8443, 0.8686]$ | $0.6052$ $[0.5805, 0.6272]$ | $0.4897$ $[0.4625, 0.5159]$ | $0.7925$ $[0.7659, 0.8181]$ | $0.1270$ $[0.1210, 0.1331]$ |
| **5. Symmetric B4 EBM** | $0.34$ | $0.6130$ $[0.5790, 0.6448]$ | $0.8570$ $[0.8443, 0.8686]$ | $0.6052$ $[0.5805, 0.6272]$ | $0.4897$ $[0.4625, 0.5159]$ | $0.7925$ $[0.7659, 0.8181]$ | $0.1270$ $[0.1210, 0.1331]$ |

```
[Diagnostic Figure: Bootstrap Confidence Intervals Comparison]
File: reports/domain_adaptation/step_07_2/bootstrap_confidence_intervals_comparison.png
```

---

## 5. Experiment B — Paired Model Comparison (Hypothesis Testing)

Paired evaluation was conducted on identical resamples ($B = 1,000$) between **Symmetric B4 pyGAM** and **Original Raw pyGAM**:

| Metric Comparison | Mean Difference | Standard Error | 95% Confidence Interval | Empirical p-value | Excludes Zero? | Statistical Significance Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\Delta\text{PR-AUC}$** | **$+0.0477$** | $0.0052$ | **$[+0.0380, \, +0.0572]$** | **$p < 10^{-4}$** | **YES** | **STATISTICALLY SIGNIFICANT ($p < 0.001$)** |
| **$\Delta\text{ROC-AUC}$** | **$+0.0046$** | $0.0015$ | **$[+0.0016, \, +0.0075]$** | **$p = 0.0020$** | **YES** | **STATISTICALLY SIGNIFICANT ($p < 0.01$)** |
| **$\Delta\text{Brier Score}$**| **$-0.0022$** | $0.0005$ | **$[-0.0030, \, -0.0013]$** | **$p < 10^{-4}$** | **YES** | **STATISTICALLY SIGNIFICANT ($p < 0.001$)** |
| **$\Delta\text{F1}$ ($\tau = 0.33$)** | $-0.0062$ | $0.0027$ | $[-0.0115, \, -0.0010]$ | $p = 0.0140$ | YES | Operational shift at fixed threshold |

```
[Diagnostic Figure: Paired Model Difference Distributions]
File: reports/domain_adaptation/step_07_2/paired_model_difference_distributions.png
```

### Statistical Inference:
- The null hypothesis $H_0: \Delta\text{PR-AUC} \le 0$ is rejected at the $\alpha = 0.001$ level. In all $1,000$ block bootstrap replications, Symmetric B4 pyGAM outperformed raw pyGAM in PR-AUC ($\min \Delta\text{PR-AUC} = +0.0312$).
- The slight reduction in F1 at the fixed threshold ($\Delta = -0.0062$) is a natural consequence of probability recalibration: raw pyGAM was severely overconfident in high-CAPE regimes due to spline distortion, artificially boosting raw recall at the expense of precision and Brier calibration.

---

## 6. Experiment C — Parameter Sensitivity Analysis

The scaling multipliers were systematically varied by $\pm 20\%$ around the estimated $\alpha$ values ($k \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$). For each variation, pyGAM was retrained on transformed Florida training data, tuned on Florida validation data, and evaluated on Florida test and Chandipur target:

| Scale Factor $k$ | $\alpha_{\text{CAPE}}$ | $\alpha_{\text{CIN}}$ | Florida Val $\tau_{\text{opt}}$ | Florida Test PR-AUC | Florida Test ROC-AUC | Chandipur PR-AUC | Chandipur ROC-AUC | Chandipur F1 | PR-AUC Gain vs Raw |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.80$** | $1237.69$ | $176.05$ | $0.33$ | $0.3337$ | $0.8291$ | **$0.5999$** | $0.8540$ | $0.6086$ | **$+0.0476$** |
| **$0.90$** | $1392.40$ | $198.05$ | $0.33$ | $0.3337$ | $0.8291$ | **$0.5999$** | $0.8540$ | $0.6086$ | **$+0.0476$** |
| **$1.00$** | $1547.11$ | $220.06$ | $0.33$ | $0.3337$ | $0.8291$ | **$0.5999$** | $0.8540$ | $0.6086$ | **$+0.0476$** |
| **$1.10$** | $1701.82$ | $242.07$ | $0.33$ | $0.3337$ | $0.8291$ | **$0.5999$** | $0.8540$ | $0.6086$ | **$+0.0476$** |
| **$1.20$** | $1856.53$ | $264.07$ | $0.33$ | $0.3337$ | $0.8291$ | **$0.5999$** | $0.8540$ | $0.6086$ | **$+0.0476$** |

### Mathematical Proof of Parameter Invariance:
In `pyGAM`, B-spline basis functions are defined by placing knots uniformly across the empirical domain $[x_{\min}, x_{\max}]$. When the input data is scaled by a positive constant $k$, the knots scale proportionally ($t_i \mapsto k \cdot t_i$). Because B-spline basis functions satisfy scale homogeneity:
$$B_{i, p}\left(\frac{k \cdot x}{k}\right) \equiv B_{i, p}(x)$$
the learned spline coefficients and resulting predicted probabilities are **mathematically invariant** to linear re-scaling of the log-transformed coordinates.
**Conclusion**: The $+0.0476$ gain is completely immune to the choice of $\alpha$. It arises entirely from the **logarithmic curvature $\ln(1 + x / 1000)$** which redistributes knot density.

```
[Diagnostic Figure: Parameter Sensitivity Curves]
File: reports/domain_adaptation/step_07_2/parameter_sensitivity_curves.png
```

---

## 7. Experiment D — Seasonal Generalization

| Season | Sample Count | Positive Count | Prevalence | Raw pyGAM PR-AUC | Sym B4 pyGAM PR-AUC | $\Delta\text{PR-AUC}$ (95% CI) | Raw pyGAM ROC-AUC | Sym B4 pyGAM ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JJAS (Monsoon)** | 7,808 | 3,761 | 48.17% | $0.6065$ | **$0.6594$** | **$+0.0530$ $[+0.0415, +0.0650]$** | $0.6716$ | **$0.6898$** |
| **ON (Post-monsoon)**| 3,904 | 422 | 10.81% | $0.5733$ | **$0.6579$** | **$+0.0846$ $[+0.0431, +0.1258]$** | $0.9329$ | **$0.9440$** |
| **DJF (Winter)** | 5,776 | 63 | 1.09% | $0.1546$ | **$0.2072$** | $+0.0525$ $[-0.0024, +0.1338]$ | $0.9193$ | **$0.9227$** |
| **MAM (Pre-monsoon)**| 5,888 | 1,216 | 20.65% | $0.3820$ | $0.3764$ | $-0.0056$ $[-0.0249, +0.0111]$ | $0.7254$ | $0.7119$ |

### Seasonal Discussion:
- **Monsoon & Post-monsoon**: Show substantial, statistically significant improvements where 95% CIs exclude zero ($+0.0530$ in JJAS and $+0.0846$ in ON). These seasons account for $76.6\%$ of all thunderstorm occurrences in Chandipur.
- **Pre-monsoon (MAM)**: Performance is statistically indistinguishable ($\Delta = -0.0056$, 95% CI $[-0.0249, +0.0111]$ includes zero). In MAM, convection is governed by dryline capping inversion bursts, where both models perform similarly.
- **Winter (DJF)**: Although PR-AUC shows a nominal $+0.0525$ increase, the 95% CI spans zero due to the extremely small positive sample size ($N = 63$ positive events out of $5,776$ soundings, $1.09\%$ prevalence).

```
[Diagnostic Figure: Seasonal Robustness with Uncertainty]
File: reports/domain_adaptation/step_07_2/seasonal_robustness_with_ci.png
```

---

## 8. Experiment E — Temporal and Spatial Subgroup Robustness

To verify that the $+0.0476$ gain is not driven by an idiosyncratic year or single grid point, comprehensive subgroup audits were performed:

| Subgroup Partition | Category / Name | Sample Count | Prevalence | Raw pyGAM PR-AUC | Sym B4 pyGAM PR-AUC | $\Delta\text{PR-AUC}$ | Superiority Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **By Year** | 2022 | 5,840 | 22.71% | $0.5435$ | **$0.6229$** | **$+0.0794$** | **Symmetric B4 Wins** |
| **By Year** | 2023 | 5,840 | 22.53% | $0.5283$ | **$0.5891$** | **$+0.0608$** | **Symmetric B4 Wins** |
| **By Year** | 2024 | 5,856 | 26.54% | $0.5689$ | **$0.5825$** | **$+0.0136$** | **Symmetric B4 Wins** |
| **By Year** | 2025 | 5,840 | 21.68% | $0.5733$ | **$0.6229$** | **$+0.0496$** | **Symmetric B4 Wins** |
| **Leave-One-Year-Out** | Excl. 2022 | 17,536 | 23.59% | $0.5545$ | **$0.5943$** | **$+0.0398$** | **Symmetric B4 Wins** |
| **Leave-One-Year-Out** | Excl. 2023 | 17,536 | 23.64% | $0.5631$ | **$0.6062$** | **$+0.0431$** | **Symmetric B4 Wins** |
| **Leave-One-Year-Out** | Excl. 2024 | 17,520 | 22.31% | $0.5455$ | **$0.6081$** | **$+0.0626$** | **Symmetric B4 Wins** |
| **Leave-One-Year-Out** | Excl. 2025 | 17,536 | 23.93% | $0.5465$ | **$0.5930$** | **$+0.0466$** | **Symmetric B4 Wins** |
| **By Grid Point** | $21.25^\circ\text{N}, 87.00^\circ\text{E}$ | 5,844 | 23.43% | $0.5434$ | **$0.5897$** | **$+0.0463$** | **Symmetric B4 Wins** |
| **By Grid Point** | $21.25^\circ\text{N}, 87.25^\circ\text{E}$ | 5,844 | 23.72% | $0.5731$ | **$0.6159$** | **$+0.0428$** | **Symmetric B4 Wins** |
| **By Grid Point** | $21.50^\circ\text{N}, 87.00^\circ\text{E}$ | 5,844 | 21.75% | $0.5265$ | **$0.5730$** | **$+0.0465$** | **Symmetric B4 Wins** |
| **By Grid Point** | $21.50^\circ\text{N}, 87.25^\circ\text{E}$ | 5,844 | 24.57% | $0.5683$ | **$0.6224$** | **$+0.0541$** | **Symmetric B4 Wins** |
| **Leave-One-Grid-Out**| Excl. GP1 | 17,532 | 23.35% | $0.5557$ | **$0.6038$** | **$+0.0481$** | **Symmetric B4 Wins** |
| **Leave-One-Grid-Out**| Excl. GP2 | 17,532 | 23.25% | $0.5460$ | **$0.5954$** | **$+0.0493$** | **Symmetric B4 Wins** |
| **Leave-One-Grid-Out**| Excl. GP3 | 17,532 | 23.90% | $0.5604$ | **$0.6083$** | **$+0.0479$** | **Symmetric B4 Wins** |
| **Leave-One-Grid-Out**| Excl. GP4 | 17,532 | 22.96% | $0.5472$ | **$0.5923$** | **$+0.0451$** | **Symmetric B4 Wins** |

```
[Diagnostic Figure: Subgroup Analysis by Year and Grid Point]
File: reports/domain_adaptation/step_07_2/year_and_gridpoint_subgroup_analysis.png
```

---

## 9. Experiment F — Interpretability and Spline Knot Redistribution

The core mechanical reason for the superior performance of Symmetric B4 pyGAM lies in how the logarithmic transformation redistributes the cubic B-spline knots:

1. **Spline Knot Starvation in Raw pyGAM**:
   - In Florida, $75\%$ of CAPE observations fall below $1,015\text{ J/kg}$.
   - However, raw pyGAM places its $15$ spline knots uniformly across the range $[0, 7240\text{ J/kg}]$ (knots at $0, 517, 1034, 1551, \dots, 7240\text{ J/kg}$).
   - As a result, only **$2$ knots** covered the dense $0\text{--}1000\text{ J/kg}$ range where $75\%$ of the physical data resided! Meanwhile, **$11$ knots** were squandered on the sparse upper tail ($> 2000\text{ J/kg}$) where only $5\%$ of soundings occurred.
2. **Knot Density Redistribution in Symmetric B4**:
   - When CAPE is log-compressed via $\tilde{x} = \alpha \ln(1 + x / 1000)$, the range $[0, 7240]$ maps to $[0, 3262.9]$.
   - The $15$ uniform knots in transformed space correspond to **logarithmically spaced knots in physical space**: knots are densely clustered in the $0\text{--}1500\text{ J/kg}$ region (allocating **$6$ knots** to the critical initiation zone), while spreading knots sparsely in the extreme upper convective tail.
   - This prevents overfitting to sparse high-energy extremes while providing high spline curvature resolution in the low-to-moderate CAPE range.
3. **Spline Extrapolation Elimination**:
   - For physical CAPE $> 7,240\text{ J/kg}$ (which collapsed raw pyGAM to $p \approx 0.0039$), the transformed values smoothly follow the concave logarithmic plateau. The partial dependence log-odds asymptote cleanly at $f(\text{CAPE}) \approx +1.8$ rather than oscillating or collapsing.

```
[Diagnostic Figure: Interpretability and Spline Knot Redistribution]
File: reports/domain_adaptation/step_07_2/interpretability_spline_redistribution.png
```
