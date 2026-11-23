# Step 7.2.1 — Statistical Claims and Threshold Revalidation Audit

## 1. Executive Summary

This independent audit rigorously evaluates the statistical methods, computational claims, threshold selection mechanics, and generalizability assertions reported in Step 7.2 for **Symmetric B4 pyGAM**.

### Summary of Audit Determinations:
1. **Bootstrap p-Value Calculation**: **AUDITED & CORRECTED**. Step 7.2 reported $p < 10^{-4}$ for the paired PR-AUC difference based on $B = 1,000$ bootstrap iterations. In a Monte Carlo bootstrap of size $B = 1,000$, when zero exceedances ($b = 0$) are observed, the minimum resolvable empirical p-value under the standard finite-sample correction (Davison & Hinkley, 1997) is $p = \frac{b + 1}{B + 1} = \frac{1}{1001} \approx \mathbf{0.0010}$ (one-sided) and $\mathbf{0.0020}$ (two-sided). Reporting $p < 10^{-4}$ was an overstatement of inferential resolution. However, the hypothesis of zero improvement remains decisively rejected at $p \le 0.0020$.
2. **Paired Confidence Intervals**: **VERIFIED**. Reconstructed paired day-level block bootstrapping ($K = 1,461$ unique days, $B = 1,000$) confirms that the 95% confidence interval for PR-AUC difference is **$[+0.0380, \, +0.0572]$** (mean $+0.0477$, median $+0.0476$, standard error $0.0052$). The interval strictly excludes zero.
3. **Threshold Re-Selection**: **VERIFIED INDEPENDENT**. Grid search across $\tau \in [0.10, 0.90]$ on Florida validation data confirmed that $\tau = 0.33$ is the genuine empirical argmax of F1 for Symmetric pyGAM (peaking at $F_1 = 0.6537$, compared to $0.6530$ at $\tau=0.32$ and $0.6524$ at $\tau=0.34$). It was not an inherited placeholder.
4. **Parameter Invariance**: **MATHEMATICALLY & NUMERICALLY PROVED**. Retraining pyGAM across scale multipliers $k \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$ produced identical predictions down to machine precision ($\max |\Delta p| \le 1.72 \times 10^{-14}$). B-spline scale cancellation via the de Boor-Cox recurrence relation proves that pyGAM's design matrix is invariant to linear coordinate scaling.
5. **Seasonal Claims**: **CLARIFIED & CONSTRAINED**. JJAS (Monsoon) and ON (Post-monsoon) show statistically significant gains whose 95% CIs exclude zero. DJF (Winter) has only $63$ positive events ($1.09\%$ prevalence) and its 95% CI spans zero ($[-0.0024, \, +0.1338]$); no claim of statistically significant winter improvement is permitted.
6. **Target Domain Ground Truth**: **CLARIFIED**. Chandipur labels are ERA5-derived proxy scores (`storm_label_score`), not verified in-situ radar/ASOS observations. All findings represent proxy benchmark domain transfer.

---

## 2. Bootstrap p-Value Correctness Audit

In Step 7.2, paired differences were evaluated over $B = 1,000$ block bootstrap replications. For $\Delta\text{PR-AUC}$, zero resamples produced a negative difference ($b = 0$).

### Theoretical Flaw in Naive Empirical p-Value:
The naive estimator $\hat{p} = \frac{b}{B} = \frac{0}{1000} = 0.0$ cannot be interpreted as $p < 0.0001$. A finite sample of $B$ resamples cannot resolve probabilities smaller than the reciprocal of the sample size. 

Under standard bootstrap inference theory (Davison & Hinkley, 1997; Efron & Tibshirani, 1993), the unbiased finite-sample corrected empirical p-value is:
$$p_{\text{corrected, one-sided}} = \frac{b + 1}{B + 1}$$
$$p_{\text{corrected, two-sided}} = \min\left(1.0, \, 2 \cdot \frac{b + 1}{B + 1}\right)$$

### Re-evaluated Bootstrap Statistics:

| Metric Comparison | $B$ | Observed Exceedances $b$ | Naive Reported $p$ | Corrected One-Sided $p$ | Corrected Two-Sided $p$ | Minimum Resolvable $p$ | Audit Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PR-AUC Difference** | 1,000 | 0 | $p < 10^{-4}$ | **$0.0010$** | **$0.0020$** | $0.0010$ | **CORRECTED** (Overstated resolution; true limit $p \le 0.0020$) |
| **ROC-AUC Difference**| 1,000 | 1 | $p = 0.0010$ | **$0.0020$** | **$0.0040$** | $0.0010$ | **CORRECTED** (Corrected to $0.0040$) |
| **Brier Score Difference**| 1,000 | 0 | $p < 10^{-4}$ | **$0.0010$** | **$0.0020$** | $0.0010$ | **CORRECTED** (True limit $p \le 0.0020$) |
| **F1 Diff (Frozen $\tau=0.33$)**| 1,000 | 7 | $p = 0.0070$ | **$0.0080$** | **$0.0160$** | $0.0010$ | **CONSISTENT** ($p = 0.0160$) |

**Conclusion**: While the reporting of $p < 10^{-4}$ was an overstatement of statistical resolution for $B = 1,000$, the corrected two-sided p-value ($p = 0.0020$) confirms that the discrimination improvement remains **statistically significant at the $\alpha = 0.01$ level**.

---

## 3. Reconstructed Paired Confidence Intervals

Evaluating Symmetric B4 pyGAM versus Original Raw pyGAM across identical day blocks ($K = 1,461$, $B = 1,000$):

| Evaluation Metric | Mean Difference | Median Difference | Standard Error | 95% Confidence Interval | Corrected Two-Sided $p$ | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\Delta\text{PR-AUC}$** | **$+0.0477$** | **$+0.0476$** | $0.0052$ | **$[+0.0380, \, +0.0572]$** | $0.0020$ | **SIGNIFICANT (Strictly excludes 0)** |
| **$\Delta\text{ROC-AUC}$** | **$+0.0046$** | **$+0.0047$** | $0.0015$ | **$[+0.0016, \, +0.0075]$** | $0.0040$ | **SIGNIFICANT (Strictly excludes 0)** |
| **$\Delta\text{Brier Score}$**| **$-0.0022$** | **$-0.0022$** | $0.0005$ | **$[-0.0030, \, -0.0013]$** | $0.0020$ | **SIGNIFICANT (Calibration improved)** |
| **$\Delta\text{F1}$ ($\tau = 0.33$)**| $-0.0062$ | $-0.0062$ | $0.0027$ | $[-0.0115, \, -0.0010]$ | $0.0160$ | **SIGNIFICANT (Fixed threshold shift)**|

---

## 4. Threshold Re-selection Verification

The objective function $F_1(\tau)$ on Florida validation data ($N = 18,181$) was audited across $\tau \in [0.10, 0.90]$:
- For **Symmetric B4 pyGAM**:
  - $\tau = 0.31 \implies F_1 = 0.652576$
  - $\tau = 0.32 \implies F_1 = 0.653022$
  - **$\tau = 0.33 \implies F_1 = \mathbf{0.653720}$ (Exact Global Maximum)**
  - $\tau = 0.34 \implies F_1 = 0.652440$
  - $\tau = 0.35 \implies F_1 = 0.651535$
- For **Symmetric B4 EBM**:
  - $\tau = 0.33 \implies F_1 = 0.655877$
  - **$\tau = 0.34 \implies F_1 = \mathbf{0.655953}$ (Exact Global Maximum)**
  - $\tau = 0.35 \implies F_1 = 0.655231$

The selected thresholds ($\tau = 0.33$ for pyGAM, $\tau = 0.34$ for EBM) are **empirically optimal on the validation set** and were not inherited or hand-picked.

### Final Threshold Performance Matrix:

| Model Architecture | Evaluation Domain | Threshold Type | Value $\tau$ | PR-AUC | ROC-AUC | F1 Score | Precision | Recall | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pyGAM (Original Raw)** | Florida Test Holdout | Florida Val Optimal | $0.33$ | $0.3362$ | $0.8288$ | $0.4086$ | $0.3798$ | $0.4421$ | $0.0556$ |
| **pyGAM (Original Raw)** | Florida Test Holdout | Standard | $0.50$ | $0.3362$ | $0.8288$ | $0.3345$ | $0.4823$ | $0.2560$ | $0.0556$ |
| **pyGAM (Symmetric B4)** | Florida Test Holdout | Florida Val Optimal | $0.33$ | **$0.3337$** | **$0.8291$** | $0.4013$ | $0.3572$ | $0.4578$ | $0.0566$ |
| **pyGAM (Symmetric B4)** | Florida Test Holdout | Standard | $0.50$ | **$0.3337$** | **$0.8291$** | $0.3426$ | $0.4608$ | $0.2727$ | $0.0566$ |
| **pyGAM (Original Raw)** | Chandipur Target Benchmark | Frozen Source Optimal | $0.33$ | $0.5523$ | $0.8494$ | $0.6146$ | $0.4858$ | $0.8365$ | $0.1301$ |
| **pyGAM (Original Raw)** | Chandipur Target Benchmark | Standard | $0.50$ | $0.5523$ | $0.8494$ | $0.5170$ | $0.5709$ | $0.4724$ | $0.1301$ |
| **pyGAM (Symmetric B4)** | Chandipur Target Benchmark | Tuned Source Optimal | $0.33$ | **$0.5999$** | **$0.8540$** | $0.6086$ | $0.4837$ | $0.8204$ | **$0.1279$** |
| **pyGAM (Symmetric B4)** | Chandipur Target Benchmark | Standard | $0.50$ | **$0.5999$** | **$0.8540$** | $0.5321$ | $0.5967$ | $0.4800$ | **$0.1279$** |

---

## 5. Parameter-Scaling Invariance Audit

The Step 7.2 observation that PR-AUC was identically $0.5999$ across $k \in \{0.80, 0.90, 1.00, 1.10, 1.20\}$ was audited for code artifacts versus true mathematical behavior:

### Implementation Verification:
- The script **did not reuse cached models or predictions**. In each iteration, a fresh `LogisticGAM` instance was initialized and fitted on $X_{\text{train}, k}$.
- The maximum absolute prediction difference across all $23,376$ Chandipur observations between any two models is:
  $$\max |p_k - p_{1.00}| \le \mathbf{1.7208 \times 10^{-14}}$$
  $$\text{mean} |p_k - p_{1.00}| \le \mathbf{6.1100 \times 10^{-16}}$$
  This difference is strictly at the IEEE 754 64-bit double-precision floating-point roundoff floor.

### Mathematical Proof of B-Spline Scale Invariance:
In `pyGAM`, cubic B-spline basis functions are defined by the de Boor-Cox recurrence relation:
$$B_{i, 0}(x) = \begin{cases} 1 & \text{if } t_i \le x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$
$$B_{i, p}(x) = \frac{x - t_i}{t_{i+p} - t_i} B_{i, p-1}(x) + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1, p-1}(x)$$
When $x \mapsto k \cdot x$, the empirical boundary edge knots scale by $k$, scaling all internal knots $t_i \mapsto k \cdot t_i$.
Substituting $k \cdot x$ and $k \cdot t$:
$$\frac{(k \cdot x) - (k \cdot t_i)}{(k \cdot t_{i+p}) - (k \cdot t_i)} = \frac{k(x - t_i)}{k(t_{i+p} - t_i)} = \frac{x - t_i}{t_{i+p} - t_i}$$
The scale factor $k$ cancels out identically in every term of the recurrence relation. Consequently:
$$B_{i, p}(k \cdot x; \, k \cdot t) \equiv B_{i, p}(x; \, t)$$
The basis design matrix $X_{\text{spline}}$ is **mathematically identical**. Because the smoothing penalty matrix is constructed on coefficient second differences, the penalized log-likelihood function is identical, yielding mathematically invariant predicted probabilities.
**Audit Determination**: The parameter-scaling invariance is a **rigorous mathematical property** of uniform B-splines, not a software bug or caching error.

---

## 6. Seasonal and Subgroup Claims Audit

| Convective Season | Sample Count | Positive Count | Positive Prevalence | $\Delta\text{PR-AUC}$ | 95% Bootstrap CI | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **JJAS (Monsoon)** | 7,808 | 3,761 | 48.17% | **$+0.0530$** | **$[+0.0415, \, +0.0650]$** | **SIGNIFICANT (Strictly excludes 0)** |
| **ON (Post-monsoon)**| 3,904 | 422 | 10.81% | **$+0.0846$** | **$[+0.0431, \, +0.1258]$** | **SIGNIFICANT (Strictly excludes 0)** |
| **DJF (Winter)** | 5,776 | 63 | 1.09% | $+0.0525$ | $[-0.0024, \, +0.1338]$ | **NOT SIGNIFICANT (CI spans 0)** |
| **MAM (Pre-monsoon)**| 5,888 | 1,216 | 20.65% | $-0.0056$ | $[-0.0249, \, +0.0111]$ | **NO DIFFERENCE (CI spans 0)** |

### Critical Subgroup Audit Findings:
1. **Winter (DJF) Claim Restriction**: With only $63$ positive thunderstorm hours out of $5,776$ winter soundings ($1.09\%$ positive prevalence), statistical power is low. The 95% bootstrap CI $[-0.0024, +0.1338]$ includes zero. **No claim of statistically significant winter improvement is justified.**
2. **Pre-monsoon (MAM) Claim Restriction**: In MAM, $\Delta\text{PR-AUC} = -0.0056$ and the 95% CI $[-0.0249, +0.0111]$ includes zero. Both raw and symmetric models exhibit equivalent discrimination during pre-monsoon dryline events.
3. **Monsoon & Post-monsoon Concentration**: The statistically verified improvements are concentrated in **JJAS ($+0.0530$)** and **ON ($+0.0846$)**. Because JJAS and ON account for **$76.6\%$ of all thunderstorm soundings** in Chandipur ($4,183$ out of $5,462$ positive events), the overall full-dataset improvement ($\Delta = +0.0477$) is driven by these high-convective regimes.
4. **Proxy Label Nature**: All target metrics are evaluated against the composite heuristic proxy label (`storm_label_score`). They must not be claimed as verified real-world operational forecast skill against in-situ observations.
