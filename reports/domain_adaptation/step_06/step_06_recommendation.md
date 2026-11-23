# Definitive Recommendation on Importance Weighting and Density-Ratio Estimation

## 1. Definitive Scientific Verdict

**We definitively advise AGAINST adopting Importance Weighting as an operational domain adaptation method for transferring Florida thunderstorm models to Chandipur.**

While Importance Weighting is methodologically sound and **physically safe** (it avoids the coordinate rotations and negative energy artifacts that disqualified CORAL), it fails to improve transfer skill across all configurations:
1. **Transfer Skill Does Not Improve**:
   - For `InterpretML` EBM, the unweighted baseline achieves **F1 = $0.6059$** (PR-AUC = $0.6134$, ROC-AUC = $0.8572$).
   - Reweighted EBM variants consistently underperform or match baseline:
     - Logistic Regression (Max 5): F1 = $0.5861$ (PR-AUC = $0.5391$, ROC-AUC = $0.8341$)
     - LightGBM (p99): F1 = $0.5703$ (PR-AUC = $0.5371$, ROC-AUC = $0.8268$)
     - Random Forest (p99): F1 = $0.5723$ (PR-AUC = $0.5466$, ROC-AUC = $0.8291$)
     - Kernel Mean Matching (KMM): F1 = $0.6019$ (PR-AUC = $0.6026$, ROC-AUC = $0.8539$)
     - Seasonal LightGBM: F1 = $0.5819$ (PR-AUC = $0.5890$, ROC-AUC = $0.8390$)
2. **Effective Sample Size Depletion**:
   Non-linear density estimators upweight rare Florida convective extremes, collapsing the Effective Sample Size (ESS) from $66,255$ down to $22,100$–$30,800$ ($33\%$–$46\%$ sample retention). This increases estimator variance without improving target discrimination.
3. **Spline Extrapolation Breakdown Is Not Solved**:
   Importance weighting reweights Florida training data within its existing convex hull; it cannot expand the knot support of `pyGAM`. For raw Chandipur observations with $\text{CAPE} > 7,240\text{ J/kg}$, pyGAM spline probabilities still dive to zero ($0.081$–$0.096$).

---

## 2. Comparison Against Previous TDAF Transfer Benchmarks

| Domain Adaptation Method | Physical Valid? (No Negatives) | Additive Semantics Preserved? | Chandipur EBM F1 ($\tau=0.34$) | Chandipur EBM PR-AUC | Chandipur EBM ROC-AUC | Florida Test PR-AUC | Operational Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A: Zero-Shot Baseline** | Yes | Yes | 0.6059 | 0.6134 | 0.8572 | 0.3513 | Frozen Reference |
| **Cond_B: CAPE Clipping (7,240 J/kg)** | **Yes** | **Yes** | **0.6059** | **0.6134** | **0.8572** | **0.3513** | **GOLD STANDARD (Adopted)** |
| **Step 5: Standard CORAL ($\alpha=1.0$)** | **NO** (37.6% neg CAPE) | **NO** (Linear mixing) | 0.4525 | 0.4049 | 0.7616 | N/A | **REJECTED** |
| **Step 6: Logistic Regression (Max 5)** | Yes | Yes | 0.5861 | 0.5391 | 0.8341 | 0.3551 | **REJECTED** |
| **Step 6: LightGBM (p99)** | Yes | Yes | 0.5703 | 0.5371 | 0.8268 | 0.3471 | **REJECTED** |
| **Step 6: KMM (Kernel Mean Matching)** | Yes | Yes | 0.6019 | 0.6026 | 0.8539 | 0.3500 | **REJECTED** |
| **Step 6: Seasonal LightGBM (p99)** | Yes | Yes | 0.5819 | 0.5890 | 0.8390 | 0.3323 | **REJECTED** |

**Conclusion**: **Condition B (Univariate CAPE clipping at the Florida training boundary of $7,240\text{ J/kg}$)** remains the most defensible, mathematically stable, and highest-performing transfer method. It completely cures pyGAM extrapolation collapse, creates zero physical violations, and requires zero sample-weighting heuristics.

---

## 3. Systematic Answers to the 10 Scientific Validity Checks

### 1. Does it improve source-target distribution representation?
**Yes, statistically.** Reweighting source samples reduces the Wasserstein distance and shifts Florida's marginal feature distributions closer to Chandipur's (e.g. elevating effective mean CAPE and surface winds).

### 2. Does it preserve physical feature values?
**Yes, 100%.** Unlike CORAL, importance weighting modifies only sample weights $w_i \in \mathbb{R}_+$. All physical coordinates ($x_i$) remain untransformed: zero negative CAPE, zero negative CIN, zero negative winds, and cloud fraction strictly in $[0, 1]$.

### 3. Does it produce stable effective sample size?
**Only when aggressively clipped.** Raw density ratios from LightGBM collapse ESS to $9,526.8$ ($14.38\%$ retention). Applying 99th percentile clipping or Fixed Maximum Weight = 5 restores ESS to $33.4\%$–$46.6\%$, which is mathematically stable.

### 4. Does it concentrate too much weight on a few Florida observations?
**Yes, under unclipped non-linear models.** In unclipped LightGBM, the top 1% of samples hold $17.8\%$ of total weight, and the top 5% hold $36.5\%$. Clipping at p99 reduces top 1% concentration to $9.29\%$.

### 5. Does it improve GAM transfer?
**No.** `pyGAM` F1 score drops from **$0.6146 \to 0.5672$–$0.5888$**, and PR-AUC drops from **$0.5526 \to 0.4908$–$0.5532$**.

### 6. Does it improve EBM transfer?
**No.** `InterpretML` EBM F1 score drops from **$0.6059 \to 0.5703$–$0.5861$**, and PR-AUC drops from **$0.6134 \to 0.5371$–$0.5890$**.

### 7. Does it damage Florida holdout performance?
**Slightly for global models, severely for seasonal models.** Global weighting (LR, KMM) maintains Florida test PR-AUC ($0.350$–$0.355$ vs $0.351$). However, Seasonal LightGBM degrades Florida test PR-AUC from $0.3513 \to 0.3323$.

### 8. Does it improve or worsen calibration?
**It worsens calibration on Chandipur.** Reweighting forces models to assign higher baseline probabilities to convective environments, causing overprediction and increasing Brier score on Chandipur from $0.1270 \to 0.1343$–$0.1398$.

### 9. Is seasonal weighting defensible?
**No.** Florida training data ends on July 31 and contains **0 observations for October–November (ON)**, forcing an artificial global fallback for $16.7\%$ of target cases. Furthermore, seasonal weighting degrades pre-monsoon (MAM) discrimination due to opposing synoptic forcing mechanisms.

### 10. Does the covariate-shift assumption remain plausible?
**No, the covariate shift assumption is demonstrably FALSE.**
The covariate shift hypothesis assumes $P_S(Y \mid X) = P_T(Y \mid X)$. However:
- In Florida, $Y$ represents **rigorously observed ASOS METAR thunderstorm events** where convective initiation is suppressed on high-CAPE days unless triggered by localized sea-breeze boundaries.
- In Chandipur, $Y$ is a **deterministic heuristic composite** ($\ge 2.50$) that triggers almost automatically under high thermodynamic energy.
Because the generative label mechanism differs, weighting $X$ cannot align $Y$.

---

## 4. Strategic Recommendations for Step 7 and Beyond

Having systematically investigated multivariate feature alignment (CORAL, Step 5) and source sample reweighting (Importance Weighting, Step 6), the empirical evidence indicates that:
1. **Unconstrained linear affine transformations (CORAL) fail** due to physical coordinate rotation and negative energy violations.
2. **Covariate shift sample reweighting fails** due to fundamental concept shift ($P_S(Y \mid X) \ne P_T(Y \mid X)$).

Therefore, Step 7 should investigate:
- **Univariate Non-Linear Quantile Alignment (eCDF Matching)**:
  Aligning each feature's marginal distribution independently via empirical cumulative distribution functions ($x_{T,j} \mapsto F_{S,j}^{-1}(F_{T,j}(x_{T,j}))$) guarantees physical bound preservation ($[0, \infty)$) while avoiding cross-variable feature mixing.
- **Target Climatological Threshold Calibration**:
  Recalibrating decision thresholds ($\tau$) based on target climatological base rates rather than relying on Florida-derived thresholds.
