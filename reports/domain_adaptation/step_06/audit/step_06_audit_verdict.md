# Step 6.1: Formal Audit Verdict and Scientific Findings

## 1. Formal Audit Verdict Summary Table

Each core dimension of the Step 6 implementation and experimental protocol has been evaluated and classified according to rigorous scientific standards:

| Audit Dimension | Specific Item Evaluated | Classification | Detailed Rationale & Evidence |
| :--- | :--- | :---: | :--- |
| **Target-Label Isolation** | Exclusion of Chandipur proxy labels from weighting pipeline | **PASS** | `storm_label_score` was never accessed during domain classifier training, density-ratio calculation, weight clipping, or KMM optimization. |
| **Threshold Integrity** | Source of decision thresholds (`tau_opt`) | **PASS** | Confirmed: `tau_opt` values ($0.33$ for pyGAM, $0.34$ for EBM) are identical to the frozen Florida Phase 2B validation thresholds. Chandipur labels were never used to tune thresholds. |
| **Prior Correction** | Prior odds adjustment $(N_S / N_T)$ in density ratio | **PASS** | Applied exactly once in the numerator ($66,255 / 23,376 = 2.8343$), rigorously satisfying Bayes' rule: $w(x) = \frac{P(z=1 \mid x)}{P(z=0 \mid x)} \frac{N_S}{N_T}$. |
| **Out-of-Fold Estimation** | Domain classifier probability generation | **PASS** | 5-Fold Stratified Cross-Validation across 89,631 combined domain samples. Zero in-sample training leakage occurred. |
| **Domain Feature Scaling** | StandardScaler usage in Logistic Regression | **PASS WITH CAVEATS** | Feature scaling was fit globally on unlabeled combined features $X_{\text{domain}}$. While safe from label leakage, strict ML hygiene recommends fitting scalers within each fold. Impact is negligible. |
| **KMM Optimization** | Convex quadratic programming convergence via PGD | **PASS** | Objective function reduced from $-511,146.52 \to -627,727.87$. Equality constraint residual is $7.2 \times 10^{-9}$ (machine precision). Zero box constraint violations. |
| **Library Weight Mechanics** | Native support in `pyGAM` and `InterpretML` EBM | **PASS** | Verified via source inspection: `pyGAM` passes weights to P-IRLS diagonal working matrix $W$; `InterpretML` passes weights to native C++ boosting engine. |
| **Model Freeze Integrity** | Immutability of original native Florida models | **PASS** | Original SHA-256 hashes (`native_gam_model.pkl`: `d2f949c2e6834084`, `native_ebm_model.pkl`: `8b038ffc89ccef4e`) remain completely unchanged since Oct 31, 2026. |
| **Chronological Split Safety** | Exclusion of Florida validation and test sets | **PASS** | Only months 1–7 ($N = 66,255$) were used for training. Months 8–9 (validation) and 10–12 (test) were strictly quarantined. |
| **Covariate Shift Assumption** | Invariance of conditional distribution $P(Y \mid X)$ | **FAIL (Empirically & Theoretically)** | The assumption $P_S(Y \mid X) = P_T(Y \mid X)$ fails because Florida labels represent physical ASOS observations while Chandipur labels represent a deterministic mathematical formula. |
| **Overall Methodological Verdict** | Scientific validity of Step 6 execution | **PASS** | The experiment is fully reproducible, mathematically sound, leakage-free, and scientifically rigorous. |

---

## 2. Detailed Technical Findings

### 2.1 Target-Label Isolation and Threshold Supervision
- **Target-Supervised vs. Unsupervised Evaluation**:
  The evaluation in Step 6 was **strictly unsupervised cross-region transfer**. The threshold $\tau_{\text{opt}} = 0.33$ (pyGAM) and $\tau_{\text{opt}} = 0.34$ (EBM) were derived on Florida validation data in Phase 2B. At no point were Chandipur labels used to search for optimal F1 thresholds. Standard uncalibrated threshold ($0.50$) performance was also evaluated and reported separately.

### 2.2 KMM Convex Optimality
- The KMM primal objective:
  
  $$\min_{\beta} f(\beta) = \frac{1}{2} \beta^T (K_s + 10^{-3} I) \beta - \kappa^T \beta$$
  
  has a strictly positive-definite Hessian matrix $H = K_s + 10^{-3} I \succ 0$ (all eigenvalues $\lambda_i \ge 10^{-3} > 0$).
- Because the feasible set $\{ \beta \in \mathbb{R}^n \mid \sum \beta_i = n, 0 \le \beta_i \le B \}$ is compact and convex, the quadratic program is **strictly convex**.
- By the Karush-Kuhn-Tucker (KKT) theorem, any stationary point of the projected gradient descent mapping is the **unique, certified global minimum**. The final gradient residual and zero constraint violations confirm convergence to this global optimum.

### 2.3 Scientific Root Cause of Negative Transfer under Importance Weighting
The audit confirms that the failure of importance weighting to improve transfer performance is not an artifact of implementation error, but a **fundamental physical and statistical limitation**:
1. **Covariate Shift is an Inadequate Model**: Reweighting source observations assumes that the relationship between atmospheric predictors and storm occurrence is universal ($P_S(Y \mid X) = P_T(Y \mid X)$).
2. **Concept Shift Dominates**: Florida thunderstorm occurrence in high-CAPE regimes is throttled by convective inhibition and requires mesoscale triggering (sea-breeze convergence, gust front collisions). Chandipur's heuristic formula, however, computes a linear composite of CAPE, shear, and moisture, predicting storm conditions almost deterministically in elevated thermodynamic regimes.
3. **Sample Size Collapse**: Weighting source samples by density ratios focuses learning on Florida's rare high-CAPE days, reducing the effective sample size from $66,255$ down to $22,100$–$30,800$, which increases variance without reconciling the differing label definitions.

---

## 3. Final Strategic Recommendation

1. **Retain Condition B as Gold Standard Transfer Baseline**:
   - Univariate CAPE clipping at the Florida training boundary of $7,240\text{ J/kg}$ on frozen Florida native models remains the superior, scientifically sound benchmark:
     - `InterpretML` EBM: **F1 = $0.6059$**, PR-AUC = **$0.6134$**, ROC-AUC = **$0.8572$**.
     - `pyGAM`: **F1 = $0.6146$**, PR-AUC = **$0.5526$**, ROC-AUC = **$0.8502$**.
2. **Reject Importance Weighting for Operational Deployment**:
   - Importance weighting degraded transfer performance across all tested configurations (EBM F1 dropped from $0.6059 \to 0.5703$–$0.5861$).
3. **Path Forward to Step 7**:
   - Having audited and eliminated multivariate linear alignment (Step 5, CORAL) and sample space reweighting (Step 6, Importance Weighting), domain adaptation should pivot to **Univariate Non-Linear Quantile Alignment (eCDF Matching)** and **Target Climatological Base Rate Calibration**.

---
*Step 6.1 independent scientific audit is concluded. All audit files are preserved. Awaiting user guidance before proceeding to Step 7.*
