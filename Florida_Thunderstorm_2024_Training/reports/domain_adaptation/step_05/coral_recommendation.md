# Definitive Recommendation on Correlation Alignment (CORAL) for Cross-Region Thunderstorm Transfer

## 1. Definitive Scientific Verdict

**We definitively advise AGAINST adopting Standard Unconstrained CORAL (or any full linear affine feature alignment) as an operational domain adaptation method for transferring Florida thunderstorm models to Chandipur.**

Although CORAL completely minimizes the algebraic distance between covariance matrices ($\|C_{\hat{X}_T} - C_{X_S}\|_F \to 0$), it causes severe degradation in real transfer performance and violates basic physical and thermodynamic laws:
1. **Severe Performance Collapse**: Standard CORAL reduces `pyGAM` F1 score from **$0.6146 \to 0.4090$** (a $33.5\%$ relative drop) and `InterpretML` EBM F1 score from **$0.6059 \to 0.4525$** (a $25.3\%$ relative drop).
2. **Massive Physical Law Violations**: Over **$37.6\%$ of observations produce negative CAPE** (down to $-219.4\text{ J/kg}$), **$46.0\%$ produce negative CIN**, surface wind speeds become negative (down to $-2.71\text{ m/s}$), and total cloud cover fraction exceeds $1.0$ (up to $1.43$).
3. **Loss of Additive Interpretability**: Generalized Additive Models (`pyGAM`) and Explainable Boosting Machines (`InterpretML` EBM) represent predictions as a sum of univariate shape functions:
   
   $$g(E[Y]) = \beta_0 + \sum_{i=1}^d f_i(x_i) + \sum_{i \ne j} f_{ij}(x_i, x_j)$$
   
   When features are rotated by a dense linear matrix $A = C_T^{-1/2} C_S^{1/2}$, the transformed coordinate $\hat{x}_i = \sum_{j} A_{ji} x_j$ is no longer a pure physical measurement (e.g., CAPE), but an uninterpretable linear amalgam of thermodynamic energy, kinematic shear, and cloud cover. Passing this mixture into the Florida shape functions $f_i$ destroys their physical semantics.

---

## 2. Comparison Against Existing Benchmarks

| Adaptation Technique | pyGAM F1 ($\tau=0.33$) | EBM F1 ($\tau=0.34$) | Physical Violations | Additive Semantics Preserved? | Implementation Complexity | Recommendation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond_A: Zero-Shot Baseline** | 0.6146 | 0.6059 | None (Raw ERA5) | Yes | None | Retained as Frozen Baseline |
| **Cond_B: CAPE Boundary Clipping (7,240 J/kg)** | **0.6146** | **0.6059** | **None** | **Yes** | **Trivial ($\mathcal{O}(1)$)** | **CURRENT BEST METHOD (Adopted)** |
| **Cond_C: Standard CORAL ($\alpha=1.0$)** | 0.4090 | 0.4525 | Extreme (37.6% neg CAPE) | No (Dense linear mixing) | Matrix square roots | **REJECTED** |
| **Cond_D: CORAL + Physical Clamping** | 0.4638 | 0.4525 | None (Post-hoc clamped) | No | Clamping heuristics | **REJECTED** |
| **Cond_F: Partial CORAL ($\alpha=0.25$)** | 0.5744 | 0.5685 | High (28.3% neg CAPE) | Weakened | Parameter interpolation | **REJECTED** |
| **Cond_G: Seasonal CORAL** | 0.5252 | 0.5508 | Extreme (34.0% neg CAPE) | No (Fails in MAM) | Seasonal stratification + Fallback | **REJECTED** |

**Conclusion**: Condition B (Univariate CAPE clipping at $7,240\text{ J/kg}$) remains the superior, scientifically sound, and physically defensible transfer method. It eliminates GAM extrapolation breakdown on extreme convective days without corrupting any other physical variable.

---

## 3. Physical Hazards of Multivariate Linear Alignment in Meteorology

The failure of CORAL in this setting highlights four critical principles of atmospheric data science:
1. **Physical Quantities Have Hard Natural Boundaries**: Thermodynamic variables (CAPE, CIN) and kinematic variables (wind speed, wind shear) are strictly non-negative scalars ($\in [0, \infty)$). Linear affine operators assume unconstrained Gaussian support on $\mathbb{R}^d$ and cannot respect hard boundaries.
2. **Atmospheric Covariances Are Non-Stationary and Regime-Dependent**: In Florida, summer afternoon convection occurs in a humid, maritime subtropical environment with moderate shear ($10$–$15\text{ m/s}$) and sea-breeze convergence. In Chandipur, pre-monsoon thunderstorms (Nor'westers) require extreme CAPE ($> 5,000\text{ J/kg}$) overcoming a strong capping inversion under continental dry-air intrusion. Forcing Chandipur to adopt Florida's covariance structure suppresses the exact thermodynamic signatures that initiate Chandipur storms.
3. **Correlation Does Not Equal Causation**: A high correlation between shear and CAPE in one climatic zone is an emergent property of local synoptic setups, not a universal physical law. Imposing source cross-correlations onto target features forces artificial anti-correlations where none exist in nature.
4. **Post-Hoc Physical Clamping is Insufficient**: As shown in Condition D, clamping negative values to zero after CORAL only recovers pyGAM F1 from $0.4090 \to 0.4638$. It cannot reconstruct the underlying physical relationships that were scrambled during the whitening step.

---

## 4. Path Forward: Strategic Recommendations for Future Steps

To advance domain adaptation without sacrificing physical validity or model interpretability, future research steps in the TDAF framework should prioritize:

1. **Univariate Non-Linear Quantile Alignment (Quantile Matching / CDF Alignment)**:
   - Align each feature's marginal distribution independently via empirical cumulative distribution functions (eCDF):
     
     $$\hat{x}_{T,j} = F_{S,j}^{-1}(F_{T,j}(x_{T,j}))$$
     
   - **Advantage**: Respects physical boundaries naturally (e.g. $[0, \infty)$), avoids cross-variable feature mixing, and directly preserves the univariate interpretability of GAM and EBM shape functions.
2. **Sample Reweighting / Importance Weighting (Density Ratio Estimation)**:
   - Instead of transforming target features $X_T$, compute covariate shift weights $w(x) = p_S(x) / p_T(x)$ using Kernel Mean Matching (KMM) or Nearest-Neighbor Density Ratios.
   - **Advantage**: Features remain 100% unaltered and physically intact.
3. **Constrained Optimal Transport (OT)**:
   - If multivariate alignment is pursued, use Optimal Transport with physical boundary constraints (e.g., Wasserstein barycenters constrained to the positive orthant $\mathbb{R}_+^d$) rather than unconstrained quadratic forms.
4. **Target Temperature / Threshold Calibration**:
   - Investigate unsupervised threshold recalibration based on target climatological thunderstorm base rates rather than relying on Florida-derived optimal thresholds.
