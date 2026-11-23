# TDAF Step 4: Mathematical Methodology of CAPE Transformations

**Framework**: Thunderstorm Domain Adaptation Framework (TDAF)  
**Phase**: Step 4 — Controlled CAPE Adaptation and Extrapolation Experiments  
**Date**: November 2026  
**Status**: **METHODOLOGY SPECIFICATION & VERIFICATION**  

---

## 1. Mathematical Formulation of Candidate Transformations

This document details the exact mathematical equations, boundary constraints, and algorithmic implementations for the five controlled CAPE transformations evaluated in Step 4.

All transformations are strictly **unsupervised and inference-only**. No target labels are utilized, and the pre-trained model weights of `pyGAM` (`LogisticGAM`) and `InterpretML` (`ExplainableBoostingClassifier`) remain completely unaltered.

---

### 1.1 Experiment A: Unmodified Zero-Shot Baseline (Identity Mapping)
- **Equation**:
  $$\mathcal{T}_A(x) = x$$
- **Description**: Evaluates the saved models directly on the raw, unadjusted Chandipur ECMWF ERA5 Surface CAPE values:
  $$x \in [0.0, 9,906.0]\text{ J/kg}$$
- **Role**: Serves as the frozen negative control reference against which all adaptation methods are benchmarked.

---

### 1.2 Experiment B: Hard Boundary Clipping (Domain Max Clamping)
- **Equation**:
  $$\mathcal{T}_B(x) = \min(x, \text{CAPE}_{\text{MAX}})$$
  where $\text{CAPE}_{\text{MAX}} = 7,240.0\text{ J/kg}$ is the absolute maximum Surface CAPE observed in the Florida training set (January–July 2024, $N=60,637$).
- **Mathematical Properties**:
  - Continuous, piecewise-linear, non-decreasing.
  - Derivative:
    $$\frac{d\mathcal{T}_B}{dx} = \begin{cases} 1 & \text{if } x < 7,240.0 \\ 0 & \text{if } x > 7,240.0 \end{cases}$$
  - Range: $[0.0, 7,240.0]\text{ J/kg}$.
- **Observations Altered**: Exactly $81$ observations ($0.3465\%$) where $x > 7,240.0\text{ J/kg}$. The remaining $23,295$ observations ($99.6535\%$) are strictly bit-for-bit identical to the raw data.
- **Physical Limitation**: Discards all gradient information for extreme buoyancy above $7,240\text{ J/kg}$, treating a $9,906\text{ J/kg}$ event identically to a $7,240\text{ J/kg}$ event.

---

### 1.3 Experiment C: Global Empirical Quantile Mapping (CDF Matching)
- **Equation**:
  $$\mathcal{T}_C(x) = F_S^{-1}\left(F_T(x)\right)$$
  where:
  - $F_T(x) = P(X_T \le x)$ is the empirical cumulative distribution function (ECDF) of the Chandipur target dataset ($N=23,376$).
  - $F_S^{-1}(u) = \inf\{s \in \mathbb{R} : F_S(s) \ge u\}$ is the empirical quantile function of the Florida training dataset ($N=60,637$).
- **Algorithmic Implementation**:
  1. For each target value $x_i \in X_T$, compute its empirical rank percentile with standard continuity correction:
     $$u_i = \frac{\text{rank}(x_i) - 0.5}{N_T}$$
  2. Map $u_i \in (0, 1)$ to the Florida training empirical quantile via linear percentile interpolation:
     $$\mathcal{T}_C(x_i) = \text{Percentile}\left(X_{S, \text{train}}, 100 \times u_i\right)$$
- **Mathematical Properties**:
  - Monotonic by construction.
  - Matches the marginal cumulative distribution of the target to the source training set:
    $$F_{\mathcal{T}_C(X_T)}(s) \approx F_S(s)$$
- **Acute Meteorological Vulnerability (Zero-Inflation Distortion)**:
  - In Florida, due to winter nights and stable marine flow, **$34.90\%$ of training observations have $\text{CAPE} = 0.0\text{ J/kg}$**.
  - In Chandipur, only **$20.86\%$ of observations have $\text{CAPE} = 0.0\text{ J/kg}$**.
  - As a result, Chandipur observations between the 20.86th and 34.90th percentiles (which correspond to moderate tropical instability of $14.5\text{ to }320.0\text{ J/kg}$) are artificially compressed to **$0.0\text{ J/kg}$**. This destroys genuine physical convective energy.

---

### 1.4 Experiment D: Smooth Monotonic Hyperbolic Tangent Saturation
- **Equation**:
  $$\mathcal{T}_D(x) = \text{CAPE}_{\text{MAX}} \times \tanh\left(\frac{x}{\text{CAPE}_{\text{SCALE}}}\right)$$
  where:
  - $\text{CAPE}_{\text{MAX}} = 7,240.0\text{ J/kg}$ (Florida training ceiling).
  - $\text{CAPE}_{\text{SCALE}} = 3,744.28\text{ J/kg}$ (Florida training 99th percentile, selected a priori without examining Chandipur labels).
- **Mathematical Properties**:
  - Smooth ($C^\infty$), strictly monotonic ($\frac{d\mathcal{T}_D}{dx} > 0$ for all $x \ge 0$).
  - First derivative:
    $$\frac{d\mathcal{T}_D}{dx} = \frac{\text{CAPE}_{\text{MAX}}}{\text{CAPE}_{\text{SCALE}}} \operatorname{sech}^2\left(\frac{x}{\text{CAPE}_{\text{SCALE}}}\right)$$
  - Linear near the origin: For $x \ll \text{CAPE}_{\text{SCALE}}$, $\tanh(u) \approx u$, so:
    $$\mathcal{T}_D(x) \approx x \times \frac{7,240.0}{3,744.28} \approx 1.934 x$$
  - Asymptotically bounded:
    $$\lim_{x \to \infty} \mathcal{T}_D(x) = 7,240.0\text{ J/kg}$$
  - At the Chandipur maximum ($x = 9,906.0\text{ J/kg}$):
    $$\frac{9,906.0}{3,744.28} = 2.6456 \implies \tanh(2.6456) = 0.98998 \implies \mathcal{T}_D(9,906) = 7,167.45\text{ J/kg}$$
- **Advantages**:
  - Strictly preserves rank order across all observations (Spearman $\rho = 1.000000$).
  - Prevents non-differentiable sharp corners at $7,240\text{ J/kg}$.
  - Completely bounds target CAPE within the stable support of pyGAM's training knots.

---

### 1.5 Experiment E: Seasonal Empirical Quantile Mapping
- **Equation**:
  $$\mathcal{T}_E(x, s) = F_{S, s}^{-1}\left(F_{T, s}(x)\right)$$
  where $s \in \{\text{DJF}, \text{MAM}, \text{JJAS}, \text{ON}\}$ denotes the meteorological season.
- **Seasonal Group Sample Sizes & Verification**:
  - **Chandipur Target**:
    - DJF (Winter): $N = 5,776$
    - MAM (Pre-monsoon): $N = 5,888$
    - JJAS (Monsoon): $N = 7,808$
    - ON (Post-monsoon): $N = 3,904$
  - **Florida Training Reference (Months 1–7)**:
    - DJF (Winter, Jan–Feb): $N = 18,688$ (Adequate)
    - MAM (Pre-monsoon, Mar–May): $N = 28,407$ (Adequate)
    - JJAS (Monsoon, Jun–Jul): $N = 19,160$ (Adequate)
    - **ON (Post-monsoon, Oct–Nov)**: **$N = 0$** (Months 10–11 reside in Florida test set!).
- **Documented Fallback Strategy for Season ON**:
  - Because Florida training data contains zero observations during October and November, forcing an ON-specific mapping from the training set is mathematically impossible without data leakage.
  - **Fallback Rule**: Target observations occurring in season ON are mapped using the **global Florida training empirical quantile function** ($N=60,637$):
    $$\mathcal{T}_E(x, \text{ON}) = F_{S, \text{global}}^{-1}\left(F_{T, \text{ON}}(x)\right)$$
  - This avoids test-set leakage while ensuring deterministic, complete mapping for all $23,376$ rows.

---

## 2. Summary Comparison of Transformation Properties

| Metric / Property | Exp A (Raw) | Exp B (Clipping) | Exp C (Global Quantile) | Exp D (Tanh Saturation) | Exp E (Seasonal Quantile) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Output Range ($\text{J/kg}$)** | $[0.0, 9,906.0]$ | $[0.0, 7,240.0]$ | $[0.0, 7,105.1]$ | $[0.0, 7,167.5]$ | $[0.0, 7,166.6]$ |
| **Observations Altered** | $0$ ($0.0\%$) | **$81$ ($0.35\%$)** | $23,376$ ($100.0\%$) | $23,376$ ($100.0\%$) | $23,376$ ($100.0\%$) |
| **Rank Preservation ($\rho$)**| $1.000000$ | **$1.000000$** | $0.982998$ | **$1.000000$** | $0.729583$ |
| **Differentiability** | $C^\infty$ | Piecewise $C^0$ | Non-parametric | **Smooth $C^\infty$** | Non-parametric |
| **Preserves CAPE-CIN $\rho$**| $0.5853$ | **$0.5853$** | $0.5566$ | **$0.5853$** | $0.4799$ |
| **Preserves CAPE-Shear $\rho$**| $-0.3518$ | **$-0.3518$** | $-0.3386$ | **$-0.3518$** | $-0.4728$ |
| **Zero-Inflation Distortion** | None | None | **Severe ($+14\%$ zeroes)**| None | Moderate |
| **Removes Extrapolation?** | No | **Yes** | **Yes** | **Yes** | **Yes** |
