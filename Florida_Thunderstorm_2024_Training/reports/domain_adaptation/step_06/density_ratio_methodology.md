# Importance Weighting and Density-Ratio Estimation Methodology

## 1. Executive Summary & Objective

This document defines the mathematical formulation, numerical optimization algorithms, regularisation schemes, and stabilization policies for **Importance Weighting and Density-Ratio Estimation** within the **Thunderstorm Domain Adaptation Framework (TDAF)**.

The objective of Step 6 is to evaluate whether reweighting source training samples (Florida ASOS/ERA5 data, $D_S$) can bridge the covariate shift between Florida and Chandipur ($D_T$) without transforming the target features or violating physical laws. Unlike Correlation Alignment (CORAL), which linearly rotates feature axes and generates physically impossible states (such as negative CAPE and CIN), importance weighting operates purely in sample space, leaving the physical units and coordinate axes of all six meteorological predictors completely unaltered.

---

## 2. Mathematical Formulation of Covariate Shift

### 2.1 The Covariate Shift Hypothesis

Let $x \in \mathcal{X} \subset \mathbb{R}^6$ denote the vector of compatible meteorological predictors:
1. `cape_J_kg` (Convective Available Potential Energy)
2. `cin_J_kg` (Convective Inhibition)
3. `bulk_shear_850_300_ms` (Deep-layer bulk wind shear)
4. `cloud_cover_fraction` (Total cloud cover fraction)
5. `wind_speed_10m_ms` (10-metre horizontal wind speed)
6. `wind_speed_850hPa_ms` (850 hPa horizontal wind speed)

Let $y \in \{0, 1\}$ denote the binary thunderstorm occurrence indicator. Under the classic **Covariate Shift Assumption** (Shimodaira, 2000; Sugiyama & Kawanabe, 2012):
1. The marginal feature distributions differ across domains: $p_S(x) \ne p_T(x)$.
2. The conditional label distribution is assumed to be invariant: $P_S(y \mid x) = P_T(y \mid x)$.

Under these assumptions, the expected risk on the target domain $\mathcal{R}_T(f) = \mathbb{E}_{(x, y) \sim P_T}[\ell(f(x), y)]$ can be rewritten as a weighted expectation over the source domain:

$$\mathcal{R}_T(f) = \int \ell(f(x), y) p_T(x, y) \, dx dy = \int \ell(f(x), y) P(y \mid x) p_T(x) \, dx dy$$

$$\mathcal{R}_T(f) = \int \ell(f(x), y) P(y \mid x) \frac{p_T(x)}{p_S(x)} p_S(x) \, dx dy = \mathbb{E}_{(x, y) \sim P_S}[w(x) \ell(f(x), y)]$$

where the optimal importance weight function is the **density ratio**:

$$w(x) = \frac{p_T(x)}{p_S(x)}$$

---

## 3. Density-Ratio Estimation Methods

### 3.1 Method A: Domain Classifier Density Ratio

Following Bickel et al. (2007) and Qin (1998), the density ratio can be recovered from a probabilistic binary domain classifier trained to distinguish between source and target observations.

Let $z \in \{0, 1\}$ be the domain indicator variable, where $z = 0$ denotes a Florida source training observation ($N_S = 66,255$) and $z = 1$ denotes a Chandipur target observation ($N_T = 23,376$). By Bayes' Theorem:

$$p(x \mid z=1) = \frac{P(z=1 \mid x) p(x)}{P(z=1)}, \quad p(x \mid z=0) = \frac{P(z=0 \mid x) p(x)}{P(z=0)}$$

Taking the ratio:

$$w(x) = \frac{p_T(x)}{p_S(x)} = \frac{p(x \mid z=1)}{p(x \mid z=0)} = \frac{P(z=1 \mid x)}{P(z=0 \mid x)} \cdot \frac{P(z=0)}{P(z=1)} = \frac{P(z=1 \mid x)}{1 - P(z=1 \mid x)} \cdot \frac{N_S}{N_T}$$

#### Leakage-Safe Estimation via 5-Fold Stratified Cross-Validation
To prevent out-of-sample overfitting and artificial separation:
1. The combined domain dataset ($N = 89,631$) is partitioned into 5 stratified folds.
2. In each fold $k \in \{1, \dots, 5\}$, the domain classifier is trained on the remaining 4 folds and predicts out-of-fold probabilities $\hat{p}_i = \hat{P}(z=1 \mid x_i)$ for the held-out fold.
3. Every source training sample receives an out-of-fold predicted probability $\hat{p}_i$ from an estimator that never saw $x_i$ during training.
4. Numerical safeguards are applied to prevent infinite weights: $\hat{p}_i \leftarrow \text{clip}(\hat{p}_i, \epsilon, 1 - \epsilon)$ with $\epsilon = 10^{-4}$.

#### Classifiers Evaluated
- **Logistic Regression**: Linear log-odds model with `StandardScaler` feature preprocessing.
- **Random Forest**: Regularized non-linear ensemble (`n_estimators=100`, `max_depth=8`, `min_samples_leaf=20`).
- **LightGBM**: Gradient-boosted decision trees (`n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `num_leaves=31`).

---

### 3.2 Method B: Kernel Mean Matching (KMM)

Kernel Mean Matching (Huang et al., 2007) avoids density estimation altogether by directly optimizing source sample weights $\beta \in \mathbb{R}^{N_S}$ that minimize the Maximum Mean Discrepancy (MMD) between the weighted source distribution and the unweighted target distribution in a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$:

$$\min_{\beta} \left\| \frac{1}{N_S} \sum_{i=1}^{N_S} \beta_i \Phi(x_{S,i}) - \frac{1}{N_T} \sum_{j=1}^{N_T} \Phi(x_{T,j}) \right\|_{\mathcal{H}}^2$$

Expanding the squared norm yields a quadratic programming (QP) problem:

$$\min_{\beta} \frac{1}{2} \beta^T K \beta - \kappa^T \beta \quad \text{subject to} \quad 0 \le \beta_i \le B, \quad \left| \sum_{i=1}^{N_S} \beta_i - N_S \right| \le N_S \epsilon$$

where:
- $K_{ij} = k(x_{S,i}, x_{S,j})$ is the source-source kernel Gram matrix.
- $\kappa_i = \frac{N_S}{N_T} \sum_{j=1}^{N_T} k(x_{S,i}, x_{T,j})$ is the source-target kernel interaction vector.
- $k(x, x') = \exp(-\gamma \|x - x'\|^2)$ is the Radial Basis Function (RBF) kernel with bandwidth $\gamma = 1 / (2 \sigma^2)$ set via the median heuristic over pairwise distances.

#### Fast Projected Gradient Descent (PGD) with Capped-Simplex Projection
For computational scalability on large datasets, we draw a reproducible, representative stratified subsample of $n_S = 2,000$ Florida samples and $n_T = 2,000$ Chandipur samples (`random_state=42`). 

To avoid the $\mathcal{O}(N^3)$ computational bottleneck of general active-set solvers (like SciPy's `SLSQP`), we implement a vectorized **Projected Gradient Descent (PGD)** solver:

$$\beta^{(t+1)} = \Pi_{\mathcal{C}}\left( \beta^{(t)} - \eta (H \beta^{(t)} - \kappa) \right)$$

where $H = K + 10^{-3} I$, step size $\eta = 1 / \|H\|_2$, and $\Pi_{\mathcal{C}}$ is the exact projection onto the capped simplex $\{x \in \mathbb{R}^n \mid \sum x_i = n, 0 \le x_i \le B\}$. The projection is solved via bisection on the dual variable $\theta$ satisfying $\sum \text{clip}(y_i - \theta, 0, B) = n$, which converges in $\le 40$ iterations to machine precision.

#### Inductive Out-of-Sample Smoothing
The optimized subsample weights $\beta^*$ are extended inductively to all 66,255 Florida training observations via Nadaraya-Watson kernel regression:

$$w_{\text{KMM}}(x) = \frac{\sum_{j=1}^{n_S} \beta_j^* k(x, x_{S,j})}{\sum_{j=1}^{n_S} k(x, x_{S,j})}$$

followed by normalization to mean 1.0.

---

## 4. Weight Stabilization and Clipping Policies

Unbounded density ratios often suffer from severe variance and weight concentration, where a handful of uncharacteristic source samples receive astronomical weights, effectively collapsing the training sample size.

We evaluate five predefined, label-blind clipping policies:
1. **Policy A (No Clipping)**: $w \leftarrow w / \bar{w}$.
2. **Policy B (95th Percentile Clipping)**: $w \leftarrow \min(w, q_{0.95})$, then normalized to mean 1.0.
3. **Policy C (99th Percentile Clipping)**: $w \leftarrow \min(w, q_{0.99})$, then normalized to mean 1.0.
4. **Policy D (Fixed Maximum Weight = 5)**: $w \leftarrow \min(w, 5.0)$, then normalized to mean 1.0.
5. **Policy E (Fixed Maximum Weight = 10)**: $w \leftarrow \min(w, 10.0)$, then normalized to mean 1.0.

### Effective Sample Size (ESS)
To quantify sample dispersion and weight degeneracy, we compute Kish's Effective Sample Size (Kish, 1965):

$$\text{ESS} = \frac{\left( \sum_{i=1}^{N_S} w_i \right)^2}{\sum_{i=1}^{N_S} w_i^2}$$

and the sample retention ratio $\text{ESS} / N_S \in (0, 1]$. An ESS ratio below $0.20$ signals severe variance inflation, while an ESS ratio above $0.40$ indicates a well-stabilized reweighting scheme.

---

## 5. Season-Aware Importance Weighting & Fallback Policy

Chandipur undergoes drastic seasonal transitions between dry winter (DJF), severe pre-monsoon Nor'westers (MAM), monsoon troughs (JJAS), and post-monsoon cyclonic regimes (ON).

We estimate seasonal density ratios $w_m(x)$ by partitioning both domains by meteorological season:
- **DJF (Winter)**: Florida train Jan–Feb ($N_S = 18,688$), Chandipur ($N_T = 5,776$).
- **MAM (Pre-monsoon)**: Florida train Mar–May ($N_S = 28,407$), Chandipur ($N_T = 5,888$).
- **JJAS (Monsoon)**: Florida train Jun–Jul ($N_S = 19,160$), Chandipur ($N_T = 7,808$).
- **ON (Post-monsoon)**: Florida training data covers months 1 to 7 and has **0 observations** in October–November ($N_S = 0$), whereas Chandipur has $N_T = 3,904$.

### Documented Fallback Policy for Season ON
Because Florida source training observations are absent in post-monsoon months, seasonal domain classifiers cannot be trained for ON. We strictly enforce a documented **Global Fallback Policy**:
- For Chandipur observations in ON, sample weighting defaults to the global annual density-ratio model.
- Florida training observations are reweighted according to their respective active training season (DJF, MAM, or JJAS).

---

## 6. Weighted GAM and EBM Training Protocols

Weighted models are trained on the Florida source training set ($N_S = 66,255$) using the original binary labels ($y_{\text{ASOS}}$) and the estimated sample weights $w_i$.

### 6.1 pyGAM (`LogisticGAM`)
Trained with native sample weights passed to `fit(X, y, weights=w)`.
- Formula: $\text{logit}(P) = \beta_0 + \sum_{j=1}^6 s(j, \text{n\_splines}=15, \text{lam}=1.0)$
- Optimization: Penalized Iteratively Reweighted Least Squares (P-IRLS) with `max_iter=100`, `tol=1e-4`.

### 6.2 InterpretML (`ExplainableBoostingClassifier`)
Trained with native sample weights passed to `fit(X, y, sample_weight=w)`.
- Architecture: Pure main-effects EBM (`interactions=0`, `max_leaves=2`, `max_bins=128`, `outer_bags=16`, `learning_rate=0.01`, `random_state=42`).

### 6.3 Dual Inference Conditions
Each trained model is evaluated under:
- **Condition A (Raw Features)**: Input directly to the model.
- **Condition B (CAPE Clipped at 7,240 J/kg)**: $\text{CAPE} \leftarrow \min(\text{CAPE}, 7240\text{ J/kg})$ to prevent spline extrapolation breakdown.
