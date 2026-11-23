# GOES-16 GLM Feature Definitions & Causal Boundary Conventions
**Experiment Directory:** `experiments/glm_integration_2024/`  

---

## 1. Mathematical Formulation & Leakage Controls

To guarantee absolute absence of temporal leakage, all predictor features for a forecast initialization time $t$ are constructed **exclusively from observations timestamped strictly at or before $t$**.

Let $t$ be the forecast initialization time (e.g. `2024-05-15 06:00:00 UTC`).  
The 6-hour forecast verification target window is the strictly half-open future interval:
$$\mathcal{T}_{\text{target}} = (t, t + 6\text{h}]$$

All predictor features must be computed from the strictly half-open historical lookback intervals:
$$\mathcal{T}_{\text{prior}}(\Delta t) = (t - \Delta t, t]$$
where $\Delta t \in \{1\text{h}, 3\text{h}, 6\text{h}\}$.

### Crucial Leakage Prevention Rule:
Any satellite flash observation with timestamp $t_{\text{flash}} > t$ (even by a fraction of a millisecond) is **strictly excluded** from feature generation at time $t$.  
For example, flashes in granule `s20241360000000` occurring at `00:00:06 UTC` fall into $(00\text{Z}, 00\text{Z} + 1\text{min}]$, which belongs to the *target window* of $t = 00\text{Z}$. Therefore, they are **never** used at $00\text{Z}$; they become eligible predictors only for later cycles such as $t = 06\text{Z}$.

---

## 2. Feature Definitions & Units

| Feature Name | Lookback Interval | Spatial Radius | Unit | Description | Missing / Sentinel Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `glm_flash_count_25km_1h` | $(t - 1\text{h}, t]$ | $R \le 25\text{ km}$ | count | Number of GLM flashes in station aerodrome vicinity | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_count_50km_1h` | $(t - 1\text{h}, t]$ | $R \le 50\text{ km}$ | count | Number of GLM flashes in regional sub-synoptic buffer | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_count_25km_3h` | $(t - 3\text{h}, t]$ | $R \le 25\text{ km}$ | count | Number of GLM flashes in station aerodrome vicinity | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_count_50km_3h` | $(t - 3\text{h}, t]$ | $R \le 50\text{ km}$ | count | Number of GLM flashes in regional sub-synoptic buffer | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_count_25km_6h` | $(t - 6\text{h}, t]$ | $R \le 25\text{ km}$ | count | Number of GLM flashes in station aerodrome vicinity | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_count_50km_6h` | $(t - 6\text{h}, t]$ | $R \le 50\text{ km}$ | count | Number of GLM flashes in regional sub-synoptic buffer | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_flash_density_50km_6h`| $(t - 6\text{h}, t]$ | $R \le 50\text{ km}$ | flashes/$\text{km}^2$| `glm_flash_count_50km_6h` / $(\pi \times 50^2)$ | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_time_since_last_flash_min` | $(t - 6\text{h}, t]$ | $R \le 50\text{ km}$ | minutes | Minutes from most recent flash in buffer to $t$ | `360.0` min (max lookback) or NaN |
| `glm_statewide_flash_count_6h` | $(t - 6\text{h}, t]$ | Florida State | count | Statewide total accepted flashes | NaN (indicator: `glm_coverage_available == 0`) |
| `has_prior_flash_50km_6h` | $(t - 6\text{h}, t]$ | $R \le 50\text{ km}$ | binary (0/1) | Whether $\ge 1$ flash was observed in 50 km buffer | NaN (indicator: `glm_coverage_available == 0`) |
| `glm_coverage_available` | $(t - 6\text{h}, t]$ | Domain | binary (0/1) | Whether valid GLM observations were retrieved on disk | `0` (missing coverage flag) |

---

## 3. Distance Computation

Great-circle distances between ASOS station coordinates $(\phi_s, \lambda_s)$ and GLM flash locations $(\phi_f, \lambda_f)$ are calculated using the spherical Haversine formula with mean Earth radius $R_{\text{earth}} = 6371.0\text{ km}$:
$$d = 2 R_{\text{earth}} \arcsin \left(\sqrt{\sin^2\left(\frac{\phi_f - \phi_s}{2}\right) + \cos \phi_s \cos \phi_f \sin^2\left(\frac{\lambda_f - \lambda_s}{2}\right)}\right)$$
