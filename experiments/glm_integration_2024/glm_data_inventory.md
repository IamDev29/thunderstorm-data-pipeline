# GOES-16 GLM Data Inventory & Coverage Audit (2024)
**Experiment Directory:** `experiments/glm_integration_2024/`  
**Generated:** 2026-10-28 05:08:57 UTC  

---

## 1. Sensor Specifications & Instrument Provenance

- **Platform:** GOES-16 (`G16`), Flight Model 1 (`FM1`), Geostationary East ($75.2^\circ\text{W}$).
- **Instrument:** Geostationary Lightning Mapper (GLM).
- **Product:** Level-2 Lightning Cluster Filter Algorithm (`OR_GLM-L2-LCFA_G16`), Version 2.
- **Granularity:** Physical flash observations with measured constituent group and event hierarchies.
- **Spatial Resolution:** ~8 km nominal nadir resolution; ~10–14 km over the Florida domain ($[24.5^\circ\text{N}, 31.5^\circ\text{N}], [-88.0^\circ\text{W}, -79.5^\circ\text{W}]$).
- **Coordinate Reference System:** Geodetic GRS80 / WGS84 latitude and longitude coordinates reconstructed from fixed-grid projection angles.
- **Time Representation:** Microsecond-precision UTC timestamps reconstructed from J2000 epoch product seconds and event-level microsecond offsets. Zero synoptic-hour collapse.

---

## 2. On-Disk Raw Granule Inventory

All raw NetCDF files on disk are located under:  
`Florida_Thunderstorm_2024_Training/data/raw/glm/`

| Granule Filename | Coverage Start (UTC) | Coverage End (UTC) | Raw Flashes | Accepted FL Flashes | L0 Error % | Navigated % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `OR_GLM-L2-LCFA_G16_s20241360000000_...nc` | 2024-05-15 00:00:00 | 2024-05-15 00:00:20 | 280 | 6 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241360000200_...nc` | 2024-05-15 00:00:20 | 2024-05-15 00:00:40 | 286 | 6 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241360000400_...nc` | 2024-05-15 00:00:40 | 2024-05-15 00:01:00 | 291 | 8 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600000_...nc` | 2024-05-15 06:00:00 | 2024-05-15 06:00:20 | 166 | 3 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600200_...nc` | 2024-05-15 06:00:20 | 2024-05-15 06:00:40 | 183 | 6 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241360600400_...nc` | 2024-05-15 06:00:40 | 2024-05-15 06:01:00 | 163 | 6 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200000_...nc` | 2024-05-15 12:00:00 | 2024-05-15 12:00:20 | 185 | 31 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200200_...nc` | 2024-05-15 12:00:20 | 2024-05-15 12:00:40 | 181 | 31 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361200400_...nc` | 2024-05-15 12:00:40 | 2024-05-15 12:01:00 | 186 | 32 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800000_...nc` | 2024-05-15 18:00:00 | 2024-05-15 18:00:20 | 245 | 16 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800200_...nc` | 2024-05-15 18:00:20 | 2024-05-15 18:00:40 | 249 | 19 | 0.0% | 100.0% |
| `OR_GLM-L2-LCFA_G16_s20241361800400_...nc` | 2024-05-15 18:00:40 | 2024-05-15 18:01:00 | 243 | 17 | 0.0% | 100.0% |
| **Total (12 granules)** | — | — | **2,658** | **181** | **0.0%** | **100.0%** |

---

## 3. Rejection Accounting & Hierarchy Measurements

- **Total Raw Flash Records Processed:** 2,658
- **Rejected Outside Bounding Box ($[24.5^\circ, 31.5^\circ]\text{N}, [-88.0^\circ, -79.5^\circ]\text{W}$):** 1,970 (74.12%)
- **Rejected Outside Florida State Polygon:** 507 (19.07%)
- **Accepted In-Florida Flashes:** **181** (6.81%)
  - *Active Terrestrial ASOS / ERA5 Grid Cells:* **173** flashes
  - *Coastal Marine Grid Cells:* **8** flashes
- **Measured Hierarchy Counts:**
  - Constituent Groups: **4,534** measured (mean: 25.05 groups/flash vs. legacy 12.0 synthetic multiplier)
  - Constituent Events: **9,243** measured (mean: 51.07 events/flash vs. legacy 28.0 synthetic multiplier)
- **Quality Flags:**
  - Good Quality (`flash_quality_flag == 0`): 172 flashes (95.03%)
  - Degraded Event Count (`flash_quality_flag == 3`): 9 flashes (4.97%, due to deep convective event burst)
  - Duplicate Flashes Detected: **0** across all granules.

---

## 4. Coverage & Missing Data Analysis

- **Total Synoptic Cycles in 2024 (366 days $\times$ 4 cycles/day):** 1,464 cycles
- **Synoptic Cycles with Valid GLM Granules on Disk:** **4 cycles** (0.27%)
- **Synoptic Cycles Missing GLM Granules on Disk:** **1,460 cycles** (99.73%)
- **Total Monitored ASOS Station-Windows (Criterion C):** 112,509 windows
- **Common Evaluation Cohort Windows (May 15, 2024):** **320 windows** (0.28%)
- **Excluded Monitored Windows due to Missing GLM:** **112,189 windows** (99.72%)
- **October–December Holdout Coverage:** **0 granules (100% missing)**.

### Scientific Implications:
1. Missing granules **must not** be treated as confirmed zero lightning.
2. The GLM experiment on available data is strictly a **pilot validation comparison** on the May 15 common cohort ($N=320$).
3. The October–December 2024 holdout cannot be evaluated for GLM without full-year streaming acquisition.
