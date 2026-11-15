# Test Architecture & Empirical Validation Benchmarks

## 1. Four-Layer Test Suite (`tests/`)

Instead of ad-hoc script assertions, WINTELLISYT organizes verification around four engineering contract layers:

| Layer | Directory | Focus |
| :--- | :--- | :--- |
| **1. Physics** | `tests/physics/` | Circular arithmetic & wrapping (`0°/360°`), FROM/TO wake projections, active vs. passive donor stratification, moist-air density & hub-height shear, half-meter PC binning, operating-state union & interval integration. |
| **2. Diagnostics** | `tests/diagnostics/` | Deterministic synthetic fault injections: NREL 2-stage consensus & wake-trough true-north recovery, CUSUM/EWMA calm-wind state holding & left-censoring, yaw $\cos^2$ bias & NTF audits, blade A/B/C pitch asymmetry & motor current spikes. |
| **3. Leakage** | `tests/leakage/` | Strict chronological & spatial isolation: target exclusion from peer consensus and donor pools, pre-cutoff fitting for PC baselines & peer MCP models, embargoed transfer boundaries, and frozen calibration radii. |
| **4. Contracts** | `tests/contracts/` | Greenbyte ZIP parsing, duplicate-timestamp full quarantine (`keep=False`), UTC timezone enforcement, status event deduplication, Open-Meteo cache manifest integrity, UI HTML export, and Starlette routes (`/`, `/health`, `/api/manifest`, `/api/tables/{name}`). |

---

## 2. Empirical Results on Cubico 2016–2017 (`artifacts/engineering_suite_2017/`)

All suite outputs cover **20 active turbines** (`315,360` Kelmarsh 2017 rows + `49,106` unique events; `735,840` Penmanshiel 2017 rows + `34,861` unique events; `0` duplicate timestamps left unquarantined).

### 2.1 Wind Direction Consensus & Wake-Trough True North (`direction_consensus.csv`, `manifest.json`)
- **Kelmarsh (6 turbines)**: Mean peer MAE drops from `3.11°` raw to `2.74°` after relative consensus alignment (strongest improvement on `Kelmarsh 5`: `4.31° -> 2.76°`, `relative_offset_deg = -3.03°`, `total_offset_deg = -4.41°`; `Kelmarsh 3`: `3.46° -> 2.79°`, `total_offset_deg = +1.25°`). True-north wake-trough anchoring achieves `status = "anchored_via_wake_trough"` (`common_true_north_offset_deg = -1.38°`, `pair_mad_deg = 2.14°`, `28/28` strong pairs, `mean_wake_depth_from = 0.729` vs `mean_wake_depth_to = 0.346`).
- **Penmanshiel (14 turbines)**: Mean peer MAE improves from `4.76°` raw to `4.58°` relative (`Penmanshiel 15`: `5.34° -> 4.56°`, `total_offset_deg = +0.54°`). Dense multi-row array wakes result in `status = "low_confidence"` (`common_true_north_offset_deg = -1.47°`, `pair_mad_deg = 13.06°`, `84/84` pairs, `mean_wake_depth_from = 0.461` vs `mean_wake_depth_to = 0.352`).
- **ERA5 Cross-Check**: `Kelmarsh` `n = 19,401` matched hours (`median SCADA - ERA5 = -3.39°`, `MAE = 7.51°`); `Penmanshiel` `n = 25,269` matched hours (`median SCADA - ERA5 = -6.45°`, `MAE = 9.99°`).
- **Left-Censored Stops**: Only `Penmanshiel 06` exhibits `left_censored_stop_events = 1`; the remaining 19 turbines record `0`.

### 2.2 Wake Regime Stratification (`wake_regime_effects.csv`)
Across donor-stratified partial-load regimes, `active_wake` sectors exhibit elevated turbulence intensity (`~0.14–0.165`) compared with `freestream` (`~0.092–0.125`), accompanied by depressed partial-load peer Power Performance Ratios (`ppr_peer_mean = 0.509–0.960` in `active_wake` vs `0.806–1.380` in `freestream`). Comparing `active_wake` against `passive_wake_donor_stopped` separates genuine rotor wake deficits from site terrain effects.

### 2.3 Three-Baseline Power Curve Health Monitoring (`pc_health_summary.csv`)
Evaluated across Nov–Dec 2017 (`57–61` evaluation days after `31` calibration days [`29–30` on `Penmanshiel 05`]), `59/60` turbine-baseline pairs remain nominal while `Kelmarsh 3` under `self_historical` fires `3` negative-drift alert days starting `2017-11-02` (`final_cusum_pos = 257.93`). `any_left_censored_onset = False` across all `60` pairs.

### 2.4 Peer MCP Wind Speed Reconstruction & Multi-Horizon Gap Benchmark (`peer_drift_summary.csv`, `peer_gap_benchmark.csv`)
- **Held-Out Reconstruction**: Across all 20 turbines, test MAE ranges `0.4802–0.7374 m/s`, signed bias `-0.130` to `+0.135 m/s`, `q99` false exceedance rate `0.88%–3.52%`, and synthetic `+2 m/s` fault detection `17.0%–59.56%`.
- **Gap Imputation vs. Persistence (Fleet Averages Across All 20 Turbines)**:

| Gap Horizon | Donor Dropout | Abstention Rate | Peer MCP MAE (m/s) | Peer MCP RMSE (m/s) | Persistence MAE (m/s) | 90% Interval Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` step (10 min) | `False` | `0.089` | `0.717` | `0.946` | `0.736` | `0.852` |
| `1` step (10 min) | `True` | `0.078` | `0.796` | `1.110` | `0.735` | `0.824` |
| `6` steps (1 h) | `False` | `0.028` | `0.664` | `0.885` | `1.390` | `0.851` |
| `6` steps (1 h) | `True` | `0.026` | `0.715` | `0.949` | `1.389` | `0.827` |
| `36` steps (6 h) | `False` | `0.022` | `0.660` | `0.905` | `2.070` | `0.851` |
| `36` steps (6 h) | `True` | `0.022` | `0.704` | `0.948` | `2.068` | `0.831` |
| `144` steps (24 h) | `False` | `0.026` | `0.595` | `0.805` | `2.757` | `0.885` |
| `144` steps (24 h) | `True` | `0.027` | `0.659` | `0.875` | `2.752` | `0.852` |

### 2.5 Air-Density Correction Holdout Benchmark (`density_benchmark.csv`, 2016 -> 2017 across 11 Turbines)
Trained purely on 2016 power curves (Kelmarsh 1–6, Penmanshiel 11–15) and evaluated chronologically on 2017 power predictions:

| Season / Air-Density Stratum | `ws_raw` MAE (kW) | `ws_pub_adj` MAE (kW) | `ws_iec` MAE (kW) | `ws_svenningsen` MAE (kW) |
| :--- | :---: | :---: | :---: | :---: |
| `overall` | `62.75` (`87.68` RMSE) | `59.91` (`83.50` RMSE) | `60.03` (`83.83` RMSE) | **`59.75`** (`83.35` RMSE) |
| `high_density_gt_1.25` | `77.59` (`107.58` RMSE) | `67.52` (`94.05` RMSE) | `65.38` (`91.02` RMSE) | **`63.89`** (`88.78` RMSE) |
| `low_density_lt_1.20` | `58.25` (`80.86` RMSE) | **`56.58`** (`78.86` RMSE) | `56.75` (`79.30` RMSE) | `56.94` (`79.61` RMSE) |
| `winter_DJF` | `80.35` (`107.90` RMSE) | `74.17` (`99.82` RMSE) | `74.85` (`100.88` RMSE) | **`73.95`** (`99.61` RMSE) |
| `summer_JJA` | `52.17` (`71.84` RMSE) | `50.00` (`69.15` RMSE) | **`49.78`** (`68.97` RMSE) | `49.79` (`69.03` RMSE) |

Full ERA5 moist-air WindPRO/EMD Svenningsen scaling (`ws_svenningsen`) consistently delivers the largest accuracy gains in winter and cold high-density regimes (`>1.25 kg/m³`), cutting fleet mean MAE by `13.70 kW` (`-17.7%`) over unnormalized SCADA wind speed.
