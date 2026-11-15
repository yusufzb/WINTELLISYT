# Actionable Asset Intelligence Report (2017 Audit)

This report translates the automated outputs in `artifacts/engineering_suite_2017/` into prioritized O&M inspection actions across Kelmarsh (`MM92`, 6 turbines) and Penmanshiel (`MM82`, 14 turbines).

---

## 1. Fleet IEC Energy Waterfall (`energy_waterfall.csv`)

Across 2017, the 20 turbines exported **117,425.60 MWh** (`117.43 GWh`) while recording **2,021.71 MWh** of downtime losses (`0.0 MWh` in exported curtailment counters):

| Farm (`MM92` / `MM82`) | Turbines | Exported Energy (MWh) | Net Energy (MWh) | Downtime Loss (MWh) | Potential MPC (MWh) | Potential Learned (MWh) | MPC - Learned Gap (MWh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kelmarsh** | `1–6` | `32,556.99` | `32,542.90` | `576.30` | `31,854.29` | `31,989.40` | `-135.11` |
| **Penmanshiel** | `01,02,04–15` | `84,868.61` | `84,826.75` | `1,445.41` | `86,939.98` | `85,774.51` | `+1,165.48` |

### Key Outliers Requiring Action
- **Kelmarsh 6 (`MM92`)**: Lowest fleet export (`4,439.84 MWh`), highest Kelmarsh downtime loss (`162.77 MWh`), and largest contractual-minus-learned gap (`+498.61 MWh`, comparing `potential_mpc_mwh = 4,370.66` vs `potential_learned_mwh = 3,872.05`). Recommend full aerodynamic/controller baseline inspection and wake-sector reconciliation against its `68.5 m` lower hub height.
- **Penmanshiel 06 (`MM82`)**: Highest fleet downtime loss (`253.42 MWh` on `5,819.66 MWh` exported; also the only turbine with a `left_censored_stop_events = 1` flag in `direction_consensus.csv`).
- **Penmanshiel 10, 02, 11, 05, 01**: Largest contractual-vs-learned underperformance gaps at Penmanshiel (`+266.80 MWh`, `+239.65 MWh`, `+199.22 MWh`, `+198.02 MWh`, and `+185.56 MWh`, respectively).

---

## 2. Anemometry & Power Curve Health Alerts

1. **Kelmarsh 1 Cup Anemometer Divergence (`anemometer_ntf_audit.csv`)**:
   - `median_s1_minus_s2_ms = +0.504 m/s` between `Sensor 1` and `Sensor 2` on `Kelmarsh 1`. Schedule nacelle cup swap / boom alignment check, as controller wind estimation directly impacts partial-load torque control and potential energy accounting.
2. **Kelmarsh 3 Negative Power-Curve Drift (`pc_health_summary.csv`)**:
   - Under `self_historical` monitoring, `Kelmarsh 3` is the **only** turbine-baseline pair triggering negative-drift CUSUM alerts across Nov–Dec 2017 (`3` alert days starting `2017-11-02`, `final_cusum_pos = 257.93`). Notably, `Kelmarsh 3` also recorded `+0.0385` PPR recovery after an `8.98 h` stop on `2017-08-16` (`pitch_ppr_recovery.csv`) prior to the November drift onset.
3. **Wind Vane Calibration (`direction_consensus.csv`)**:
   - `Kelmarsh 5` carries the largest relative directional bias (`relative_offset_deg = -3.03°`, `total_offset_deg = -4.41°`), reducing peer directional MAE from `4.31°` to `2.76°` once corrected.

---

## 3. Pitch System & Three-Axis Motor Health

- **Blade Angle Asymmetry (`pitch_asymmetry.csv`)**: All 6 Kelmarsh turbines show `max_pair_offset_deg = 0.0°` (`asymmetry_excursion_pct <= 0.03%`, `flag_asymmetry = False`). Penmanshiel lacks Blade A/B/C pitch angle telemetry.
- **Maintenance Pre/Post Recovery (`pitch_ppr_recovery.csv`, 413 events)**: Across both farms, `26` stop events achieved measurable `improved_ppr` recovery (`3` at Kelmarsh, `23` at Penmanshiel), highlighted by `Kelmarsh 3` (`2017-08-16`, `8.98 h`, `+0.0385` PPR) and `Kelmarsh 5` (`2017-01-11` Pitch run-away stop, `+0.0362` PPR); `64` events showed post-stop `degraded_ppr`, `64` `unchanged_ppr`, and `259` `insufficient_or_confounded`.
- **Penmanshiel Pitch Motor Thermal Spread (`pitch_actuator_health.csv`)**: **10 of 14 Penmanshiel turbines** exceed the `8.0°C` inter-axis `p95_motor_temp_spread_c` anomaly threshold:

| Turbine | `p95_motor_temp_spread_c` (°C) | Priority |
| :--- | :---: | :---: |
| `Penmanshiel 04` | `13.22` | P1 — Inspect pitch motor cooling / slip-ring brushes |
| `Penmanshiel 15` | `13.16` | P1 — Combined with `-7.83°` yaw bias candidate |
| `Penmanshiel 13` | `11.90` | P1 |
| `Penmanshiel 06` | `11.54` | P2 — Co-located with `253.42 MWh` top downtime |
| `Penmanshiel 05` | `11.22` | P2 |
| `Penmanshiel 12` | `10.58` | P2 |
| `Penmanshiel 02` | `10.22` | P2 |
| `Penmanshiel 01` | `10.13` | P2 |
| `Penmanshiel 07` | `9.84` | P3 |
| `Penmanshiel 11` | `8.17` | P3 — Combined with `-7.23°` yaw bias candidate |

---

## 4. Static Yaw Alignment & Drivetrain CMS / Wake Fatigue

1. **Static Yaw Bias Ranking (`yaw_misalignment_summary.csv`)**:
   - On Penmanshiel (where optimization curves remain interior to search bounds), **Penmanshiel 15** (`estimated_static_yaw_bias_deg = -7.83°`, `61.04 MWh` recoverable), **Penmanshiel 11** (`-7.23°`, `47.44 MWh`), **Penmanshiel 14** (`-5.67°`, `31.13 MWh`), and **Penmanshiel 02** (`-5.41°`, `26.55 MWh`) lead candidate vane re-zeroing (`317.54 MWh` farm total).
   - *Caution*: Kelmarsh fits clip at the `+15.0°` boundary (`842.24 MWh` upper bound) due to wind-speed confounding across the partial-load knee and must **not** trigger blind yaw offsets without speed-binned LiDAR verification.
   - Kelmarsh cable-twist counters operate smoothly up to `1.98` turns (`WT4`, `19` unwind steps; `18–23` annual step cycles across Kelmarsh 1–6).
2. **Drivetrain & Bearing Thermal Outliers (`drivetrain_cms_summary.csv`)**:
   - **Kelmarsh 1**: Highest fleet thermal excess (`+3.73°C` on `rotor_bearing_temp_c`) co-occurring with the highest Kelmarsh 95th-percentile drivetrain acceleration (`174.898 mm/s²`) — high-priority borescope/grease inspection target.
   - **Penmanshiel 12**: Highest Penmanshiel thermal residual (`+2.79°C` on `gen_bearing_front_temp_c`), despite producing the highest annual export (`6,843.78 MWh`).
3. **Wake-Induced Vibration Uplift (`wake_vibration_fatigue.csv`)**:
   - Operating inside `active_wake` elevates median tower and drivetrain accelerations sharply over `freestream`: **Kelmarsh 3** exhibits `+71.04%` uplift on `tower_accel_x_mmss` (`97.324` vs `56.902`), `+50.67%` on `tower_accel_y_mmss`, and `+48.73%` on `drivetrain_accel_mmss`; **Penmanshiel 10** exhibits `+48.28%` drivetrain acceleration uplift (`0.860` vs `0.580`). Incorporate active wake sector load exposure into main-bearing and tower-bolt fatigue inspections.
