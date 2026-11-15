# Assumptions, Constraints, and Known Limitations

This document records the explicit physical assumptions, SCADA provenance findings, sensor coverage gaps, and analytical caveats uncovered during the Principal Engineer audit of the Kelmarsh (`MM92`) and Penmanshiel (`MM82`) datasets.

---

## 1. Speed Measurement Source & Transfer Function Constraints (NTF)

### Empirical Audit Findings (`anemometer_ntf_audit.csv` & `vane_nacelle_audit.csv`)
- **Cup Anemometer Averaging**: Across all 20 turbines, `Wind speed (m/s)` closely matches the arithmetic mean of `Wind speed Sensor 1 (m/s)` and `Wind speed Sensor 2 (m/s)`, with `median_ws_over_mean_s12` lying tightly in `0.9985–1.0001`.
  - Penmanshiel `02, 05, 09` classify directly as `matches_mean_s1_s2` (MAE `0.0128–0.0193 m/s` vs mean S1/S2).
  - The remaining 17 turbines exhibit small but systematic deviations (`MAE 0.0211–0.1131 m/s`), reflecting Greenbyte/controller quality filtering, cup-selection logic, or Nacelle Transfer Function (NTF) shaping.
- **Implied NTF Assumption**: Raw high-frequency anemometer signals (prior to PLC low-pass filtering, 1P/2P blade-passing notch attenuation, and internal NTF lookup tables) are **not** included in 10-minute SCADA exports. All analyses treat `Wind speed (m/s)` as a post-PLC/post-NTF 10-minute nacelle-anemometer estimate rather than an IEC 61400-12-1 free-stream met-mast reference.
- **Sensor Calibration Divergence**: Kelmarsh 1 shows `median_s1_minus_s2_ms = +0.504 m/s`, indicating a substantial primary-vs-redundant cup discrepancy that warrants physical inspection and prohibits blindly treating Sensor 1 as unbiased.
- **Vane-Nacelle Geometric Identity**: Across all 20 turbines, `median_wind_minus_nacelle_plus_vane_deg` is `-0.047°` to `+0.007°` (Kelmarsh) and `0.000°` (Penmanshiel), empirically verifying the Greenbyte identity `Wind direction ≈ (Nacelle position + Vane position) mod 360°`.

---

## 2. Publisher Density Adjustment vs. ERA5 Moist-Air Scaling

- Auditing `Density adjusted wind speed (m/s)` against `Wind speed (m/s)` and `Nacelle ambient temperature (°C)` yields `implied_density_temp_corr = -1.0` on **all 20 turbines** (`implied_density_median_kg_m3 = 1.2010–1.2128 kg/m³`).
- **Limitation**: The SCADA publisher's density-adjusted channel uses a simplified constant-pressure ideal-gas temperature ratio (`T`), omitting time-varying barometric pressure and water vapor partial pressure (relative humidity). Furthermore, nacelle ambient sensors can suffer solar/housing radiative bias relative to free-stream atmosphere.
- **Mitigation**: `atmosphere.py` computes moist-air hub-height density from Open-Meteo ERA5 surface pressure, 2m temperature, and relative humidity scaled to exact per-turbine hub + terrain elevation (`layout.csv`), comparing both IEC (`m = 1/3`) and WindPRO/EMD Svenningsen (`m(v)`) formulations out-of-sample.

---

## 3. True-North Anchoring Confidence Limits

- **Kelmarsh (6 turbines)**: Well-separated geometric layout delivers `status = "anchored_via_wake_trough"` with `28/28` strong donor-target pairs, low inter-pair scatter (`pair_mad_deg = 2.14°`), clear FROM/TO contrast (`mean_wake_depth_from = 0.729` vs `mean_wake_depth_to = 0.346`), and `common_true_north_offset_deg = -1.38°` (ERA5 cross-check: `median_scada_minus_era5_deg = -3.39°`, MAE `7.51°`).
- **Penmanshiel (14 turbines)**: Dense multi-row array with overlapping double/triple wakes yields `status = "low_confidence"` across `84/84` pairs (`pair_mad_deg = 13.06°`, `common_true_north_offset_deg = -1.47°`, `mean_wake_depth_from = 0.461` vs `mean_wake_depth_to = 0.352`, ERA5 cross-check `-6.45°`, MAE `9.99°`).
- **Constraint**: True-north offsets at Penmanshiel should be treated as diagnostic estimates rather than authoritative absolute compass calibrations until validated against site met-mast or nacelle LiDAR measurements.

---

## 4. Cross-Farm Channel Asymmetries & Missing Sensors

Comparison across the two Senvion farms (`Senvion MM92` vs `Senvion MM82`) reveals major structural telemetry differences that prohibit naive cross-farm feature sharing:

| Signal Group | Kelmarsh (`MM92`, 6 WT) | Penmanshiel (`MM82`, 14 WT) | Engineering Impact |
| :--- | :--- | :--- | :--- |
| Blade Pitch Angles (`A/B/C`) | Available in 2016 & 2017 (`6/6`) | Absent / 100% NaN (`0/14`) | Blade-to-blade `pitch_asymmetry.csv` runs on Kelmarsh only |
| 3-Axis Pitch Motor Temperatures | Absent (`n_temp = 0`) | Available (`n_temp = 51,781–52,188`) | Thermal spread anomaly flags (`>8.0°C`) run on Penmanshiel only |
| 3-Axis Pitch Motor Currents | Available (`6/6`) | Available (`14/14`) | Cross-farm comparison valid within farm; Penmanshiel currents higher/different drive |
| Vane StDev (`vane_std_median_deg`) | Absent (NaN) | Available (`10.14°–12.45°`) | Yaw turbulence spread available on Penmanshiel only |
| Cable Windings & Unwind Steps | Available (`1.73–1.98` turns, `18–23` steps) | Absent (`0` steps / NaN turns) | Twist-limit diagnostics run on Kelmarsh only |
| Wind Speed Sensors 1/2 & StDev | Absent in 2016; available in 2017 | Available in 2016 (`5 WT`) & 2017 (`14 WT`) | 2016->2017 TI / S1-S2 studies constrained on Kelmarsh 2016 |
| Gear Oil DP / Drivetrain Accel Scale | `312–357` DP / `65–79` mm/s² median | `3.51–4.11` DP / `0.67–0.83` median | ~100× unit/scaling shift between farms; CMS models must normalize strictly within-farm |
| Hub Height Heterogeneity | Mixed (`78.5 m` WT1,2,4,5; `68.5 m` WT3,6) | Uniform `59.0 m` (`WT03` absent) | ERA5 density & shear adjustments must use per-turbine `hub_height_m + elevation_m` |

---

## 5. Methodological Caveats & Exclusions

1. **Static Yaw $\cos^2$ Speed-Confounding (`yaw_misalignment_summary.csv`)**:
   - At Kelmarsh, broad partial-load fitting (`4–11 m/s`) saturates against the `±15.0°` search clip on 4/6 turbines (`WT2, WT3, WT4, WT6`), producing an inflated upper-bound recovery estimate (`842.24 MWh` fleet total). This reflects correlation between wind-speed-dependent pitch/vane trim around the power curve knee and measured nacelle vane offset rather than pure physical yaw misalignment.
   - At Penmanshiel, fitted optima stay interior to the search bounds (`-2.83°` to `-7.83°`, summing to `317.54 MWh`), offering cleaner relative ranking across `WT15` and `WT11`, yet still requiring wind-speed-stratified bin validation before physical vane re-zeroing.
2. **Penmanshiel WT03 Absence**: Turbine `03` does not appear in any published Cubico Zenodo Penmanshiel archive (`14` turbines: `01, 02, 04..15`). Downstream wake matrices explicitly exclude `WT03`.
3. **Zero Exported Curtailment Counters**: All `Energy/Lost Production to Curtailment (*)` SCADA counters sum to `0.0 MWh` in 2017 on both farms, meaning grid/noise/shadow/bat curtailment cannot be quantified from those publisher counters alone without cross-referencing `Status_*.csv` stop codes and `mpc_vs_learned_gap.csv`.
4. **No High-Frequency 1 Hz / 50 Hz SCADA**: All diagnostics operate strictly on 10-minute averaged and standard-deviation statistics; transient controller dynamics (1P/2P blade passing loads, tower resonance damping ratios, and blade root bending moments) cannot be resolved.
5. **Contractual Baseline vs. Site Air Density (`oem_power_curves.json`)**: OEM 1.0 m/s knots represent standard sea-level `1.225 kg/m³` freestream conditions, whereas field turbines operate at lower terrain-adjusted densities (`~1.18–1.22 kg/m³` depending on elevation `135–228 m` and temperature) and experience sector wakes—explaining part of `mpc_minus_learned_mwh` gaps independently of mechanical degradation.
