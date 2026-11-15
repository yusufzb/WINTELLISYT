# Architecture and Engineering Methodology

WINTELLISYT (`v0.2.0`) is structured along clean ISO-style repository boundaries separating installable domain code (`src/wintellisyt/`), immutable raw/external/reference datasets (`data/`), orchestration scripts (`scripts/`), a four-layer test pyramid (`tests/`), generated outputs (`artifacts/`), and documentation (`docs/`).

## 1. Repository Layout

```text
WINTELLISYT/
|-- src/wintellisyt/
|   |-- __init__.py
|   |-- cli.py                  # CLI: demo, run, transfer, sources, ui
|   |-- wind_geometry.py        # Geodesic pair vectors, FROM/TO bearings, wake overlap
|   |-- direction.py            # Two-stage NREL circular consensus + wake-trough true north
|   |-- wake.py                 # Geometric wake sectors + active/passive donor stratification
|   |-- pc_health.py            # Three-baseline power curve monitoring (EWMA/CUSUM + left censoring)
|   |-- pitch_health.py         # Blade A/B/C asymmetry + maintenance pre/post PPR recovery
|   |-- peer_reconstruction.py  # Leave-one-out MCP wind reconstruction & multi-horizon gap imputation
|   |-- atmosphere.py           # Moist-air density, IEC 1/3 scaling, WindPRO/EMD Svenningsen scaling
|   |-- asset_intelligence.py   # Phase 2.5 suite: energy waterfall, yaw cos^2, motors, CMS, fatigue
|   |-- power_curve/            # Core 0.5 m/s method-of-bins reference & PHYSICS.md
|   |-- data/                   # Greenbyte SCADA/status loaders & SHA-256 manifest verification
|   |-- ui/                     # NiceGUI primary app, Starlette fallback, JSON API, standalone HTML
|   |-- pipeline.py             # End-to-end single-turbine audit runner
|   |-- operating_state.py      # Physical + event-based turbine state reconciliation
|   |-- energy.py               # Interval-safe energy integration
|   |-- transfer.py             # Cross-turbine chronological transfer scoring
|   `-- uncertainty.py          # Uncertainty propagation helpers
|-- data/
|   |-- raw/{kelmarsh,penmanshiel}/  # SHA-256 verified Cubico Greenbyte SCADA ZIPs & static CSVs
|   |-- external/era5/              # Cached Open-Meteo ERA5 atmospheric reanalysis JSONs
|   `-- reference/                  # SHA-256 manifest, attribution, OEM PC knots, signal inventory
|-- scripts/
|   `-- run_engineering_suite.py    # Executes full multi-farm suite -> 24 CSVs + manifest.json
|-- tests/
|   |-- physics/                    # Unit laws: circular wrapping, wake geometry, atmosphere, PC bins
|   |-- diagnostics/                # Synthetic fault injection (CUSUM, true north, yaw cos^2, pitch)
|   |-- leakage/                    # Strict future/target exclusion & causal fitting guards
|   `-- contracts/                  # SCADA ingestion, duplicate quarantine, UI & Starlette route tests
|-- docs/                           # English engineering reference documentation
`-- artifacts/                      # Reproducible CSV suites, JSON manifests, interactive HTML dashboard
```

---

## 2. Core Physical & Diagnostic Modules

### 2.1 Spatial Wind Geometry (`wind_geometry.py`)
Given turbine WGS-84 coordinates and rotor diameters `rotor_m` in `LayoutPoint`, pairwise separation vectors `(d_east_m, d_north_m)` establish Euclidean spacing and donor-to-target azimuths:
- `source_bearing(donor, target)`: compass bearing pointing **from** `target` **toward** `donor` (`atan2(-dx, -dy)` mod 360 deg), representing the meteorological **FROM** inflow direction under which `donor` sits upstream of `target`.
- `wake_projection(donor, target, wind_from_deg)`: longitudinal downstream distance `x_m` (in meters and rotor diameters `x_d = x_m / D_donor`) and lateral cross-wind offset `y_m`.
- `potential_overlap(donor, target, wind_from_deg)`: compares `|y_m|` against the linear wake expansion radius `r_w = 0.5 * D_donor + k_w * x_m` plus `0.5 * D_target` (`k_w` expansion coefficient) whenever `x_m > 0`.

### 2.2 Two-Stage NREL Circular Consensus & True-North Anchoring (`direction.py`)
1. **Stage 1 (Relative Consensus)**: For each target turbine `i`, excluded from its own reference pool, compute a distance-weighted circular median vector across peer turbines calibrated strictly prior to the evaluation cutoff.
2. **Stage 2 (Relative Offset Alignment)**: Estimate per-turbine circular median offset `relative_offset_deg` so turbines align onto a common farm angular reference frame (`raw_peer_mae_deg -> relative_peer_mae_deg`).
3. **True-North Anchoring via Wake Troughs**: Rotate candidate global offsets across neighbor-pair inflow sectors and measure partial-load peer Power Performance Ratio (`ppr_peer_mean`) deficits.
   - Disambiguate **FROM** vs **TO** conventions by comparing mean wake trough depth under meteorological FROM alignment (`mean_wake_depth_from`) against TO alignment (`mean_wake_depth_to`).
   - Kelmarsh (6 turbines): `anchored_via_wake_trough` (`pair_mad_deg = 2.14°`, `common_true_north_offset_deg = -1.38°`, `mean_wake_depth_from = 0.729` vs `mean_wake_depth_to = 0.346`).
   - Penmanshiel (14 turbines): dense overlapping multi-row array yields higher cross-pair spread (`pair_mad_deg = 13.06°`, `common_true_north_offset_deg = -1.47°`, `status = "low_confidence"`), cross-checked against ERA5 (`median_scada_minus_era5_deg = -6.45°`).

### 2.3 Active vs. Passive Wake Stratification (`wake.py`)
Unlike naive geometric wind-sector filters, `wake.py` separates downstream observations into:
- `active_wake`: upstream donor is geometrically aligned (`potential_overlap`) **and** actively generating (`power > cutoff`, extracting momentum).
- `passive_wake_donor_stopped`: identical directional corridor when upstream donor is stopped/offline (natural terrain/orography control group).
- `freestream`: no upstream turbine within the expanded wake cone.

Evaluating partial-load turbulence intensity (`wind_speed_std_ms / wind_speed_ms`) and peer Power Performance Ratio (`ppr_peer_mean`) against both `freestream` and `passive_wake_donor_stopped` isolates turbine-induced wake losses from topographic shelter effects.

### 2.4 Power Curve Health Monitoring (`pc_health.py`)
Residual monitoring `r_t = P_actual(t) - P_expected(v_t)` runs simultaneously against **three independent baselines** fitted strictly on training/calibration history:
1. `self_historical`: turbine's own historical 0.5 m/s normal power curve.
2. `fleet_median`: cross-fleet reference normalized per farm/model cohort.
3. `contractual_oem`: interpolated OEM 1.0 m/s power curve knots (`data/reference/oem_power_curves.json`).

**State Holding & Left-Censoring**: During calm wind (`v < cut_in`) or stopped periods, EWMA and two-sided tabular CUSUM accumulators hold state instead of decaying toward zero (`cusum_pos` for systematic underperformance, `cusum_neg` for overperformance). Consecutive missing/censored timestamps before a step fault are tagged (`any_left_censored_onset`) so operators can distinguish abrupt faults masked by downtime from smooth progressive degradation.

### 2.5 Pitch System Health (`pitch_health.py`)
- **Blade A/B/C Asymmetry**: Quantifies pairwise differences (`A-B`, `B-C`, `C-A`) where all three pitch angle channels exist (Kelmarsh), flagging sustained excursions (`flag_asymmetry`).
- **Pre/Post Maintenance Recovery (`pitch_ppr_recovery.csv`)**: Aligns deduplicated `Status_*.csv` stop intervals against telemetry windows before and after each event (requiring sufficient supported wind-speed bins and flagging `competing_maintenance` or `full_followup` completeness) to classify outcomes (`improved_ppr`, `unchanged_ppr`, `degraded_ppr`, `insufficient_or_confounded`).

### 2.6 Peer MCP Wind Speed Reconstruction & Gap Imputation (`peer_reconstruction.py`)
Trains leave-one-turbine-out direction-aware linear/binned transfer models strictly on pre-cutoff training splits:
- **Anemometer Drift Audit**: Quantifies test-set MAE, signed bias (`obs - peer`), calibrated conformal `q90`/`q99` interval thresholds, and out-of-sample synthetic `+2 m/s` drift detection rates.
- **Multi-Horizon Gap Imputation**: Evaluates reconstruction accuracy under synthetic contiguous outages of `1, 6, 36, and 144` ten-minute intervals (`10 min` through `24 h`) with and without donor dropout (`donor_dropout = False/True`), benchmarked directly against persistence.

### 2.7 Atmospheric Density & Hub-Height Normalization (`atmosphere.py`)
Using cached Open-Meteo ERA5 surface temperature, pressure, relative humidity, and `100 m` wind components (`data/external/era5/`):
- **IEC 61400-12-1 Standard Scaling (`ws_iec`)**:

$$v_{\text{IEC}} = v_{\text{obs}} \left(\frac{\rho_{\text{moist}}(z_{\text{hub}})}{\rho_0}\right)^{1/3}, \qquad \rho_0 = 1.225\ \text{kg/m}^3$$

- **WindPRO / EMD Svenningsen Speed-Dependent Exponent (`ws_svenningsen`)**: Accounts for pitch-regulated torque limiting above rated transition using piecewise exponent $m(v)$:

$$m(v) = \begin{cases} \frac{1}{3}, & v \le 7.5\ \text{m/s} \\ \frac{1}{3} + \frac{v - 7.5}{5.0}\cdot \frac{1}{3}, & 7.5 < v < 12.5\ \text{m/s} \\ \frac{2}{3}, & v \ge 12.5\ \text{m/s} \end{cases}$$

### 2.8 Phase 2.5 Actionable Asset Intelligence (`asset_intelligence.py`)
1. **IEC Energy Waterfall (`energy_waterfall.csv` & `mpc_vs_learned_gap.csv`)**: Reconciles `energy_export_mwh`, `parasitic_import_mwh`, `net_energy_mwh`, theoretical/MPC/learned potential energy, downtime losses, and binned contractual-minus-learned deficits.
2. **Static Yaw Misalignment & Cable Windings (`yaw_misalignment_summary.csv`)**: Regresses partial-load normalized power against relative nacelle vane angle ($\cos^2$ loss curve) and tracks yaw twist saturation / unwinding step resets.
3. **3-Axis Pitch Motor Current & Thermal Health (`pitch_actuator_health.csv`)**: Detects stator/drive overload and inter-axis thermal spread anomalies (`p95_motor_temp_spread_c > 8.0°C`).
4. **Peer-Normalized Drivetrain & Bearing CMS (`drivetrain_cms_summary.csv`)**: Fits within-farm baseline expectations across gear oil differential pressure, metallic particle counts, gearbox/bearing/stator temperatures, and drivetrain/tower accelerometers to rank per-turbine thermal residuals.
5. **Wake-Induced Structural Fatigue (`wake_vibration_fatigue.csv`)**: Quantifies median percentage uplift in nacelle/drivetrain and fore-aft/side-side tower accelerations between `active_wake` and `freestream` inflows.

### 2.9 Interactive UI & Standalone Dashboard (`src/wintellisyt/ui/`)
`wintellisyt ui` supports three execution modes:
1. `--export-only`: Renders an offline-capable ~5 MB self-contained interactive Plotly HTML report (`interactive_dashboard.html`) across 6 thematic tabs (`Overview`, `Direction & Wake`, `PC & Pitch Health`, `Peer & Density`, `Asset Intelligence`, `Provenance & Inventory`).
2. `--backend auto` (or `nicegui`): Launches a live reactive NiceGUI web UI when `nicegui` is installed, automatically falling back to Starlette + Uvicorn when `nicegui` is absent.
3. `--backend starlette`: Serves the interactive HTML app plus REST JSON endpoints (`/health`, `/api/manifest`, `/api/tables/{name}`).
