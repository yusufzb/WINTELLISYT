# Signal Inventory & Fleet Geometry Reference

## 1. Fleet Layout Summary (`artifacts/engineering_suite_2017/layout.csv`)

Both sites operate Senvion 2.05 MW doubly-fed induction turbines under UK onshore conditions:
- **Kelmarsh (Northamptonshire, UK)**: 6 × `Senvion MM92` (`2,050 kW` rated, `92.0 m` rotor diameter, aggregate `12.3 MW`). Hub heights are **mixed**: `78.5 m` on `WT1, WT2, WT4, WT5` and `68.5 m` on `WT3, WT6`. Terrain elevations span `135.04 m` (`WT6`) to `156.58 m` (`WT2`).
- **Penmanshiel (Scottish Borders, UK)**: 14 active × `Senvion MM82` (`2,050 kW` rated, `82.0 m` rotor diameter, aggregate `28.7 MW`; `WT03` is absent from all published archives). Uniform hub height is `59.0 m`; terrain elevations span `180.24 m` (`WT07`) to `228.15 m` (`WT15`).

---

## 2. Inventory Coverage Audit (`data/reference/signal_inventory.csv`)

Auditing all raw telemetry CSVs across both farms and years yields **9,288** channel-turbine-year entries spanning **19** canonical signals:

| Inventory Dimension | Observed Breakdown |
| :--- | :--- |
| Total channel-turbine-year rows | `9,288` |
| Unique canonical signals mapped | `19` |
| Kelmarsh turbines (`2016` / `2017`) | `6` / `6` (`WT1`–`WT6`) |
| Penmanshiel turbines (`2016` / `2017`) | `5` (`WT11`–`WT15`) / `14` (`WT01, 02, 04..15`) |
| Eligibility: `key` | `31` |
| Eligibility: `candidate; see channel-specific caveats` | `446` |
| Eligibility: `inventory only; semantic/use review required` | `4,962` |
| Eligibility: `unavailable` (100% NaN or absent) | `3,849` |

---

## 3. Canonical Channel Matrix (Populated Turbine Counts with `finite_count > 0`)

| Canonical Signal (`signal_inventory.csv`) | Kelmarsh 2016 (max 6) | Kelmarsh 2017 (max 6) | Penmanshiel 2016 (max 5) | Penmanshiel 2017 (max 14) | Primary Diagnostic Usage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `active_power_kw` | 6 | 6 | 5 | 14 | PC health, wake PPR, yaw cos^2, energy waterfall |
| `reported_ws` (`Wind speed (m/s)`) | 6 | 6 | 5 | 14 | Primary binning, peer MCP target/donors |
| `wind_speed_sensor_1_ms` | **0** | 6 | 5 | 14 | NTF provenance & dual-anemometer calibration check |
| `wind_speed_sensor_2_ms` | **0** | 6 | 5 | 14 | NTF provenance & dual-anemometer calibration check |
| `wind_speed_std_ms` | **0** | 6 | 5 | 14 | Wake turbulence intensity (`TI = sigma / v`) |
| `publisher_density_adjusted_ws_ms` | 6 | 6 | 5 | 14 | Audited against ERA5 moist-air IEC/Svenningsen |
| `reported_wind_direction_deg` | 6 | 6 | 5 | 14 | NREL circular consensus, wake sectors, ERA5 check |
| `reported_nacelle_position_deg` | 6 | 6 | 5 | 14 | Vane-nacelle closure verification |
| `reported_combined_vane_deg` | **0** | 6 | 5 | 14 | Vane-nacelle closure & static yaw regression |
| `ambient_temperature_c` | 6 | 6 | 5 | 14 | Thermal CMS normalization & density audit |
| `blade_a_pitch_deg` / `B` / `C` | 6 | 6 | **0** | **0** | Kelmarsh blade pitch asymmetry & recovery |
| `rotor_speed_rpm` / `generator_rpm` | 6 | 6 | 5 | 14 | Operating regime gating (`pc_health`, `wake`) |
| `publisher_energy_export_kwh` | 6 | 6 | 5 | 14 | IEC energy waterfall exported MWh |
| `publisher_energy_import_kwh` | 6 | 6 | 5 | 14 | IEC parasitic import MWh |

---

## 4. Rich Ingestion Extensions (`RICH_CHANNEL_MAP` in `src/wintellisyt/data/scada.py`)

Beyond the 19 core signals inventoried for the base study, `load_rich_telemetry` extracts extended Greenbyte SCADA channels needed for Phase 2.5 Asset Intelligence:
- **Yaw & Nacelle**: `Vane position 1+2, StdDev (°)` (`Penmanshiel`), `Cable windings from calibration point` (`Kelmarsh`).
- **Energy & Potential**: `Potential power default PC (kW)`, `Potential power MPC (kW)`, `Potential power learned PC (kW)`, `Energy Theoretical (kWh)`, `Lost Production (Contractual) (kWh)`, `Lost Production to Downtime (kWh)`, `Lost Production to Performance (kWh)`, and 11 `Lost Production to Curtailment (*)` counters.
- **Pitch Drives & Motors**: `Motor current axis 1/2/3 (A)` (`Kelmarsh` + `Penmanshiel`), `Motor temperature axis 1/2/3 (°C)` (`Penmanshiel` only).
- **Drivetrain & Tower CMS**: `Gear oil differential pressure`, `Metal particles` (`Kelmarsh`), `Drive train acceleration`, `Tower acceleration x/y`, plus gearbox oil, rotor/front/rear/generator bearing, and stator temperature channels (normalized strictly within-farm due to the ~100× gear-oil DP and acceleration unit shift documented in `ASSUMPTIONS_AND_LIMITATIONS.md`).
