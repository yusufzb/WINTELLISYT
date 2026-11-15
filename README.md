# WINTELLISYT (`v0.2.0`)

**Open-Source Wind Turbine SCADA Diagnostics, IEC 61400 Performance Auditing & Interactive Asset Intelligence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](data/reference/DATA_ATTRIBUTION.md)

**WINTELLISYT** is a reproducible wind-engineering analytics platform and interactive NiceGUI + Folium + Plotly workbench designed for fleet-scale operational SCADA auditing. Built and validated across two commercial wind farms (**Kelmarsh 6x Senvion MM92** and **Penmanshiel 14x Senvion MM82**, totaling 20 turbines and >1.05M 10-minute intervals in 2017), it unifies leakage-free statistical monitoring, boundary-layer wake & air-density physics, and root-cause failure synthesis inside a single-screen diagnostic workflow.

---

## Why WINTELLISYT?

Conventional SCADA reporting often masks mechanical faults behind uncalibrated anemometers, cold-air density biases, missing status-log masking, and fragmented charts that force reliability engineers to jump across disconnected tools. WINTELLISYT solves this end-to-end:

1. **Leakage-Free Operational Modeling**: Strict temporal out-of-sample evaluation splits, event-status debouncing, and out-of-time calibration prevent optimistic bias in power-curve monitoring and peer MCP wind reconstruction.
2. **IEC & Empirical Atmospheric Physics**: Benchmarks ERA5 moist-air density normalization (`IEC 61400-12-1` exponent vs. **WindPRO / EMD Svenningsen** speed-dependent scaling `ws_svenningsen`) and quantifies `<=10D` inter-turbine wake deficits using active-vs-stopped upwind donor placebos.
3. **NREL True-North Directional Consensus**: Two-stage circular sensor calibration paired with FROM-convention wake-trough true-north anchoring eliminates long-term nacelle northing drift.
4. **Integrated 3-Workspace Visual Workbench**: Moves seamlessly from a fleet-wide **Folium geospatial cockpit** (with Net/Theoretical Capacity Factor `CF %` and Clean Bin Deviation) straight into a **Single-Screen 360° Turbine Diagnostic Chain** with synchronized Wind-Speed and Time axes.

---

## Interactive Workbench Architecture (`3 Cohesive Workspaces`)

### 1. Site Cockpit & Interactive Folium Geo-Map
- **Multi-Basemap Geospatial View**: Interactive **Folium / Leaflet** map supporting 5 switchable layers (*Esri World Imagery Satellite*, *OpenTopoMap*, *CartoDB Positron*, *CartoDB DarkMatter*, *OpenStreetMap*), ConvexHull site boundary, scaled `1D` rotor rings, `<=10D` prevailing wake corridors, and hover/click diagnostic popups.
- **Fleet Watchlist & CF / Power-Curve Scorecard**: Automatically flags underperforming units (by Clean Power Curve Deviation `%` and Net vs. Theoretical Capacity Factor `CF %` gap) with 1-click navigation straight into the single-turbine diagnostic chain, alongside IEC 61400-26 B.2.2 vs. Event-Mask availability waterfalls.

### 2. Single-Screen 360° Turbine Diagnostic Chain
- **Automated Root-Cause Verdict Banner**: Synthesizes M1 power-curve drift (`CUSUM-` / `EWMA`), M2 yaw `cos²` misalignment bias, M3 thermal CMS / 3-blade pitch asymmetry, and M4 anemometer S1–S2 divergence into a single priority diagnosis.
- **Wind-Speed Locked Stack (`x = ws_svenningsen / ws_iec`)**: Combines density-normalized `Scattergl` telemetry (color-coded by clean normal vs. curtailed vs. waked vs. stopped regimes), OEM MPC / Learned PC / Peer baselines, `0.5 m/s` Method-of-Bins delta, and blade pitch + rotor speed overlays.
- **Time-Locked Chronological Stack (`shared_xaxes=True`, `hovermode="x unified"`)**: Aligns Daily Performance Index & `CUSUM-` onset curves, Monthly IEC B.2.2 vs. Event-Mask availability cross-checks, and debounced Stop-Outage Gantt episodes.
- **Subsystem Deep Diagnostics**: Guarded quadratic yaw parabola fitting, active-vs-stopped wake polar rose, load-matched thermal CMS residuals (`250 kW` bins), 3-axis pitch motor current asymmetry, and Virtual Anemometer peer MCP drift.

### 3. Fleet Outage Pareto & Sensor Provenance
- **Debounced Event Dictionary & Outage Pareto**: Merges high-frequency Senvion alarm chatter into discrete downtime episodes (`MWh` & `Hours` lost by RDS-PP subsystem and root code), with searchable `T0 +/- 60 min` pre/post-fault cascade timelines.
- **NTF & Anemometer Provenance**: Audits nacelle transfer functions, primary vs. secondary cup offsets (`S1 - S2`), and completeness matrices across the entire archive.

---

## Key Verified 2017 Findings

- **Fleet Scale & Integrity**: `1,051,200` 10-minute telemetry intervals (`315,360` Kelmarsh + `735,840` Penmanshiel), `83,967` deduplicated status events, `117.43 GWh` exported generation, and `2,021.71 MWh` recorded downtime losses.
- **Directional Consensus & True North**: Reduces circular peer MAE (`3.11° -> 2.74°` at Kelmarsh; `4.76° -> 4.58°` at Penmanshiel) and anchors Kelmarsh true north via wake troughs (`-1.38°`, `pair_mad_deg = 2.14°`), while explicitly flagging Penmanshiel dense multi-row wake overlap as `low_confidence` (`pair_mad_deg = 13.06°`).
- **Air Density Normalization (`2016 -> 2017`)**: WindPRO/EMD Svenningsen speed-dependent scaling (`ws_svenningsen`) reduces high-density (`>1.25 kg/m³`) holdout prediction MAE from `77.59 kW` (`ws_raw`) and `67.52 kW` (`ws_pub_adj`) down to **`63.89 kW`**.
- **Actionable Asset Intelligence**: Detects `Kelmarsh 3` negative `CUSUM` power-curve drift (`2017-11-02` onset), `Kelmarsh 1` `+0.504 m/s` S1–S2 anemometer divergence and `+3.73°C` rotor-bearing thermal residual, **10/14** Penmanshiel units exceeding `8.0°C` 3-axis pitch motor thermal spread (peaking at `13.22°C` on `WT04`), and up to `+71.04%` wake-induced tower X acceleration uplift.

---

## Installation & Package Requirements

Requires **Python `>=3.10`**. All dependencies are declared in [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml):

| Component | Packages |
|---|---|
| **Core Engine** | `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `pyarrow>=14.0` |
| **ML Benchmarks (`[v2]`)** | `scikit-learn>=1.4`, `threadpoolctl>=3.2` |
| **Workbench & Geo UI (`[ui]`)** | `nicegui>=2.0`, `plotly>=5.18`, `folium>=0.16`, `starlette>=0.36`, `uvicorn>=0.27` |
| **Quality & Plots (`[dev]`)** | `httpx>=0.27`, `matplotlib>=3.8`, `ruff>=0.4,<1`, `mypy>=1.8` |

Install the full environment from the repository root:

```bash
pip install -r requirements.txt
pip install -e ".[all]"
```

---

## Raw Dataset Acquisition (`data/raw/`)

Raw upstream SCADA ZIP archives (`~755 MB`) are **not tracked in Git** so the repository remains lightweight and faithful to upstream licensing. The empty directory hierarchy (`data/raw/kelmarsh/` and `data/raw/penmanshiel/`) and a complete file-by-file download/checksum guide are provided in [`data/raw/README.md`](data/raw/README.md).

- **Kelmarsh Wind Farm Data (`v4`)**: <https://doi.org/10.5281/zenodo.16807551> (Cubico Sustainable Investments Ltd, CC BY 4.0)
- **Penmanshiel Wind Farm Data (`v3`)**: <https://doi.org/10.5281/zenodo.16807304> (Cubico Sustainable Investments Ltd, CC BY 4.0)

Download and byte/MD5/SHA-256 verify all 9 raw files automatically via the CLI:

```bash
# Download and verify all Cubico SCADA ZIPs, static coordinate CSVs, and signal mappings
python -m wintellisyt.data.acquire --all

# Verify existing local files against data/reference/DATA_FILE_SHA256.csv
python -m wintellisyt.data.acquire --verify-only

# Fetch cached 2016-2017 hourly ERA5 weather data from Open-Meteo (optional for full density suite)
python scripts/fetch_era5_weather.py
```

---

## Quickstart

Run from the repository root (`$env:PYTHONPATH = "src"` in PowerShell or `export PYTHONPATH=src` in bash):

```powershell
# 1. Generate all 24 multi-farm suite CSVs + manifest.json
python scripts/run_engineering_suite.py artifacts/engineering_suite_2017 --overwrite

# 2. Launch the interactive NiceGUI + Folium workbench (strictly bound to 127.0.0.1 loopback)
python -m wintellisyt.cli ui --suite-dir artifacts/engineering_suite_2017 --host 127.0.0.1 --port 8080
# On Windows, you can also double-click: run_dashboard.bat

# 3. Export a standalone offline interactive HTML dashboard (Folium + Plotly)
python -m wintellisyt.cli ui --suite-dir artifacts/engineering_suite_2017 --export-only

# 4. Run the synthetic smoke demo, linter, static type checker, and 4-layer unit tests
python -m wintellisyt.cli demo --output artifacts/runs/demo
python -m ruff check src tests scripts
python -m mypy src/wintellisyt
python -m unittest discover -s tests -v
```

> **Security & Backend Notes**: The live web dashboard strictly enforces loopback-only binding (`127.0.0.1`, `localhost`, or `::1`) and refuses external network interfaces. When `nicegui` is absent, `--backend auto` falls back seamlessly to the built-in **Starlette + Uvicorn** server serving identical Plotly dashboards and JSON APIs (`/`, `/health`, `/api/manifest`, `/api/tables/{name}`).

---

## Repository Structure

```text
├── src/wintellisyt/          # Modular core library, Parquet store, CLI, Folium geo-map & NiceGUI UI
├── data/
│   ├── raw/                  # Local raw SCADA drop-in folders (kelmarsh/, penmanshiel/) + download guide
│   └── reference/            # Signal inventory, SHA-256 checksums, OEM curves, CC BY 4.0 attribution
├── scripts/                  # End-to-end suite runner, ERA5 fetcher, benchmark & figure scripts
├── tests/                    # 4-layer test suite: physics/, diagnostics/, leakage/, contracts/ + UI workbench
├── docs/                     # Full English methodology, architecture, benchmarks & reproducibility guides
├── .github/workflows/ci.yml  # GitHub Actions CI (Python 3.10 & 3.12: ruff, mypy, unittest, CLI smoke)
├── requirements.txt          # Complete pinned minimum dependencies
├── pyproject.toml            # Packaging metadata, optional extras ([v2], [ui], [dev], [all]), tool config
├── run_dashboard.bat         # Portable Windows localhost launcher for the NiceGUI workbench
├── CITATION.cff              # Software & upstream SCADA dataset citation metadata
└── LICENSE                   # MIT License + third-party data attribution
```

---

## Engineering Documentation

- [Architecture & Engineering Methods](docs/ARCHITECTURE_AND_METHODS.md)
- [Assumptions, Provenance & Limitations](docs/ASSUMPTIONS_AND_LIMITATIONS.md)
- [Signal Inventory & Fleet Layout](docs/SIGNAL_INVENTORY.md)
- [Validation Architecture & Empirical Benchmarks](docs/VALIDATION_AND_BENCHMARKS.md)
- [Actionable Asset Intelligence Report (2017)](docs/ACTIONABLE_INSIGHTS_REPORT.md)
- [Reproducibility & Checksums](docs/REPRODUCIBILITY.md)
- [Raw Dataset Download & Verification Guide](data/raw/README.md)
