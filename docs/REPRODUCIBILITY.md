# Reproducibility & Verification Guide

## 1. Environment Requirements & Package Dependencies

WINTELLISYT runs on Python `>=3.10` with pinned scientific, geospatial UI, and static-check dependencies defined in [`requirements.txt`](../requirements.txt) and [`pyproject.toml`](../pyproject.toml):

- **Core Computational Engine**: `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `pyarrow>=14.0`
- **Machine Learning Benchmarks (`[v2]`)**: `scikit-learn>=1.4`, `threadpoolctl>=3.2`
- **Interactive Workbench & Geospatial UI (`[ui]`)**: `nicegui>=2.0`, `plotly>=5.18`, `folium>=0.16`, `starlette>=0.36`, `uvicorn>=0.27`
- **Quality Gates & Figures (`[dev]`)**: `httpx>=0.27`, `matplotlib>=3.8`, `ruff>=0.4,<1`, `mypy>=1.8`

Install everything from the repository root:

```bash
pip install -r requirements.txt
pip install -e ".[all]"
```

- **UI Backends**: With `nicegui` and `folium` installed, `wintellisyt ui` serves the reactive 3-workspace NiceGUI `@ui.page("/")` web dashboard on local loopback (`127.0.0.1` / `localhost`) by default (`--backend auto` / `--backend nicegui`), with built-in Starlette + Uvicorn (`--backend starlette`) and standalone self-contained HTML export (`--export-only`) modes.
- **Full Suite & Static Typing**: With `[all]` extras installed, all `77/77` unit tests run with `0` skips and `mypy src/wintellisyt` verifies the package cleanly.

---

## 2. Raw SCADA & ERA5 Integrity Pins (`data/reference/DATA_FILE_SHA256.csv`)

Raw Cubico SCADA ZIP archives and static layout files are excluded from Git tracking and reside under `data/raw/`. See [`data/raw/README.md`](../data/raw/README.md) for full Zenodo download URLs, expected sizes, MD5 digests, and SHA-256 verification tables:

- **Kelmarsh v4** (`https://doi.org/10.5281/zenodo.16807551`): `Kelmarsh_SCADA_2016_3082.zip`, `Kelmarsh_SCADA_2017_3083.zip`, `Kelmarsh_WT_dataSignalMapping.csv`, `Kelmarsh_WT_static.csv`.
- **Penmanshiel v3** (`https://doi.org/10.5281/zenodo.16807304`): `Penmanshiel_SCADA_2016_WT11-15_3107.zip`, `Penmanshiel_SCADA_2017_WT01-10_3114.zip`, `Penmanshiel_SCADA_2017_WT11-15_3115.zip`, `Penmanshiel_WT_dataSignalMapping.xlsx`, `Penmanshiel_WT_static.csv`.

Automated raw acquisition and checksum verification from the repository root:

```bash
# Download and SHA-256/MD5 verify all 9 raw files into data/raw/
python -m wintellisyt.data.acquire --all

# Verify already-downloaded files against data/reference/DATA_FILE_SHA256.csv
python -m wintellisyt.data.acquire --verify-only

# Fetch 2016-2017 hourly ERA5 atmospheric reanalysis from Open-Meteo (requires static CSVs)
python scripts/fetch_era5_weather.py
```

Cached Open-Meteo ERA5 weather files are written to `data/external/era5/{kelmarsh,penmanshiel}.json` with recorded manifest SHA-256 digests:
- Kelmarsh ERA5 SHA-256: `31229ad395fafd9c0fef5df2b9366abccd572b1bab614b9e3fc59d3c3a717672`
- Penmanshiel ERA5 SHA-256: `a9294de5d5989d2baa068bf9d597256bc5ea384085eacddd47a371a5ab2e9e7b`

---

## 3. Exact Commands to Reproduce Suite, UI, and Tests

Run all commands from the repository root:

### A. Run the Full Multi-Farm Engineering & Asset Intelligence Suite (`24 CSVs + manifest.json`)
```powershell
$env:PYTHONPATH = "src"
python scripts/run_engineering_suite.py artifacts/engineering_suite_2017 --overwrite
```

### B. Export Standalone Interactive HTML Dashboard or Serve Live Localhost UI
```powershell
$env:PYTHONPATH = "src"
# Offline self-contained Folium + Plotly HTML dashboard:
python -m wintellisyt.cli ui --suite-dir artifacts/engineering_suite_2017 --export-only

# Live interactive web server on loopback (Windows users can also double-click run_dashboard.bat):
python -m wintellisyt.cli ui --suite-dir artifacts/engineering_suite_2017 --host 127.0.0.1 --port 8080
```

### C. Synthetic Smoke Demo & Source Metadata Check
```powershell
$env:PYTHONPATH = "src"
python -m wintellisyt.cli sources
python -m wintellisyt.cli demo --output artifacts/runs/demo
```

### D. Static Lint, Type Check & Four-Layer Test Verification
```powershell
$env:PYTHONPATH = "src"
python -m ruff check src tests scripts
python -m mypy src/wintellisyt
python -m unittest discover -s tests -v
```
