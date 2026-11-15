# Raw Wind Farm SCADA Datasets (`data/raw/`)

Heavy upstream SCADA archives (`*.zip`) and vendor raw tables are **intentionally excluded from Git version control** (see `.gitignore`) so the repository stays lightweight, fast to clone, and strictly faithful to upstream data provenance.

This folder preserves the exact target directory layout expected by WINTELLISYT:

```text
data/raw/
├── README.md
├── kelmarsh/
│   ├── Kelmarsh_SCADA_2016_3082.zip
│   ├── Kelmarsh_SCADA_2017_3083.zip
│   ├── Kelmarsh_WT_dataSignalMapping.csv
│   └── Kelmarsh_WT_static.csv
└── penmanshiel/
    ├── Penmanshiel_SCADA_2016_WT11-15_3107.zip
    ├── Penmanshiel_SCADA_2017_WT01-10_3114.zip
    ├── Penmanshiel_SCADA_2017_WT11-15_3115.zip
    ├── Penmanshiel_WT_dataSignalMapping.xlsx
    └── Penmanshiel_WT_static.csv
```

---

## 1. Upstream Data Sources & Licensing (`CC BY 4.0`)

Both wind farm datasets are openly published on Zenodo by **Cubico Sustainable Investments Ltd** (Charlie Plumley and Roberta Takeuchi, 2025) under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license:

| Wind Farm | Turbine Fleet | Upstream Record (DOI) | Version | License |
|---|---|---|---|---|
| **Kelmarsh Wind Farm** | 6 x Senvion MM92 (`2,050 kW` rated) | <https://doi.org/10.5281/zenodo.16807551> | `v4` | CC BY 4.0 |
| **Penmanshiel Wind Farm** | 14 x Senvion MM82 (`2,050 kW` rated, excluding WT03) | <https://doi.org/10.5281/zenodo.16807304> | `v3` | CC BY 4.0 |

> **Atmospheric Reanalysis (`data/external/era5/`)**: Hourly boundary-layer air density fields (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `pressure_msl`, `wind_direction_100m`, `wind_speed_100m`) are fetched directly from the **Open-Meteo Historical Weather API / Copernicus ERA5** using `python scripts/fetch_era5_weather.py` once the static coordinate CSVs are present.

---

## 2. Automated Download & Checksum Verification

Run the built-in acquisition module from the repository root (`$env:PYTHONPATH="src"` on Windows or `PYTHONPATH=src` on Linux/macOS):

```bash
# Download and byte/MD5/SHA-256 verify all 9 upstream files (~755 MB total)
python -m wintellisyt.data.acquire --all

# Verify already-downloaded local files against data/reference/DATA_FILE_SHA256.csv
python -m wintellisyt.data.acquire --verify-only

# Download only the primary 2016 archive for a single site (source_manifest.json)
python -m wintellisyt.data.acquire kelmarsh
python -m wintellisyt.data.acquire penmanshiel

# Optionally download cached hourly ERA5 weather JSON files
python scripts/fetch_era5_weather.py
```

---

## 3. Manual Download Links & SHA-256 / MD5 Checksum Manifest

All digests below are enforced against [`data/reference/DATA_FILE_SHA256.csv`](../reference/DATA_FILE_SHA256.csv):

### Kelmarsh (`https://zenodo.org/records/16807551`)

| Target Relative Path | Size (Bytes / MB) | Publisher MD5 | SHA-256 Digest | Direct Download URL |
|---|---|---|---|---|
| `data/raw/kelmarsh/Kelmarsh_SCADA_2016_3082.zip` | `97,915,553` (93.4 MB) | `ca8efbb6dbcfae3516583937eb5b5402` | `4e8a73aeb823aeb21a2c7829f14434e44eea523f1d7f9be0177ff1b76e3ded27` | [Download ZIP](https://zenodo.org/records/16807551/files/Kelmarsh_SCADA_2016_3082.zip) |
| `data/raw/kelmarsh/Kelmarsh_SCADA_2017_3083.zip` | `174,604,722` (166.5 MB) | `c78263ee52ee0e48e2cb4bbaa1ba211a` | `528c942caec09bf02a25651cb13b6ecb13d4a09d287b2bb546f10adc614a14e3` | [Download ZIP](https://zenodo.org/records/16807551/files/Kelmarsh_SCADA_2017_3083.zip) |
| `data/raw/kelmarsh/Kelmarsh_WT_dataSignalMapping.csv` | `3,150` (3.1 KB) | `ca8fe399ab15ae7111d3214c632efd36` | `2dad8552ba4d45d59b9a778c68025b8c3614179d48dfb938c3af07c90ebd06af` | [Download CSV](https://zenodo.org/records/16807551/files/Kelmarsh_WT_dataSignalMapping.csv) |
| `data/raw/kelmarsh/Kelmarsh_WT_static.csv` | `807` (0.8 KB) | `af3a038f0f7fddfc1608ad0bfc8cf5ba` | `a249919471580244d41a547e407ef371d45f2d5347a59bc23cf3c980f607ee93` | [Download CSV](https://zenodo.org/records/16807551/files/Kelmarsh_WT_static.csv) |

### Penmanshiel (`https://zenodo.org/records/16807304`)

| Target Relative Path | Size (Bytes / MB) | Publisher MD5 | SHA-256 Digest | Direct Download URL |
|---|---|---|---|---|
| `data/raw/penmanshiel/Penmanshiel_SCADA_2016_WT11-15_3107.zip` | `55,917,173` (53.3 MB) | `92c30453d43a38e985af3e2e305eedd4` | `1d1f36873afe0cbd9ffaee3c2b8fd92d44905fd1282c1e615bda9e18a63fd30f` | [Download ZIP](https://zenodo.org/records/16807304/files/Penmanshiel_SCADA_2016_WT11-15_3107.zip) |
| `data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT01-10_3114.zip` | `273,531,678` (260.9 MB) | `c8279e34a5e33d688e2219aa9abab74e` | `2c08ee7566a4dc37b8e42e83f8324a543bdf47cbf484a8e40b7ab3eec035fbcb` | [Download ZIP](https://zenodo.org/records/16807304/files/Penmanshiel_SCADA_2017_WT01-10_3114.zip) |
| `data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT11-15_3115.zip` | `152,896,438` (145.8 MB) | `d41874e3d195e88506949492b5758cf6` | `2da262bbc55aef030cfdcd2fa11b6a664bcaaafdf916364e9c4c26fdd6adfb7c` | [Download ZIP](https://zenodo.org/records/16807304/files/Penmanshiel_SCADA_2017_WT11-15_3115.zip) |
| `data/raw/penmanshiel/Penmanshiel_WT_dataSignalMapping.xlsx` | `19,359` (19.4 KB) | `2eae1b84a014619268a6e4f53e6b1c91` | `428392004a391bc3646e415dbd1cd5c723f6499d0f3c9ea2697736eeb3c59c63` | [Download XLSX](https://zenodo.org/records/16807304/files/Penmanshiel_WT_dataSignalMapping.xlsx) |
| `data/raw/penmanshiel/Penmanshiel_WT_static.csv` | `2,029` (2.0 KB) | `c4cd4191234c1a67a391fe5d2978256b` | `2a742dfd5c8486bf54c1afb647bd803954f4c56aea31635e5aa93828f56ebd7b` | [Download CSV](https://zenodo.org/records/16807304/files/Penmanshiel_WT_static.csv) |
