# Cubico SCADA Dataset Attribution & Software License

## Underlying SCADA & Static Datasets (`CC BY 4.0`)

- Charlie Plumley and Roberta Takeuchi (2025), *Kelmarsh wind farm data*, v4. [Zenodo Record](https://doi.org/10.5281/zenodo.16807551)
- Charlie Plumley and Roberta Takeuchi (2025), *Penmanshiel wind farm data*, v3. [Zenodo Record](https://doi.org/10.5281/zenodo.16807304)

Both datasets are published by **Cubico Sustainable Investments Ltd** under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license:  
<https://creativecommons.org/licenses/by/4.0/>

The five raw SCADA ZIP archives, two static layout CSVs, and two signal mapping files match the upstream Zenodo release files byte-for-byte. Publisher MD5 checksums and local SHA-256 digests are pinned in [`DATA_FILE_SHA256.csv`](DATA_FILE_SHA256.csv). Operational event logs correspond to the `Status_*` CSV members inside each ZIP archive. [`ARCHIVE_MEMBERS.csv`](ARCHIVE_MEMBERS.csv) indexes archive member locations across telemetry and status files, and [`SCADA_HEADERS.csv`](SCADA_HEADERS.csv) catalogs raw column headers extracted from publisher exports. Derived engineering tables and diagnostics computed by WINTELLISYT preserve provenance links back to these CC BY 4.0 sources.

## Software License (`MIT`)

The WINTELLISYT Python codebase, diagnostics, test suite, and documentation are licensed under the **MIT License** (see [`../../LICENSE`](../../LICENSE)). Atmospheric reanalysis fields used for density benchmarking originate from the Open-Meteo Historical Weather API / Copernicus ERA5.
