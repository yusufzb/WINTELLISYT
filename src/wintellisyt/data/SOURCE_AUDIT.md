# SCADA source audit

## Coverage and attribution

The audited input is one publisher archive per Cubico site, with no telemetry
sampling within the retrieved archives. Both raw ZIPs were fetched on
2026-11-14 UTC, matched their publisher-listed MD5, and had their SHA-256
computed locally. The paths, URLs, version pins and hashes are in
[the source manifest](source_manifest.json). Local copies are read-only and
excluded from Git under `data/raw/`. Acquisition verifies existing files and
rejects a byte count or digest mismatch.

| Source | Primary release evidence | Empirical input |
| --- | --- | --- |
| Kelmarsh | Charlie Plumley and Roberta Takeuchi, Cubico Sustainable Investments Ltd, v4, CC BY 4.0. The six Senvion MM92 turbines have a 12.3 MW aggregate nameplate and publisher coverage from 2016 through 2024. [Publisher deposit](https://doi.org/10.5281/zenodo.16807551) | 2016 archive, turbines 1–6, 314,496 SCADA rows. |
| Penmanshiel | Charlie Plumley and Roberta Takeuchi, Cubico Sustainable Investments Ltd, v3, CC BY 4.0. The 14 Senvion MM82 turbines have a 28.7 MW aggregate nameplate. Publisher coverage runs from 2016 through 2024, with variable turbine and signal coverage and no WT03. [Publisher deposit](https://doi.org/10.5281/zenodo.16807304) | 2016 archive, turbines 11–15, 126,402 SCADA rows. |
| La Haute Borne | ENGIE Green identifies its open data portal as the publisher channel. The historical catalog and records API returned curl exit 6 (DNS resolution failure), including a direct check outside the sandbox. The applicable license and release version could not be confirmed from publisher records. [ENGIE Green brochure](https://www.engie-green.fr/wp-content/uploads/2020/12/brochure-bu-2020-engie-france-renouvelables-vfr.pdf), [publisher catalog](https://opendata-renewables.engie.com/explore/dataset/la-haute-borne-data-2013-2016/) | Acquisition blocked. No raw data, SHA-256, or empirical coverage. The schema adapter has synthetic tests based on historical export documentation. [OpenOA QA example](https://openoa.readthedocs.io/en/develop/examples/00_intro_to_plant_data.html) |

Cubico redistributions must follow the license, including attribution and
indicating changes. Wintellisyt maps channels to a common schema and parses
missing values. Use the version-specific DOI and credit the deposit creators
and Cubico Sustainable Investments Ltd when citing data-derived outputs.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Timestamp and status

The retrieved Greenbyte telemetry and status preambles explicitly declare
`UTC`. Row timestamps are timezone-naive strings. The adapter attaches UTC and
preserves source order and duplicates. It rejects a conflicting timezone
declaration. The 10-minute cadence is publisher-documented and observed in
both archives. The interval start or end convention for SCADA labels remains
unverified. Penmanshiel turbine availability varies within the overall
export date range declared by the preamble. ENGIE's historical offsets
`+01:00` and `+02:00` disambiguate repeated local clock times. The adapter
requires an explicit offset and converts it to UTC.
[OpenOA ENGIE offset audit](https://openoa.readthedocs.io/en/develop/examples/00_intro_to_plant_data.html)

Cubico includes separate `Status_*.csv` members with event start, end, duration,
status, code, message, comment, and service contract and IEC categories.
The retrieved logs contain Stop and Warning records as well as informational
events, multiple event categories, and blank IEC categories. The adapter
leaves these logs separate. It emits null `status_code` and
`operational_state="unknown"` for SCADA rows. The current evidence supports an
observed electrical power curve with unknown status, including rows that may
reflect restrictions or outages. A labeled curve requires an explicit
interval overlap rule, precedence and coverage audits for event codes, and
separate operation strata. ENGIE status evidence remains unavailable.

## Source field mapping

| Canonical field | Cubico Greenbyte headers in both ZIPs | Historical ENGIE headers documented by OpenOA |
| --- | --- | --- |
| `turbine_id` | Publisher turbine number in telemetry member name | `Wind_turbine_name` |
| `timestamp_utc` | `# Date and time` in declared UTC | `Date_time` with offset |
| `wind_speed_ms` | `Wind speed (m/s)` | `Ws_avg` |
| `active_power_kw` | `Power (kW)` | `P_avg` |
| `ambient_temperature_c` | `Nacelle ambient temperature (°C)` | `Ot_avg` |
| `wind_direction_deg` | `Wind direction (°)` | `Wa_avg`, absolute wind direction |
| `rotor_speed_rpm` | `Rotor speed (RPM)` | Unavailable in this mapping |
| `pitch_deg` | `Blade angle (pitch position) A (°)` | `Ba_avg`, as exported |

The units require no scaling on these fields. The publisher's NaN sentinel
represents missing or erroneous data. The adapter preserves it as null and
adds a missing-field flag. Unmapped optional fields stay null. The source
location is a path, optionally with `!member`, plus the one-based physical
line number. ENGIE's mapping is provisional until publisher material can be
retrieved and licensed. [OpenOA plant schema](https://openoa.readthedocs.io/en/latest/api/schema.html)

## Retrieved archives

All eleven turbine streams were strictly increasing with a 10-minute
interval between consecutive UTC keys. Within each observed turbine span,
there were zero repeated timestamp keys, zero reversed timestamps, and zero
gaps in the timestamp grid. Many timestamps have missing channels. Zero
timestamp gaps therefore does not establish full measurement coverage.

| Site and turbine | First UTC row (2016) | Raw rows | Finite wind and power pair |
| --- | --- | ---: | ---: |
| Kelmarsh 1 | Jan 03 00:00 | 52,416 | 48,485 |
| Kelmarsh 2 | Jan 03 00:00 | 52,416 | 48,483 |
| Kelmarsh 3 | Jan 03 00:00 | 52,416 | 47,229 |
| Kelmarsh 4 | Jan 03 00:00 | 52,416 | 46,882 |
| Kelmarsh 5 | Jan 03 00:00 | 52,416 | 47,755 |
| Kelmarsh 6 | Jan 03 00:00 | 52,416 | 46,596 |
| Penmanshiel 11 | Jul 19 09:40 | 23,846 | 19,286 |
| Penmanshiel 12 | Jul 02 16:20 | 26,254 | 19,354 |
| Penmanshiel 13 | Jun 30 14:40 | 26,552 | 19,038 |
| Penmanshiel 14 | Jul 09 16:20 | 25,246 | 18,554 |
| Penmanshiel 15 | Jul 14 20:00 | 24,504 | 19,175 |

Every turbine's last row is 2016-12-31 23:50 UTC. All 126,402 Penmanshiel
rows have null blade-A pitch. A finite pair is a completeness measure,
before physical screening or state selection. The audit used
`read_scada(path, farm_id)` over each full ZIP and counted timestamps and
non-null wind and power values by turbine.

## Reproduce acquisition

```shell
python -m wintellisyt.data.acquire kelmarsh
python -m wintellisyt.data.acquire penmanshiel
```

Once publisher access is restored, obtain the ENGIE export and current
release terms from the catalog, record the exact file and SHA-256, and check
the historical schema and timezone assumptions against the retrieved bytes.
