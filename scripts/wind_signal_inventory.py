"""Audit exact Cubico export headers and values from the five pinned archives."""

import argparse
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from wintellisyt.pipeline import file_hash

ROOT = Path(__file__).resolve().parents[1]
STATIC_PINS = {
    "kelmarsh": "a249919471580244d41a547e407ef371d45f2d5347a59bc23cf3c980f607ee93",
    "penmanshiel": "2a742dfd5c8486bf54c1afb647bd803954f4c56aea31635e5aa93828f56ebd7b",
}
MAP_PINS = {
    "kelmarsh": (
        "Kelmarsh_WT_dataSignalMapping.csv",
        "2dad8552ba4d45d59b9a778c68025b8c3614179d48dfb938c3af07c90ebd06af",
    ),
    "penmanshiel": (
        "Penmanshiel_WT_dataSignalMapping.xlsx",
        "428392004a391bc3646e415dbd1cd5c723f6499d0f3c9ea2697736eeb3c59c63",
    ),
}
CANONICAL = {
    "Wind speed (m/s)": "reported_ws",
    "Wind speed, Standard deviation (m/s)": "wind_speed_std_ms",
    "Wind speed Sensor 1 (m/s)": "wind_speed_sensor_1_ms",
    "Wind speed Sensor 2 (m/s)": "wind_speed_sensor_2_ms",
    "Density adjusted wind speed (m/s)": "publisher_density_adjusted_ws_ms",
    "Wind direction (°)": "reported_wind_direction_deg",
    "Nacelle position (°)": "reported_nacelle_position_deg",
    "Vane position 1+2 (°)": "reported_combined_vane_deg",
    "Power (kW)": "active_power_kw",
    "Nacelle ambient temperature (°C)": "ambient_temperature_c",
    "Rotor speed (RPM)": "rotor_speed_rpm",
    "Generator RPM (RPM)": "generator_rpm",
    **{
        f"Blade angle (pitch position) {blade} (°)": f"blade_{blade.lower()}_pitch_deg"
        for blade in "ABC"
    },
    "Energy Export (kWh)": "publisher_energy_export_kwh",
    "Energy Import (kWh)": "publisher_energy_import_kwh",
    "Turbine Power setpoint (kW)": "turbine_power_setpoint_kw",
}
AGGREGATES = {
    "standard deviation": "standard_deviation", "stddev": "standard_deviation",
    "min": "minimum", "minimum": "minimum", "max": "maximum", "maximum": "maximum",
}
XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
TIMESTAMP = "# Date and time"


def require_hash(path: Path, sha256: str) -> None:
    if file_hash(path) != sha256:
        raise ValueError(f"{path}: SHA-256 mismatch")


def mapping(farm: str) -> tuple[dict[str, tuple[str, str, str]], str, str]:
    name, sha256 = MAP_PINS[farm]
    path = ROOT / "data/raw" / farm / name
    require_hash(path, sha256)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            rows = list(reader)
    else:
        rows = []
        with ZipFile(path) as workbook:
            strings = [
                "".join(text.text or "" for text in cell.iter(f"{XML}t"))
                for cell in ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            ]
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        for worksheet_row in list(sheet.iter(f"{XML}row"))[1:]:
            cells = {}
            for cell in worksheet_row:
                value = cell.findtext(f"{XML}v") or ""
                cells[cell.attrib["r"][0]] = strings[int(value)] if cell.get("t") == "s" else value
            rows.append([cells.get(column, "") for column in "ABCD"])
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        if len(row) >= 4 and row[0].strip().isdigit() and row[1].strip():
            groups.setdefault(row[1].strip().casefold(), []).append((row[0], row[2], row[3]))
    unique = {title: items[0] for title, items in groups.items() if len(set(items)) == 1}
    return unique, name, sha256


def geometry(farm: str) -> dict[int, dict[str, str]]:
    path = ROOT / "data/raw" / farm / f"{farm.title()}_WT_static.csv"
    require_hash(path, STATIC_PINS[farm])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            int(row["Title"].split()[-1]): row
            for row in csv.DictReader(stream) if row["Title"]
        }


def pins() -> Iterator[tuple[str, int, dict]]:
    manifest = json.loads((ROOT / "src/wintellisyt/data/source_manifest.json").read_text("utf-8"))
    later = json.loads((ROOT / "scripts/transfer_2017_inputs.json").read_text("utf-8"))
    extra = json.loads((ROOT / "scripts/control_2017_target_input.json").read_text("utf-8"))
    for farm in ("kelmarsh", "penmanshiel"):
        yield farm, 2016, manifest[farm]["acquisition"]
        yield farm, 2017, later[farm]
    yield "penmanshiel", 2017, extra


def parse_header(stream: io.TextIOWrapper) -> list[str]:
    utc = False
    for line in stream:
        if line.rstrip() == "# Time zone: UTC":
            utc = True
        if line.startswith(f"{TIMESTAMP},") and utc:
            header = next(csv.reader([line]))
            if len(header) != len(set(header)):
                raise ValueError("Duplicate field names")
            return header
        if not line.startswith("#"):
            break
    raise ValueError("Expected Greenbyte UTC preamble and a SCADA timestamp header")


def metadata(source: str, tags: dict, hub_height: str) -> dict:
    title, unit = source, ""
    if match := re.fullmatch(r"(.+) \(([^()]*)\)", source):
        title, unit = match.groups()
    base, _, suffix = title.rpartition(", ")
    statistic = AGGREGATES.get(suffix.lower(), "unknown")
    if statistic != "unknown":
        title = base
    tag, description, tag_unit = tags.get(title.casefold(), ("", "", ""))
    canonical = CANONICAL.get(source, "")
    placement = "unknown"
    correction = "undocumented"
    processing = "unknown"
    if source.startswith("Nacelle ambient temperature"):
        placement = f"nacelle exterior per map/title; hub height {hub_height} m; sensor height unknown"
    if source.startswith(("Wind speed", "Density adjusted wind speed")):
        placement = f"wind channel; turbine hub height {hub_height} m; sensor mount/height unverified"
        correction = "NTF, fallback and filtering unverified"
    if statistic != "unknown":
        processing = "named statistic; aggregation producer unverified"
    if source.startswith("Density adjusted wind speed"):
        processing = "adjusted per export label"
        correction = "density adjustment named; density source/formula, NTF unverified"
    if source in ("Power (kW)", "Wind speed (m/s)"):
        statistic = "nominal mean; exact aggregation unverified"
    if title.startswith(("Wind direction", "Nacelle position", "Vane position")):
        correction = "angular convention/calibration undocumented"
    if source.startswith(("Energy", "Lost Production")):
        statistic = "counter" if "counter" in source.casefold() else "energy; support unverified"
    return {
        "canonical_name": canonical, "physical_meaning": description or title,
        "unit_in_header": unit, "statistic": statistic, "greenbyte_signal_id": tag,
        "unit_in_mapping": tag_unit, "sensor_placement_height": placement,
        "processing": processing, "correction_status": correction,
        "semantics_basis": "publisher mapping + export header" if tag else "export header only",
        "dependencies": "read SIGNAL_INVENTORY.md",
    }


def summarize(
    stream: io.TextIOWrapper, header: list[str], year: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    chunks = pd.read_csv(
        stream, names=header, header=None, dtype=str, na_filter=False,
        chunksize=4096, index_col=False,
    )
    timestamps = []
    arrays = []
    blank = np.zeros(len(header) - 1, dtype=np.int64)
    sentinel = blank.copy()
    infinite = blank.copy()
    invalid = blank.copy()
    for chunk in chunks:
        timestamps.extend(chunk.pop(TIMESTAMP))
        raw = chunk.to_numpy(dtype=object, copy=True)
        blank_mask = raw == ""
        blank += np.count_nonzero(blank_mask, axis=0)
        sentinel_mask = (raw == "NaN") | (raw == "nan") | (raw == "NAN")
        sentinel += np.count_nonzero(sentinel_mask, axis=0)
        raw[blank_mask] = "NaN"
        try:
            numbers = raw.astype(np.float64)
        except ValueError:
            numbers = chunk.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        infinite += np.count_nonzero(np.isinf(numbers), axis=0)
        invalid += np.count_nonzero(np.isnan(numbers) & ~blank_mask & ~sentinel_mask, axis=0)
        arrays.append(numbers)
    values = np.concatenate(arrays, axis=0)
    time_index = pd.to_datetime(
        timestamps, format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise"
    )
    delta_ns = np.diff(time_index.as_unit("ns").asi8)
    step = 600_000_000_000
    if set(time_index.year) != {year}:
        raise ValueError(f"Expected only {year} UTC timestamps")
    contiguous = np.concatenate(([False], delta_ns == step))
    finite = np.isfinite(values)
    bounds = np.where(finite, values, np.nan)
    minimum = np.fmin.reduce(bounds, axis=0)
    maximum = np.fmax.reduce(bounds, axis=0)
    repeats = np.zeros(len(header)-1, dtype=np.int64)
    runs = repeats.copy()
    for idx, column in enumerate(bounds.T):
        repeated = contiguous[1:] & (column[1:] == column[:-1])
        repeats[idx] = repeated.sum()
        starts = np.flatnonzero(np.concatenate(([True], ~repeated, [True])))
        runs[idx] = np.diff(starts).max() if finite[:, idx].any() else 0
    time_qa = {
        "first_utc": time_index.min().isoformat(), "last_utc": time_index.max().isoformat(),
        "rows": len(time_index), "duplicate_keys": int(time_index.duplicated().sum()),
        "backward_steps": int((delta_ns < 0).sum()),
        "gaps_gt_10_min": int((delta_ns > step).sum()),
        "off_cadence_positive_steps": int(((delta_ns > 0) & (delta_ns != step)).sum()),
    }
    counts = {
        "finite_count": finite.sum(axis=0), "blank_count": blank, "nan_token_count": sentinel,
        "infinite_count": infinite, "invalid_token_count": invalid, "min": minimum,
        "max": maximum, "equal_previous_10min_count": repeats, "longest_equal_run_rows": runs,
    }
    if not np.all(sum(counts[key] for key in tuple(counts)[:5]) == len(time_index)):
        raise ValueError("Signal counts do not partition source rows")
    return time_qa, counts


def inventory() -> Iterator[dict]:
    tags = {farm: mapping(farm) for farm in MAP_PINS}
    statics = {farm: geometry(farm) for farm in MAP_PINS}
    for farm, year, pin in pins():
        path = ROOT / pin["path"]
        require_hash(path, pin["sha256"])
        if path.stat().st_size != pin["bytes"]:
            raise ValueError(f"{path}: byte count mismatch")
        signal_tags, mapping_file, mapping_hash = tags[farm]
        with ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if Path(n).name.startswith("Turbine_Data_"))
            for member in members:
                turbine = int(Path(member).name.split("_")[3])
                with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
                    header = parse_header(stream)
                    time_qa, stats = summarize(stream, header, year)
                print(f"{farm} {year} WT{turbine:02} {time_qa}", flush=True)
                common = {
                    "farm": farm, "year": year, "turbine": f"{turbine:02}",
                    "archive": pin["path"], "archive_sha256": pin["sha256"],
                    "archive_url": pin["url"], "member": member,
                    "signal_mapping_file": mapping_file, "signal_mapping_sha256": mapping_hash,
                    "time_support": "10-minute UTC grid; anchor unverified", **time_qa,
                }
                yield {
                    **common, "source_header": TIMESTAMP, "canonical_name": "timestamp_utc",
                    "column": 1, "physical_meaning": "SCADA time label", "analysis_eligibility": "key",
                }
                for i, source in enumerate(header[1:]):
                    support = {key: values[i].item() for key, values in stats.items()}
                    if not support["finite_count"]:
                        support.update(min="", max="")
                    info = metadata(source, signal_tags, statics[farm][turbine]["Hub Height (m)"])
                    if not support["finite_count"]:
                        eligibility = "unavailable"
                    elif info["canonical_name"]:
                        eligibility = "candidate; see channel-specific caveats"
                    else:
                        eligibility = "inventory only; semantic/use review required"
                    yield {
                        **common, "source_header": source, "column": i + 2, **info, **support,
                        "analysis_eligibility": eligibility,
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    rows = list(inventory())
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if len(Counter((r["farm"], r["year"], r["turbine"], r["column"]) for r in rows)) != len(rows):
        raise ValueError("Duplicate inventory identity")
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} header entries, SHA-256 {file_hash(args.output)}")


if __name__ == "__main__":
    main()
