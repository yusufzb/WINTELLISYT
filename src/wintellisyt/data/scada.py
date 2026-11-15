"""Read provider exports with row-level provenance and multi-channel support."""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

CADENCE_MINUTES = 10
_CUBICO = {"kelmarsh", "penmanshiel"}
_CUBICO_FILENAME = re.compile(
    r"^Turbine_Data_(Kelmarsh|Penmanshiel)_(\d+)_.*\.csv$", re.IGNORECASE
)
_STATUS_FILENAME = re.compile(
    r"^Status_(Kelmarsh|Penmanshiel)_(\d+)_.*\.csv$", re.IGNORECASE
)
_CUBICO_FIELDS = {
    "wind_speed_ms": "Wind speed (m/s)",
    "active_power_kw": "Power (kW)",
    "ambient_temperature_c": "Nacelle ambient temperature (°C)",
    "wind_direction_deg": "Wind direction (°)",
    "rotor_speed_rpm": "Rotor speed (RPM)",
    "pitch_deg": "Blade angle (pitch position) A (°)",
}
_ENGIE_FIELDS = {
    "wind_speed_ms": "Ws_avg",
    "active_power_kw": "P_avg",
    "ambient_temperature_c": "Ot_avg",
    "wind_direction_deg": "Wa_avg",
    "rotor_speed_rpm": None,
    "pitch_deg": "Ba_avg",
}

RICH_CHANNEL_MAP: dict[str, str] = {
    "Wind speed (m/s)": "wind_speed_ms",
    "Wind speed, Standard deviation (m/s)": "wind_speed_std_ms",
    "Wind speed Sensor 1 (m/s)": "wind_speed_s1_ms",
    "Wind speed Sensor 2 (m/s)": "wind_speed_s2_ms",
    "Density adjusted wind speed (m/s)": "density_adjusted_ws_ms",
    "Wind direction (°)": "wind_direction_deg",
    "Nacelle position (°)": "nacelle_position_deg",
    "Vane position 1+2 (°)": "vane_position_deg",
    "Vane position 1+2, StdDev (°)": "vane_position_std_deg",
    "Cable windings from calibration point": "cable_windings",
    "Power (kW)": "active_power_kw",
    "Potential power default PC (kW)": "potential_power_default_kw",
    "Potential power MPC (kW)": "potential_power_mpc_kw",
    "Potential power learned PC (kW)": "potential_power_learned_kw",
    "Potential power reference turbines (kW)": "potential_power_ref_kw",
    "Potential power primary reference turbines (kW)": "potential_power_primary_ref_kw",
    "Potential power secondary reference turbines (kW)": "potential_power_secondary_ref_kw",
    "Cascading potential power (kW)": "cascading_potential_power_kw",
    "Cascading potential power for performance (kW)": "cascading_potential_power_perf_kw",
    "Available Capacity for Production (kW)": "available_capacity_kw",
    "Energy Theoretical (kWh)": "energy_theoretical_kwh",
    "Energy Export (kWh)": "energy_export_kwh",
    "Energy Import (kWh)": "energy_import_kwh",
    "Lost Production (Contractual) (kWh)": "lost_contractual_kwh",
    "Lost Production (Time-based IEC B.2.2) (kWh)": "lost_time_iec_b22_kwh",
    "Lost Production (Production-based IEC B.2.2) (kWh)": "lost_prod_iec_b22_kwh",
    "Lost Production to Downtime (kWh)": "lost_downtime_kwh",
    "Lost Production to Performance (kWh)": "lost_performance_kwh",
    "Lost Production to Curtailment (Total) (kWh)": "lost_curtailment_total_kwh",
    "Lost Production to Curtailment (Grid) (kWh)": "lost_curtailment_grid_kwh",
    "Lost Production to Curtailment (Noise) (kWh)": "lost_curtailment_noise_kwh",
    "Lost Production to Curtailment (Shadow) (kWh)": "lost_curtailment_shadow_kwh",
    "Lost Production to Curtailment (Bats) (kWh)": "lost_curtailment_bats_kwh",
    "Lost Production to Curtailment (Birds) (kWh)": "lost_curtailment_birds_kwh",
    "Lost Production to Curtailment (Ice) (kWh)": "lost_curtailment_ice_kwh",
    "Lost Production to Curtailment (Sector Management) (kWh)": "lost_curtailment_sector_kwh",
    "Lost Production to Curtailment (Technical) (kWh)": "lost_curtailment_technical_kwh",
    "Lost Production to Curtailment (Marketing) (kWh)": "lost_curtailment_marketing_kwh",
    "Lost Production to Curtailment (Boat Action) (kWh)": "lost_curtailment_boat_kwh",
    "Lost Production to Curtailment (Grid Constraint) (kWh)": "lost_curtailment_grid_constraint_kwh",
    "Performance Index": "performance_index",
    "Production Factor": "production_factor",
    "Rotor speed (RPM)": "rotor_speed_rpm",
    "Generator RPM (RPM)": "generator_rpm",
    "Blade angle (pitch position) A (°)": "pitch_a_deg",
    "Blade angle (pitch position) B (°)": "pitch_b_deg",
    "Blade angle (pitch position) C (°)": "pitch_c_deg",
    "Motor current axis 1 (A)": "motor_current_1_a",
    "Motor current axis 2 (A)": "motor_current_2_a",
    "Motor current axis 3 (A)": "motor_current_3_a",
    "Temperature motor axis 1 (°C)": "motor_temp_1_c",
    "Temperature motor axis 2 (°C)": "motor_temp_2_c",
    "Temperature motor axis 3 (°C)": "motor_temp_3_c",
    "Drive train acceleration (mm/ss)": "drivetrain_accel_mmss",
    "Tower Acceleration X (mm/ss)": "tower_accel_x_mmss",
    "Tower Acceleration y (mm/ss)": "tower_accel_y_mmss",
    "Gear oil inlet pressure (bar)": "gear_oil_inlet_bar",
    "Gear oil pump pressure (bar)": "gear_oil_pump_bar",
    "Metal particle count": "metal_particle_count",
    "Nacelle ambient temperature (°C)": "ambient_temperature_c",
    "Rotor bearing temp (°C)": "rotor_bearing_temp_c",
    "Front bearing temperature (°C)": "front_bearing_temp_c",
    "Rear bearing temperature (°C)": "rear_bearing_temp_c",
    "Gear oil temperature (°C)": "gear_oil_temp_c",
    "Generator bearing front temperature (°C)": "gen_bearing_front_temp_c",
    "Generator bearing rear temperature (°C)": "gen_bearing_rear_temp_c",
    "Stator temperature 1 (°C)": "stator_temp_1_c",
    "Transformer temperature (°C)": "transformer_temp_c",
    "Data Availability": "data_availability",
    "Time-based Contractual Avail.": "avail_time_contractual",
    "Time-based IEC B.2.2 (Users View)": "avail_time_iec_b22",
    "Time-based IEC B.2.3 (Users View)": "avail_time_iec_b23",
    "Time-based IEC B.2.4 (Users View)": "avail_time_iec_b24",
    "Time-based IEC B.3.2 (Manufacturers View)": "avail_time_iec_b32",
    "Production-based IEC B.2.2 (Users View)": "avail_prod_iec_b22",
    "Production-based IEC B.2.3 (Users View)": "avail_prod_iec_b23",
    "Production-based IEC B.3.2 (Manufacturers View)": "avail_prod_iec_b32",
    "Time-based System Avail.": "avail_time_system",
    "Production-based System Avail.": "avail_prod_system",
}


@dataclass(frozen=True, slots=True)
class ScadaRecord:
    """A source observation, including nullable channels and original location."""

    farm_id: str
    turbine_id: str
    timestamp_utc: datetime
    wind_speed_ms: float | None
    active_power_kw: float | None
    ambient_temperature_c: float | None
    wind_direction_deg: float | None
    rotor_speed_rpm: float | None
    pitch_deg: float | None
    status_code: str | None
    source_file: str
    source_row: int
    quality_flags: tuple[str, ...] = ()
    operational_state: str = "unknown"
    wind_speed_std_ms: float | None = None
    generator_rpm: float | None = None
    power_setpoint_kw: float | None = None
    wind_speed_s1_ms: float | None = None
    wind_speed_s2_ms: float | None = None
    density_adjusted_ws_ms: float | None = None
    nacelle_position_deg: float | None = None
    vane_position_deg: float | None = None
    pitch_b_deg: float | None = None
    pitch_c_deg: float | None = None


def load_static_layout(farm_id: str, root: str | Path = ".") -> dict[str, dict]:
    """Load turbine static metadata (coordinates, rotor, hub height, elevation)."""
    farm = farm_id.lower()
    if farm not in _CUBICO:
        raise ValueError(f"Unsupported static layout farm: {farm_id}")
    path = Path(root) / "data" / "raw" / farm / f"{farm.title()}_WT_static.csv"
    layout: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            title = (row.get("Title") or "").strip()
            if not title:
                continue
            turbine_id = title.split()[-1]
            layout[turbine_id] = {
                "farm": farm,
                "turbine_id": turbine_id,
                "title": title,
                "model": (row.get("Model") or "").strip(),
                "rated_power_kw": float(row["Rated power (kW)"]),
                "hub_height_m": float(row["Hub Height (m)"]),
                "rotor_diameter_m": float(row["Rotor Diameter (m)"]),
                "latitude": float(row["Latitude"]),
                "longitude": float(row["Longitude"]),
                "elevation_m": float(row["Elevation (m)"]),
            }
    return layout


def _parse_greenbyte_header(stream: io.TextIOWrapper, source: str) -> tuple[list[str], int]:
    """Skip Greenbyte preamble comments and return normalized column names + data start row."""
    utc_declared = False
    for idx, line in enumerate(stream):
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if body.lower().startswith("time zone:") and body.partition(":")[2].strip().upper() == "UTC":
                utc_declared = True
            if body.startswith("Date and time,"):
                if not utc_declared:
                    raise ValueError(f"{source}: expected Greenbyte UTC preamble")
                header = [name.strip() for name in next(csv.reader([body]))]
                if len(header) != len(set(header)):
                    raise ValueError(f"{source}: duplicate header columns")
                return header, idx + 1
        elif stripped:
            raise ValueError(f"{source}: unexpected non-comment row before header")
    raise ValueError(f"{source}: missing '# Date and time,' header")


def load_rich_telemetry(
    paths: str | Path | Sequence[str | Path],
    farm_id: str,
    *,
    year: int | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load multi-channel Cubico telemetry from one or more ZIPs into a deduplicated DataFrame.

    Returns `(df, quarantined_duplicate_rows)`. Duplicate `(timestamp_utc, turbine_id)` keys are
    quarantined (dropped completely so neither copy can bias statistics).
    """
    farm = farm_id.lower()
    if farm not in _CUBICO:
        raise ValueError(f"load_rich_telemetry supports Cubico farms only, got {farm_id}")
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    selected_source_cols = list(columns) if columns is not None else list(RICH_CHANNEL_MAP.keys())

    frames: list[pd.DataFrame] = []
    for archive_path in path_list:
        with ZipFile(archive_path) as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if (match := _CUBICO_FILENAME.fullmatch(Path(info.filename).name))
                    and match[1].lower() == farm
                ),
                key=lambda info: info.filename,
            )
            for info in members:
                match = _CUBICO_FILENAME.fullmatch(Path(info.filename).name)
                assert match is not None
                turbine_id = match[2]
                source_label = f"{archive_path.name}!{info.filename}"
                with io.TextIOWrapper(archive.open(info), encoding="utf-8-sig", newline="") as stream:
                    header, skip_rows = _parse_greenbyte_header(stream, source_label)
                available = [col for col in selected_source_cols if col in header]
                usecols = ["Date and time", *available]
                with io.TextIOWrapper(archive.open(info), encoding="utf-8-sig", newline="") as stream:
                    chunk = pd.read_csv(
                        stream,
                        skiprows=skip_rows,
                        names=header,
                        usecols=usecols,
                        na_values=["", "NA", "NaN", "nan", "NAN", "null", "NULL"],
                        low_memory=False,
                    )
                chunk["timestamp_utc"] = pd.to_datetime(
                    chunk.pop("Date and time"), format="%Y-%m-%d %H:%M:%S", utc=True
                )
                if year is not None:
                    chunk = chunk[chunk["timestamp_utc"].dt.year == year]
                if chunk.empty:
                    continue
                rename_map = {src: RICH_CHANNEL_MAP.get(src, src) for src in available}
                chunk = chunk.rename(columns=rename_map)
                for canonical in set(rename_map.values()):
                    chunk[canonical] = pd.to_numeric(chunk[canonical], errors="coerce")
                chunk["farm_id"] = farm
                chunk["turbine_id"] = turbine_id
                frames.append(chunk)

    if not frames:
        return pd.DataFrame(), 0
    combined = pd.concat(frames, ignore_index=True)
    dup_mask = combined.duplicated(subset=["timestamp_utc", "turbine_id"], keep=False)
    quarantined = int(dup_mask.sum())
    if quarantined:
        combined = combined[~dup_mask].copy()
    combined = combined.sort_values(["turbine_id", "timestamp_utc"]).reset_index(drop=True)
    return combined, quarantined


def read_status_events(
    paths: str | Path | Sequence[str | Path],
    farm_id: str,
    *,
    year: int | None = None,
) -> pd.DataFrame:
    """Read and deduplicate Greenbyte Status_*.csv event logs from Cubico ZIP archives."""
    farm = farm_id.lower()
    if farm not in _CUBICO:
        raise ValueError(f"read_status_events supports Cubico farms only, got {farm_id}")
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    records: list[dict] = []

    for archive_path in path_list:
        with ZipFile(archive_path) as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if (match := _STATUS_FILENAME.fullmatch(Path(info.filename).name))
                    and match[1].lower() == farm
                ),
                key=lambda info: info.filename,
            )
            for info in members:
                match = _STATUS_FILENAME.fullmatch(Path(info.filename).name)
                assert match is not None
                turbine_id = match[2]
                with io.TextIOWrapper(archive.open(info), encoding="utf-8-sig", newline="") as stream:
                    lines = [line for line in stream if line.strip() and not line.lstrip().startswith("#")]
                if not lines:
                    continue
                for row in csv.DictReader(lines):
                    records.append(
                        {
                            "farm_id": farm,
                            "turbine_id": turbine_id,
                            "timestamp_start_raw": (row.get("Timestamp start") or "").strip(),
                            "timestamp_end_raw": (row.get("Timestamp end") or "").strip(),
                            "duration_raw": (row.get("Duration") or "").strip(),
                            "status": (row.get("Status") or "").strip(),
                            "code": (row.get("Code") or "").strip(),
                            "message": (row.get("Message") or "").strip(),
                            "comment": (row.get("Comment") or "").strip(),
                            "service_contract_category": (row.get("Service contract category") or "").strip(),
                            "iec_category": (row.get("IEC category") or "").strip(),
                        }
                    )

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp_start"] = pd.to_datetime(
        df.pop("timestamp_start_raw"), utc=True, errors="coerce", format="mixed"
    )
    df["timestamp_end"] = pd.to_datetime(
        df.pop("timestamp_end_raw"), utc=True, errors="coerce", format="mixed"
    )
    df["open_ended"] = df["timestamp_end"].isna() | (
        df["timestamp_start"].notna() & (df["timestamp_end"] <= df["timestamp_start"])
    )
    df = df.drop_duplicates(
        subset=["farm_id", "turbine_id", "timestamp_start", "timestamp_end", "status", "code", "message", "iec_category"]
    ).reset_index(drop=True)
    if year is not None:
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        df["left_censored"] = df["timestamp_start"].notna() & (df["timestamp_start"] < year_start)
        keep = (
            (df["timestamp_start"].notna() & (df["timestamp_start"] < year_end))
            & (df["timestamp_end"].isna() | (df["timestamp_end"] > year_start))
        )
        df = df[keep].reset_index(drop=True)
    else:
        df["left_censored"] = False
    return df


def read_scada(
    path: str | Path, farm_id: str, *, turbine_id: str | None = None
) -> Iterator[ScadaRecord]:
    """Yield native CSV rows or Cubico ZIP member rows, in source order.

    In ZIP files, turbine_id selects a provider turbine number. For Cubico
    CSVs with renamed files, supply turbine_id to recover the lost identifier.
    For ENGIE, turbine_id filters the provider identifier in each CSV row.
    """
    path = Path(path)
    if farm_id not in _CUBICO and farm_id != "la_haute_borne":
        raise ValueError(f"Unsupported SCADA farm: {farm_id}")
    if path.suffix.lower() == ".zip":
        if farm_id not in _CUBICO:
            raise ValueError("ZIP ingestion supports Cubico turbine archives")
        with ZipFile(path) as archive:
            members = [
                info for info in archive.infolist()
                if (match := _CUBICO_FILENAME.fullmatch(Path(info.filename).name))
                and match[1].lower() == farm_id
                and (turbine_id is None or match[2] == turbine_id)
            ]
            if not members:
                raise ValueError(f"{path}: no matching {farm_id} turbine CSVs")
            for info in members:
                with io.TextIOWrapper(
                    archive.open(info), encoding="utf-8-sig", newline=""
                ) as stream:
                    yield from _read_csv(
                        stream, f"{path}!{info.filename}", farm_id, turbine_id
                    )
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from _read_csv(stream, str(path), farm_id, turbine_id)


def _read_csv(
    lines: Iterable[str], source: str, farm: str, turbine_id: str | None
) -> Iterator[ScadaRecord]:
    cubico = farm in _CUBICO
    if cubico:
        filename = Path(source.rsplit("!", 1)[-1]).name
        match = _CUBICO_FILENAME.fullmatch(filename)
        if match:
            if match[1].lower() != farm or turbine_id not in (None, match[2]):
                raise ValueError(f"{source}: supplied identity conflicts with filename")
            turbine_id = match[2]
        if not turbine_id:
            raise ValueError(f"{source}: Cubico CSV requires its original filename or turbine_id")
    time_field = "Date and time" if cubico else "Date_time"
    fields = _CUBICO_FIELDS if cubico else _ENGIE_FIELDS
    lines = iter(lines)
    for header_line, line in enumerate(lines, start=1):
        header = line.strip().removeprefix("#").lstrip()
        if cubico and header.lower().startswith("time zone:"):
            if header.partition(":")[2].strip().upper() != "UTC":
                raise ValueError(f"{source}:{header_line}: expected Cubico UTC export")
        if header.startswith(f"{time_field},") and cubico:
            delimiter = ","
            break
        if not cubico and header and not line.startswith("#"):
            delimiter = ";" if header.count(";") > header.count(",") else ","
            break
        if line.strip() and not line.lstrip().startswith("#"):
            raise ValueError(f"{source}:{header_line}: expected {time_field} header")
    else:
        raise ValueError(f"{source}: missing {time_field} header")

    column_names = [name.strip() for name in next(csv.reader([header], delimiter=delimiter))]
    required = {time_field}
    for channel in ("wind_speed_ms", "active_power_kw"):
        column = fields[channel]
        if column is None:
            raise ValueError(f"{farm}: required channel {channel} is unmapped")
        required.add(column)
    if not cubico:
        required.add("Wind_turbine_name")
    missing = required.difference(column_names)
    if missing or len(column_names) != len(set(column_names)):
        raise ValueError(f"{source}: missing columns {sorted(missing)} or duplicate header")

    reader = csv.reader(lines, delimiter=delimiter, strict=True)
    for values in reader:
        if not values:
            continue
        row_number = header_line + reader.line_num
        location = f"{source}:{row_number}"
        if len(values) != len(column_names):
            raise ValueError(f"{location}: expected {len(column_names)} fields, got {len(values)}")
        row = dict(zip(column_names, values, strict=True))
        row_turbine = turbine_id if cubico else row["Wind_turbine_name"].strip()
        if not row_turbine:
            raise ValueError(f"{location}: missing turbine identifier")
        if not cubico and turbine_id is not None and row_turbine != turbine_id:
            continue
        timestamp = _timestamp(row[time_field], cubico, location)
        flags: list[str] = []
        numeric_fields = {
            field: _numeric(row.get(column) if column else None, field, flags, delimiter == ";")
            for field, column in fields.items()
        }
        yield ScadaRecord(
            farm_id=farm,
            turbine_id=row_turbine,
            timestamp_utc=timestamp,
            status_code=None,
            source_file=source,
            source_row=row_number,
            quality_flags=tuple(flags),
            wind_speed_ms=numeric_fields["wind_speed_ms"],
            active_power_kw=numeric_fields["active_power_kw"],
            ambient_temperature_c=numeric_fields["ambient_temperature_c"],
            wind_direction_deg=numeric_fields["wind_direction_deg"],
            rotor_speed_rpm=numeric_fields["rotor_speed_rpm"],
            pitch_deg=numeric_fields["pitch_deg"],
            generator_rpm=_numeric(
                row.get("Generator RPM (RPM)") if cubico else None,
                "generator_rpm", [], False,
            ),
            power_setpoint_kw=_numeric(
                row.get("Turbine Power setpoint (kW)") if cubico else None,
                "power_setpoint_kw", [], False,
            ),
            wind_speed_std_ms=_numeric(
                row.get("Wind speed, Standard deviation (m/s)") if cubico else None,
                "wind_speed_std_ms", [], False,
            ),
            wind_speed_s1_ms=_numeric(
                row.get("Wind speed Sensor 1 (m/s)") if cubico else None,
                "wind_speed_s1_ms", [], False,
            ),
            wind_speed_s2_ms=_numeric(
                row.get("Wind speed Sensor 2 (m/s)") if cubico else None,
                "wind_speed_s2_ms", [], False,
            ),
            density_adjusted_ws_ms=_numeric(
                row.get("Density adjusted wind speed (m/s)") if cubico else None,
                "density_adjusted_ws_ms", [], False,
            ),
            nacelle_position_deg=_numeric(
                row.get("Nacelle position (°)") if cubico else None,
                "nacelle_position_deg", [], False,
            ),
            vane_position_deg=_numeric(
                row.get("Vane position 1+2 (°)") if cubico else None,
                "vane_position_deg", [], False,
            ),
            pitch_b_deg=_numeric(
                row.get("Blade angle (pitch position) B (°)") if cubico else None,
                "pitch_b_deg", [], False,
            ),
            pitch_c_deg=_numeric(
                row.get("Blade angle (pitch position) C (°)") if cubico else None,
                "pitch_c_deg", [], False,
            ),
        )


def _timestamp(value: str, cubico: bool, location: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{location}: invalid timestamp {value!r}") from exc
    if timestamp.tzinfo is None:
        if not cubico:
            raise ValueError(f"{location}: ENGIE timestamp requires an explicit UTC offset")
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _numeric(
    value: str | None, field: str, flags: list[str], decimal_comma: bool
) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "na", "nan", "null"}:
        flags.append(f"missing_{field}")
        return None
    try:
        number = float(value.replace(",", ".") if decimal_comma else value)
        if math.isfinite(number):
            return number
    except ValueError:
        pass
    flags.append(f"invalid_{field}")
    return None
