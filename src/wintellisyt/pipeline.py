"""Normalize, audit and write a versioned set of SCADA artifacts."""

import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files

from . import __version__
from .uncertainty import block_mae

SCHEMA_VERSION = 1
CADENCE_MINUTES = 10
STATES = {"normal", "curtailed", "stopped", "faulted", "unknown"}


@dataclass(frozen=True)
class Observation:
    timestamp_utc: datetime
    turbine_id: str
    wind_speed_ms: float | None
    active_power_kw: float | None
    operational_state: str
    quality_flags: tuple[str, ...] = ()
    source_row: int = 0
    source_file: str = ""


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else _stream_hash(stream)


def _stream_hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    return json.loads(
        files("wintellisyt.data").joinpath("source_manifest.json").read_text(encoding="utf-8")
    )


def load_rows(input_path, source, turbine=None):
    if source == "synthetic":
        with input_path.open(newline="", encoding="utf-8") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if turbine is None or row["turbine_id"] == turbine:
                    timestamp = datetime.fromisoformat(row["timestamp_utc"])
                    if timestamp.tzinfo is None or row["state"] not in STATES:
                        raise ValueError(f"Invalid synthetic timestamp or state at row {line}")
                    yield Observation(
                        timestamp.astimezone(timezone.utc),
                        row["turbine_id"],
                        float(row["wind_speed_ms"]) if row["wind_speed_ms"] else None,
                        float(row["active_power_kw"]) if row["active_power_kw"] else None,
                        row["state"],
                        source_row=line,
                        source_file=input_path.name,
                    )
        return

    from .data import read_scada

    for record in read_scada(input_path, source, turbine_id=turbine):
        yield Observation(
            record.timestamp_utc, record.turbine_id, record.wind_speed_ms, record.active_power_kw,
            record.operational_state, record.quality_flags, record.source_row, record.source_file,
        )


def assess(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.turbine_id, row.timestamp_utc].append(row)

    clean = []
    times = defaultdict(list)
    duplicates = []
    for (turbine, timestamp), copies in sorted(groups.items()):
        times[turbine].append(timestamp)
        if len(copies) == 1:
            clean.append(copies[0])
            continue
        # Full source-row equivalence is unavailable once values are normalized.
        # Quarantine the entire timestamp key so no choice can bias a baseline.
        duplicates.append({
            "turbine_id": turbine,
            "timestamp_utc": timestamp.isoformat(),
            "source_rows": [
                {"file": row.source_file, "row": row.source_row} for row in copies
            ],
        })

    cadence = timedelta(minutes=CADENCE_MINUTES)
    gaps = []
    off_cadence_intervals = 0
    for turbine, timestamps in times.items():
        for previous, current in zip(timestamps, timestamps[1:]):
            delta = current - previous
            off_cadence_intervals += int(delta % cadence != timedelta())
            if delta > cadence:
                gaps.append({
                    "turbine_id": turbine,
                    "after": previous.isoformat(),
                    "before": current.isoformat(),
                    "unobserved_expected_slots": (delta - timedelta(microseconds=1)) // cadence,
                })

    return clean, {
        "rows": len(rows),
        "unique_turbine_timestamps": len(groups),
        "rows_after_duplicate_quarantine": len(clean),
        "duplicate_groups": duplicates,
        "quarantined_rows": sum(len(group["source_rows"]) for group in duplicates),
        "state_counts": dict(Counter(row.operational_state for row in rows)),
        "quality_flags": dict(Counter(flag for row in rows for flag in row.quality_flags)),
        "missing_wind_speed_rows": sum(row.wind_speed_ms is None for row in rows),
        "missing_active_power_rows": sum(row.active_power_kw is None for row in rows),
        "gaps": gaps,
        "cadence_minutes": CADENCE_MINUTES,
        "off_cadence_intervals": off_cadence_intervals,
        "coverage": {
            turbine: {
                "first_utc": timestamps[0].isoformat(),
                "last_utc": timestamps[-1].isoformat(),
                "observed_unique_timestamps": len(timestamps),
            }
            for turbine, timestamps in times.items()
        },
    }


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def curve_input(rows):
    return (
        {"wind_speed_mps": row.wind_speed_ms, "power_kw": row.active_power_kw,
         "operational_state": row.operational_state}
        for row in rows
    )


def run(input_path, output, source, turbine=None, min_bin_count=5):
    from .power_curve import fit_power_curve

    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if min_bin_count < 1:
        raise ValueError("Minimum bin count must be at least 1")
    digest = file_hash(input_path)
    source_info = None if source == "synthetic" else source_manifest()[source]
    if source_info is None:
        evidence = "synthetic"
    elif source_info["acquisition"].get("sha256") == digest:
        evidence = "verified_real"
    else:
        evidence = "unverified_input"
    rows = list(load_rows(input_path, source, turbine))
    if not rows:
        raise ValueError("No observations match the source and turbine selection")
    if file_hash(input_path) != digest:
        raise ValueError("Input changed during ingestion")
    clean, qa = assess(rows)
    turbine_groups = defaultdict(list)
    clean_groups = defaultdict(list)
    for row in rows:
        turbine_groups[row.turbine_id].append(row)
    for row in clean:
        clean_groups[row.turbine_id].append(row)

    bins: list[dict[str, object]] = []
    normal_status = {}
    for turbine_id, observations in sorted(turbine_groups.items()):
        observed = fit_power_curve(curve_input(observations), min_bin_count=min_bin_count)
        screened = fit_power_curve(curve_input(clean_groups[turbine_id]), min_bin_count=min_bin_count)
        normal_status[turbine_id] = screened.normal_status
        for stratum, stats in (("raw", observed.raw_bins), ("screened", screened.raw_bins),
                               ("normal", screened.normal_bins)):
            bins.extend(
                {"turbine_id": turbine_id, "stratum": stratum, **asdict(item)} for item in stats
            )

    from .evaluation import evaluate

    metrics, residuals = evaluate(clean, min_bin_count=min_bin_count)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "qa.json", qa)
    write_json(output / "metrics.json", metrics)
    write_json(output / "uncertainty.json", block_mae(residuals))
    write_csv(output / "bins.csv",
              ("turbine_id", "stratum", "index", "lower_mps", "upper_mps", "count",
               "mean_power_kw", "supported"), bins)
    write_csv(output / "observations.csv",
              ("timestamp_utc", "turbine_id", "wind_speed_ms", "active_power_kw",
               "operational_state", "quality_flags", "source_file", "source_row"),
              (asdict(row) for row in rows))
    write_csv(output / "residuals.csv",
              ("timestamp_utc", "turbine_id", "operational_state", "wind_speed_ms",
               "actual_kw", "predicted_kw", "residual_kw"), residuals)
    from .reporting import write_report

    write_report(output, source=source, evidence=evidence)
    hashes = {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_evidence": evidence,
        "source_provenance": source_info,
        "input_file": input_path.name,
        "input_sha256": digest,
        "parameters": {"turbine": turbine, "min_bin_count": min_bin_count},
        "normal_status": normal_status,
        "artifact_sha256": hashes,
    })
    return evidence
