"""Count observed canonical SCADA channels in each pinned Cubico archive."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from wintellisyt.pipeline import file_hash, source_manifest
from wintellisyt.data import read_scada

CHANNELS = ("wind_speed_ms", "active_power_kw", "ambient_temperature_c",
            "wind_direction_deg", "rotor_speed_rpm", "pitch_deg", "status_code")


def coverage(path, site, year, digest):
    if file_hash(path) != digest:
        raise ValueError(f"Input checksum differs: {path}")
    totals = Counter()
    valid = Counter()
    for row in read_scada(path, site):
        if row.timestamp_utc.year != year:
            continue
        totals[row.turbine_id] += 1
        for channel in CHANNELS:
            valid[row.turbine_id, channel] += getattr(row, channel) is not None
        valid[row.turbine_id, "wind_and_power"] += (
            row.wind_speed_ms is not None and row.active_power_kw is not None
        )
    if file_hash(path) != digest:
        raise ValueError(f"Input changed during ingestion: {path}")
    return [
        {"site": site, "year_utc": year, "turbine": turbine, "channel": channel,
         "observed_rows": n, "nonmissing_rows": valid[turbine, channel],
         "archive_sha256": digest}
        for turbine, n in sorted(totals.items(), key=lambda item: int(item[0]))
        for channel in (*CHANNELS, "wind_and_power")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    scripts = Path(__file__).resolve().parent
    root = scripts.parent
    pins_2016 = {site: source_manifest()[site]["acquisition"]
                 for site in ("kelmarsh", "penmanshiel")}
    pins_2017 = json.loads((scripts / "transfer_2017_inputs.json").read_text(encoding="utf-8"))
    new = json.loads((scripts / "control_2017_target_input.json").read_text(encoding="utf-8"))
    rows = []
    for year, pins in ((2016, pins_2016), (2017, pins_2017), (2017, {"penmanshiel": new})):
        for site, pin in pins.items():
            raw_path = Path(pin["path"])
            rows.extend(coverage(raw_path if raw_path.is_absolute() else (root / raw_path), site, year, pin["sha256"]))
    if not rows:
        raise ValueError("No pinned archive observations found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
