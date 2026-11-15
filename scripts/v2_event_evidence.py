"""Audit event evidence on the pinned V2 2017 archives without assigning states."""

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from zipfile import ZipFile

from wintellisyt.pipeline import file_hash, source_manifest

if __package__:
    from .audit_cubico_events import CADENCE, MEMBER, alignment, inspect_events, stop_sensitivity
    from .v2_benchmark import archive_groups
else:
    from audit_cubico_events import CADENCE, MEMBER, alignment, inspect_events, stop_sensitivity
    from v2_benchmark import archive_groups


def audit_pin(pin, site, year):
    groups = archive_groups(pin, site, year)
    records = {}
    with ZipFile(pin["path"]) as archive:
        for member in archive.namelist():
            match = MEMBER.fullmatch(Path(member).name)
            if match is None:
                continue
            farm, turbine = match.groups()
            if farm.lower() != site or turbine in records or turbine not in groups:
                raise ValueError(f"Unexpected event member: {member}")
            events, open_stops, counts, codes = inspect_events(archive, member)
            times = Counter(r.timestamp_utc.replace(tzinfo=None) for r in groups[turbine])
            timestamps = sorted(t for t, count in times.items() if count == 1)
            start_counts, start_stops = alignment(timestamps, events, timedelta())
            end_counts, end_stops = alignment(timestamps, events, -CADENCE)
            disagreement = start_stops ^ end_stops
            records[turbine] = {
                "year": year, "member": member, "raw_scada_rows": sum(times.values()),
                "duplicate_rows_quarantined": sum(times.values()) - len(timestamps),
                "unique_scada_rows": len(timestamps), "event_counts": counts,
                "scada_status_code_nonmissing": sum(row.status_code is not None for row in groups[turbine]),
                "verified_state_rows": 0, "unknown_state_rows": len(timestamps),
                "unknown_fraction": 1.0 if timestamps else None,
                "timestamp_at_start": start_counts, "timestamp_at_end": end_counts,
                "stop_overlap_both_anchors": len(start_stops & end_stops),
                "stop_overlap_anchor_disagreement": len(disagreement),
                "stop_disagreement_first_utc": min(disagreement).isoformat() + "Z" if disagreement else None,
                "stop_disagreement_last_utc": max(disagreement).isoformat() + "Z" if disagreement else None,
                "open_stop_persistence_sensitivity": stop_sensitivity(
                    timestamps, events, open_stops, start_stops, end_stops
                ),
                "event_codes": [{"status": status, "code": code, "category": category, "rows": n}
                                for (status, code, category), n in sorted(codes.items())],
            }
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(pin["path"])
    pin_path = raw_path if raw_path.is_absolute() else (root / raw_path)
    if records.keys() != groups.keys() or file_hash(pin_path) != pin["sha256"]:
        raise ValueError(f"SCADA/status member or archive mismatch for {site}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    source = args.benchmark / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["year"] != 2017 or manifest["smoke"]:
        raise ValueError("Expected the full 2017 benchmark")
    inputs = manifest["inputs"]
    sites = source_manifest()
    records = {}
    for site, pin in inputs["2017"].items():
        records[site] = audit_pin(pin, site, 2017)
    extra = audit_pin(inputs["2017_new"], "penmanshiel", 2017)
    if records["penmanshiel"].keys() & extra.keys():
        raise ValueError("Overlapping Penmanshiel archive coverage")
    records["penmanshiel"].update(extra)
    payload = {
        "year": 2017, "class": "retrospective_event_overlap_candidates",
        "state": "unknown", "reason": "Unresolved timestamp anchor and state reference",
        "policies": ["half_open_timestamp_at_start", "half_open_timestamp_at_end"],
        "benchmark_manifest_sha256": file_hash(source),
        "script_sha256": file_hash(Path(__file__)),
        "pins": {site: {k: sites[site][k] for k in ("source_url", "license", "provider", "creators")}
                 for site in records},
        "archives": {"2017": inputs["2017"], "2017_new": inputs["2017_new"]},
        "sites": records,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
