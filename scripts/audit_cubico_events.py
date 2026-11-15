"""Audit event exports and candidate SCADA interval alignment on pinned archives."""

import argparse
import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from heapq import heappop, heappush
from pathlib import Path
from zipfile import ZipFile

from wintellisyt.data import read_scada
from wintellisyt.data.acquire import acquire_archive
from wintellisyt.pipeline import file_hash, source_manifest

MEMBER = re.compile(r"Status_(Kelmarsh|Penmanshiel)_(\d+)_.*\.csv")
CADENCE = timedelta(minutes=10)


def inspect_events(archive, member):
    with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
        utc = False
        for line in stream:
            if line.strip() == "# Time zone: UTC":
                utc = True
            if line.startswith("Timestamp start,"):
                break
        else:
            raise ValueError(f"{member}: missing event header")
        if not utc:
            raise ValueError(f"{member}: expected UTC event export")
        reader = csv.DictReader(stream, fieldnames=next(csv.reader([line])), strict=True)
        counts = Counter()
        codes = Counter()
        seen = set()
        events = []
        open_stops = []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{member}: malformed event row {reader.line_num}")
            counts["rows"] += 1
            key = tuple(row.values())
            counts["exact_repeat_rows"] += key in seen
            seen.add(key)
            start_text, end_text = row["Timestamp start"], row["Timestamp end"]
            status, code, category = row["Status"], row["Code"], row["IEC category"]
            codes[status, code, category] += 1
            if start_text in {"", "-"} or end_text in {"", "-"}:
                counts["missing_endpoint_rows"] += 1
                counts[f"missing_endpoint_status_{status}"] += 1
                if end_text in {"", "-"} and start_text not in {"", "-"}:
                    counts["start_only_rows"] += 1
                    if status == "Stop":
                        start = datetime.fromisoformat(start_text)
                        if start.tzinfo is not None:
                            raise ValueError(f"{member}: expected naive UTC start")
                        open_stops.append((start, datetime.max, status))
                continue
            start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
            if start.tzinfo is not None or end.tzinfo is not None:
                raise ValueError(f"{member}: expected naive UTC endpoints")
            if end <= start:
                counts["zero_length_rows" if end == start else "reversed_rows"] += 1
                continue
            hours, minutes, seconds = map(int, row["Duration"].split(":"))
            counts["duration_mismatch_rows"] += end - start != timedelta(
                hours=hours, minutes=minutes, seconds=seconds
            )
            events.append((start, end, status))
        return sorted(events), open_stops, dict(counts), codes


def alignment(times, events, offset):
    """Count event presence in half-open candidate windows in a forward sweep."""
    active = []
    statuses = Counter()
    index = 0
    counts = Counter()
    stop_times = set()
    for timestamp in sorted(times):
        start = timestamp + offset
        end = start + CADENCE
        while index < len(events) and events[index][0] < end:
            _, event_end, status = events[index]
            heappush(active, (event_end, status))
            statuses[status] += 1
            index += 1
        while active and active[0][0] <= start:
            _, status = heappop(active)
            statuses[status] -= 1
        counts["any_event"] += bool(active)
        if statuses["Stop"] > 0:
            counts["stop_event"] += 1
            stop_times.add(timestamp)
        counts["multiple_events"] += len(active) > 1
        counts["stop_and_other"] += 0 < statuses["Stop"] < len(active)
    return dict(counts), stop_times


def stop_sensitivity(timestamps, events, open_stops, start_stops, end_stops):
    """Bound additional overlap if each recorded open Stop persists."""
    if open_stops:
        persistent = sorted(events + open_stops)
        _, expanded_start = alignment(timestamps, persistent, timedelta())
        _, expanded_end = alignment(timestamps, persistent, -CADENCE)
    else:
        expanded_start, expanded_end = start_stops, end_stops
    return {
        "additional_timestamp_at_start": len(expanded_start - start_stops),
        "additional_timestamp_at_end": len(expanded_end - end_stops),
        "additional_under_either_anchor": len(
            (expanded_start | expanded_end) - (start_stops | end_stops)
        ),
    }


def audit(source):
    provenance = source_manifest()[source]
    path = acquire_archive(source)
    times = defaultdict(list)
    for record in read_scada(path, source):
        times[record.turbine_id].append(record.timestamp_utc.replace(tzinfo=None))
    turbines = {}
    code_counts = Counter()
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            match = MEMBER.fullmatch(Path(member).name)
            if not match:
                continue
            farm, turbine = match.groups()
            if farm.lower() != source or turbine in turbines or turbine not in times:
                raise ValueError(f"Unexpected event member: {member}")
            events, open_stops, counts, codes = inspect_events(archive, member)
            code_counts.update(codes)
            timestamps = times[turbine]
            start_counts, start_stops = alignment(timestamps, events, timedelta())
            end_counts, end_stops = alignment(timestamps, events, -CADENCE)
            turbines[turbine] = {
                "member": member,
                "event_counts": counts,
                "scada_rows": len(timestamps),
                "first_event_start_utc": events[0][0].isoformat() if events else None,
                "last_event_end_utc": max(event[1] for event in events).isoformat() if events else None,
                "timestamp_at_start": start_counts,
                "timestamp_at_end": end_counts,
                "stop_overlap_both_conventions": len(start_stops & end_stops),
                "stop_overlap_conventions_disagree": len(start_stops ^ end_stops),
                "open_stop_persistence_sensitivity": stop_sensitivity(
                    timestamps, events, open_stops, start_stops, end_stops
                ),
            }
    if turbines.keys() != times.keys():
        raise ValueError("SCADA and status turbines differ")
    return {
        "source": source,
        "source_url": provenance["source_url"],
        "license": provenance["license"],
        "creators": provenance["creators"],
        "provider": provenance["provider"],
        "sha256": file_hash(path),
        "timezone": "UTC",
        "method": "Positive overlap with candidate half-open 10-minute SCADA windows",
        "turbines": turbines,
        "event_codes": [
            {"status": status, "code": code, "iec_category": category, "rows": count}
            for (status, code, category), count in sorted(code_counts.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = {source: audit(source) for source in ("kelmarsh", "penmanshiel")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
