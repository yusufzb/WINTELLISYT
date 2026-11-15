"""Apply the HOSE protocol to pinned Cubico data, one turbine at a time."""

import argparse
import csv
import gzip
import io
import json
import subprocess
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import astuple
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

from wintellisyt.data import read_scada
from wintellisyt.operating_state import (
    CADENCE, CATEGORIES, PRESETS, RATED_KW, SITE_MODELS, VERSION, Event, Signals, fuse,
    physical_proxy, retrospective_diagnostic, window,
)
from wintellisyt.pipeline import file_hash

if __package__:
    from .audit_cubico_events import MEMBER, alignment
else:
    from audit_cubico_events import MEMBER, alignment


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


def parse_events(archive, member):
    with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
        utc = False
        for line in stream:
            utc |= line.strip() == "# Time zone: UTC"
            if line.startswith("Timestamp start,"):
                break
        else:
            raise ValueError(f"Missing event header: {member}")
        if not utc:
            raise ValueError(f"Missing UTC declaration: {member}")
        reader = csv.DictReader(stream, fieldnames=next(csv.reader([line])), strict=True)
        catalog, bounded, legacy, unresolved = [], [], [], []
        for index, raw in enumerate(reader):
            if None in raw or None in raw.values():
                raise ValueError(f"Malformed event row: {member} {index}")
            start, end = [None if raw[k] in ("", "-") else datetime.fromisoformat(raw[k])
                          for k in ("Timestamp start", "Timestamp end")]
            if any(d is not None and d.tzinfo is not None for d in (start, end)):
                raise ValueError("Expected naive UTC event time")
            status, category = raw["Status"], raw["IEC category"]
            semantic, cause = CATEGORIES.get(category, ("unmapped", "unknown"))
            issue = "complete"
            if start is None or end is None:
                issue = "missing_endpoint"
            elif end <= start:
                issue = "nonpositive_duration"
            else:
                legacy.append((start, end, status))
                try:
                    h, m, s = map(int, raw["Duration"].split(":"))
                    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
                        raise ValueError("Invalid duration")
                    if (end - start).total_seconds() != 3600 * h + 60 * m + s:
                        raise ValueError("Duration mismatch")
                except ValueError:
                    issue = "contradictory_duration"
            if issue == "complete":
                bounded.append(Event(index, start, end, status, category))
            elif start is not None:
                unresolved.append(start)
            catalog.append({"id": index, **raw, "interval_quality": issue,
                            "semantic_event": "logged_stop" if status == "Stop" else semantic,
                            "category_semantic": semantic, "logged_cause": cause,
                            "mapping_confidence": "literal_logged_category" if category in CATEGORIES else "unmapped",
                            "mapping_id": f"{VERSION}:literal-category", "source_id": member})
        return catalog, sorted(bounded, key=lambda e: e.start), sorted(legacy), sorted(unresolved)


def analyze_turbine(rows, catalog, events, legacy, unresolved, output):
    duplicates = Counter(r.timestamp_utc for r in rows)
    clean = sorted((r for r in rows if duplicates[r.timestamp_utc] == 1), key=lambda r: r.timestamp_utc)
    timestamps = [r.timestamp_utc.replace(tzinfo=None) for r in clean]
    _, left = alignment(timestamps, legacy, -CADENCE)
    _, right = alignment(timestamps, legacy, timedelta())
    counters = defaultdict(Counter)
    episodes = defaultdict(Counter)
    lookup = {e.id: e for e in events}
    canonical = {}
    episode_id = {}
    for e in events:
        raw = catalog[e.id]
        key = (e.start, e.end, e.status, raw["Code"], raw["Message"], e.category)
        episode_id[e.id] = canonical.setdefault(key, e.id)
    year_start = datetime(2017, 1, 1)
    year_end = datetime(2018, 1, 1)
    reference = {i for i in canonical.values()
                 if lookup[i].status == "Stop" and lookup[i].category in CATEGORIES
                 and lookup[i].start < year_end and year_start < lookup[i].end}
    active = []
    index = 0
    with output.open("wb") as raw_output, gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0,
                                                      filename="", compresslevel=3) as compressed:
        for row, stamp in zip(clean, timestamps, strict=True):
            start, end = stamp - CADENCE, stamp + CADENCE
            while index < len(events) and events[index].start < end:
                active.append(events[index])
                index += 1
            active = [e for e in active if start < e.end]
            before, after = window(active, start), window(active, stamp)
            uncertain = bisect_left(unresolved, end) > bisect_left(unresolved, start)
            signals = Signals(row.wind_speed_ms, row.rotor_speed_rpm, row.generator_rpm,
                              row.pitch_deg, row.power_setpoint_kw)
            proxy = physical_proxy(row.farm_id, signals)
            alternate = physical_proxy(row.farm_id, signals, "sensitivity")
            diagnostic = retrospective_diagnostic(proxy, row.active_power_kw, row.wind_speed_ms)
            tier, restriction, cause, flags = fuse(proxy, before, after, lookup, uncertain)
            shared = set(before.full_stop_ids) & set(after.full_stop_ids)
            candidates = {episode_id[i] for i in shared} & reference
            if candidates and not {"concurrent_event_categories", "unresolved_event_start"}.intersection(flags):
                counters["categorized_core_rows"][proxy.regime] += 1
                for category in {lookup[i].category for i in candidates}:
                    counters[f"core_category:{category}"][proxy.regime] += 1
                for i in candidates:
                    episodes[i][proxy.regime] += 1
            elif before.stop_seconds or after.stop_seconds:
                counters["other_stop_rows"][proxy.regime] += 1
            for k, v in (("tiers", tier), ("proxy_regimes", proxy.regime), ("diagnostic_regimes", diagnostic.regime),
                         ("restrictions", restriction), ("causes", cause), ("proxy_rules", proxy.rule)):
                counters[k][v] += 1
            counters["missing"].update(proxy.missing)
            counters["flags"].update(flags)
            counters["sensitivity"]["regime_changed"] += proxy.regime != alternate.regime
            counters["sensitivity"]["restriction_changed"] += proxy.restriction != alternate.restriction
            counters["sensitivity"]["tier_changed"] += tier != fuse(alternate, before, after, lookup, uncertain)[0]
            a, b = bool(before.stop_seconds), bool(after.stop_seconds)
            am = any(lookup[i].status == "Stop" for i in before.midpoint_ids)
            bm = any(lookup[i].status == "Stop" for i in after.midpoint_ids)
            for name, count in (("start", b), ("end", a), ("both", a and b), ("one", a != b),
                                ("start_midpoint", bm), ("end_midpoint", am), ("one_midpoint", am != bm)):
                counters["stop_membership"][name] += count
            record = {
                "farm": row.farm_id, "turbine": row.turbine_id, "timestamp_utc": row.timestamp_utc.isoformat(),
                "config": VERSION, "source_row": row.source_row, "verified_state": None,
                "signals": {**vars(signals), "power_kw": row.active_power_kw},
                "exogenous_operational_proxy": vars(proxy), "retrospective_diagnostic": vars(diagnostic),
                "start_anchor": astuple(after), "end_anchor": astuple(before),
                "anchor_invariant_full_stop": bool(shared), "tier": tier, "restriction": restriction,
                "logged_cause": cause, "flags": flags,
            }
            compressed.write((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode())
    summary = {
        "raw_rows": len(rows), "rows": len(clean), "duplicates_quarantined": len(rows) - len(clean),
        "legacy_stop_disagreement": len(left ^ right),
        "event_quality": dict(Counter(x["interval_quality"] for x in catalog)),
        "category_presence": dict(Counter(x["mapping_confidence"] for x in catalog)),
        "categorized_stop_episodes": len(reference),
        "categorized_stop_episodes_with_core_rows": len(episodes),
        "categorized_stop_episodes_with_standstill_rows": sum(e["standstill_like"] > 0 for e in episodes.values()),
        "categorized_stop_episode_seconds": sum(
            (min(lookup[i].end, year_end) - max(lookup[i].start, year_start)).total_seconds()
            for i in reference
        ),
        "categorized_core_episode_rows": {str(i): dict(v) for i, v in sorted(episodes.items())},
        "categorized_stop_episode_durations_seconds": {
            str(i): (min(lookup[i].end, year_end) - max(lookup[i].start, year_start)).total_seconds()
            for i in sorted(reference)
        },
        "stop_code_category_counts": [
            {"code": code, "category": category, "n": n} for (code, category), n in sorted(Counter(
                (r["Code"], r["IEC category"]) for r in catalog if r["Status"] == "Stop"
            ).items())
        ],
    }
    return {**summary, **{k: dict(v) for k, v in counters.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.benchmark / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["year"] != 2017 or manifest["smoke"] or manifest["rated_kw"] != RATED_KW:
        raise ValueError("Expected the full pinned 2017 benchmark")
    for site, digest in manifest["static_sha256"].items():
        if file_hash(ROOT / "data/raw" / site / f"{site.title()}_WT_static.csv") != digest:
            raise ValueError(f"Static checksum mismatch: {site}")
    args.output.mkdir(parents=True)
    summary = {}
    for site, pin in [*manifest["inputs"]["2017"].items(), ("penmanshiel", manifest["inputs"]["2017_new"])]:
        path = _repo_path(pin["path"])
        if file_hash(path) != pin["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path}")
        with ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                match = MEMBER.fullmatch(Path(member).name)
                if match is None:
                    continue
                farm, turbine = match.groups()
                key = f"{site}/{turbine}"
                if farm.lower() != site or key in summary:
                    raise ValueError(f"Unexpected event member: {member}")
                catalog, events, legacy, unresolved = parse_events(archive, member)
                rows = [r for r in read_scada(path, site, turbine_id=turbine) if r.timestamp_utc.year == 2017]
                prefix = f"{site}-{turbine}"
                result = analyze_turbine(rows, catalog, events, legacy, unresolved,
                                         args.output / f"{prefix}.jsonl.gz")
                (args.output / f"{prefix}-events.json").write_text(json.dumps(catalog), encoding="utf-8")
                summary[key] = {**result, "source_archive": path.name, "source_member": rows[0].source_file,
                                "event_member": member}
                print(key, result["rows"], flush=True)
        if file_hash(path) != pin["sha256"]:
            raise ValueError(f"Archive changed: {path}")
    frozen = json.loads((args.benchmark / "metrics.json").read_text(encoding="utf-8"))
    if summary.keys() != frozen.keys():
        raise ValueError("Frozen target set mismatch")
    payload = {
        "version": VERSION, "evidence": "retrospective_cubico_2017",
        "benchmark_manifest_sha256": file_hash(source),
        "code_commit": _git_commit(),
        "code_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/v2_state_engine.py", "scripts/audit_cubico_events.py",
            "src/wintellisyt/operating_state.py", "src/wintellisyt/data/scada.py", "docs/ARCHITECTURE_AND_METHODS.md",
        )},
        "inputs": manifest["inputs"], "static_sha256": manifest["static_sha256"],
        "preset": PRESETS, "site_models": SITE_MODELS, "rated_kw": RATED_KW,
        "window_field_order": ["event_ids", "union_seconds", "stop_seconds", "midpoint_event_ids", "full_stop_ids"],
        "window_fraction_denominator_seconds": CADENCE.total_seconds(),
        "turbines": summary, "sha256": {p.name: file_hash(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
