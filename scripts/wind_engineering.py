"""Run frozen 2017 directional and power-curve engineering diagnostics."""

import argparse
import csv
import gzip
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from wintellisyt.data import read_scada
from wintellisyt.pc_health import monitor
from wintellisyt.pipeline import file_hash
from wintellisyt.wind_geometry import LayoutPoint, pair_vector, source_bearing

ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


CUTOFF = pd.Timestamp("2017-10-01", tz="UTC")
TEST = pd.Timestamp("2017-11-01", tz="UTC")
SIGNALS = ["reported_ws", "power_kw", "direction", "sigma", "pitch", "ambient_c"]
HOSE = ROOT / "artifacts/runs/v2-hose-2017-rc"


def wrap(angle):
    return (angle + 180) % 360 - 180


def load_site(site, pins, static_digest):
    path = ROOT / "data/raw" / site / f"{site.title()}_WT_static.csv"
    if file_hash(path) != static_digest:
        raise ValueError(f"Static hash mismatch: {site}")
    layout = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if not row["Title"].strip():
            continue
        key = row["Title"].split()[-1]
        if key in layout or float(row["Rated power (kW)"]) != 2050:
            raise ValueError(f"Invalid turbine metadata: {key}")
        layout[key] = LayoutPoint(float(row["Latitude"]), float(row["Longitude"]), float(row["Rotor Diameter (m)"]))
    records = []
    for pin in pins:
        path = _repo_path(pin["path"])
        if file_hash(path) != pin["sha256"]:
            raise ValueError(f"Archive hash mismatch: {path.name}")
        records.extend((r.timestamp_utc, r.turbine_id, r.wind_speed_ms, r.active_power_kw,
                        r.wind_direction_deg, r.wind_speed_std_ms, r.pitch_deg, r.ambient_temperature_c)
                       for r in read_scada(path, site) if r.timestamp_utc.year == 2017)
    frame = pd.DataFrame(records, columns=["utc", "turbine", *SIGNALS])
    duplicates = int(frame.duplicated(["utc", "turbine"], keep=False).sum())
    frame = frame.drop_duplicates(["utc", "turbine"], keep=False).set_index(["utc", "turbine"]).sort_index()
    if set(frame.index.get_level_values("turbine")) != layout.keys():
        raise ValueError("Layout and SCADA turbine IDs differ")
    wide = frame.unstack()
    wide["reported_ws"] = wide["reported_ws"].where(wide["reported_ws"] >= 0)
    wide["direction"] = wide["direction"].where(wide["direction"].ge(0) & wide["direction"].le(360)) % 360
    return wide, layout, duplicates


def direction_model(wide, layout):
    direction = wide["direction"]
    ids = list(direction)
    vectors, pair_info = {}, []
    for i, target in enumerate(ids):
        for j, donor in enumerate(ids):
            if i == j:
                continue
            east, north = pair_vector(layout[target], layout[donor])
            distance = float(np.hypot(east, north))
            vectors[target, donor] = (east, north, distance, source_bearing(east, north))
            if i < j:
                diff = wrap((direction[target] - direction[donor]).loc[direction.index < CUTOFF]).dropna()
                avg = np.exp(1j * np.deg2rad(diff)).mean()
                if len(diff) >= 144 and abs(avg) >= .5:
                    constraint = np.eye(len(ids))[i] - np.eye(len(ids))[j]
                    weight = float(np.sqrt(len(diff) * abs(avg) / distance))
                    pair_info.append((constraint * weight, float(np.median(diff)) * weight))
    matrix = np.array([r for r, _ in pair_info] + [np.ones(len(ids))])
    if np.linalg.matrix_rank(matrix) < len(ids):
        raise ValueError("Relative direction graph disconnected")
    offsets = np.linalg.lstsq(matrix, [y for _, y in pair_info] + [0.], rcond=None)[0]
    offset = pd.Series(offsets, index=ids)
    corrected = (direction - offset) % 360
    peer, summaries, monthly = {}, [], []
    for target in ids:
        donors = [key for key in ids if key != target]
        weights = np.array([1 / vectors[target, key][2] for key in donors])
        refs = []
        for values in (direction, corrected):
            donor_angles = values[donors]
            n = donor_angles.notna().sum(axis=1)
            z = (np.exp(1j * np.deg2rad(donor_angles)) * weights).sum(axis=1)
            length = z.abs() / (donor_angles.notna() * weights).sum(axis=1)
            refs.append(((np.rad2deg(np.angle(z)) % 360).where((n >= 2) & (length >= .7)), n, length))
        (raw, _, _), (adjusted, n, length) = refs
        peer[target] = adjusted
        before = wrap(direction[target] - raw)
        after = wrap(corrected[target] - adjusted)
        mask = (direction.index >= TEST) & before.notna() & after.notna()
        summaries.append(dict(turbine=target, relative_offset_deg=offset[target], status="relative_only",
                              test_n=int(mask.sum()), raw_mae_deg=before[mask].abs().mean(),
                              relative_mae_deg=after[mask].abs().mean(), peer_count_mean=n[mask].mean(),
                              resultant_mean=length[mask].mean()))
        for month, values in after.groupby(after.index.strftime("%Y-%m")):
            valid = values.dropna()
            if len(valid):
                monthly.append(dict(turbine=target, month=month, n=len(valid),
                                    median_peer_difference_deg=float(valid.median())))
    return vectors, peer, corrected, summaries, monthly


def event_masks(site, turbine, index, manifest):
    name = f"{site}-{turbine}"
    rows = HOSE / f"{name}.jsonl.gz"
    event_path = HOSE / f"{name}-events.json"
    for path in (rows, event_path):
        if file_hash(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Event artifact hash mismatch: {path.name}")
    events = json.loads(event_path.read_text(encoding="utf-8"))
    stops = {e["id"] for e in events if e["Status"] == "Stop" and e["interval_quality"] == "complete"}
    with gzip.open(rows, "rt", encoding="utf-8") as stream:
        mappings = [(r["timestamp_utc"], r["start_anchor"][2] > 0, r["end_anchor"][2] > 0)
                    for r in map(json.loads, stream)]
    out = pd.DataFrame(mappings, columns=["utc", "exclude_start", "exclude_end"])
    out["utc"] = pd.to_datetime(out["utc"], utc=True)
    out = out.set_index("utc").reindex(index)
    if out.isna().any().any():
        raise ValueError("HOSE timestamp join failed")
    return out, events, stops


def block_range(days, delta):
    if len(days) < 7:
        return None, None
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(days), size=(1000, len(days)))
    return tuple(np.quantile(np.asarray(delta)[draws].mean(axis=1), [.025, .975]))


def evaluate_target(site, key, wide, layout, vectors, peer, events_manifest, outputs):
    times = wide.index
    ws, power = wide["reported_ws"][key], wide["power_kw"][key]
    test = times >= TEST
    others = [other for other in layout if other != key]
    peer_speed = wide["reported_ws"][others].median(axis=1).where(wide["reported_ws"][others].count(axis=1) >= 2)
    roles, events, stops = event_masks(site, key, times, events_manifest)
    for event in events:
        if event["Status"] != "Stop":
            continue
        start = pd.to_datetime(event["Timestamp start"], utc=True, errors="coerce")
        if pd.notna(start) and start < times.min():
            outputs["left_censored_events"].append(dict(farm=site, turbine=key, event_id=event["id"],
                                                        start_utc=str(start), first_telemetry=str(times.min())))
    ti = (wide["sigma"][key] / ws).where((ws >= 3) & (wide["sigma"][key] >= 0))
    direction = peer[key]
    spread = np.full(len(times), False)
    stable = np.full(len(times), False)
    for donor in others:
        east, north, distance, bearing = vectors[key, donor]
        if distance > 12 * layout[donor].rotor_m:
            continue
        angle = np.deg2rad(direction)
        x, y = east * np.sin(angle) + north * np.cos(angle), east * np.cos(angle) - north * np.sin(angle)
        spread |= ((x > 0) & (abs(y) <= (layout[key].rotor_m + layout[donor].rotor_m)/2 + .05*x)).to_numpy()
        difference = abs(wrap(direction - bearing)).to_numpy()
        stable |= difference <= 5
    near = np.zeros(len(times), dtype=bool)
    for donor in others:
        _, _, distance, bearing = vectors[key, donor]
        if distance <= 12 * layout[donor].rotor_m:
            near |= abs(wrap(direction - bearing)).to_numpy() <= 25
    exposure = np.where(direction.isna(), "unknown", np.where(stable, "potential",
                        np.where(near, "boundary", "non_exposed_layout")))
    outputs["geometry_exposure"].extend(
        dict(farm=site, turbine=key, scenario="relative_angles_assumed_from",
             classification=k, n=int(v)) for k, v in pd.Series(exposure[test]).value_counts().items())
    outputs["geometry_exposure"].append(dict(farm=site, turbine=key, scenario="rotor_intersection",
                                             classification="potential", n=int(spread[test].sum())))
    target = pd.DataFrame(dict(power=power, exposure=exposure, ti=ti))
    bins = np.floor(ws * 2 + .5) / 2
    for role, keep in (("all_qa", pd.Series(True, index=times)), ("start", ~roles["exclude_start"]),
                       ("end", ~roles["exclude_end"])):
        groups = power[test & keep & ws.notna()].groupby(bins).agg(
            n="count", mean_kw="mean", median_kw="median", sd_kw="std")
        for b, row in groups[groups["n"] > 0].iterrows():
            outputs["power_curve"].append(dict(farm=site, turbine=key, filter=role, speed_bin=b, **row.to_dict()))
    for axis, speed in (("reported_ws", ws), ("peer_index", peer_speed)):
        donor_speed = wide["reported_ws"][others].stack()
        if axis == "peer_index":
            donor_speed = pd.DataFrame({
                donor: wide["reported_ws"][others].drop(columns=donor).median(axis=1)
                for donor in others
            }).stack()
        donor_bins = np.floor(donor_speed * 2 + .5) / 2
        donors = wide["power_kw"][others].stack()
        donor_reference = donors.loc[donors.index.get_level_values(0) < CUTOFF].groupby(
            donor_bins).agg(["mean", "count"])
        ref = (np.floor(speed * 2 + .5) / 2).map(donor_reference["mean"].where(donor_reference["count"] >= 30))
        deviation = power - ref
        eligible = test & power.notna() & ref.gt(50)
        if eligible.any():
            outputs["ppr"].append(dict(farm=site, turbine=key, axis=axis, n=int(eligible.sum()),
                                       ratio_of_sums=power[eligible].sum()/ref[eligible].sum(),
                                       mae_kw=deviation[eligible].abs().mean()))
        if axis == "reported_ws":
            daily = deviation.groupby(times.normalize()).agg(["mean", "count"])
            daily = daily[daily["count"] >= 24]["mean"]
            calibration = daily[(daily.index >= CUTOFF) & (daily.index < TEST)]
            if len(calibration) >= 7:
                values = [(d.date(), v) for d, v in daily[daily.index >= TEST].items()]
                for row in monitor(values, calibration.tolist()):
                    outputs["pc_health"].append(dict(farm=site, turbine=key, **asdict(row)))
        if axis == "peer_index":
            target["reference"] = ref.where(ref > 50)
            target["peer_bin"] = np.floor(speed*2 + .5)/2
    for axis, columns, measure in (("reported_ws", bins, "ti"), ("peer_index", target["peer_bin"], "ppr")):
        frame = target.loc[test].copy()
        frame["bin"] = columns
        if measure == "ppr":
            frame["ppr"] = frame["power"] / frame["reference"]
        for speed_bin, group in frame.groupby("bin"):
            eligible = group[[measure, "exposure"]].dropna().copy()
            a, b = [eligible.loc[eligible["exposure"] == k, measure] for k in ("potential", "non_exposed_layout")]
            if min(len(a), len(b)) < 30:
                continue
            days = pd.concat([a.groupby(a.index.normalize()).mean().rename("a"),
                              b.groupby(b.index.normalize()).mean().rename("b")], axis=1).dropna()
            if len(days) < 7:
                continue
            delta = days["a"] - days["b"]
            low, high = block_range(days, delta)
            outputs["matched_associations"].append(dict(
                farm=site, turbine=key, axis=axis, measure=measure, speed_bin=speed_bin,
                n_potential=len(a), n_non_exposed=len(b), matched_days=len(days),
                potential_median=a.median(), non_exposed_median=b.median(),
                paired_daily_difference=delta.mean(), low95=low, high95=high))
    pitch = wide["pitch"][key]
    for e in events:
        if e["id"] not in stops or e["IEC category"] != "Scheduled Maintenance" or "pitch" not in (
            e["Message"] + " " + e.get("Comment", "")
        ).lower():
            continue
        start, end = [pd.Timestamp(e[field], tz="UTC") for field in ("Timestamp start", "Timestamp end")]
        seven = pd.Timedelta(days=7)
        competing = any(other["id"] != e["id"] and other["id"] in stops
                        and other["IEC category"] == "Scheduled Maintenance"
                        and pd.Timestamp(other["Timestamp start"], tz="UTC") < end+seven
                        and pd.Timestamp(other["Timestamp end"], tz="UTC") > start-seven for other in events)
        pre = pitch[(times >= start-seven) & (times < start) & ws.between(5, 12)].groupby(bins).agg(["median", "count"])
        post = pitch[(times >= end) & (times < end+seven) & ws.between(5, 12)].groupby(bins).agg(["median", "count"])
        combined = pre.join(post, lsuffix="_pre", rsuffix="_post")
        supported = combined[(combined["count_pre"] >= 24) & (combined["count_post"] >= 24)]
        outputs["pitch_events"].append(dict(
            farm=site, turbine=key, event_id=e["id"], competing_maintenance=competing,
            full_followup=start-seven >= times.min() and end+seven <= times.max(),
            supported_bins=len(supported), pitch_change_deg=float(
                (supported["median_post"]-supported["median_pre"]).median()) if not competing
            and start-seven >= times.min() and end+seven <= times.max() and len(supported) else None))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    from collections import defaultdict

    output = args.output
    if output.exists():
        cli.error("Use a fresh output directory")
    source = ROOT / "artifacts/runs/v2-cubico-2017-final/manifest.json"
    manifest = json.loads(source.read_text())
    event_source = HOSE / "summary.json"
    hose = json.loads(event_source.read_text())
    if hose["benchmark_manifest_sha256"] != file_hash(source):
        raise ValueError("Frozen source mismatch")
    result = defaultdict(list)
    for site, pin in manifest["inputs"]["2017"].items():
        print(f"Reading {site}", flush=True)
        pins = [pin] if site == "kelmarsh" else [pin, manifest["inputs"]["2017_new"]]
        wide, layout, duplicates = load_site(site, pins, manifest["static_sha256"][site])
        vectors, peers, corrected, summaries, monthly = direction_model(wide, layout)
        result["northing_offsets"].extend(dict(farm=site, **entry) for entry in summaries)
        result["direction_monthly"].extend(dict(farm=site, **entry) for entry in monthly)
        result["site_totals"].append(dict(farm=site, rows=len(wide)*len(layout), duplicates=duplicates))
        for (target, donor), (_, _, distance, bearing) in vectors.items():
            result["wake_sectors"].append(dict(farm=site, target=target, source=donor, bearing_from=bearing,
                                               distance_m=distance, distance_source_d=distance/layout[donor].rotor_m,
                                               sector_halfwidth_deg=15, in_range=distance <= 12*layout[donor].rotor_m))
        for key, point in layout.items():
            result["layout"].append(dict(farm=site, turbine=key, **asdict(point)))
            speed = wide["reported_ws"][key]
            for mode, series in (("reported", wide["direction"][key]), ("relative_only", corrected[key])):
                rose = pd.DataFrame({"direction": np.floor((series + 15) % 360/30)*30,
                                     "speed_band": pd.cut(speed, [0, 5, 10, 15, np.inf], right=False)})
                for (direction, band), n in rose.value_counts().items():
                    result["wind_rose"].append(dict(farm=site, turbine=key, basis=mode,
                                                    direction_deg=direction, speed_band=str(band), n=n))
            evaluate_target(site, key, wide, layout, vectors, peers, hose, result)
    output.mkdir(parents=True)
    for name, records in result.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(dict(
        benchmark_manifest_sha256=file_hash(source), hose_manifest_sha256=file_hash(event_source),
        code_commit=_git_commit(),
        code_sha256={name: file_hash(ROOT / name) for name in [
            "scripts/wind_engineering.py","src/wintellisyt/wind_geometry.py","src/wintellisyt/pc_health.py",
            "docs/VALIDATION_AND_BENCHMARKS.md"]},
        numpy=np.__version__, pandas=pd.__version__, direction_status="relative_only",
        output_hashes={p.name:file_hash(p) for p in sorted(output.glob("*.csv"))},
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
