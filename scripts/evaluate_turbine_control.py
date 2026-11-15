"""Run the matched within-farm held-out-turbine control."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from wintellisyt.data.acquire import acquire_archive
from wintellisyt.pipeline import assess, file_hash, load_rows, source_manifest

if __package__:
    from . import evaluate_transfer_intervals as base
else:
    import evaluate_transfer_intervals as base

BUDGETS = ("source_only", "within_farm", "local")
NEW_TARGETS = {f"{i:02d}" for i in range(1, 11)} - {"03"}


def donor_ids(splits, target=None):
    candidates = sorted((turbine for turbine in splits if turbine != target), key=int)
    if len(candidates) < 4:
        raise ValueError("Four distinct donor turbines are required")
    return tuple(candidates[:4])


def donor_model(splits, ids):
    train = [row for turbine in ids for row in splits[turbine]["train"]]
    calibration = [row for turbine in ids for row in splits[turbine]["calibration"]]
    curve = base.fit(train)
    return (curve, base.calibrate(curve, calibration)), {
        "donors": ids, "train_rows": len(train), "calibration_rows": len(calibration),
    }


def load(path, site, year, digest):
    if file_hash(path) != digest:
        raise ValueError(f"Archive SHA256 mismatch: {path}")
    rows, qa = assess(list(load_rows(path, site)))
    if file_hash(path) != digest:
        raise ValueError(f"Archive changed during ingestion: {path}")
    groups = defaultdict(list)
    for row in rows:
        if row.timestamp_utc.year == year:
            groups[row.turbine_id].append(row)
    splits = {t: base.partition(observations, year) for t, observations in groups.items()}
    return splits, {k: qa[k] for k in ("rows", "quarantined_rows", "off_cadence_intervals")}


def compare(cross_farm, within_farm, targets, year):
    donor, donor_info = donor_model(cross_farm, donor_ids(cross_farm))
    controls = {}
    reports, scored_by_turbine = {}, {}
    for turbine, split in sorted(targets.items(), key=lambda item: int(item[0])):
        ids = donor_ids(within_farm, turbine)
        if ids not in controls:
            controls[ids] = donor_model(within_farm, ids)
        control, control_info = controls[ids]
        scored, report = base.evaluate_target(*donor, split, within_farm=control)
        if not scored:
            raise ValueError(f"No shared test support for turbine {turbine}")
        reports[turbine] = {
            **report, "matched": base.metrics(scored, BUDGETS),
            "within_farm_donor": control_info,
            "bands_mps": {band: base.metrics([r for r in scored if r[1] == band], BUDGETS)
                          for band in base.BANDS},
        }
        scored_by_turbine[turbine] = scored
    origin = datetime(year, 11, 1, tzinfo=timezone.utc)
    summary = base.macro([r["matched"] for r in reports.values()], BUDGETS)
    differences = {}
    for method in base.METHODS:
        cross = summary["budgets"]["source_only"][method]
        within = summary["budgets"]["within_farm"][method]
        differences[method] = {
            "coverage_difference": cross["coverage"] - within["coverage"],
            "interval_score_difference_kw": cross["interval_score_kw"] - within["interval_score_kw"],
            "block_bootstrap": {
                str(days): base.bootstrap(scored_by_turbine, origin, days, method,
                                          reference="within_farm")
                for days in (2, 7)
            },
        }
    return {"cross_farm_donor": donor_info, "turbines": reports, "summary": summary,
            "cross_minus_within": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--year", type=int, choices=(2016, 2017), required=True)
    parser.add_argument("--new-targets", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (args.new_targets and args.year != 2017):
        parser.error("Use a new output path and select 2017 for the new target cohort")
    manifest = source_manifest()
    if args.year == 2017:
        pins = json.loads(Path(__file__).with_name("transfer_2017_inputs.json")
                          .read_text(encoding="utf-8"))
    else:
        pins = {site: manifest[site]["acquisition"] for site in base.SOURCES}
    root = Path(__file__).resolve().parents[1]
    sites, provenance = {}, {}
    for site, pin in pins.items():
        raw_path = Path(pin["path"])
        path = (raw_path if raw_path.is_absolute() else (root / raw_path)) if args.year == 2017 else acquire_archive(site, root=root)
        sites[site], qa = load(path, site, args.year, pin["sha256"])
        provenance[site] = {**pin, "qa": qa, "source_url": manifest[site]["source_url"],
                            "license": manifest[site]["license"]}
    if args.new_targets:
        pin = json.loads(Path(__file__).with_name("control_2017_target_input.json")
                         .read_text(encoding="utf-8"))
        tgt_raw = Path(pin["path"])
        targets, qa = load(tgt_raw if tgt_raw.is_absolute() else (root / tgt_raw), "penmanshiel", 2017, pin["sha256"])
        if targets.keys() != NEW_TARGETS or targets.keys() & sites["penmanshiel"].keys():
            raise ValueError("New target archive has unexpected or overlapping turbine IDs")
        provenance["penmanshiel_new_targets"] = {**pin, "qa": qa}
        comparisons = {"penmanshiel_new_targets":
                       compare(sites["kelmarsh"], sites["penmanshiel"], targets, 2017)}
    else:
        comparisons = {
            target: compare(sites[source], sites[target], sites[target], args.year)
            for source, target in (base.SOURCES, base.SOURCES[::-1])
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"year": args.year, "source_evidence": "verified_real",
                   "protocol": "docs/VALIDATION_AND_BENCHMARKS.md",
                   "provenance": provenance, "comparisons": comparisons},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
