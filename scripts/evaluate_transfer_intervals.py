"""Run the E4/E5 bidirectional calendar transfer benchmark."""

import argparse
import json
import math
import random
import platform
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wintellisyt import __version__
from wintellisyt.data.acquire import acquire_archive
from wintellisyt.pipeline import assess, curve_input, file_hash, load_rows, source_manifest
from wintellisyt.power_curve import fit_power_curve

if __package__:
    from .evaluate_intervals import BANDS, EDGES, LEVEL, eligible, radius
else:
    from evaluate_intervals import BANDS, EDGES, LEVEL, eligible, radius

RATED_POWER_KW = 2050
BUDGETS = ("source_only", "target_recalibrated", "local")
METHODS = ("pooled", "wind_band")
SOURCES = ("kelmarsh", "penmanshiel")


def partition(rows, year):
    calibration = datetime(year, 10, 1, tzinfo=timezone.utc)
    test = datetime(year, 11, 1, tzinfo=timezone.utc)
    gap = timedelta(minutes=10)
    return {
        "train": [r for r in rows if r.timestamp_utc < calibration - gap],
        "calibration": [r for r in rows if calibration <= r.timestamp_utc < test - gap],
        "test": [r for r in rows if r.timestamp_utc >= test],
    }


def fit(rows):
    return fit_power_curve(curve_input(rows), min_bin_count=5)


def calibrate(model, rows):
    scores = [[] for _ in BANDS]
    for row in rows:
        if eligible(row):
            prediction = model.predict(row.wind_speed_ms, stratum="raw")
            if prediction is not None:
                scores[bisect_right(EDGES, row.wind_speed_ms)].append(
                    abs(row.active_power_kw - prediction)
                )
    pooled = radius([error for band in scores for error in band])
    return pooled, [radius(band) for band in scores], [len(band) for band in scores]


def evaluate_target(donor, donor_calibration, split, *, within_farm=None):
    local = fit(split["train"])
    calibrated = {
        "source_only": (donor, donor_calibration),
        "local": (local, calibrate(local, split["calibration"])),
    }
    if within_farm is None:
        calibrated["target_recalibrated"] = (donor, calibrate(donor, split["calibration"]))
    else:
        calibrated["within_farm"] = within_farm
    availability = {budget: {method: 0 for method in METHODS} for budget in calibrated}
    scored = []
    total = 0
    eligible_bands = Counter()
    for row in split["test"]:
        if not eligible(row):
            continue
        total += 1
        index = bisect_right(EDGES, row.wind_speed_ms)
        eligible_bands[BANDS[index]] += 1
        results = {}
        for budget, (model, (pooled, bands, _)) in calibrated.items():
            prediction = model.predict(row.wind_speed_ms, stratum="raw")
            if prediction is None:
                continue
            error = row.active_power_kw - prediction
            for method, width in zip(METHODS, (pooled, bands[index] if bands[index] is not None
                                              else pooled), strict=True):
                if width is not None:
                    availability[budget][method] += 1
                    results[budget, method] = (error, width)
        if len(results) == len(calibrated) * len(METHODS):
            scored.append((row.timestamp_utc, BANDS[index], results))
    return scored, {
        "split_rows": {key: len(value) for key, value in split.items()},
        "eligible": total,
        "eligible_bands": dict(eligible_bands),
        "available": availability,
        "calibration": {
            budget: {"pooled_radius_kw": pooled, "band_radii_kw": bands,
                     "band_scored_rows": counts}
            for budget, (_, (pooled, bands, counts)) in calibrated.items()
        },
    }


def interval_score(error, width):
    return 2 * width + 20 * max(0, abs(error) - width)


def metrics(scored, budgets=BUDGETS):
    if not scored:
        return {"n": 0}
    n = len(scored)
    values = {}
    for budget in budgets:
        values[budget] = {}
        for method in METHODS:
            pairs = [results[budget, method] for _, _, results in scored]
            lower = sum(error < -width for error, width in pairs)
            upper = sum(error > width for error, width in pairs)
            coverage = (n - lower - upper) / n
            values[budget][method] = {
                "coverage": coverage, "coverage_shortfall": max(0, LEVEL - coverage),
                "width_kw": 2 * math.fsum(width for _, width in pairs) / n,
                "interval_score_kw": math.fsum(interval_score(*pair) for pair in pairs) / n,
                "lower_misses": lower, "upper_misses": upper,
                "bias_kw": math.fsum(error for error, _ in pairs) / n,
                "mae_kw": math.fsum(abs(error) for error, _ in pairs) / n,
                "rmse_kw": math.sqrt(math.fsum(error**2 for error, _ in pairs) / n),
                "nmae": math.fsum(abs(error) for error, _ in pairs) / n / RATED_POWER_KW,
                "nrmse": math.sqrt(math.fsum(error**2 for error, _ in pairs) / n) / RATED_POWER_KW,
            }
    for budget in budgets:
        for method in METHODS:
            values[budget][method]["paired_score_delta_vs_local_kw"] = (
                values[budget][method]["interval_score_kw"]
                - values["local"][method]["interval_score_kw"]
            )
    return {"n": n, "descriptive_small_stratum": n < 50, "budgets": values}


def bootstrap(site_scored, origin, block_days, method, band=None, *, reference="local"):
    blocks = defaultdict(dict)
    for turbine, scored in site_scored.items():
        for timestamp, speed_band, values in scored:
            if band is not None and band != speed_band:
                continue
            key = (timestamp - origin).days // block_days
            entry = blocks[key].setdefault(turbine, [0.0, 0])
            entry[0] += (interval_score(*values["source_only", method])
                         - interval_score(*values[reference, method]))
            entry[1] += 1
    if len(blocks) < 8:
        return {"blocks": len(blocks), "status": "insufficient_blocks"}
    rng = random.Random(20261114)
    samples = []
    options = [blocks[key] for key in sorted(blocks)]
    represented_turbines = {turbine for block in options for turbine in block}
    for _ in range(1000):
        totals = defaultdict(lambda: [0.0, 0])
        for block in rng.choices(options, k=len(options)):
            for turbine, (score, count) in block.items():
                totals[turbine][0] += score
                totals[turbine][1] += count
        if len(totals) == len(represented_turbines):
            samples.append(math.fsum(score / count for score, count in totals.values()) / len(totals))
    if len(samples) < 950:
        return {"blocks": len(blocks), "status": "insufficient_complete_draws",
                "complete_draws": len(samples)}
    samples.sort()
    return {"blocks": len(blocks), "status": "descriptive",
            "complete_draws": len(samples),
            "range_kw": [samples[int(len(samples) * 0.025)],
                         samples[int(len(samples) * 0.975) - 1]]}


def macro(reports, budgets=BUDGETS):
    reports = [report for report in reports if report["n"]]
    result = {"scored_turbines": len(reports), "n": sum(report["n"] for report in reports)}
    if reports:
        result["budgets"] = {
            budget: {
                method: {
                    key: math.fsum(r["budgets"][budget][method][key] for r in reports)
                    / len(reports)
                    for key in reports[0]["budgets"][budget][method]
                    if key not in {"lower_misses", "upper_misses"}
                }
                for method in METHODS
            }
            for budget in budgets
        }
    return result


def direction(donor, target, year):
    model = fit(row for split in donor.values() for row in split["train"])
    calibration = calibrate(model, (row for split in donor.values() for row in split["calibration"]))
    reports = {}
    site_scored = {}
    for turbine, split in sorted(target.items()):
        scored, report = evaluate_target(model, calibration, split)
        site_scored[turbine] = scored
        report["matched"] = metrics(scored)
        report["bands_mps"] = {
            band: metrics([row for row in scored if row[1] == band]) for band in BANDS
        }
        reports[turbine] = report
    origin = datetime(year, 11, 1, tzinfo=timezone.utc)
    return {
        "turbines": reports,
        "site_turbine_weighted": macro([report["matched"] for report in reports.values()]),
        "site_bands_mps": {
            band: macro([report["bands_mps"][band] for report in reports.values()])
            for band in BANDS
        },
        "source_only_score_delta_bootstrap": {
            method: {str(days): bootstrap(site_scored, origin, days, method) for days in (2, 7)}
            for method in METHODS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--year", type=int, choices=(2016, 2017), default=2016)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    splits, provenance, qa_reports = {}, {}, {}
    year = args.year
    manifest = source_manifest()
    independent = (json.loads(Path(__file__).with_name("transfer_2017_inputs.json")
                              .read_text(encoding="utf-8")) if year == 2017 else {})
    root = Path(__file__).resolve().parents[1]
    for source in SOURCES:
        info = manifest[source]
        acquisition = independent[source] if independent else info["acquisition"]
        raw_path = Path(acquisition["path"])
        path = (raw_path if raw_path.is_absolute() else (root / raw_path)) if independent else acquire_archive(source, root=root)
        if path.stat().st_size != acquisition["bytes"] or file_hash(path) != acquisition["sha256"]:
            raise ValueError(f"Pinned archive verification failed: {path}")
        rows, qa = assess(list(load_rows(path, source)))
        if file_hash(path) != acquisition["sha256"]:
            raise ValueError(f"Input changed during ingestion: {source}")
        groups = defaultdict(list)
        for row in rows:
            if row.timestamp_utc.year == year:
                groups[row.turbine_id].append(row)
        if year == 2017 and (qa["quarantined_rows"] or qa["off_cadence_intervals"]
                             or len(groups) != (6 if source == "kelmarsh" else 5)):
            raise ValueError(f"Independent archive failed timestamp/turbine QA: {source}")
        splits[source] = {turbine: partition(group, year) for turbine, group in groups.items()}
        if year == 2017:
            for turbine, split in splits[source].items():
                if min(sum(map(eligible, values)) for values in split.values()) < 50:
                    raise ValueError(f"Independent archive has low split support: {source}/{turbine}")
        provenance[source] = {"url": info["source_url"], "license": info["license"],
                              "sha256": acquisition["sha256"], "file": path.name,
                              "publisher_md5": acquisition["publisher_md5"],
                              "download_url": acquisition["url"]}
        qa_reports[source] = {"rows": qa["rows"], "quarantined_rows": qa["quarantined_rows"],
                              "turbine_splits": {
                                  t: {k: len(v) for k, v in split.items()}
                                  for t, split in splits[source].items()
                              }}
    result = {
        "protocol": "docs/VALIDATION_AND_BENCHMARKS.md", "year": year,
        "source_evidence": "verified_real",
        "study_stage": "development" if year == 2016 else "independent_year",
        "package_version": __version__, "python_version": platform.python_version(),
        "rated_power_kw": RATED_POWER_KW,
        "nominal_coverage": LEVEL, "provenance": provenance, "qa": qa_reports,
        "bootstrap": {"replicates": 1000, "seed": 20261114, "minimum_blocks": 8},
        "directions": {
            f"{donor}_to_{target}": direction(splits[donor], splits[target], year)
            for donor, target in (SOURCES, SOURCES[::-1])
        },
    }
    if year == 2017 and any(
        not turbine["matched"]["n"]
        for outcome in result["directions"].values() for turbine in outcome["turbines"].values()
    ):
        raise ValueError("Independent-year transfer lacks matched interval support")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
