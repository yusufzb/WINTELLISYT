"""Run the fixed E5 interval protocol on verified Cubico archives."""

import argparse
import json
import math
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

from wintellisyt.data.acquire import acquire_archive
from wintellisyt.pipeline import assess, curve_input, load_rows, source_manifest
from wintellisyt.power_curve import fit_power_curve

EDGES = (5, 10, 15)
BANDS = ("0–<5", "5–<10", "10–<15", "15+")
MIN_CALIBRATION = 50
LEVEL = 0.9


def radius(errors):
    if len(errors) < MIN_CALIBRATION:
        return None
    rank = (9 * (len(errors) + 1) + 9) // 10
    return sorted(errors)[rank - 1]


def eligible(row):
    return (
        row.wind_speed_ms is not None and math.isfinite(row.wind_speed_ms)
        and row.wind_speed_ms >= 0
        and row.active_power_kw is not None and math.isfinite(row.active_power_kw)
    )


def summarize(errors, radii):
    pairs = [(error, width) for error, width in zip(errors, radii, strict=True)
             if width is not None]
    lower = sum(error < -width for error, width in pairs)
    upper = sum(error > width for error, width in pairs)
    n = len(pairs)
    return {
        "intervals": n,
        "coverage": (n - lower - upper) / n if n else None,
        "mean_width_kw": 2 * math.fsum(width for _, width in pairs) / n if n else None,
        "lower_misses": lower,
        "upper_misses": upper,
    }


def evaluate(rows):
    ordered = sorted(rows, key=lambda row: row.timestamp_utc)
    timestamps = list(dict.fromkeys(row.timestamp_utc for row in ordered))
    if len(timestamps) < 5:
        raise ValueError("At least five unique timestamps are required")
    calibration_start = timestamps[3 * len(timestamps) // 5]
    test_start = timestamps[4 * len(timestamps) // 5]
    train = [row for row in ordered if row.timestamp_utc < calibration_start]
    calibration = [row for row in ordered if calibration_start <= row.timestamp_utc < test_start]
    test = [row for row in ordered if row.timestamp_utc >= test_start]
    model = fit_power_curve(curve_input(train), min_bin_count=5)

    def errors(partition):
        valid = [0] * len(BANDS)
        groups = [[] for _ in BANDS]
        for row in partition:
            if not eligible(row):
                continue
            band = bisect_right(EDGES, row.wind_speed_ms)
            valid[band] += 1
            prediction = model.predict(row.wind_speed_ms, stratum="raw")
            if prediction is not None:
                groups[band].append(row.active_power_kw - prediction)
        return valid, groups

    calibration_valid, calibration_groups = errors(calibration)
    pooled = radius([abs(error) for group in calibration_groups for error in group])
    band_radii = [radius([abs(error) for error in group]) for group in calibration_groups]
    test_valid, test_groups = errors(test)
    reports = {}
    all_errors, all_pooled, all_conditional = [], [], []
    for idx, band in enumerate(BANDS):
        group = test_groups[idx]
        local = band_radii[idx]
        conditional = pooled if local is None else local
        reports[band] = {
            "calibration_eligible": calibration_valid[idx],
            "calibration_supported": len(calibration_groups[idx]),
            "test_eligible": test_valid[idx],
            "test_supported": len(group),
            "band_radius_kw": local,
            "pooled_fallback_intervals": len(group) if local is None and pooled is not None else 0,
            "pooled": summarize(group, [pooled] * len(group)),
            "wind_band": summarize(group, [conditional] * len(group)),
        }
        all_errors.extend(group)
        all_pooled.extend([pooled] * len(group))
        all_conditional.extend([conditional] * len(group))
    return {
        "split": {
            "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "train_end_utc": train[-1].timestamp_utc.isoformat(),
            "calibration_start_utc": calibration_start.isoformat(),
            "calibration_end_utc": calibration[-1].timestamp_utc.isoformat(),
            "test_start_utc": test_start.isoformat(),
        },
        "pooled_radius_kw": pooled,
        "bands_mps": reports,
        "test_eligible": sum(test_valid),
        "test_supported": len(all_errors),
        "pooled": summarize(all_errors, all_pooled),
        "wind_band": summarize(all_errors, all_conditional),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = {
        "protocol": "docs/VALIDATION_AND_BENCHMARKS.md",
        "nominal_coverage": LEVEL,
        "source_evidence": "verified_real",
        "created_utc": datetime.now().astimezone().isoformat(),
        "turbines": {},
    }
    manifest = source_manifest()
    for source, turbine in (("kelmarsh", "1"), ("penmanshiel", "11")):
        path = acquire_archive(source)
        rows, qa = assess(list(load_rows(path, source, turbine)))
        provenance = manifest[source]
        result["turbines"][f"{source}/{turbine}"] = {
            "source_url": provenance["source_url"],
            "input_sha256": provenance["acquisition"]["sha256"],
            "license": provenance["license"],
            "quarantined_rows": qa["quarantined_rows"],
            **evaluate(rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
