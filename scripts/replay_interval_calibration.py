"""Replay rolling pooled calibration with strictly lagged target feedback."""

import argparse
import json
from bisect import bisect_right, insort
from collections import deque
from datetime import timedelta
from pathlib import Path

if __package__:
    from . import evaluate_turbine_control as control
else:
    import evaluate_turbine_control as control

base = control.base
MODELS = ("cross_farm", "within_farm", "rolling_cross_farm")
WINDOW = 1008
MIN_HISTORY = 50
FEEDBACK_DELAYS_HOURS = (0, 1, 24)


def replay(rows, cross, cross_radius, within, within_radius, *, feedback_delay_hours=0):
    if cross_radius is None or within_radius is None:
        raise ValueError("Both donor models require a pooled calibration radius")
    if feedback_delay_hours not in FEEDBACK_DELAYS_HOURS:
        raise ValueError(f"Feedback delay must be one of {FEEDBACK_DELAYS_HOURS} hours")
    delay = timedelta(hours=feedback_delay_hours)
    pending = deque()
    history = deque()
    ordered_errors = []
    scored = []
    eligible_count = warmup = feedback_count = 0
    for row in sorted(rows, key=lambda item: item.timestamp_utc):
        if not base.eligible(row):
            continue
        eligible_count += 1
        prediction = cross.predict(row.wind_speed_ms, stratum="raw")
        if prediction is None:
            continue
        cutoff = row.timestamp_utc - delay
        while pending and pending[0][0] <= cutoff and pending[0][0] < row.timestamp_utc:
            _, arrived_error = pending.popleft()
            if len(history) == WINDOW:
                ordered_errors.remove(history.popleft())
            history.append(arrived_error)
            insort(ordered_errors, arrived_error)

        n = len(history)
        rank = (9 * (n + 1) + 9) // 10
        radius = ordered_errors[rank - 1] if n >= MIN_HISTORY else cross_radius
        error = row.active_power_kw - prediction
        reference = within.predict(row.wind_speed_ms, stratum="raw")
        if reference is not None:
            warmup += n < MIN_HISTORY
            scored.append((
                base.BANDS[bisect_right(base.EDGES, row.wind_speed_ms)],
                (error, row.active_power_kw - reference, error),
                (cross_radius, within_radius, radius),
            ))
        pending.append((row.timestamp_utc, abs(error)))
        feedback_count += 1
    return scored, {"eligible": eligible_count, "scored": len(scored),
                    "feedback_rows": feedback_count, "matched_warmup_rows": warmup}


def summarize(scored):
    if not scored:
        return {}
    result = {}
    n = len(scored)
    for i, name in enumerate(MODELS):
        pairs = [(errors[i], radii[i]) for _, errors, radii in scored]
        lower = sum(error < -width for error, width in pairs)
        upper = sum(error > width for error, width in pairs)
        result[name] = {
            "coverage": 1 - (lower + upper) / n,
            "width_kw": 2 * sum(width for _, width in pairs) / n,
            "interval_score_kw": sum(base.interval_score(*p) for p in pairs) / n,
            "lower_misses": lower, "upper_misses": upper,
        }
    return result


def report(targets, cross, cross_radius, within, within_radius, *, feedback_delay_hours=0):
    reports = {}
    for turbine, split in sorted(targets.items()):
        scored, counts = replay(
            split["test"], cross, cross_radius, within, within_radius,
            feedback_delay_hours=feedback_delay_hours,
        )
        if not scored:
            raise ValueError(f"No shared support: {turbine}")
        reports[turbine] = {**counts, "metrics": summarize(scored), "bands": {
            band: {"n": len(group), "metrics": summarize(group)}
            for band in base.BANDS
            if (group := [r for r in scored if r[0] == band])
        }}
    macro = {model: {
        key: sum(r["metrics"][model][key] for r in reports.values()) / len(reports)
        for key in ("coverage", "width_kw", "interval_score_kw")
    } for model in MODELS}
    return {
        "turbines": reports,
        "turbine_weighted": macro,
        "matched_rows": sum(r["scored"] for r in reports.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--delay-sensitivity",
        action="store_true",
        help="Compare feedback delays of 0, 1, and 24 hours on the pinned target cohort.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    pins = json.loads((script_dir / "transfer_2017_inputs.json").read_text(encoding="utf-8"))
    target_pin = json.loads((script_dir / "control_2017_target_input.json").read_text(encoding="utf-8"))
    donors = {}
    for site, pin in pins.items():
        raw_path = Path(pin["path"])
        splits, _ = control.load(raw_path if raw_path.is_absolute() else (repo_root / raw_path), site, 2017, pin["sha256"])
        donors[site] = control.donor_model(splits, control.donor_ids(splits))
    tgt_raw = Path(target_pin["path"])
    targets, qa = control.load(tgt_raw if tgt_raw.is_absolute() else (repo_root / tgt_raw), "penmanshiel", 2017, target_pin["sha256"])
    if targets.keys() != control.NEW_TARGETS or qa["quarantined_rows"]:
        raise ValueError("Unexpected target cohort or duplicate timestamps")
    (cross, cross_cal), cross_info = donors["kelmarsh"]
    (within, within_cal), within_info = donors["penmanshiel"]
    result = {
        "protocol": "docs/VALIDATION_AND_BENCHMARKS.md", "source_evidence": "verified_real",
        "stage": "retrospective_replay", "year": 2017, "window_observations": WINDOW,
        "min_history": MIN_HISTORY, "donor_pins": pins, "target_pin": target_pin,
        "donors": {"cross_farm": cross_info, "within_farm": within_info},
        "target_qa": qa,
    }
    models = (targets, cross, cross_cal[0], within, within_cal[0])
    if args.delay_sensitivity:
        result.update({
            "schema_version": 2,
            "feedback_delays_hours": FEEDBACK_DELAYS_HOURS,
            "feedback_rule": (
                "Separate history per turbine. Only eligible earlier target outcomes supported by "
                "the frozen cross-farm model may update its radius. An earlier outcome becomes "
                "available when elapsed UTC time is at least the configured delay."
            ),
            "feedback_rows_definition": (
                "Eligible cross-farm-supported outcomes in the replay, including outcomes "
                "whose feedback arrives after the final prediction."
            ),
            "evaluation_scope": "Previously evaluated cohort under simulated feedback delays.",
            "sensitivity": {
                str(hours): report(*models, feedback_delay_hours=hours)
                for hours in FEEDBACK_DELAYS_HOURS
            },
        })
    else:
        result.update(report(*models))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
