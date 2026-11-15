"""Run the frozen V2 Cubico benchmark and write an immutable result directory."""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from wintellisyt.data import read_scada
from wintellisyt.physics import STANDARD_DENSITY, cp_electrical, tip_speed_ratio, turbulence_proxy
from wintellisyt.pipeline import file_hash, source_manifest
from wintellisyt.v2_models import CONFIG, RATED_KW, bin_index, fit_models

if __package__:
    from . import evaluate_turbine_control as control
    from . import replay_interval_calibration as feedback
else:
    import evaluate_turbine_control as control
    import replay_interval_calibration as feedback

RADII = {"kelmarsh": 46, "penmanshiel": 41}
STATIC_HASHES = {
    "kelmarsh": "a249919471580244d41a547e407ef371d45f2d5347a59bc23cf3c980f607ee93",
    "penmanshiel": "2a742dfd5c8486bf54c1afb647bd803954f4c56aea31635e5aa93828f56ebd7b",
}
CHANNELS = ("wind_speed_ms", "active_power_kw", "ambient_temperature_c", "wind_direction_deg",
            "rotor_speed_rpm", "pitch_deg", "wind_speed_std_ms", "status_code")


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


def archive_groups(pin, site, year):
    path = _repo_path(pin["path"])
    if file_hash(path) != pin["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    groups = defaultdict(list)
    for row in read_scada(path, site):
        if row.timestamp_utc.year == year:
            groups[row.turbine_id].append(row)
    if file_hash(path) != pin["sha256"]:
        raise ValueError(f"Archive changed: {path}")
    return groups


def prepare(groups, site, year):
    availability, turbines, physics = {}, {}, {}
    for turbine, rows in groups.items():
        counts = Counter(row.timestamp_utc for row in rows)
        clean = [r for r in rows if counts[r.timestamp_utc] == 1]
        availability[turbine] = {
            "rows": len(rows), "quarantined": len(rows) - len(clean),
            "nonmissing": {c: sum(getattr(r, c) is not None for r in rows) for c in CHANNELS},
        }
        speed_bins = defaultdict(list)
        for row in clean:
            v, p = row.wind_speed_ms, row.active_power_kw
            if v is None or p is None or not 0 <= v <= 60:
                continue
            cp = cp_electrical(p, v, RADII[site], STANDARD_DENSITY)
            rpm = row.rotor_speed_rpm
            tsr = tip_speed_ratio(rpm, v, RADII[site]) if rpm is not None else None
            sigma = row.wind_speed_std_ms
            ti = turbulence_proxy(sigma, v) if sigma is not None else None
            speed_bins[int(2 * v + 0.5)].append(
                (p, cp, tsr, rpm, row.pitch_deg, ti)
            )
        physics[turbine] = {
            str(key / 2): {"n": len(values), "signals": {
                name: {"n": len(finite), "mean": float(np.mean(finite)),
                       "median": float(np.median(finite)), "sd": float(np.std(finite))}
                for i, name in enumerate(("power_kw", "cp_elec", "tsr", "rotor_rpm", "pitch_deg", "ti"))
                if (finite := [v[i] for v in values if v[i] is not None])
            }, "cp_below_zero": sum(v[1] is not None and v[1] < 0 for v in values),
                "cp_above_betz": sum(v[1] is not None and v[1] > 16/27 for v in values)}
            for key, values in sorted(speed_bins.items())
        }
        turbines[turbine] = {
            part: [r for r in records if r.wind_speed_ms is not None and r.active_power_kw is not None
                   and 3 <= r.wind_speed_ms <= 15]
            for part, records in control.base.partition(clean, year).items()
        }
    return turbines, availability, physics


def features(rows, diagnostic=False, radius=0):
    x, y, retained = [], [], []
    for r in rows:
        v = r.wind_speed_ms
        if diagnostic:
            t, d, rpm = r.ambient_temperature_c, r.wind_direction_deg, r.rotor_speed_rpm
            if t is None or d is None or rpm is None or not 0 <= d <= 360:
                continue
            tsr = tip_speed_ratio(rpm, v, radius)
            if tsr is None:
                continue
            angle = np.radians(d)
            x.append([v, t, np.sin(angle), np.cos(angle), rpm, tsr])
        else:
            x.append([v])
        y.append(r.active_power_kw)
        retained.append(r)
    return np.asarray(x, dtype=float).reshape(-1, 6 if diagnostic else 1), np.asarray(y), retained


def fit_partition(train, calibration, diagnostic=False, radius=0):
    x, y, _ = features(train, diagnostic, radius)
    c, cy, _ = features(calibration, diagnostic, radius)
    models, support, info = fit_models(x, y)
    mask = support(c)
    if int(mask.sum()) < 50:
        raise ValueError("Insufficient calibration support")
    radii = {key: control.base.radius(np.abs(cy[mask] - model(c[mask])).tolist())
             for key, model in models.items()}
    return models, support, radii, {**info, "train_n": len(y), "cal_n": int(mask.sum()),
                                    "radii_kw": radii}


def score(y, pred, radius):
    error = y - pred
    absolute = np.abs(error)
    mae, rmse = float(np.mean(absolute)), float(np.sqrt(np.mean(error**2)))
    return {"n": len(y), "mae_kw": mae, "rmse_kw": rmse, "bias_kw": float(np.mean(error)),
            "nmae": mae / RATED_KW, "nrmse": rmse / RATED_KW,
            "coverage": float(np.mean(absolute <= radius)), "width_kw": 2 * radius,
            "interval_score_kw": float(np.mean(2 * radius + 20 * np.maximum(absolute-radius, 0))),
            "lower_misses": int(np.sum(error < -radius)), "upper_misses": int(np.sum(error > radius))}


def paired_blocks(rows, delta):
    blocks = defaultdict(list)
    for row, value in zip(rows, delta, strict=True):
        blocks[row.timestamp_utc.date().toordinal() // 7].append(float(value))
    sums = [(sum(v), len(v)) for v in blocks.values()]
    if len(sums) < 8:
        return {"blocks": len(sums), "range_kw": None}
    rng = random.Random(42)
    draws = []
    for _ in range(1000):
        sample = rng.choices(sums, k=len(sums))
        draws.append(sum(a for a, _ in sample) / sum(b for _, b in sample))
    return {"blocks": len(sums), "range_kw": np.quantile(draws, [0.025, 0.975]).tolist()}


def evaluate(fits, test, diagnostic=False, radius=0):
    x, y, rows = features(test, diagnostic, radius)
    if not len(y):
        return {"eligible_n": 0, "matched_n": 0}
    mask = np.logical_and.reduce([fit[1](x) for fit in fits.values()])
    if not mask.any():
        return {"eligible_n": len(y), "matched_n": 0}
    x, y = x[mask], y[mask]
    rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
    bins = bin_index(x[:, 0])
    result = {}
    for budget, (models, _, radii, _) in fits.items():
        model_x = x[:, :1] if diagnostic and budget == "F0" else x
        predictions = {key: model(model_x) for key, model in models.items()}
        result[budget] = {
            "models": {key: score(y, pred, radii[key]) for key, pred in predictions.items()},
            "hybrid_clipped": int(np.count_nonzero(predictions["hybrid"] != predictions["hybrid_bounded"])),
            "hybrid_minus_binned_mae": paired_blocks(
                rows, np.abs(y - predictions["hybrid"]) - np.abs(y - predictions["binned"])
            ),
            "bias_by_bin": {key: {str(b / 2): {"n": int(np.sum(bins == b)),
                                              "bias_kw": float(np.mean((y - pred)[bins == b]))}
                                  for b in np.unique(bins)}
                            for key, pred in predictions.items()},
        }
    return {"eligible_n": len(mask), "matched_n": len(y), "budgets": result}


def pooled_fit(groups, ids):
    return fit_partition(*([r for turbine in ids for r in groups[turbine][part]]
                           for part in ("train", "calibration")))


def hybrid_replay(cross, within, rows):
    x, _, rows = features(rows)

    class Curve:
        def __init__(self, fitted):
            models, support, _, _ = fitted
            mask = support(x)
            self.lookup = dict(zip(x[mask, 0], models["hybrid"](x[mask]).tolist(), strict=True))

        def predict(self, speed, *, stratum):
            return self.lookup.get(speed)

    cross_curve, within_curve = Curve(cross), Curve(within)
    result = {}
    for delay in feedback.FEEDBACK_DELAYS_HOURS:
        scored, counts = feedback.replay(rows, cross_curve, cross[2]["hybrid"],
                                         within_curve, within[2]["hybrid"],
                                         feedback_delay_hours=delay)
        result[str(delay)] = {**counts, "metrics": feedback.summarize(scored)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    scripts = Path(__file__).parent
    pins = json.loads((scripts / "transfer_2017_inputs.json").read_text(encoding="utf-8"))
    new = json.loads((scripts / "control_2017_target_input.json").read_text(encoding="utf-8"))
    manifest = source_manifest()
    for site, digest in STATIC_HASHES.items():
        path = ROOT / "data/raw" / site / f"{site.title()}_WT_static.csv"
        if file_hash(path) != digest:
            raise ValueError(f"Static metadata checksum mismatch: {path}")
    code_paths = [
        ROOT / "docs/VALIDATION_AND_BENCHMARKS.md",
        ROOT / "pyproject.toml",
        *(scripts / name for name in (
            "v2_benchmark.py", "evaluate_intervals.py", "evaluate_transfer_intervals.py",
            "evaluate_turbine_control.py", "replay_interval_calibration.py",
        )),
        *(ROOT / "src/wintellisyt" / name for name in (
            "physics.py", "v2_models.py", "data/scada.py", "pipeline.py",
        )),
    ]
    code_hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in code_paths}
    data, old_ids, availability, physics = {}, {}, {}, {}
    for site, pin in pins.items():
        old = archive_groups(pin, site, 2017)
        old_ids[site] = tuple(old)
        if site == "penmanshiel" and not args.smoke:
            extra = archive_groups(new, site, 2017)
            if extra.keys() != control.NEW_TARGETS or old.keys() & extra.keys():
                raise ValueError("Unexpected target IDs")
            old.update(extra)
        data[site], availability[site], physics[site] = prepare(old, site, 2017)
        if not args.smoke:
            _, availability[f"{site}_2016"], _ = prepare(
                archive_groups(manifest[site]["acquisition"], site, 2016), site, 2016
            )

    report = {}
    with threadpool_limits(limits=2):
        for site, other in (("kelmarsh", "penmanshiel"), ("penmanshiel", "kelmarsh")):
            print(f"Fitting {other} donor models for {site}", flush=True)
            donor_ids = control.donor_ids(old_ids[other])
            cross = pooled_fit(data[other], donor_ids)
            cached = {}
            for turbine, split in data[site].items():
                train, cal = split["train"], split["calibration"]
                if args.smoke:
                    train, cal = train[:2000], cal[:300]
                ids = control.donor_ids(old_ids[site], turbine)
                if ids not in cached:
                    cached[ids] = pooled_fit(data[site], ids)
                within = cached[ids]
                local = fit_partition(train, cal)
                entry = evaluate({"local": local, "within": within, "cross": cross},
                                 split["test"][:400] if args.smoke else split["test"])
                entry["donors"] = {"within": ids, "cross": donor_ids}
                entry["fit"] = {"local": local[3], "within": within[3], "cross": cross[3]}
                if not args.smoke:
                    r = RADII[site]
                    complete = {part: features(rows, True, r)[2] for part, rows in split.items()}
                    if min(map(len, complete.values())) >= 200:
                        f0 = fit_partition(complete["train"], complete["calibration"])
                        try:
                            f1 = fit_partition(complete["train"], complete["calibration"], True, r)
                        except ValueError as exc:
                            entry["operational"] = {"status": str(exc)}
                        else:
                            entry["operational"] = evaluate({"F0": f0, "F1": f1},
                                                            complete["test"], True, r)
                    else:
                        entry["operational"] = {"status": "insufficient complete rows"}
                    if site == "penmanshiel" and turbine in control.NEW_TARGETS:
                        entry["feedback"] = hybrid_replay(cross, within, split["test"])
                report[f"{site}/{turbine}"] = entry
                if args.smoke:
                    break
    if any(file_hash(path) != code_hashes[path.relative_to(ROOT).as_posix()] for path in code_paths):
        raise ValueError("Benchmark sources changed during the run")
    payloads = {"metrics.json": report, "physics.json": physics, "availability.json": availability}
    encoded = {name: json.dumps(data_set, allow_nan=False) for name, data_set in payloads.items()}
    args.output.mkdir(parents=True)
    for name, data_set in encoded.items():
        (args.output / name).write_text(data_set, encoding="utf-8")
    metadata = {
        "schema_version": 2, "evidence": "verified_cubico_retrospective",
        "smoke": args.smoke, "protocol": "docs/VALIDATION_AND_BENCHMARKS.md", "year": 2017,
        "python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__,
        "code_commit": _git_commit(),
        "code_sha256": code_hashes,
        "config": CONFIG, "rated_kw": RATED_KW, "rotor_radii_m": RADII, "static_sha256": STATIC_HASHES,
        "density_scenario": STANDARD_DENSITY, "model_domain_ms": [3, 15],
        "feature_sets": {"F0": ["wind"], "F1": ["wind", "temperature", "sin_direction",
                        "cos_direction", "rotor_rpm", "tsr"]},
        "inputs": {"2017": pins, "2017_new": new,
                   "2016": {site: manifest[site]["acquisition"] for site in pins}},
        "artifact_sha256": {name: file_hash(args.output / name) for name in payloads},
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
