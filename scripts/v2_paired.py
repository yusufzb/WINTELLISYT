"""Recover and compare two frozen cross-farm predictors on their shared 2017 rows."""

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from wintellisyt.pipeline import file_hash
from wintellisyt.v2_models import CONFIG, bin_index

if __package__:
    from .v2_benchmark import archive_groups, control, features, score
else:
    from v2_benchmark import archive_groups, control, features, score

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


MODELS = ("binned", "hybrid")
DELTAS = ("mae_kw", "coverage", "interval_score_kw")
BOOTSTRAP_DRAWS = 1000
SEED = 42


def bootstrap(targets, hours):
    """Resample identical UTC blocks for all turbines, including both farms."""
    block_ids = sorted(set(np.concatenate([t["seconds"] // (3600 * hours) for t in targets])))
    block_index = {key: i for i, key in enumerate(block_ids)}
    counts = np.zeros((len(block_ids), len(targets)))
    sums = np.zeros((*counts.shape, len(DELTAS)))
    for i, target in enumerate(targets):
        blocks = [block_index[key] for key in target["seconds"] // (3600 * hours)]
        np.add.at(counts[:, i], blocks, 1)
        np.add.at(sums[:, i], blocks, target["delta"])
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(block_ids), np.full(len(block_ids), 1 / len(block_ids)), BOOTSTRAP_DRAWS)
    n = draws @ counts
    total = np.einsum("db,btm->dtm", draws, sums)
    valid = (n > 0).all(axis=1)
    if len(block_ids) < 8 or valid.sum() < 0.95 * BOOTSTRAP_DRAWS:
        return {"blocks": len(block_ids), "valid_draws": int(valid.sum()), "ci95": None}
    macro = (total[valid] / n[valid, :, None]).mean(axis=1)
    pooled = total[valid].sum(axis=1) / n[valid].sum(axis=1)[:, None]
    return {"blocks": len(block_ids), "valid_draws": int(valid.sum()), "ci95": {
        label: dict(zip(DELTAS, np.quantile(values, [0.025, 0.975], axis=0).T.tolist(), strict=True))
        for label, values in (("turbine_macro", macro), ("row_pooled", pooled))
    }}


def train_cross(groups, donor_ids, fit):
    x, y, _ = features([row for key in donor_ids for row in groups[key]["train"]])
    bins = bin_index(x[:, 0])
    counts = np.bincount(bins)
    support = np.flatnonzero(counts >= CONFIG["min_bin"])
    if support.tolist() != fit["supported_bins"] or len(y) != fit["train_n"]:
        raise ValueError("Cross-farm donor support differs from frozen run")
    means = np.zeros_like(counts, dtype=float)
    means[support] = np.bincount(bins, weights=y)[support] / counts[support]

    def shape(x):
        return np.interp(x[:, 0], fit["shape_wind_ms"], fit["shape_power_kw"])

    residual = HistGradientBoostingRegressor(
        max_iter=CONFIG["boost_iterations"], max_leaf_nodes=CONFIG["boost_leaves"],
        early_stopping=False, random_state=CONFIG["seed"],
    ).fit(x, y - shape(x))

    def predict(x):
        return {"binned": means[bin_index(x[:, 0])], "hybrid": shape(x) + residual.predict(x)}

    return predict


def target_data(rows, frozen, predict):
    x, y, rows = features(rows)
    bins = bin_index(x[:, 0])
    support = np.logical_and.reduce([
        np.isin(bins, fit["supported_bins"]) for fit in frozen["fit"].values()
    ])
    if (len(y), int(support.sum())) != (frozen["eligible_n"], frozen["matched_n"]):
        raise ValueError("Frozen test cohort mismatch")
    x, y = x[support], y[support]
    pred = predict(x)
    radii = {m: frozen["fit"]["cross"]["radii_kw"][m] for m in MODELS}
    errors = {m: y - pred[m] for m in MODELS}
    for m in MODELS:
        actual = score(y, pred[m], radii[m])
        expected = frozen["budgets"]["cross"]["models"][m]
        if any(not math.isclose(v, expected[k], rel_tol=1e-11, abs_tol=1e-9)
               for k, v in actual.items()):
            raise ValueError(f"Frozen test metric mismatch for {m}: {actual} vs {expected}")
    absolute = {m: np.abs(errors[m]) for m in MODELS}
    binned, hybrid = MODELS
    interval_scores = {m: 2 * radii[m] + 20 * np.maximum(absolute[m] - radii[m], 0) for m in MODELS}
    delta = np.column_stack((
        absolute[hybrid] - absolute[binned],
        (absolute[hybrid] <= radii[hybrid]).astype(int) - (absolute[binned] <= radii[binned]),
        interval_scores[hybrid] - interval_scores[binned],
    ))
    timestamps = [r.timestamp_utc for r, keep in zip(rows, support, strict=True) if keep]
    return {
        "timestamps": timestamps, "observed": y, "predictions": pred,
        "seconds": np.asarray([int(t.timestamp()) for t in timestamps]),
        "bins": bins[support], "eligible": len(support), "errors": errors, "radii": radii, "delta": delta,
    }


def summarize(targets, wind_bin=None):
    groups = []
    for t in targets:
        mask = np.ones(len(t["bins"]), dtype=bool) if wind_bin is None else t["bins"] == wind_bin
        if mask.any():
            groups.append((t, mask))
    if not groups:
        return {"n": 0, "turbines": 0}
    models = {}
    for m in MODELS:
        def measures(error, radius):
            result = score(error, np.zeros_like(error), radius)
            return {**result, "width_kw": float(np.mean(2 * radius)),
                    "median_width_kw": float(np.median(2 * radius))}

        per_turbine = [measures(t["errors"][m][mask], np.asarray(t["radii"][m])) for t, mask in groups]
        pooled = measures(
            np.concatenate([t["errors"][m][mask] for t, mask in groups]),
            np.concatenate([np.full(int(mask.sum()), t["radii"][m]) for t, mask in groups]),
        )
        macro = {key: float(np.mean([item[key] for item in per_turbine]))
                 for key in ("mae_kw", "rmse_kw", "bias_kw", "coverage", "width_kw", "interval_score_kw")}
        models[m] = {"row_pooled": pooled, "turbine_macro": macro}
    return {"n": pooled["n"], "turbines": len(groups), "models": models}


def load_partitions(manifest):
    data = {}
    for site, pin in manifest["inputs"]["2017"].items():
        groups = archive_groups(pin, site, 2017)
        if site == "penmanshiel":
            extra = archive_groups(manifest["inputs"]["2017_new"], site, 2017)
            if extra.keys() != control.NEW_TARGETS or groups.keys() & extra.keys():
                raise ValueError("Unexpected target IDs")
            groups.update(extra)
        data[site] = {}
        for key, rows in groups.items():
            counts = Counter(r.timestamp_utc for r in rows)
            usable = [r for r in rows if counts[r.timestamp_utc] == 1 and r.active_power_kw is not None
                      and r.wind_speed_ms is not None and 3 <= r.wind_speed_ms <= 15]
            data[site][key] = control.base.partition(usable, 2017)
    return data


def recover_targets(frozen, partitions):
    targets = {}
    for site, other in (("kelmarsh", "penmanshiel"), ("penmanshiel", "kelmarsh")):
        ref = next(entry for key, entry in frozen.items() if key.startswith(site + "/"))
        predict = train_cross(partitions[other], ref["donors"]["cross"], ref["fit"]["cross"])
        for key, split in partitions[site].items():
            target = f"{site}/{key}"
            entry = frozen[target]
            if entry["donors"]["cross"] != ref["donors"]["cross"]:
                raise ValueError(f"Unexpected donors for {target}")
            targets[target] = target_data(split["test"], entry, predict)
    if targets.keys() != frozen.keys():
        raise ValueError("Frozen target set mismatch")
    return targets


def build_report(frozen, partitions):
    targets = recover_targets(frozen, partitions)

    def cohort(group):
        return {
            **summarize(group),
            "eligible": sum(t["eligible"] for t in group),
            "bootstrap_hours": {hours: bootstrap(group, hours) for hours in (24, 6, 48)},
            "wind_bins_ms": {str(b / 2): summarize(group, b) for b in range(6, 31)},
        }

    return {
        "overall": cohort(list(targets.values())),
        "directions": {site: cohort([t for key, t in targets.items() if key.startswith(site + "/")])
                       for site in partitions},
        "targets": {key: {**summarize([t]), "eligible": t["eligible"], "bootstrap_24h": bootstrap([t], 24)}
                    for key, t in targets.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    source = args.benchmark / "manifest.json"
    frozen_manifest = json.loads(source.read_text(encoding="utf-8"))
    if (frozen_manifest["year"] != 2017 or frozen_manifest["smoke"]
            or frozen_manifest["evidence"] != "verified_cubico_retrospective"
            or frozen_manifest["config"] != CONFIG):
        raise ValueError("Expected the frozen full 2017 benchmark")
    for name, digest in frozen_manifest["artifact_sha256"].items():
        if file_hash(args.benchmark / name) != digest:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    frozen = json.loads((args.benchmark / "metrics.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=2):
        report = build_report(frozen, load_partitions(frozen_manifest))
    report["provenance"] = {
        "schema_version": 1, "evidence": "verified_cubico_retrospective", "feature_set": ["wind"],
        "benchmark": args.benchmark.as_posix(), "benchmark_manifest_sha256": file_hash(source),
        "input_pins": frozen_manifest["inputs"], "benchmark_code": frozen_manifest["code_commit"],
        "code_commit": _git_commit(),
        "analysis_sha256": file_hash(Path(__file__)), "nominal_coverage": 0.9,
        "protocol_sha256": file_hash(ROOT / "docs/VALIDATION_AND_BENCHMARKS.md"),
        "seed": SEED, "draws": BOOTSTRAP_DRAWS, "python": sys.version.split()[0],
        "numpy": np.__version__, "sklearn": sklearn.__version__,
    }
    payload = json.dumps(report, allow_nan=False, indent=2)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
