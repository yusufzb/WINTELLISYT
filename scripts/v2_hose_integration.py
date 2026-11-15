"""Replay two frozen predictors and join their shared test rows to HOSE v1."""

import argparse
import csv
import gzip
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from wintellisyt.operating_state import PRESETS, VERSION, Signals, physical_proxy
from wintellisyt.pipeline import file_hash
from wintellisyt.v2_models import CONFIG

if __package__:
    from .v2_paired import MODELS, SEED, BOOTSTRAP_DRAWS, bootstrap, load_partitions, recover_targets, summarize
else:
    from v2_paired import MODELS, SEED, BOOTSTRAP_DRAWS, bootstrap, load_partitions, recover_targets, summarize

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "untracked"


REGIMES = ("rotating", "standstill_like", "transition", "unknown")
SPLIT = "2017-nov-dec-test"
SIGNALS = ("wind_ms", "rotor_rpm", "generator_rpm", "pitch_deg", "setpoint_kw")
PAIR_SHA256 = "1abedca9176a77914fe3255a9412042d170afd33eee212dc6eb1df60f523408a"
HOSE_SHA256 = "adf1597f9e0dcd6a597f2f7296c01fc0c1fb35871934a1d2f39e1997b01a4fd7"


def verified_json(path, digest):
    if file_hash(path) != digest:
        raise ValueError(f"Checksum mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def utc(timestamp):
    time = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
    if time.utcoffset() is None:
        raise ValueError("Naive timestamp")
    return time.astimezone(timezone.utc)


def join_rows(key, timestamps, rows):
    farm, turbine = key.split("/")
    wanted = [utc(t) for t in timestamps]
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"Duplicate prediction key: {key}")
    wanted_set = set(wanted)
    found, seen = {}, set()
    descriptive = Counter()
    for row in rows:
        if (row["farm"], row["turbine"], row["config"]) != (farm, turbine, VERSION):
            raise ValueError(f"HOSE experiment identity mismatch: {key}")
        stamp = utc(row["timestamp_utc"])
        if stamp in seen:
            raise ValueError(f"Duplicate HOSE key: {key} {stamp}")
        seen.add(stamp)
        if stamp in wanted_set:
            channels = Signals(**{name: row["signals"][name] for name in SIGNALS})
            fixed = physical_proxy(farm, channels)
            if fixed.regime != row["exogenous_operational_proxy"]["regime"]:
                raise ValueError(f"Stored proxy mismatch: {key} {stamp}")
            alternate = physical_proxy(farm, channels, "sensitivity")
            found[stamp] = fixed.regime, alternate.regime, fixed.missing
            descriptive[f"tier:{row['tier']}"] += 1
            descriptive["diagnostic:rotating_low_export"] += (
                row["retrospective_diagnostic"]["regime"] == "rotating_low_export"
            )
    if found.keys() != wanted_set:
        raise ValueError(f"Unmatched predictions: {key} {len(wanted_set - found.keys())}")
    return [found[t] for t in wanted], len(seen), dict(descriptive)


def reconcile(actual, expected):
    for key in ("n", "turbines"):
        if actual[key] != expected[key]:
            raise ValueError(f"Frozen aggregate cohort mismatch: {key}")
    for model in MODELS:
        for weight in ("turbine_macro", "row_pooled"):
            for key, value in expected["models"][model][weight].items():
                if not math.isclose(actual["models"][model][weight][key], value,
                                    rel_tol=1e-11, abs_tol=1e-9):
                    raise ValueError(f"Frozen aggregate mismatch: {model} {weight} {key}")


def group_stats(targets, preset_index, regime):
    subset, missing = [], Counter()
    for t in targets:
        mask = np.asarray([item[preset_index] == regime for item in t["joined"]], dtype=bool)
        if mask.any():
            subset.append({
                **{k: t[k][mask] for k in ("seconds", "bins", "delta")},
                "radii": t["radii"], "errors": {m: t["errors"][m][mask] for m in MODELS},
            })
            for item, keep in zip(t["joined"], mask, strict=True):
                if keep:
                    missing.update(item[2])
    result = summarize(subset)
    if not subset:
        return {**result, "missing": {}, "bootstrap_24h": None}
    models = result["models"]
    return {
        **result, "missing": {key: missing[key] for key in SIGNALS},
        "delta_mae_kw": {weight: models["hybrid"][weight]["mae_kw"] - models["binned"][weight]["mae_kw"]
                         for weight in ("turbine_macro", "row_pooled")},
        "bootstrap_24h": bootstrap(subset, 24) if len(subset) >= 2 and result["n"] >= 100
        and all(len(set(t["seconds"] // 86400)) >= 8 for t in subset) else None,
    }


def write_predictions(path, targets, frozen):
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["farm", "turbine", "timestamp_utc", "cohort", "split", "donors", "observed_kw",
                      *(f"{m}_{k}_kw" for m in MODELS for k in ("prediction", "lower", "upper")),
                      "proxy_regime", "alternate_regime"])
        for key, t in targets.items():
            farm, turbine = key.split("/")
            for i, stamp in enumerate(t["timestamps"]):
                bounds = []
                for model in MODELS:
                    p, r = t["predictions"][model][i], t["radii"][model]
                    bounds.extend((p, p - r, p + r))
                out.writerow([farm, turbine, utc(stamp).isoformat(), "cross_farm_source_only", SPLIT,
                              "/".join(frozen[key]["donors"]["cross"]), t["observed"][i], *bounds,
                              *t["joined"][i][:2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "paired", "hose", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    manifest_path = args.benchmark / "manifest.json"
    manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = verified_json(args.paired, PAIR_SHA256)
    hose = verified_json(args.hose / "summary.json", HOSE_SHA256)
    if (manifest["year"] != 2017 or manifest["smoke"] or manifest["config"] != CONFIG
            or pair["provenance"]["benchmark_manifest_sha256"] != manifest_sha
            or hose["benchmark_manifest_sha256"] != manifest_sha or hose["inputs"] != manifest["inputs"]
            or hose["preset"] != {key: list(value) for key, value in PRESETS.items()}
            or hose["version"] != VERSION):
        raise ValueError("Incompatible experiment provenance")
    for name, digest in manifest["artifact_sha256"].items():
        verified_json(args.benchmark / name, digest)
    frozen = verified_json(args.benchmark / "metrics.json", manifest["artifact_sha256"]["metrics.json"])
    with threadpool_limits(limits=2):
        targets = recover_targets(frozen, load_partitions(manifest))
    groups = {"overall": list(targets.values())}
    groups.update({site: [t for k, t in targets.items() if k.split("/")[0] == site]
                   for site in manifest["inputs"]["2017"]})
    canonical = {"overall": pair["overall"], **pair["directions"]}
    for scope, group in groups.items():
        reconcile(summarize(group), canonical[scope])
    if targets.keys() != hose["turbines"].keys():
        raise ValueError("HOSE target cohort differs")
    qa = {}
    for key, target in targets.items():
        partition = key.replace("/", "-") + ".jsonl.gz"
        path = args.hose / partition
        if file_hash(path) != hose["sha256"][partition]:
            raise ValueError(f"HOSE partition checksum mismatch: {key}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            joined, n, descriptive = join_rows(key, target["timestamps"], map(json.loads, stream))
        if n != hose["turbines"][key]["rows"]:
            raise ValueError(f"HOSE count mismatch: {key}")
        target["joined"] = joined
        qa[key] = {"hose_rows": n, "eligible_predictions": target["eligible"], "matched": len(joined),
                   "no_common_support": target["eligible"] - len(joined), "unmatched_predictions": 0,
                   "hose_only": n - len(joined), "prediction_duplicates": 0, "hose_duplicates": 0,
                   "retrospective_descriptive_only": descriptive}
    summary = {
        "evidence": "exploratory_retrospective_cubico_2017", "benchmark_manifest_sha256": manifest_sha,
        "full": canonical, "join_qa": qa, "nominal_coverage": 0.9,
        "strata": {scope: {preset: {regime: group_stats(group, i, regime) for regime in REGIMES}
                           for i, preset in enumerate(PRESETS)}
                   for scope, group in groups.items()},
    }
    args.output.mkdir(parents=True)
    prediction_path = args.output / "predictions.csv.gz"
    write_predictions(prediction_path, targets, frozen)
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        "code_commit": _git_commit(),
        "source_sha256": {"benchmark_manifest": manifest_sha, "paired": PAIR_SHA256, "hose": HOSE_SHA256},
        "input_pins": manifest["inputs"], "model_config": CONFIG, "proxy_presets": PRESETS,
        "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "split": SPLIT,
        "python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
        "code_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/v2_hose_integration.py", "scripts/v2_paired.py", "scripts/v2_benchmark.py",
            "src/wintellisyt/operating_state.py", "docs/ARCHITECTURE_AND_METHODS.md")},
        "sha256": {p.name: file_hash(p) for p in (prediction_path, summary_path)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Joined {sum(t['matched'] for t in qa.values()):,} frozen prediction rows")


if __name__ == "__main__":
    main()
