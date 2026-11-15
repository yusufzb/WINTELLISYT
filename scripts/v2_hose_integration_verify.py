"""Verify the integrated output and generate a matched subgroup figure."""

import argparse
import csv
import gzip
import json
import subprocess
from collections import Counter
from pathlib import Path

from wintellisyt.pipeline import file_hash

ROOT = Path(__file__).resolve().parents[1]

SOURCES = (
    "scripts/v2_hose_integration.py", "scripts/v2_paired.py", "scripts/v2_benchmark.py",
    "scripts/evaluate_turbine_control.py", "scripts/evaluate_transfer_intervals.py",
    "scripts/evaluate_intervals.py", "scripts/replay_interval_calibration.py",
    "src/wintellisyt/v2_models.py", "src/wintellisyt/data/scada.py", "src/wintellisyt/pipeline.py",
    "src/wintellisyt/operating_state.py", "src/wintellisyt/physics.py", "src/wintellisyt/data/__init__.py",
    "src/wintellisyt/power_curve/__init__.py", "docs/ARCHITECTURE_AND_METHODS.md", "pyproject.toml",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hose", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if file_hash(args.run / name) != digest:
            raise ValueError(f"Output checksum mismatch: {name}")
    revision = manifest.get("code_commit", "untracked")
    source_blobs = {}
    for path in SOURCES:
        file_p = ROOT / path
        git_blob = "untracked"
        if revision != "untracked":
            try:
                current = subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()
                committed = subprocess.check_output(["git", "rev-parse", f"{revision}:{path}"], cwd=ROOT, text=True).strip()
                if current != committed:
                    raise ValueError(f"Source differs from replay commit: {path}")
                git_blob = committed
            except (OSError, subprocess.CalledProcessError):
                pass
        source_blobs[path] = {"git_blob": git_blob, "sha256": file_hash(file_p)}
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    keys = {}
    counts = {"conservative": Counter(), "sensitivity": Counter()}
    with gzip.open(args.run / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["farm"], row["turbine"], row["timestamp_utc"])
            if key in keys or row["cohort"] != "cross_farm_source_only" or row["split"] != manifest["split"]:
                raise ValueError(f"Invalid output identity: {key}")
            keys[key] = (row["proxy_regime"], row["alternate_regime"])
            counts["conservative"][row["proxy_regime"]] += 1
            counts["sensitivity"][row["alternate_regime"]] += 1
    if len(keys) != summary["full"]["overall"]["n"]:
        raise ValueError("Output row count differs")
    for preset, values in summary["strata"]["overall"].items():
        if any(counts[preset][regime] != row["n"] for regime, row in values.items()):
            raise ValueError(f"Subgroup counts differ: {preset}")

    hose_path = args.hose / "summary.json"
    if file_hash(hose_path) != manifest["source_sha256"]["hose"]:
        raise ValueError("HOSE summary checksum mismatch")
    hose = json.loads(hose_path.read_text(encoding="utf-8"))
    invalid = {}
    matched = 0
    for target in summary["join_qa"]:
        farm, turbine = target.split("/")
        name = target.replace("/", "-") + ".jsonl.gz"
        if file_hash(args.hose / name) != hose["sha256"][name]:
            raise ValueError(f"HOSE partition checksum mismatch: {target}")
        with gzip.open(args.hose / name, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (farm, turbine, row["timestamp_utc"])
                if key not in keys:
                    continue
                matched += 1
                for preset, regime in zip(counts, keys[key], strict=True):
                    record = invalid.setdefault(f"{farm}/{preset}/{regime}", Counter())
                    for sensor, upper in (("rotor_rpm", 35), ("generator_rpm", 3000)):
                        value = row["signals"][sensor]
                        record[sensor] += value is not None and not 0 <= value <= upper
    if matched != len(keys):
        raise ValueError("Invalid-channel audit does not cover matched keys")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["strata"]["overall"]["conservative"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    labels = []
    for y, (name, group) in enumerate(groups.items()):
        labels.append(f"{name.replace('_', ' ').title()}  (n={group['n']:,})")
        if not group["n"]:
            continue
        point = group["delta_mae_kw"]["turbine_macro"]
        info = group["bootstrap_24h"]
        if info is not None and info["ci95"] is not None:
            lower, upper = info["ci95"]["turbine_macro"]["mae_kw"]
            ax.hlines(y, lower, upper, color="#4477AA", linewidth=2)
        ax.plot(point, y, "o", color="#225588")
    ax.axvline(0, color="0.5", linestyle="--")
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xlabel="Hybrid minus binned mean turbine-macro MAE (kW)",
           title="2017 Cubico cross-farm transfer by sensor proxy")
    ax.invert_yaxis()
    for extension in ("svg", "png"):
        fig.savefig(args.output / f"proxy_mae.{extension}", dpi=180)
    audit = {"verified_prediction_keys": len(keys), "source_commit": revision,
             "source_blobs": source_blobs, "run_manifest_sha256": file_hash(args.run / "manifest.json"),
             "invalid_rotation_channels": invalid,
             "figures": {p.name: file_hash(p) for p in args.output.iterdir()}}
    (args.output / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Verified {len(keys):,} prediction keys and {len(source_blobs)} replay source files")


if __name__ == "__main__":
    main()
