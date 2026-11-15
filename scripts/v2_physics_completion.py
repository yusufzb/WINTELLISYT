"""Summarize joint electrical Cp/TSR and plot selected 2017 physics channels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wintellisyt.data import read_scada  # noqa: E402
from wintellisyt.physics import STANDARD_DENSITY, cp_electrical, tip_speed_ratio  # noqa: E402
from wintellisyt.pipeline import file_hash  # noqa: E402


def joint_diagnostics(pin, site, turbine, radius):
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(pin["path"])
    path = raw_path if raw_path.is_absolute() else (root / raw_path)
    if file_hash(path) != pin["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {path}")
    rows = [r for r in read_scada(path, site, turbine_id=turbine) if r.timestamp_utc.year == 2017]
    timestamps = Counter(r.timestamp_utc for r in rows)
    unique = [r for r in rows if timestamps[r.timestamp_utc] == 1]
    points = []
    for r in unique:
        if r.active_power_kw is None or r.wind_speed_ms is None or r.rotor_speed_rpm is None:
            continue
        cp = cp_electrical(r.active_power_kw, r.wind_speed_ms, radius, STANDARD_DENSITY)
        tsr = tip_speed_ratio(r.rotor_speed_rpm, r.wind_speed_ms, radius)
        if cp is not None and tsr is not None:
            points.append((tsr, cp))
    if not points:
        raise ValueError(f"No joint TSR/Cp support for {site}/{turbine}")
    return np.asarray(points), len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.benchmark / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["year"] != 2017 or manifest["smoke"]:
        raise ValueError("Expected a full 2017 benchmark")
    physics_path = args.benchmark / "physics.json"
    if file_hash(physics_path) != manifest["artifact_sha256"]["physics.json"]:
        raise ValueError("Physics checksum mismatch")
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    results = {}
    for col, (site, turbine) in enumerate((("kelmarsh", "1"), ("penmanshiel", "11"))):
        radius = manifest["rotor_radii_m"][site]
        points, unique = joint_diagnostics(manifest["inputs"]["2017"][site], site, turbine, radius)
        tsr, cp = points.T
        chart = axes[0, col].hexbin(tsr, cp, gridsize=48, bins="log", mincnt=1)
        fig.colorbar(chart, ax=axes[0, col], label="Readings per cell")
        axes[0, col].set(title=f"{site.title()} WT{turbine}", xlabel="Tip-speed ratio",
                         ylabel="Effective electrical Cp")
        bins = physics[site][turbine]
        counts = {}
        for key, label, axis in (
            ("rotor_rpm", "Rotor speed (RPM)", axes[1, col]),
            ("pitch_deg", "Blade-A pitch (degrees)", axes[1, col].twinx()),
        ):
            supported = [(float(v), b["signals"][key]) for v, b in bins.items()
                         if key in b["signals"] and b["signals"][key]["n"] >= 20]
            counts[key] = sum(row["n"] for _, row in supported)
            color = "tab:blue" if key == "rotor_rpm" else "tab:orange"
            axis.set_ylabel(label, color=color)
            if supported:
                axis.plot([v for v, _ in supported], [row["median"] for _, row in supported],
                          color=color, marker=".", linestyle="none")
            else:
                axis.set_yticks([])
                axis.text(0.98, 0.08, "Blade-A pitch unavailable", transform=axis.transAxes,
                          ha="right", color=color)
        axes[1, col].set_xlabel("Nacelle wind speed (m/s), 0.5 m/s bins")
        results[f"{site}/{turbine}"] = {
            "unique_scada_rows": unique, "joint_tsr_cp_rows": len(cp), "rotor_radius_m": radius,
            "shown_bin_rows": counts, "cp_above_betz": int(np.count_nonzero(cp > 16 / 27)),
            "cp_below_zero": int(np.count_nonzero(cp < 0)),
            "median_cp_density_scenarios_kg_m3": {
                str(density): float(np.median(cp) * STANDARD_DENSITY / density)
                for density in (1.1, STANDARD_DENSITY, 1.3)
            },
        }
    fig.suptitle("2017 Cubico diagnostics · unknown operating states · assumed density 1.225 kg/m³")
    args.output.mkdir(parents=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(args.output / f"v2_tsr_cp.{extension}", dpi=180)
    results["provenance"] = {
        "benchmark_manifest_sha256": file_hash(source),
        "script_sha256": file_hash(Path(__file__)), "matplotlib": matplotlib.__version__,
        "artifacts": {p.name: file_hash(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
