"""Generate vector and preview figures from the verified V2 run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from wintellisyt.pipeline import file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / "manifest.json").read_text())
    if meta["smoke"] or meta["evidence"] != "verified_cubico_retrospective":
        raise ValueError("Expected a full verified Cubico run")
    for name, digest in meta["artifact_sha256"].items():
        if file_hash(args.run / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    metrics = json.loads((args.run / "metrics.json").read_text())
    physics = json.loads((args.run / "physics.json").read_text())
    args.output.mkdir(parents=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "wintellisyt-v2"})
    figures = {}

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for column, (site, turbine) in enumerate((("kelmarsh", "1"), ("penmanshiel", "11"))):
        bins = physics[site][turbine]
        axes[0, column].scatter(
            [float(v) for v in bins], [r["signals"]["power_kw"]["mean"] for r in bins.values()],
            s=[min(40, 4 + r["n"] ** 0.4) for r in bins.values()],
        )
        axes[0, column].set(title=f"{site.title()} WT{turbine}", ylabel="Mean power (kW)")
        cp = [(float(v), r["signals"]["cp_elec"]["median"]) for v, r in bins.items()
              if "cp_elec" in r["signals"] and r["n"] >= 20]
        axes[1, column].scatter(*zip(*cp))
        axes[1, column].set(xlabel="Nacelle wind speed (m/s)", ylabel="Median electrical Cp")
    fig.suptitle("2017 observed bins and standard-density scenario")
    figures["v2_physics"] = fig

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    names = list(next(iter(metrics.values()))["budgets"]["local"]["models"])
    for budget, marker in (("local", "o"), ("within", "s"), ("cross", "^")):
        means = [sum(r["budgets"][budget]["models"][model]["mae_kw"] for r in metrics.values())
                 / len(metrics) for model in names]
        ax.plot(names, means, marker=marker, label=budget)
    ax.set(ylabel="Turbine-weighted MAE (kW)", title=f"Matched 2017 holdouts · {len(metrics)} turbines")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    figures["v2_model_error"] = fig

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for budget, marker in (("local", "o"), ("within", "s"), ("cross", "^")):
        for name in ("binned", "hybrid"):
            values = [r["budgets"][budget]["models"][name] for r in metrics.values()]
            ax.scatter(sum(v["width_kw"] for v in values)/len(values),
                       100*sum(v["coverage"] for v in values)/len(values),
                       marker=marker, label=f"{budget} {name}")
    ax.axhline(90, ls="--", color="gray")
    ax.set(xlabel="Mean width (kW)", ylabel="Coverage (%)", title="Frozen 90% interval transfer")
    ax.legend(fontsize=8)
    figures["v2_transfer"] = fig

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    replays = [r["feedback"] for r in metrics.values() if "feedback" in r]
    for name in ("cross_farm", "within_farm", "rolling_cross_farm"):
        values = [100*sum(r[str(delay)]["metrics"][name]["coverage"] for r in replays)/len(replays)
                  for delay in (0, 1, 24)]
        ax.plot(["0", "1", "24"], values, marker="o", label=name.replace("_", " "))
    ax.axhline(90, ls="--", color="gray")
    ax.set(xlabel="Feedback delay (hours)", ylabel="Coverage (%)",
           title=f"Hybrid interval replay · {len(replays)} Penmanshiel turbines")
    ax.legend()
    figures["v2_feedback"] = fig

    for name, fig in figures.items():
        for extension in ("svg", "pdf", "png"):
            fig.savefig(args.output / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    hashes = {p.name: file_hash(p) for p in sorted(args.output.iterdir())}
    (args.output / "manifest.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
