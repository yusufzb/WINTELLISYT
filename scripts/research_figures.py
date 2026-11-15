"""Generate research figures from the three verified turbine-control summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

RUNS = (
    ("2016", "penmanshiel", "Penmanshiel 2016"),
    ("2016", "kelmarsh", "Kelmarsh 2016"),
    ("2017", "penmanshiel", "Penmanshiel 2017"),
    ("2017", "kelmarsh", "Kelmarsh 2017"),
    ("new-2017", "penmanshiel_new_targets", "Penmanshiel 2017, new targets"),
)
MODELS = (
    ("source_only", "Cross-farm", "#A53A12", "o"),
    ("within_farm", "Within-farm", "#006B89", "s"),
    ("local", "Local history", "#424951", "^"),
)
METHODS = (("pooled", "Pooled calibration"), ("wind_band", "Wind-band calibration"))


def read_cohorts(runs):
    for suffix, cohort, label in RUNS:
        result = json.loads((runs / f"turbine-control-{suffix}.json").read_text(encoding="utf-8"))
        if result["source_evidence"] != "verified_real":
            raise ValueError(f"Expected verified real input for {label}")
        yield label, result["comparisons"][cohort]["summary"]


def save(fig, output, name):
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    fig.savefig(output / f"{name}.png", dpi=180, metadata={"Software": "WINTELLISYT"})
    plt.close(fig)


def figures(cohorts, output):
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "wintellisyt",
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for ax, (method, title) in zip(axes, METHODS, strict=True):
        for index, (model, label, color, marker) in enumerate(MODELS):
            x = [100 * summary["budgets"][model][method]["coverage"] for _, summary in cohorts]
            y = [i + (index - 1) * 0.17 for i in range(len(cohorts))]
            ax.scatter(x, y, color=color, marker=marker, label=label, s=36, zorder=3)
        ax.axvline(90, ls="--", lw=1, color="#555555", label="90% target")
        ax.set(title=title, xlabel="Empirical coverage (%)", xlim=(0, 101))
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(cohorts)), [
        f"{label}\n{summary['n']:,} matched rows" for label, summary in cohorts
    ])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Prediction-interval coverage on matched Cubico holdouts", weight="bold")
    save(fig, output, "coverage")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True, layout="constrained")
    label, summary = cohorts[-1]
    for ax, (method, title) in zip(axes, METHODS, strict=True):
        for model, model_label, color, marker in MODELS:
            stats = summary["budgets"][model][method]
            ax.scatter(stats["width_kw"], 100 * stats["coverage"], color=color,
                       marker=marker, label=model_label, s=58)
        ax.axhline(90, ls="--", lw=1, color="#555555", label="90% target")
        ax.set(title=title, xlabel="Mean interval width (kW)", ylim=(40, 100))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"{label}\n{summary['n']:,} matched readings across nine turbines", weight="bold")
    save(fig, output, "coverage_width")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=root / "artifacts/runs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f"{name}.{ext}").exists()
           for name in ("coverage", "coverage_width") for ext in ("svg", "png")):
        parser.error("Use an output directory without existing research figures")
    figures(list(read_cohorts(args.runs)), args.output)


if __name__ == "__main__":
    main()
