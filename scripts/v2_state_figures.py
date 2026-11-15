"""Plot evidence-tier coverage and anchor sensitivity from the HOSE summary."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wintellisyt.pipeline import file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    turbines = state["turbines"]
    names = [f"{site.title()} {int(key):02}" for site, key in map(lambda x: x.split("/"), turbines)]
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    left = [0.0] * len(turbines)
    for tier, label, color in (
        ("INFERRED", "Inferred", "#4477AA"), ("UNKNOWN", "Abstained", "#BBBBBB"),
        ("DOCUMENTED", "Logged event", "#EE7733"),
        ("CORROBORATED_INFERENCE", "Corroborated inference", "#228833"),
    ):
        values = [100 * row["tiers"].get(tier, 0) / row["rows"] for row in turbines.values()]
        ax.barh(names, values, left=left, color=color, label=label)
        left = [a + b for a, b in zip(left, values, strict=True)]
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Share of unique SCADA timestamps (%)",
           title="2017 Cubico operating-state evidence by turbine")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncols=2)

    fig2, ax2 = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, site in enumerate(state["site_models"]):
        counts = sum((Counter(v["stop_membership"]) for k, v in turbines.items() if k.startswith(site + "/")),
                     Counter())
        heights = [counts["one"], counts["one_midpoint"]]
        bars = ax2.bar([i - 0.17, i + 0.17], heights, width=0.32, color=["#4477AA", "#EE7733"])
        ax2.bar_label(bars, padding=3)
    ax2.set(xticks=[0, 1], xticklabels=[site.title() for site in state["site_models"]],
            ylabel="Unique timestamps", title="Stop membership disagreements between candidate anchors")
    ax2.legend(bars, ["Any overlap", "Midpoint membership"])

    for stem, figure in (("evidence_tiers", fig), ("anchor_sensitivity", fig2)):
        for extension in ("png", "svg"):
            figure.savefig(args.output / f"{stem}.{extension}", dpi=180)
        plt.close(figure)
    provenance = {"summary_sha256": file_hash(args.summary), "script_sha256": file_hash(Path(__file__)),
                  "sha256": {p.name: file_hash(p) for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
