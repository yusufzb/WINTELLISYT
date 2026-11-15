"""Write or verify hashes for the aggregate research-release artifacts."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from xml.etree import ElementTree

ASSETS = ("CHANNEL_COVERAGE.csv", "coverage.svg", "coverage.png",
          "coverage_width.svg", "coverage_width.png")
RUNS = ("turbine-control-2016.json", "turbine-control-2017.json",
        "turbine-control-new-2017.json")


def inventory(release, runs):
    paths = {f"release/{name}": release / name for name in ASSETS}
    paths.update({f"runs/{name}": runs / name for name in RUNS})
    for name in ("adaptive-interval-replay-2017.json", "adaptive-interval-delay-sensitivity-2017.json"):
        if (runs / name).is_file():
            paths[f"runs/{name}"] = runs / name
    for name in ASSETS:
        if name.endswith(".svg") and not ElementTree.parse(release / name).getroot().tag.endswith("svg"):
            raise ValueError(f"Invalid SVG: {name}")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=root / "artifacts/runs")
    parser.add_argument("--release", type=Path, default=root / "artifacts/research-release")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = args.release / "MANIFEST.json"
    digests = inventory(args.release, args.runs)
    if args.verify:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["sha256"] != digests:
            raise ValueError("Release artifact inventory or hash mismatch")
        print(f"Verified {len(digests)} artifact hashes")
        return
    manifest = {"schema_version": 2, "source_evidence": "verified_real",
                "python_version": platform.python_version(), "sha256": digests}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {len(digests)} artifact hashes to {path}")


if __name__ == "__main__":
    main()
