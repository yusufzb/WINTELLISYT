"""Fetch and cache 2016-2017 hourly ERA5 reanalysis data from Open-Meteo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "external" / "era5"
VARIABLES = (
    "temperature_2m,relative_humidity_2m,surface_pressure,"
    "pressure_msl,wind_direction_100m,wind_speed_100m"
)


def fetch_farm_era5(farm: str, out_dir: Path, *, overwrite: bool = False) -> Path:
    """Download 2016-2017 hourly ERA5 data for a farm centroid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    static_path = ROOT / "data" / "raw" / farm / f"{farm.title()}_WT_static.csv"
    with static_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("Latitude") and r.get("Longitude")]
    lat = sum(float(r["Latitude"]) for r in rows) / len(rows)
    lon = sum(float(r["Longitude"]) for r in rows) / len(rows)
    elev_vals = [float(r["Elevation (m)"]) for r in rows if r.get("Elevation (m)")]
    mean_elev = sum(elev_vals) / len(elev_vals) if elev_vals else None

    output = out_dir / f"{farm}.json"
    manifest = out_dir / f"{farm}.manifest.json"
    if output.exists() and manifest.exists() and not overwrite:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        actual_sha = hashlib.sha256(output.read_bytes()).hexdigest()
        if actual_sha == meta.get("sha256"):
            print(f"{farm}: cached and verified ({actual_sha[:12]})")
            return output

    query = urlencode(
        {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "start_date": "2016-01-01",
            "end_date": "2017-12-31",
            "hourly": VARIABLES,
            "models": "era5",
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"
    with urlopen(url, timeout=45) as response:
        data = response.read()
    parsed = json.loads(data)
    hourly = parsed.get("hourly", {})
    if parsed.get("utc_offset_seconds") != 0 or any(
        col not in hourly for col in VARIABLES.split(",")
    ):
        raise ValueError(f"Unexpected ERA5 schema or UTC offset for {farm}")

    sha256 = hashlib.sha256(data).hexdigest()
    output.write_bytes(data)
    manifest.write_text(
        json.dumps(
            {
                "farm": farm,
                "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256,
                "bytes": len(data),
                "requested_coordinates": [lat, lon],
                "mean_turbine_ground_elevation_m": mean_elev,
                "returned_coordinates": [parsed.get("latitude"), parsed.get("longitude")],
                "returned_elevation_m": parsed.get("elevation"),
                "units": parsed.get("hourly_units"),
                "model": "era5",
                "attribution": "Open-Meteo Historical Weather API / Copernicus ERA5",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{farm}: downloaded {len(data):,} bytes (sha256={sha256[:12]})")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for farm in ("kelmarsh", "penmanshiel"):
        try:
            fetch_farm_era5(farm, args.out_dir, overwrite=args.overwrite)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"WARN: Failed to fetch ERA5 for {farm}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
