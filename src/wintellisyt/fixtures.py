"""Deterministic, explicitly synthetic exercise data."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


def write_fixture(path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp_utc", "turbine_id", "wind_speed_ms", "active_power_kw", "state"))
        for interval in range(120):
            for index, turbine in enumerate(("SYN-01", "SYN-02")):
                if interval == 19 and index == 0:
                    continue
                timestamp = start + timedelta(minutes=10 * interval)
                speed = 4 + ((interval + index) % 16) * 0.5 + (interval % 3) * 0.1
                power = round(min(2200, 1.25 * speed**3) + 8 * ((interval % 5) - 2), 2)
                state = "curtailed" if interval % 11 == 0 else "normal"
                if state == "curtailed":
                    power = round(0.6 * power, 2)
                row = (timestamp.isoformat(), turbine, speed, power, state)
                writer.writerow(row)
                if interval == 9 and index == 0:
                    writer.writerow(row)
