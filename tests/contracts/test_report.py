import csv
import json
import tempfile
import unittest
from pathlib import Path

from wintellisyt.reporting import write_report


class ReportTests(unittest.TestCase):
    def test_report_without_manifest_and_with_empty_residuals(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            tables = {
                "bins.csv": (["turbine_id", "stratum", "lower_mps", "upper_mps", "count", "mean_power_kw", "supported"],
                             [["A", "raw", 2, 2.5, 4, 30, True]]),
                "observations.csv": (["wind_speed_ms", "active_power_kw"], [[2.2, 30]]),
                "residuals.csv": (["wind_speed_ms", "residual_kw"], []),
            }
            for name, (fields, rows) in tables.items():
                with (path / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerows(rows)
            (path / "qa.json").write_text(json.dumps({"rows": 1}), encoding="utf-8")
            write_report(path, source="synthetic", evidence="synthetic")
            self.assertIn("synthetic", (path / "report.html").read_text(encoding="utf-8"))
            self.assertIn("Wind speed (m/s)", (path / "curves.svg").read_text(encoding="utf-8"))
            self.assertIn("n=0", (path / "residuals.svg").read_text(encoding="utf-8"))
            self.assertFalse((path / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
