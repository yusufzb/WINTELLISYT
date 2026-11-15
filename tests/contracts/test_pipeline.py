import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from wintellisyt.cli import main
from wintellisyt.pipeline import Observation, assess


class PipelineTests(unittest.TestCase):
    def test_duplicate_quarantine_and_cadence_are_per_turbine(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        first = Observation(start, "A", 7, 90, "unknown", source_row=2)
        copy = Observation(start, "A", 7, 100, "unknown", source_row=3)
        later = Observation(start + timedelta(minutes=30), "A", 7, 90, "unknown", source_row=4)
        other = Observation(start, "B", 7, 90, "unknown", source_row=5)
        clean, qa = assess([first, copy, later, other])
        self.assertEqual(clean, [later, other])
        self.assertEqual(qa["quarantined_rows"], 2)
        self.assertEqual(qa["duplicate_groups"][0]["source_rows"][1]["row"], 3)
        self.assertEqual(len(qa["gaps"]), 1)
        self.assertEqual(qa["gaps"][0]["unobserved_expected_slots"], 2)

    def test_demo_generates_hashed_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "demo"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(["demo", "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertIn("synthetic", stdout.getvalue())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_evidence"], "synthetic")
            self.assertEqual(manifest["schema_version"], 1)
            for artifact, digest in manifest["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((output / artifact).read_bytes()).hexdigest(), digest)
            qa = json.loads((output / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["rows"], 240)
            self.assertEqual(qa["quarantined_rows"], 2)
            self.assertEqual(qa["gaps"][0]["unobserved_expected_slots"], 1)
            with (output / "bins.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(
                    {row["stratum"] for row in csv.DictReader(stream)},
                    {"raw", "screened", "normal"},
                )
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as caught:
                    main(["demo", "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
