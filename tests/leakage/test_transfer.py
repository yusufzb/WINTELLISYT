import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from wintellisyt.pipeline import Observation
from wintellisyt.transfer import transfer


class TransferTests(unittest.TestCase):
    def test_target_labels_and_future_source_rows_do_not_affect_fit(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def rows(turbine, powers):
            return [
                Observation(start + timedelta(minutes=index * 10), turbine, 7, power, "unknown")
                for index, power in enumerate(powers)
            ]

        def ingest(path, source, turbine):
            return iter(rows(turbine, [10] * 8 + [900] * 2 if turbine == "A" else [500] * 10))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("wintellisyt.transfer.load_rows", side_effect=ingest), patch(
                "wintellisyt.transfer.file_hash", return_value="hash"
            ), patch(
                "wintellisyt.transfer.source_manifest",
                return_value={"site": {"acquisition": {"sha256": "hash"},
                                       "source_url": "https://example.invalid",
                                       "license": None}},
            ):
                result = transfer(root / "in.csv", "site", "A", root / "in.csv", "site", "B",
                                  root / "out.json", 1)
        self.assertLess(result["latest_train_utc"], result["target_cutoff_utc"])
        self.assertEqual(result["train_median_kw"], 10)
        self.assertEqual(result["binned_shared_support"]["mae_kw"], 490)
        self.assertEqual(result["test_supported_pairs"], 2)

    def test_rejects_same_turbine(self):
        with self.assertRaisesRegex(ValueError, "different turbines"):
            transfer(Path("in"), "site", "A", Path("in"), "site", "A", Path("out"))

    def test_rejects_unselected_turbine(self):
        with self.assertRaisesRegex(ValueError, "turbine IDs are required"):
            transfer(Path("in"), "site", None, Path("in"), "site", "A", Path("out"))
