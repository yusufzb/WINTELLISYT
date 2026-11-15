import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install the V2 extra")
class HoseIntegrationTests(unittest.TestCase):
    def setUp(self):
        from scripts import v2_hose_integration

        self.module = v2_hose_integration
        self.stamp = "2017-11-01T00:00:00.123456+00:00"
        self.row = {
            "farm": "kelmarsh", "turbine": "1", "config": "hose-2017-v1",
            "timestamp_utc": "2017-11-01T01:00:00.123456+01:00",
            "signals": {name: None for name in self.module.SIGNALS},
            "exogenous_operational_proxy": {"regime": "unknown"},
            "retrospective_diagnostic": {"regime": "rotating_low_export"},
            "tier": "DOCUMENTED",
        }

    def test_utc_join_preserves_unknown_and_excludes_outcomes(self):
        row = self.row
        join = self.module.join_rows
        joined, count, descriptive = join("kelmarsh/1", [self.stamp], [row])
        self.assertEqual(joined[0][:2], ("unknown", "unknown"))
        self.assertEqual(count, 1)
        self.assertEqual(descriptive["diagnostic:rotating_low_export"], 1)
        row["signals"]["power_kw"] = 2050
        row["retrospective_diagnostic"]["regime"] = "rated_like"
        row["tier"] = "INFERRED"
        self.assertEqual(join("kelmarsh/1", [self.stamp], [row])[0], joined)

    def test_rejects_duplicates_naive_unmatched_and_mixed_identity(self):
        join = self.module.join_rows
        for stamps, rows, message in (
            ([self.stamp, self.row["timestamp_utc"]], [self.row], "Duplicate prediction"),
            ([self.stamp], [self.row, self.row], "Duplicate HOSE"),
            ([self.stamp[:-6]], [self.row], "Naive"),
            ([self.stamp], [], "Unmatched"),
            ([self.stamp], [{**self.row, "config": "other"}], "identity"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                join("kelmarsh/1", stamps, rows)

    def test_aggregate_mismatch_and_checksum(self):
        from hashlib import sha256
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with self.assertRaisesRegex(ValueError, "cohort mismatch"):
            self.module.reconcile({"n": 1}, {"n": 2})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_bytes(b"{}")
            self.assertEqual(self.module.verified_json(path, sha256(b"{}").hexdigest()), {})
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                self.module.verified_json(path, "0" * 64)

    def test_sparse_unknown_group_keeps_identical_model_support(self):
        import numpy as np

        target = {
            "joined": [("unknown", "unknown", ("rotor_rpm",)), ("rotating", "rotating", ())],
            "seconds": np.array([0, 600]), "bins": np.array([14, 14]),
            "delta": np.zeros((2, 3)), "radii": {"binned": 1, "hybrid": 2},
            "errors": {"binned": np.array([2, 99]), "hybrid": np.array([1, 99])},
        }
        result = self.module.group_stats([target], 0, "unknown")
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["missing"]["rotor_rpm"], 1)
        self.assertEqual(result["delta_mae_kw"]["turbine_macro"], -1)
        self.assertIsNone(result["bootstrap_24h"])
        self.assertEqual(result["models"]["binned"]["row_pooled"]["n"],
                         result["models"]["hybrid"]["row_pooled"]["n"])
