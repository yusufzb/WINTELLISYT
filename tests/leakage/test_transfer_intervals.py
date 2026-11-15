import unittest
from datetime import datetime, timedelta, timezone

from scripts.evaluate_transfer_intervals import (
    bootstrap,
    calibrate,
    evaluate_target,
    fit,
    interval_score,
    metrics,
    partition,
)
from wintellisyt.pipeline import Observation


class TransferIntervalTests(unittest.TestCase):
    def test_boundary_embargo(self):
        october = datetime(2016, 10, 1, tzinfo=timezone.utc)
        november = datetime(2016, 11, 1, tzinfo=timezone.utc)
        rows = [Observation(time, "1", 8, 100, "unknown") for time in (
            october - timedelta(minutes=20), october - timedelta(minutes=10), october,
            november - timedelta(minutes=10), november,
        )]
        parts = partition(rows, 2016)
        self.assertEqual([parts[part][0] for part in parts], [rows[0], rows[2], rows[4]])
        self.assertEqual(sum(map(len, parts.values())), 3)

    def test_target_labels_cannot_change_frozen_donor_interval(self):
        t = datetime(2016, 11, 1, tzinfo=timezone.utc)

        def rows(power):
            return [Observation(t + timedelta(minutes=i * 10), "1", 8, power, "unknown")
                    for i in range(60)]

        donor = fit(rows(100))
        donor_calibration = calibrate(donor, rows(110))
        target = {"train": rows(200), "calibration": rows(205), "test": rows(210)}
        scored, result = evaluate_target(donor, donor_calibration, target)
        self.assertEqual(result["eligible"], 60)
        self.assertEqual(result["calibration"]["source_only"]["pooled_radius_kw"], 10)
        self.assertEqual(result["calibration"]["target_recalibrated"]["pooled_radius_kw"], 105)
        self.assertEqual(metrics(scored)["budgets"]["source_only"]["pooled"]["coverage"], 0)
        target["test"] = rows(500)
        _, updated = evaluate_target(donor, donor_calibration, target)
        self.assertEqual(result["calibration"], updated["calibration"])

    def test_interval_score_and_synchronized_blocks(self):
        self.assertEqual(interval_score(10, 10), 20)
        self.assertEqual(interval_score(-15, 10), 120)
        origin = datetime(2016, 11, 1, tzinfo=timezone.utc)
        values = {("source_only", "pooled"): (15, 10), ("local", "pooled"): (0, 10)}
        scored = [(origin + timedelta(days=2 * i), "5–<10", values) for i in range(8)]
        result = bootstrap({"A": scored, "B": scored}, origin, 2, "pooled")
        self.assertEqual(result["blocks"], 8)
        self.assertEqual(result["range_kw"], [100, 100])
        self.assertEqual(bootstrap({"A": scored}, origin, 7, "pooled")["status"],
                         "insufficient_blocks")
