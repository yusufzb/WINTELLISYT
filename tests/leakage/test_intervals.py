import unittest
from datetime import datetime, timedelta, timezone

from scripts.evaluate_intervals import evaluate, radius
from wintellisyt.pipeline import Observation


class IntervalTests(unittest.TestCase):
    def test_rank_and_minimum_calibration_support(self):
        self.assertIsNone(radius(list(range(49))))
        self.assertEqual(radius(list(reversed(range(50)))), 45)

    def test_frozen_radii_and_reported_support(self):
        start = datetime(2016, 1, 1, tzinfo=timezone.utc)

        def run(test_power):
            rows = []
            for i in range(300):
                speed = 7 if i != 299 else 18
                power = 100 if i < 180 else (120 if i < 240 else test_power)
                rows.append(Observation(start + timedelta(minutes=10 * i), "1",
                                        speed, power, "unknown"))
            return evaluate(reversed(rows))

        boundary, missed = run(120), run(121)
        self.assertEqual(boundary["pooled_radius_kw"], 20)
        self.assertEqual(boundary["pooled_radius_kw"], missed["pooled_radius_kw"])
        self.assertEqual(boundary["pooled"]["coverage"], 1)
        self.assertEqual(missed["pooled"]["upper_misses"], 59)
        self.assertEqual(boundary["test_eligible"], 60)
        self.assertEqual(boundary["test_supported"], 59)
        self.assertEqual(boundary["bands_mps"]["15+"]["test_eligible"], 1)
        split = boundary["split"]
        self.assertLess(split["train_end_utc"], split["calibration_start_utc"])
        self.assertLess(split["calibration_end_utc"], split["test_start_utc"])
