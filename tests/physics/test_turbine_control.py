import unittest
from datetime import datetime, timedelta, timezone

from scripts.evaluate_transfer_intervals import calibrate, evaluate_target, fit
from scripts.evaluate_turbine_control import donor_ids
from wintellisyt.pipeline import Observation


class TurbineControlTests(unittest.TestCase):
    def test_selects_four_sorted_donors_and_omits_target(self):
        self.assertEqual(donor_ids({"10", "2", "1", "4", "5"}, "2"), ("1", "4", "5", "10"))
        with self.assertRaisesRegex(ValueError, "Four distinct"):
            donor_ids({"1", "2", "4", "5"}, "2")

    def test_scoring_intersects_within_and_cross_support(self):
        t = datetime(2017, 11, 1, tzinfo=timezone.utc)

        def rows(speed, power):
            return [Observation(t + timedelta(minutes=10 * i), "1", speed, power, "unknown")
                    for i in range(60)]

        cross = fit(rows(8, 100) + rows(10, 100))
        within = fit(rows(8, 200))
        within_cal = calibrate(within, rows(8, 205))
        split = {"train": rows(8, 300) + rows(10, 300),
                 "calibration": rows(8, 305) + rows(10, 305),
                 "test": rows(8, 300) + rows(10, 300)}
        scored, audit = evaluate_target(
            cross, calibrate(cross, rows(8, 105)), split,
            within_farm=(within, within_cal),
        )
        self.assertEqual(audit["eligible"], 120)
        self.assertEqual(audit["available"]["source_only"]["pooled"], 120)
        self.assertEqual(audit["available"]["within_farm"]["pooled"], 60)
        self.assertEqual(len(scored), 60)
        self.assertEqual(audit["calibration"]["within_farm"]["pooled_radius_kw"], 5)
