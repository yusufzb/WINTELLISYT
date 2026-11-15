import unittest
from datetime import datetime, timedelta, timezone

from scripts.replay_interval_calibration import replay
from wintellisyt.pipeline import Observation


class ConstantCurve:
    def predict(self, speed, *, stratum):
        return 100 if speed == 8 else None


class AdaptiveCalibrationTests(unittest.TestCase):
    def test_current_outcome_only_changes_later_radius(self):
        start = datetime(2017, 11, 1, tzinfo=timezone.utc)

        def run(changed_power):
            rows = [
                Observation(start + timedelta(minutes=10 * i), "01", 8,
                            changed_power if i == 50 else 110, "unknown")
                for i in range(52)
            ]
            return replay(reversed(rows), ConstantCurve(), 5, ConstantCurve(), 7)

        a, counts = run(110)
        b, _ = run(1000)
        self.assertEqual(counts["matched_warmup_rows"], 50)
        self.assertEqual(a[49][2][2], 5)
        self.assertEqual(a[50][2][2], 10)
        self.assertEqual(a[50][2], b[50][2])
        self.assertNotEqual(a[50][1], b[50][1])

    def test_outcomes_expire_from_rolling_history(self):
        start = datetime(2017, 11, 1, tzinfo=timezone.utc)
        rows = [
            Observation(start + timedelta(minutes=10 * i), "01", 8,
                        500 if i < 200 else 110, "unknown")
            for i in range(1209)
        ]
        scored, counts = replay(rows, ConstantCurve(), 5, ConstantCurve(), 7)
        self.assertEqual(counts["feedback_rows"], 1209)
        self.assertEqual(scored[1008][2][2], 400)
        self.assertEqual(scored[-1][2][2], 10)

    def test_zero_delay_does_not_use_outcomes_at_the_prediction_timestamp(self):
        start = datetime(2017, 11, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(minutes=10 * i) for i in range(50)]
        timestamps.extend((timestamps[-1], timestamps[-1] + timedelta(minutes=10)))
        rows = [Observation(timestamp, "01", 8, 110, "unknown") for timestamp in timestamps]
        scored, counts = replay(rows, ConstantCurve(), 5, ConstantCurve(), 7)
        self.assertEqual(counts["matched_warmup_rows"], 51)
        self.assertEqual([row[2][2] for row in scored[-3:]], [5, 5, 10])

    def test_delayed_outcomes_arrive_at_exact_wall_clock_boundary(self):
        start = datetime(2017, 11, 1, tzinfo=timezone.utc)
        for hours in (1, 24):
            with self.subTest(hours=hours):
                delay = timedelta(hours=hours)
                history = [start + timedelta(seconds=i) for i in range(50)]
                arrival = history[-1] + delay
                timestamps = history + [arrival - timedelta(microseconds=1), arrival]
                rows = [
                    Observation(timestamp, "01", 8, 110, "unknown") for timestamp in timestamps
                ]
                scored, counts = replay(
                    reversed(rows), ConstantCurve(), 5, ConstantCurve(), 7,
                    feedback_delay_hours=hours,
                )
                self.assertEqual(counts["scored"], len(rows))
                self.assertEqual(counts["matched_warmup_rows"], 51)
                self.assertEqual([row[2][2] for row in scored[-2:]], [5, 10])
