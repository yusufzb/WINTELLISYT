import unittest
from datetime import datetime, timedelta, timezone

from wintellisyt.energy import PowerInterval, model_referenced_energy


class EnergyDeviationTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2017, 11, 1, tzinfo=timezone.utc)
        self.step = timedelta(minutes=10)

    def interval(self, index, reference, observed):
        start = self.start + index * self.step
        return PowerInterval(start, start + self.step, reference, observed)

    def integrate(self, rows):
        return model_referenced_energy(
            rows, expected_duration=self.step,
            window_start=self.start, window_end=self.start + 4 * self.step,
        )

    def test_gaps_are_excluded_and_surplus_is_reported(self):
        result = self.integrate([self.interval(2, 60, 120), self.interval(0, 120, 0)])
        self.assertEqual(result.intervals, 2)
        self.assertAlmostEqual(result.covered_hours, 1 / 3)
        self.assertAlmostEqual(result.eligible_fraction, 0.5)
        self.assertEqual((result.signed_kwh, result.gross_shortfall_kwh,
                          result.gross_surplus_kwh), (10, 20, 10))

    def test_empty_and_duplicate_intervals(self):
        self.assertEqual(self.integrate([]).eligible_fraction, 0)
        row = self.interval(0, 120, 0)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.integrate([row, row])

    def test_rejects_invalid_intervals(self):
        invalid = (
            self.interval(-1, 100, 0),
            self.interval(0, float("nan"), 0),
            PowerInterval(self.start, self.start + 2 * self.step, 100, 0),
            PowerInterval(self.start.replace(tzinfo=None), self.start + self.step, 100, 0),
        )
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.integrate([row])
