import math
import unittest

from wintellisyt.power_curve import fit_power_curve


def observation(wind_speed_mps, power_kw, operational_state="unknown"):
    return {
        "wind_speed_mps": wind_speed_mps,
        "power_kw": power_kw,
        "operational_state": operational_state,
    }


class PowerCurveTest(unittest.TestCase):
    def test_half_metre_bins_are_centered_and_left_closed(self):
        curve = fit_power_curve(
            [
                observation(0, -5, "normal"),
                observation(math.nextafter(0.25, 0), 5, "normal"),
                observation(0.25, 100, "normal"),
                observation(math.nextafter(0.75, 0), 200, "normal"),
                observation(0.75, 400, "normal"),
            ]
        )

        self.assertEqual(
            [(b.lower_mps, b.upper_mps, b.count) for b in curve.normal_bins],
            [(-0.25, 0.25, 2), (0.25, 0.75, 2), (0.75, 1.25, 1)],
        )
        self.assertEqual(curve.predict(0.0), 0)
        self.assertEqual(curve.predict(0.25), 150)
        self.assertEqual(curve.predict(0.75), 400)
        self.assertIsNone(curve.predict(-0.1))

    def test_raw_curve_includes_all_states_and_duplicate_records(self):
        normal = observation(8.0, 900, "normal")
        curve = fit_power_curve(
            iter([normal, normal, observation(8.0, 100, "curtailed"), observation(8.0, 300)])
        )

        self.assertEqual(curve.input_count, 4)
        self.assertEqual(curve.rejected_count, 0)
        self.assertEqual(curve.normal_status, "available")
        self.assertEqual(curve.raw_bins[0].count, 4)
        self.assertEqual(curve.normal_bins[0].count, 2)
        self.assertEqual(curve.predict(8.0, stratum="raw"), 550)
        self.assertEqual(curve.predict(8.0), 900)

    def test_sparse_and_unseen_bins_do_not_predict(self):
        curve = fit_power_curve(
            [
                observation(4.0, 100, "normal"),
                observation(4.5, 200, "normal"),
                observation(4.5, 300, "normal"),
                observation(5.5, 500, "normal"),
            ],
            min_bin_count=2,
        )

        self.assertEqual(curve.normal_status, "available")
        self.assertEqual(curve.predict(4.5), 250)
        for speed, count, mean_power in [(4.0, 1, 100), (5.0, 0, None), (20.0, 0, None)]:
            with self.subTest(speed=speed):
                power_bin = curve.bin_at(speed)
                self.assertEqual(power_bin.count, count)
                self.assertEqual(power_bin.mean_power_kw, mean_power)
                self.assertFalse(power_bin.supported)
                self.assertIsNone(curve.predict(speed))

    def test_status_distinguishes_unavailable_normal_from_insufficient_support(self):
        unknown = fit_power_curve([observation(9.0, 800), {"wind_speed_mps": 9, "power_kw": 600}])
        sparse = fit_power_curve([observation(9.0, 800, "normal")], min_bin_count=2)

        self.assertEqual(unknown.normal_status, "unavailable")
        self.assertEqual(unknown.normal_bins, ())
        self.assertIsNone(unknown.predict(9.0))
        self.assertEqual(unknown.predict(9.0, stratum="raw"), 700)
        self.assertEqual(sparse.normal_status, "insufficient_support")
        self.assertIsNone(sparse.predict(9.0))
        self.assertIsNone(fit_power_curve([]).predict(9.0, stratum="raw"))

    def test_invalid_measurements_are_counted_and_queries_are_explicit(self):
        curve = fit_power_curve(
            [
                observation(9.0, 800),
                observation(9.0, math.inf, "normal"),
                observation(-1, 10),
                observation(math.nan, 10),
                observation(True, 10),
                observation(9, "800"),
                {},
            ]
        )

        self.assertEqual(curve.input_count, 7)
        self.assertEqual(curve.rejected_count, 6)
        self.assertEqual(curve.raw_bins[0].count, 1)
        self.assertEqual(curve.normal_status, "unavailable")
        self.assertIsNone(curve.bin_at(math.nan))
        with self.assertRaises(ValueError):
            curve.predict(9.0, stratum="all")
        for minimum in (0, -1, 1.5, True):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                fit_power_curve([], min_bin_count=minimum)
