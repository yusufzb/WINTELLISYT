import unittest
from datetime import date, timedelta

from wintellisyt.pc_health import monitor
from wintellisyt.wind_geometry import potential_overlap, source_bearing, wake_projection


class EngineeringTests(unittest.TestCase):
    def test_cardinal_bearings_and_from_to(self):
        for east, north, angle in ((0, 100, 0), (100, 0, 90), (0, -100, 180), (-100, 0, 270)):
            self.assertEqual(source_bearing(east, north), angle)
            self.assertAlmostEqual(wake_projection(east, north, angle)[0], 100)
            self.assertTrue(potential_overlap(east, north, angle, 90, 82))
            self.assertFalse(potential_overlap(east, north, (angle + 180) % 360, 90, 82))
        self.assertTrue(potential_overlap(0, 700, 359, 90, 82))
        self.assertFalse(potential_overlap(0, 1200, 1, 90, 82))

    def test_monitor_causality_and_left_censor(self):
        day = date(2017, 11, 1)
        series = [(day, 0), (day + timedelta(days=1), -35), (day + timedelta(days=3), -70)]
        train = [0.] * 10
        result = monitor(series, train)
        self.assertEqual(monitor(series[:-1], train), result[:-1])
        self.assertFalse(result[0].alert)
        self.assertTrue(result[1].alert)
        self.assertFalse(result[1].onset_left_censored)
        self.assertTrue(result[2].onset_left_censored)
