import math
import unittest

from wintellisyt.physics import (
    cp_electrical,
    dry_air_density,
    normalized_speed,
    relative_direction,
    swept_area,
    tip_speed_ratio,
    turbulence_proxy,
    wind_power,
)


class PhysicsTests(unittest.TestCase):
    def test_si_units(self):
        self.assertAlmostEqual(swept_area(2), 4 * math.pi)
        self.assertAlmostEqual(wind_power(4, 2, 1), 128 * math.pi)
        self.assertAlmostEqual(cp_electrical(0.128 * math.pi, 4, 2, 1), 1)
        self.assertAlmostEqual(tip_speed_ratio(60, 4, 2), math.pi)
        self.assertAlmostEqual(dry_air_density(28705, 100), 1)
        self.assertAlmostEqual(normalized_speed(4, 1, 8), 2)
        self.assertEqual(turbulence_proxy(2, 10), 0.2)
        self.assertEqual(relative_direction(1, 359), 2)

    def test_low_wind_and_invalid_values(self):
        self.assertIsNone(cp_electrical(1, 0, 2, 1))
        self.assertIsNone(tip_speed_ratio(-1, 8, 2))
        self.assertIsNone(turbulence_proxy(float("nan"), 5))
        self.assertLess(cp_electrical(-1, 8, 2, 1), 0)
        with self.assertRaises(ValueError):
            dry_air_density(100000, 0)
