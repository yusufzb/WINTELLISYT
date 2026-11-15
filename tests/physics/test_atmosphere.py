import unittest
from wintellisyt.atmosphere import hub_pressure, moist_density, reference_speed, site_speed


class AtmosphereTests(unittest.TestCase):
    def test_moist_air_and_hub_height(self):
        dry = moist_density(15, 1013.25, 0)
        self.assertAlmostEqual(dry, 101325 / (287.05 * 288.15))
        self.assertLess(moist_density(15, 1013.25, 80), dry)
        self.assertLess(hub_pressure(1000, 100, 288), 1000)
        with self.assertRaises(ValueError):
            moist_density(15, 1013, 101)

    def test_mapping_identity_sign_roundtrip_and_knots(self):
        for speed in (0, 5, 7.5, 10, 12.5, 20):
            self.assertAlmostEqual(site_speed(speed, 1.225), speed)
            self.assertAlmostEqual(reference_speed(site_speed(speed, 1.1), 1.1, 25), speed, places=12)
        self.assertGreater(site_speed(10, 1.1), 10)
        for knot in (7.5, 12.5):
            self.assertAlmostEqual(site_speed(knot - 1e-8, 1.1), site_speed(knot + 1e-8, 1.1), places=6)
        with self.assertRaises(ValueError):
            reference_speed(100, 1.1, 25)
        with self.assertRaises(ValueError):
            reference_speed(1, 100, 25)
