"""Physics unit checks for circular wind direction primitives and active-donor wake geometry."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wintellisyt.direction import circular_mean_deg, wrap_deg
from wintellisyt.wake import classify_wake_exposure
from wintellisyt.wind_geometry import LayoutPoint, pair_vector, potential_overlap, source_bearing


class DirectionAndWakePhysicsTest(unittest.TestCase):
    def test_circular_wrap_and_weighted_mean(self):
        self.assertAlmostEqual(float(wrap_deg(355.0)), -5.0, places=6)
        self.assertAlmostEqual(float(wrap_deg(-185.0)), 175.0, places=6)
        mean_angle = circular_mean_deg(np.array([350.0, 10.0]))
        self.assertAlmostEqual(min(mean_angle, 360.0 - mean_angle), 0.0, places=6)

        weighted = circular_mean_deg(np.array([350.0, 10.0]), weights=np.array([3.0, 1.0]))
        self.assertTrue(350.0 < weighted < 360.0)

    def test_from_convention_wake_projection_and_active_donor_stratification(self):
        # Donor T01 sits ~500m due north of Target T02 (5.4 diameters for D=92m)
        layout = {
            "01": LayoutPoint(latitude=52.4045, longitude=-0.9500, rotor_m=92.0),
            "02": LayoutPoint(latitude=52.4000, longitude=-0.9500, rotor_m=92.0),
        }
        east, north = pair_vector(layout["02"], layout["01"])
        self.assertAlmostEqual(source_bearing(east, north), 0.0, delta=0.5)
        self.assertTrue(potential_overlap(east, north, 0.0, 92.0, 92.0))
        self.assertFalse(potential_overlap(east, north, 180.0, 92.0, 92.0))

        idx = pd.date_range("2017-05-01", periods=3, freq="10min", tz="UTC")
        # Step 0: wind FROM north (0 deg), donor running -> T02 active_wake
        # Step 1: wind FROM north (0 deg), donor stopped -> T02 passive_wake_donor_stopped (placebo)
        # Step 2: wind FROM south (180 deg), donor running -> T02 freestream, T01 active_wake
        dir_wide = pd.DataFrame({"01": [0.0, 0.0, 180.0], "02": [0.0, 0.0, 180.0]}, index=idx)
        power_wide = pd.DataFrame({"01": [900.0, 0.0, 900.0], "02": [650.0, 900.0, 950.0]}, index=idx)
        rpm_wide = pd.DataFrame({"01": [14.0, 0.0, 14.0], "02": [13.0, 14.0, 14.2]}, index=idx)

        exposure = classify_wake_exposure(dir_wide, power_wide, rpm_wide, layout)
        self.assertEqual(exposure.loc[idx[0], "02"], "active_wake")
        self.assertEqual(exposure.loc[idx[1], "02"], "passive_wake_donor_stopped")
        self.assertEqual(exposure.loc[idx[2], "02"], "freestream")
        self.assertEqual(exposure.loc[idx[2], "01"], "active_wake")


if __name__ == "__main__":
    unittest.main()
