"""Fault-injection tests verifying recovery accuracy across diagnostic modules."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

import numpy as np
import pandas as pd

from wintellisyt.asset_intelligence import (
    analyze_pitch_actuator_currents_and_temps,
    analyze_yaw_misalignment_and_cable_windings,
)
from wintellisyt.direction import find_wake_trough_offset, fit_relative_consensus, wrap_deg
from wintellisyt.pc_health import monitor
from wintellisyt.peer_reconstruction import audit_anemometer_and_ntf
from wintellisyt.pitch_health import analyze_blade_pitch_asymmetry
from wintellisyt.wind_geometry import LayoutPoint, pair_vector, source_bearing


class FaultInjectionDiagnosticsTest(unittest.TestCase):
    def test_nrel_two_stage_consensus_and_true_north_wake_trough_recovery(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2017-01-01", periods=3600, freq="10min", tz="UTC")
        train_cutoff = pd.Timestamp("2017-01-15", tz="UTC")
        test_start = pd.Timestamp("2017-01-20", tz="UTC")

        # 4-turbine collinear N-S array (~445 m apart, D=92 m -> ~4.8 D, strictly 0/180 deg wakes)
        layout = {
            "01": LayoutPoint(52.4120, -0.9500, 92.0),
            "02": LayoutPoint(52.4080, -0.9500, 92.0),
            "03": LayoutPoint(52.4040, -0.9500, 92.0),
            "04": LayoutPoint(52.4000, -0.9500, 92.0),
        }
        true_dir = rng.uniform(0.0, 360.0, size=len(idx))
        true_ws = rng.uniform(6.0, 10.5, size=len(idx))

        # Apply planted True North rotation (+12 deg) + per-turbine offsets
        planted_true_north = 12.0
        planted_rel = {"01": 18.0, "02": -12.0, "03": -4.0, "04": -2.0}
        dir_cols = {}
        ws_cols = {}
        p_cols = {}
        for tid in layout:
            noise = rng.normal(0.0, 0.8, size=len(idx))
            dir_cols[tid] = (true_dir + planted_true_north + planted_rel[tid] + noise) % 360.0
            ws_cols[tid] = true_ws + rng.normal(0.0, 0.1, size=len(idx))
            p_cols[tid] = 900.0 + 30.0 * (ws_cols[tid] - 8.0)

        # Inject aerodynamic wake deficit along adjacent N-S FROM bearings
        ordered = ["01", "02", "03", "04"]
        for i, target in enumerate(ordered):
            for j, donor in enumerate(ordered):
                if abs(i - j) != 1:
                    continue
                east, north = pair_vector(layout[target], layout[donor])
                bearing = source_bearing(east, north)
                ang_diff = wrap_deg(true_dir - bearing)
                deficit = 0.45 * np.exp(-0.5 * (ang_diff / 5.0) ** 2)
                p_cols[target] = p_cols[target] * (1.0 - deficit)

        dir_wide = pd.DataFrame(dir_cols, index=idx)
        ws_wide = pd.DataFrame(ws_cols, index=idx)
        power_wide = pd.DataFrame(p_cols, index=idx)

        vectors, rel_dir_wide, _, dir_sum, _ = fit_relative_consensus(
            dir_wide, layout, train_cutoff, test_start
        )
        troughs_df, tn_sum = find_wake_trough_offset(
            ws_wide, power_wide, rel_dir_wide, vectors, layout
        )
        self.assertEqual(tn_sum["convention"], "FROM")
        self.assertFalse(troughs_df.empty)
        true_north_est = float(tn_sum["common_true_north_offset_deg"])
        for row in dir_sum.itertuples(index=False):
            recovered_total = wrap_deg(row.relative_offset_deg + true_north_est)
            expected_total = wrap_deg(planted_true_north + planted_rel[row.turbine_id])
            self.assertLessEqual(abs(wrap_deg(recovered_total - expected_total)), 1.5)

    def test_cusum_calm_wind_state_hold_and_left_censoring(self):
        calibration = [-4.0, 5.0, -2.0, 3.0, 0.0, 4.0, -6.0, 2.0, -1.0, 1.0]
        start = date(2017, 11, 1)
        # Mild -25 kW drift interrupted by a 4-day calm-wind gap: must hold accumulator & alert!
        days_held = [(start + timedelta(days=i), -25.0) for i in range(4)] + [
            (start + timedelta(days=8 + i), -25.0) for i in range(4)
        ]
        sigs_reset = monitor(days_held, calibration, max_hold_gap_days=1)
        sigs_hold = monitor(days_held, calibration, max_hold_gap_days=7)
        self.assertGreater(sigs_hold[-1].cusum_negative, sigs_reset[-1].cusum_negative)
        self.assertTrue(any(s.alert for s in sigs_hold))

        # Immediate Day-0 severe fault flags onset_left_censored=True
        immediate = [(start, -150.0), (start + timedelta(days=1), -150.0)]
        cens = monitor(immediate, calibration, max_hold_gap_days=7)
        self.assertTrue(cens[0].alert)
        self.assertTrue(cens[0].onset_left_censored)

    def test_pitch_asymmetry_and_actuator_current_faults(self):
        n = 300
        idx = pd.date_range("2017-03-01", periods=n, freq="10min", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp_utc": np.concatenate([idx, idx]),
                "turbine_id": ["01"] * n + ["02"] * n,
                "wind_speed_ms": np.full(2 * n, 8.0),
                "active_power_kw": np.full(2 * n, 950.0),
                "pitch_a_deg": np.full(2 * n, 0.0),
                "pitch_b_deg": np.concatenate([np.full(n, 1.25), np.full(n, 0.02)]),
                "pitch_c_deg": np.full(2 * n, 0.0),
                "motor_current_1_a": np.full(2 * n, 2.1),
                "motor_current_2_a": np.concatenate([np.full(n, 6.8), np.full(n, 2.15)]),
                "motor_current_3_a": np.full(2 * n, 2.05),
                "motor_temp_1_c": np.full(2 * n, 41.0),
                "motor_temp_2_c": np.concatenate([np.full(n, 53.0), np.full(n, 41.5)]),
                "motor_temp_3_c": np.full(2 * n, 40.8),
            }
        )
        asym = analyze_blade_pitch_asymmetry(df).set_index("turbine_id")
        self.assertTrue(bool(asym.loc["01", "flag_asymmetry"]))
        self.assertFalse(bool(asym.loc["02", "flag_asymmetry"]))
        self.assertGreaterEqual(float(asym.loc["01", "max_pair_offset_deg"]), 1.2)

        act = analyze_pitch_actuator_currents_and_temps(df).set_index("turbine_id")
        self.assertTrue(bool(act.loc["01", "flag_actuator_anomaly"]))
        self.assertFalse(bool(act.loc["02", "flag_actuator_anomaly"]))

    def test_yaw_cos2_misalignment_and_anemometer_ntf_audit(self):
        rng = np.random.default_rng(11)
        n = 800
        vane = rng.uniform(-16.0, 16.0, size=n)
        ws = rng.uniform(6.0, 9.5, size=n)
        area = np.pi * 46.0**2
        # True optimal alignment peaks at +8 deg vane angle (static 8-deg vane misalignment)
        p_kw = (0.5 * 1.225 * area * ws**3 * 0.44 * np.cos(np.deg2rad(vane - 8.0)) ** 2) / 1000.0
        # Add realistic cable winding steps
        windings = np.concatenate([np.linspace(0.0, 1.8, 400), np.linspace(0.2, -1.5, 400)])
        df = pd.DataFrame(
            {
                "turbine_id": ["01"] * n,
                "wind_speed_ms": ws * 1.05,
                "wind_speed_s1_ms": ws,
                "wind_speed_s2_ms": ws - 0.1,
                "density_adjusted_ws_ms": ws * 1.05,
                "ambient_temperature_c": np.full(n, 15.0),
                "active_power_kw": p_kw,
                "vane_position_deg": vane,
                "cable_windings": windings,
            }
        )
        yaw_sum, _ = analyze_yaw_misalignment_and_cable_windings(df, {"01": 92.0})
        self.assertEqual(len(yaw_sum), 1)
        self.assertAlmostEqual(float(yaw_sum.iloc[0]["optimal_cp_yaw_angle_deg"]), 8.0, delta=2.0)

        ntf = audit_anemometer_and_ntf(df)
        self.assertEqual(ntf.iloc[0]["ws_provenance_inference"], "oem_processed_or_ntf_scaled")
        self.assertAlmostEqual(float(ntf.iloc[0]["median_ws_over_s1"]), 1.05, places=3)


if __name__ == "__main__":
    unittest.main()
