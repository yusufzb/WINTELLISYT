"""Strict target-exclusion and chronological train/calibration/test isolation checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wintellisyt.direction import fit_relative_consensus
from wintellisyt.pc_health import evaluate_pc_health_baselines
from wintellisyt.peer_reconstruction import fit_peer_mcp, predict_peer_mcp
from wintellisyt.wind_geometry import LayoutPoint


class LeakageIsolationTest(unittest.TestCase):
    def test_direction_consensus_and_peer_ref_exclude_target_and_future(self):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2017-01-01", periods=1500, freq="10min", tz="UTC")
        train_cutoff = pd.Timestamp("2017-01-08", tz="UTC")
        test_start = pd.Timestamp("2017-01-10", tz="UTC")
        layout = {
            "01": LayoutPoint(52.404, -0.950, 92.0),
            "02": LayoutPoint(52.400, -0.950, 92.0),
            "03": LayoutPoint(52.402, -0.944, 92.0),
            "04": LayoutPoint(52.402, -0.956, 92.0),
        }
        base = rng.uniform(0.0, 360.0, size=len(idx))
        dir_wide = pd.DataFrame({k: (base + i * 5.0) % 360.0 for i, k in enumerate(layout)}, index=idx)

        _, _, peer_a, sum_a, _ = fit_relative_consensus(dir_wide, layout, train_cutoff, test_start)

        # 1. Corrupt target '01' ONLY after test_start: relative_offset_deg & peer_ref['01'] must NOT change!
        mutated = dir_wide.copy()
        mutated.loc[mutated.index >= train_cutoff, "01"] = (mutated.loc[mutated.index >= train_cutoff, "01"] + 90.0) % 360.0
        _, _, peer_b, sum_b, _ = fit_relative_consensus(mutated, layout, train_cutoff, test_start)

        off_a = sum_a.set_index("turbine_id")["relative_offset_deg"]
        off_b = sum_b.set_index("turbine_id")["relative_offset_deg"]
        pd.testing.assert_series_equal(off_a, off_b)
        pd.testing.assert_series_equal(peer_a["01"], peer_b["01"])

    def test_peer_mcp_fit_and_prediction_isolate_target_and_post_cutoff(self):
        rng = np.random.default_rng(19)
        idx = pd.date_range("2017-01-01", periods=1800, freq="10min", tz="UTC")
        cutoff = pd.Timestamp("2017-01-08", tz="UTC")

        base_ws = rng.uniform(4.0, 12.0, size=len(idx))
        base_dir = rng.uniform(0.0, 360.0, size=len(idx))
        turbines = ["01", "02", "03", "04"]
        ws_wide = pd.DataFrame({t: base_ws + rng.normal(0.0, 0.2, len(idx)) for t in turbines}, index=idx)
        p_wide = pd.DataFrame({t: 15.0 * ws_wide[t] ** 2 for t in turbines}, index=idx)
        d_wide = pd.DataFrame({t: base_dir for t in turbines}, index=idx)

        donors = ["02", "03", "04"]
        table1 = fit_peer_mcp(
            ws_wide.loc[:cutoff], p_wide.loc[:cutoff], d_wide.loc[:cutoff], "01", donors, min_cell_count=5
        )
        pred1 = predict_peer_mcp(ws_wide.loc[cutoff:], d_wide.loc[cutoff:], donors, table1)

        # Mutate target '01' on BOTH fit-after-cutoff AND test rows
        ws_mut = ws_wide.copy()
        d_mut = d_wide.copy()
        ws_mut.loc[cutoff:, "01"] = 99.0
        d_mut.loc[cutoff:, "01"] = 270.0
        table2 = fit_peer_mcp(
            ws_mut.loc[:cutoff], p_wide.loc[:cutoff], d_mut.loc[:cutoff], "01", donors, min_cell_count=5
        )
        pred2 = predict_peer_mcp(ws_mut.loc[cutoff:], d_mut.loc[cutoff:], donors, table2)

        pd.testing.assert_frame_equal(table1, table2)
        pd.testing.assert_frame_equal(pred1, pred2)

    def test_pc_health_baselines_fit_only_before_train_cutoff(self):
        rng = np.random.default_rng(23)
        idx = pd.date_range("2017-01-01", periods=30 * 144, freq="10min", tz="UTC")
        train_cutoff = pd.Timestamp("2017-01-15", tz="UTC")
        cal_end = pd.Timestamp("2017-01-23", tz="UTC")
        turbines = ["01", "02", "03"]

        ws = rng.uniform(5.0, 11.0, size=len(idx))
        ws_wide = pd.DataFrame({t: ws for t in turbines}, index=idx)
        p_wide = pd.DataFrame({t: 12.0 * ws**2 + rng.normal(0.0, 5.0, len(idx)) for t in turbines}, index=idx)
        stop_wide = pd.DataFrame(False, index=idx, columns=turbines)
        oem_knots = {str(float(k)): float(12.0 * k**2) for k in range(0, 26)}

        sig1, _ = evaluate_pc_health_baselines(ws_wide, p_wide, stop_wide, oem_knots, train_cutoff=train_cutoff, cal_end=cal_end)

        # Corrupt ALL donors after train_cutoff: neither fleet_consensus nor self_historical fit may shift!
        p_mut = p_wide.copy()
        p_mut.loc[p_mut.index >= train_cutoff, ["02", "03"]] = 10.0
        sig2, _ = evaluate_pc_health_baselines(ws_wide, p_mut, stop_wide, oem_knots, train_cutoff=train_cutoff, cal_end=cal_end)

        t1_self_a = sig1[(sig1["turbine_id"] == "01") & (sig1["baseline"] == "self_historical")].reset_index(drop=True)
        t1_self_b = sig2[(sig2["turbine_id"] == "01") & (sig2["baseline"] == "self_historical")].reset_index(drop=True)
        pd.testing.assert_frame_equal(t1_self_a, t1_self_b)

        t1_fleet_a = sig1[(sig1["turbine_id"] == "01") & (sig1["baseline"] == "fleet_consensus")].reset_index(drop=True)
        t1_fleet_b = sig2[(sig2["turbine_id"] == "01") & (sig2["baseline"] == "fleet_consensus")].reset_index(drop=True)
        pd.testing.assert_frame_equal(t1_fleet_a, t1_fleet_b)


if __name__ == "__main__":
    unittest.main()
