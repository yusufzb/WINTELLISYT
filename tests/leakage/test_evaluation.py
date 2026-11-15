import unittest
from datetime import datetime, timedelta, timezone
from math import sqrt

from wintellisyt.evaluation import evaluate
from wintellisyt.pipeline import Observation


def observation(minute, power, speed=5.0, state="unknown", turbine="T01"):
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return Observation(timestamp, turbine, speed, power, state)


class EvaluationTest(unittest.TestCase):
    def test_tied_timestamps_stay_together_and_each_turbine_fits_its_own_training(self):
        rows = [
            observation(0, 10),
            observation(10, 10),
            observation(20, 10),
            observation(30, 10),
            observation(40, 1000),
            observation(40, 2000),
            observation(60, 100, turbine="T02"),
            observation(70, 2000, turbine="T02"),
        ]
        metrics, residuals = evaluate(reversed(rows), min_bin_count=1)
        split = metrics["split"]["turbines"]
        self.assertEqual(split["T01"]["train_rows"], 4)
        self.assertEqual(split["T01"]["test_rows"], 2)
        self.assertLess(split["T01"]["train_end_utc"], split["T01"]["test_start_utc"])
        self.assertEqual(split["T02"]["train_rows"], 1)
        self.assertEqual([row["predicted_kw"] for row in residuals], [10, 10, 100])
        self.assertEqual([row["turbine_id"] for row in residuals], ["T01", "T01", "T02"])
        aggregate = metrics["strata"]["pooled_screen"]["aggregate"]
        binned = aggregate["models"]["binned_mean"]
        self.assertEqual(binned["mae_kw"], (990 + 1990 + 1900) / 3)
        self.assertEqual(binned["rmse_kw"], sqrt((990**2 + 1990**2 + 1900**2) / 3))

    def test_models_share_support_and_report_screening_denominators(self):
        rows = [
            observation(0, 10),
            observation(10, 30),
            observation(20, 60, speed=5.5),
            observation(30, 60, speed=5.5),
            observation(40, 40, speed=5.125),
            observation(40, 40, speed=10.0),
            observation(40, 40, speed=None),
        ]
        metrics, residuals = evaluate(rows, min_bin_count=2)
        strata = metrics["strata"]
        pooled = strata["pooled_screen"]
        aggregate = pooled["aggregate"]
        self.assertEqual(aggregate["test_rows"], 3)
        self.assertEqual(aggregate["eligible_test_rows"], 2)
        self.assertEqual(aggregate["scored_test_rows"], 1)
        self.assertEqual(aggregate["models"], {
            "binned_mean": {"mae_kw": 20.0, "rmse_kw": 20.0},
            "train_median": {"mae_kw": 5.0, "rmse_kw": 5.0},
            "piecewise_linear": {"mae_kw": 10.0, "rmse_kw": 10.0},
        })
        self.assertEqual(len(residuals), 1)
        self.assertEqual(residuals[0]["residual_kw"], 20)
        self.assertEqual(residuals[0]["timestamp_utc"], rows[4].timestamp_utc.isoformat())
        self.assertEqual(pooled["by_state"]["unknown"], aggregate)
        self.assertEqual(pooled["by_season_utc"]["DJF"], aggregate)
        self.assertEqual(strata["explicit_normal"]["aggregate"]["status"], "unavailable")
        self.assertIsNone(strata["explicit_normal"]["aggregate"]["models"]["binned_mean"]["mae_kw"])

    def test_explicit_normal_uses_shared_cutoff_and_only_normal_fit_rows(self):
        rows = [
            observation(0, 30, state="normal"),
            observation(10, 100, state="unknown"),
            observation(20, 40, state="normal"),
            observation(30, 0, state="curtailed"),
            observation(40, 45, state="normal"),
            observation(40, 200, state="unknown"),
        ]
        metrics, residuals = evaluate(rows, min_bin_count=2)
        strata = metrics["strata"]
        normal = strata["explicit_normal"]["aggregate"]
        self.assertEqual(normal["test_rows"], 1)
        self.assertEqual(normal["scored_test_rows"], 1)
        self.assertEqual(normal["models"]["binned_mean"]["mae_kw"], 10.0)
        self.assertEqual(normal["models"]["train_median"]["rmse_kw"], 10.0)
        self.assertEqual(strata["pooled_screen"]["aggregate"]["scored_test_rows"], 2)
        self.assertEqual([row["predicted_kw"] for row in residuals], [42.5, 42.5])

    def test_interpolation_never_bridges_unsupported_bins(self):
        rows = [
            observation(0, 10),
            observation(10, 100, speed=6.0),
            observation(20, 10),
            observation(30, 100, speed=6.0),
            observation(40, 10, speed=5.2),
            observation(40, 50, speed=5.5),
        ]
        metrics, _ = evaluate(rows, min_bin_count=2)
        aggregate = metrics["strata"]["pooled_screen"]["aggregate"]
        self.assertEqual(aggregate["scored_test_rows"], 1)
        self.assertEqual(aggregate["models"]["piecewise_linear"]["mae_kw"], 0)

    def test_single_timestamp_and_empty_input_are_unavailable(self):
        for rows in ([], [observation(10, 20), observation(10, 25)]):
            with self.subTest(rows=rows):
                metrics, residuals = evaluate(rows, min_bin_count=1)
                self.assertEqual(residuals, [])
                pooled = metrics["strata"]["pooled_screen"]["aggregate"]
                self.assertEqual(pooled["test_rows"], 0)
                self.assertEqual(pooled["status"], "unavailable")
                self.assertIsNone(pooled["models"]["train_median"]["rmse_kw"])


if __name__ == "__main__":
    unittest.main()
