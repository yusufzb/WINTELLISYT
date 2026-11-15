import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install the V2 extra")
class PairedAnalysisTests(unittest.TestCase):
    def test_synchronized_blocks_preserve_offsetting_farm_errors(self):
        import numpy as np

        from scripts.v2_paired import bootstrap

        seconds = np.arange(10) * 86400
        daily = np.column_stack((np.arange(10), np.ones(10), np.ones(10)))
        result = bootstrap([{"seconds": seconds, "delta": daily},
                            {"seconds": seconds, "delta": -daily}], 24)
        self.assertEqual(result["blocks"], 10)
        self.assertEqual(result["valid_draws"], 1000)
        self.assertEqual(result["ci95"]["turbine_macro"]["mae_kw"], [0.0, 0.0])
        self.assertIsNone(bootstrap([{"seconds": seconds[:2], "delta": daily[:2]}], 24)["ci95"])

    def test_matched_interval_metrics_and_missing_bin(self):
        import numpy as np

        from scripts.v2_paired import summarize

        target = {"bins": np.array([12, 12, 13]),
                  "errors": {"binned": np.array([10, -11, 3]), "hybrid": np.array([4, -6, 0])},
                  "radii": {"binned": 10, "hybrid": 5}}
        score = summarize([target], 12)
        self.assertEqual(score["n"], 2)
        self.assertEqual(score["models"]["binned"]["row_pooled"]["coverage"], 0.5)
        self.assertEqual(score["models"]["hybrid"]["row_pooled"]["lower_misses"], 1)
        self.assertEqual(summarize([target], 15), {"n": 0, "turbines": 0})
