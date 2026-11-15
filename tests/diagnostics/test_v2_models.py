import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install the V2 extra")
class ModelTests(unittest.TestCase):
    def test_common_support_and_bounded_predictions(self):
        import numpy as np
        from threadpoolctl import threadpool_limits

        from wintellisyt.v2_models import fit_models

        x = np.repeat([4.0, 6.0, 8.0, 10.0], 10).reshape(-1, 1)
        y = np.repeat([-20.0, 100.0, 700.0, 2500.0], 10)
        with threadpool_limits(limits=2):
            models, support, _ = fit_models(x, y)
            np.testing.assert_array_equal(support(np.array([[4.0], [5.0], [10.0]])),
                                          [True, False, True])
            pred = models["shape"](x)
            self.assertTrue(np.all(np.diff(pred) >= 0))
            self.assertTrue(np.all((pred >= 0) & (pred <= 2050)))
            np.testing.assert_array_equal(models["hybrid_bounded"](x),
                                          np.clip(models["hybrid"](x), 0, 2050))

    def test_hybrid_replay_serializes_metrics(self):
        import json
        from datetime import datetime, timedelta, timezone

        import numpy as np

        from scripts.v2_benchmark import hybrid_replay
        from wintellisyt.pipeline import Observation

        start = datetime(2017, 11, 1, tzinfo=timezone.utc)
        rows = [Observation(start + timedelta(minutes=10 * i), "01", 8, 110, "unknown")
                for i in range(55)]
        fitted = ({"hybrid": lambda x: np.full(len(x), 100.0)},
                  lambda x: np.ones(len(x), dtype=bool), {"hybrid": 10}, {})
        result = hybrid_replay(fitted, fitted, rows)
        self.assertEqual(json.loads(json.dumps(result))["0"]["scored"], 55)
