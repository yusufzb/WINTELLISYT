import unittest
from datetime import datetime, timedelta, timezone

from wintellisyt.uncertainty import block_mae


class UncertaintyTests(unittest.TestCase):
    def test_fixed_seed_and_constant_error(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"timestamp_utc": start + timedelta(days=index * 2),
             "turbine_id": "A", "residual_kw": -17}
            for index in range(12)
        ]
        rows.append({"timestamp_utc": start, "turbine_id": "B", "residual_kw": 5})
        result = block_mae(rows)
        self.assertEqual(result, block_mae(reversed(rows)))
        self.assertEqual(result["turbines"]["A"]["lower_kw"], 17)
        self.assertEqual(result["turbines"]["A"]["upper_kw"], 17)
        self.assertEqual(result["turbines"]["B"]["status"], "insufficient_blocks")
        self.assertNotIn("lower_kw", result["turbines"]["B"])
