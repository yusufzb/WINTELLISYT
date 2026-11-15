"""Descriptive block-bootstrap uncertainty for held-out binned MAE."""

import math
import random
from collections import defaultdict
from datetime import datetime, timezone

BLOCK_DAYS = 2
MIN_BLOCKS = 12
REPLICATES = 1000
SEED = 20261114


def block_mae(residuals):
    """Resample nonempty UTC blocks within turbines, retaining row weighting."""
    turbines: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in residuals:
        timestamp = row["timestamp_utc"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        timestamp = timestamp.astimezone(timezone.utc)
        block = timestamp.date().toordinal() // BLOCK_DAYS
        turbines[row["turbine_id"]][block].append(abs(row["residual_kw"]))

    rng = random.Random(SEED)
    results = {}
    for turbine, blocks in sorted(turbines.items()):
        block_stats = [(math.fsum(blocks[key]), len(blocks[key])) for key in sorted(blocks)]
        n = len(block_stats)
        result: dict[str, str | int | float] = {
            "blocks": n, "scored_rows": sum(count for _, count in block_stats)
        }
        if n < MIN_BLOCKS:
            result["status"] = "insufficient_blocks"
        else:
            samples = []
            for _ in range(REPLICATES):
                draw = rng.choices(block_stats, k=n)
                samples.append(math.fsum(total for total, _ in draw) /
                               sum(count for _, count in draw))
            samples.sort()
            result.update(status="descriptive", lower_kw=samples[24], upper_kw=samples[974])
        results[turbine] = result

    return {
        "estimand": "held-out pooled-screen MAE, conditional on fitted model and scored rows",
        "method": "percentile bootstrap of nonempty non-overlapping UTC blocks per turbine",
        "nominal_percentiles": [2.5, 97.5],
        "block_days": BLOCK_DAYS,
        "minimum_blocks": MIN_BLOCKS,
        "replicates": REPLICATES,
        "seed": SEED,
        "limitations": [
            "Intervals omit training variability and unsupported or missing rows.",
            "Block length is fixed and dependence across blocks may remain.",
            "No predictive coverage or cross-turbine calibration was tested.",
        ],
        "turbines": results,
    }
