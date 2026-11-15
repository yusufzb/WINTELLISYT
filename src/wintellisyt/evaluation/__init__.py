"""Chronological, within-turbine benchmarks for screened SCADA observations."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite, sqrt
from statistics import median
from typing import Literal

from ..pipeline import Observation, curve_input
from ..power_curve import PowerBin, fit_power_curve

__all__ = ["evaluate"]

_MODELS = ("binned_mean", "train_median", "piecewise_linear")
_STRATA = ("pooled_screen", "explicit_normal")
_SEASONS = ("DJF", "MAM", "JJA", "SON")


@dataclass
class _Scores:
    test_rows: int = 0
    eligible_test_rows: int = 0
    scored_test_rows: int = 0
    absolute_errors: list[float] = field(default_factory=lambda: [0.0] * len(_MODELS))
    squared_errors: list[float] = field(default_factory=lambda: [0.0] * len(_MODELS))

    def add_errors(self, errors):
        self.scored_test_rows += 1
        for index, error in enumerate(errors):
            self.absolute_errors[index] += abs(error)
            self.squared_errors[index] += error * error

    def report(self):
        count = self.scored_test_rows
        if count:
            status = "available"
        elif self.test_rows:
            status = "no_shared_support"
        else:
            status = "unavailable"
        return {
            "status": status,
            "test_rows": self.test_rows,
            "eligible_test_rows": self.eligible_test_rows,
            "scored_test_rows": count,
            "models": {
                model: {
                    "mae_kw": self.absolute_errors[index] / count if count else None,
                    "rmse_kw": sqrt(self.squared_errors[index] / count) if count else None,
                }
                for index, model in enumerate(_MODELS)
            },
        }


def _eligible(row: Observation) -> bool:
    return (
        row.wind_speed_ms is not None
        and isfinite(row.wind_speed_ms)
        and row.wind_speed_ms >= 0
        and row.active_power_kw is not None
        and isfinite(row.active_power_kw)
    )


def _interpolate(wind_speed: float, power_bin: PowerBin, means: dict[int, float]) -> float:
    # Adjacent supported bin centers define a line segment. At a support
    # boundary, keep the current bin's prediction through the edge of that bin.
    center = power_bin.index / 2
    direction = -1 if wind_speed < center else 1
    current = means[power_bin.index]
    neighbor = means.get(power_bin.index + direction)
    if neighbor is None:
        return current
    fraction = 2 * abs(wind_speed - center)
    return current * (1 - fraction) + neighbor * fraction


def evaluate(rows, *, min_bin_count: int):
    """Return shared-support metrics and pooled binned residuals from screened rows.

    Fit each turbine independently on the first 80% (floor) of its distinct UTC
    timestamps. The final timestamps are held out. Both strata use this cutoff.
    The caller supplies duplicate-audited observations for a single source.
    """
    if type(min_bin_count) is not int or min_bin_count < 1:
        raise ValueError("min_bin_count must be a positive integer")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.turbine_id].append(row)

    splits = {}
    residuals = []
    aggregates = {stratum: _Scores() for stratum in _STRATA}
    turbine_reports: dict[str, dict[str, dict[str, object]]] = {stratum: {} for stratum in _STRATA}
    by_state: dict[str, defaultdict[str, _Scores]] = {stratum: defaultdict(_Scores) for stratum in _STRATA}
    by_season: dict[str, defaultdict[str, _Scores]] = {stratum: defaultdict(_Scores) for stratum in _STRATA}

    for turbine, observations in sorted(grouped.items()):
        ordered = sorted(observations, key=lambda row: row.timestamp_utc)
        timestamps = list(dict.fromkeys(row.timestamp_utc for row in ordered))
        cutoff_index = max(1, len(timestamps) * 4 // 5)
        cutoff = timestamps[cutoff_index] if cutoff_index < len(timestamps) else None
        train = [row for row in ordered if cutoff is None or row.timestamp_utc < cutoff]
        test = ordered[len(train):]
        splits[turbine] = {
            "unique_timestamps": len(timestamps),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_end_utc": train[-1].timestamp_utc.isoformat(),
            "test_start_utc": cutoff.isoformat() if cutoff is not None else None,
        }

        eligible_train = [row for row in train if _eligible(row)]
        curve = fit_power_curve(curve_input(eligible_train), min_bin_count=min_bin_count)
        strata: tuple[tuple[str, Literal["raw", "normal"], tuple[PowerBin, ...]], ...] = (
            ("pooled_screen", "raw", curve.raw_bins),
            ("explicit_normal", "normal", curve.normal_bins),
        )
        for stratum, curve_stratum, bins in strata:
            means = {item.index: item.mean_power_kw for item in bins
                     if item.supported and item.mean_power_kw is not None}
            powers = [
                row.active_power_kw
                for row in eligible_train
                if curve_stratum == "raw" or row.operational_state == "normal"
            ]
            train_median = median(powers) if powers else None
            turbine_scores = _Scores()
            for row in test:
                if curve_stratum == "normal" and row.operational_state != "normal":
                    continue
                season = _SEASONS[(row.timestamp_utc.month % 12) // 3]
                score_groups = (
                    aggregates[stratum],
                    turbine_scores,
                    by_state[stratum][row.operational_state],
                    by_season[stratum][season],
                )
                eligible = _eligible(row)
                for scores in score_groups:
                    scores.test_rows += 1
                    scores.eligible_test_rows += eligible
                if not eligible:
                    continue

                power_bin = curve.bin_at(row.wind_speed_ms, stratum=curve_stratum)
                if power_bin is None or not power_bin.supported:
                    continue
                predicted = means[power_bin.index]
                actual = row.active_power_kw
                interpolated = _interpolate(row.wind_speed_ms, power_bin, means)
                errors = (actual - predicted, actual - train_median, actual - interpolated)
                for scores in score_groups:
                    scores.add_errors(errors)
                if curve_stratum == "raw":
                    residuals.append({
                        "timestamp_utc": row.timestamp_utc.isoformat(),
                        "turbine_id": turbine,
                        "operational_state": row.operational_state,
                        "wind_speed_ms": row.wind_speed_ms,
                        "actual_kw": actual,
                        "predicted_kw": predicted,
                        "residual_kw": errors[0],
                    })
            turbine_reports[stratum][turbine] = turbine_scores.report()

    return {
        "split": {
            "method": "within_turbine_chronological_distinct_utc_timestamps",
            "train_fraction": 0.8,
            "turbines": splits,
        },
        "min_bin_count": min_bin_count,
        "aggregation": "row_weighted_within_source",
        "shared_support": "eligible_test_rows_in_supported_training_bins",
        "residual_model": "pooled_screen_binned_mean",
        "strata": {
            stratum: {
                "aggregate": aggregates[stratum].report(),
                "turbines": turbine_reports[stratum],
                "by_state": {
                    key: scores.report() for key, scores in sorted(by_state[stratum].items())
                },
                "by_season_utc": {
                    key: scores.report() for key, scores in sorted(by_season[stratum].items())
                },
            }
            for stratum in _STRATA
        },
    }, residuals
