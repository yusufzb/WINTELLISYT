"""Binned electrical power references for SCADA observations."""

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, mean
from typing import Literal

__all__ = ["PowerBin", "PowerCurve", "fit_power_curve"]


@dataclass(frozen=True)
class PowerBin:
    index: int
    lower_mps: float
    upper_mps: float
    count: int
    mean_power_kw: float | None
    supported: bool


@dataclass(frozen=True)
class PowerCurve:
    raw_bins: tuple[PowerBin, ...]
    normal_bins: tuple[PowerBin, ...]
    input_count: int
    rejected_count: int
    min_bin_count: int

    @property
    def normal_status(self) -> Literal["available", "insufficient_support", "unavailable"]:
        if any(power_bin.supported for power_bin in self.normal_bins):
            return "available"
        if self.normal_bins:
            return "insufficient_support"
        return "unavailable"

    def bin_at(
        self, wind_speed_mps: float, *, stratum: Literal["normal", "raw"] = "normal"
    ) -> PowerBin | None:
        """Return bin support and summary, including empty bins for valid speeds."""
        if stratum == "normal":
            bins = self.normal_bins
        elif stratum == "raw":
            bins = self.raw_bins
        else:
            raise ValueError("stratum must be 'normal' or 'raw'")

        speed = _finite_number(wind_speed_mps)
        if speed is None or speed < 0:
            return None
        index = _bin_index(speed)
        position = bisect_left(bins, index, key=lambda power_bin: power_bin.index)
        if position < len(bins) and bins[position].index == index:
            return bins[position]
        return _make_bin(index, [], self.min_bin_count)

    def predict(
        self, wind_speed_mps: float, *, stratum: Literal["normal", "raw"] = "normal"
    ) -> float | None:
        """Return the supported bin mean in kW, or None for an unsupported speed."""
        power_bin = self.bin_at(wind_speed_mps, stratum=stratum)
        if power_bin is None or not power_bin.supported:
            return None
        return power_bin.mean_power_kw


def fit_power_curve(
    rows: Iterable[Mapping[str, object]], *, min_bin_count: int = 1
) -> PowerCurve:
    """Fit separate raw and explicitly normal references on supplied training rows.

    Each row must supply numeric `wind_speed_mps` and `power_kw` values. The
    optional `operational_state` is included in the normal reference only when
    it equals `normal`. Missing or invalid measurements increment rejected_count.
    """
    if type(min_bin_count) is not int or min_bin_count < 1:
        raise ValueError("min_bin_count must be a positive integer")

    raw: dict[int, list[float]] = {}
    normal: dict[int, list[float]] = {}
    input_count = 0
    rejected_count = 0
    for row in rows:
        input_count += 1
        speed = _finite_number(row.get("wind_speed_mps"))
        power = _finite_number(row.get("power_kw"))
        if speed is None or speed < 0 or power is None:
            rejected_count += 1
            continue
        index = _bin_index(speed)
        raw.setdefault(index, []).append(power)
        if row.get("operational_state") == "normal":
            normal.setdefault(index, []).append(power)

    return PowerCurve(
        raw_bins=tuple(_make_bin(index, raw[index], min_bin_count) for index in sorted(raw)),
        normal_bins=tuple(
            _make_bin(index, normal[index], min_bin_count) for index in sorted(normal)
        ),
        input_count=input_count,
        rejected_count=rejected_count,
        min_bin_count=min_bin_count,
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) else None


def _bin_index(speed: float) -> int:
    # Evaluate floor(2 * speed + 0.5) exactly to preserve one-sided edge values.
    numerator, denominator = speed.as_integer_ratio()
    return (4 * numerator + denominator) // (2 * denominator)


def _make_bin(index: int, powers: list[float], min_bin_count: int) -> PowerBin:
    average = None
    if powers:
        try:
            average = fmean(powers)
        except OverflowError:
            # The sum of finite values can overflow even when the mean is finite.
            average = mean(powers)
    return PowerBin(
        index=index,
        lower_mps=(2 * index - 1) / 4,
        upper_mps=(2 * index + 1) / 4,
        count=len(powers),
        mean_power_kw=average,
        supported=len(powers) >= min_bin_count,
    )
