"""Model-referenced electrical energy on explicitly bounded intervals."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import fsum, isfinite


@dataclass(frozen=True)
class PowerInterval:
    """Mean electrical power over a known half-open interval for one turbine."""

    start: datetime
    end: datetime
    reference_kw: float
    observed_kw: float


@dataclass(frozen=True)
class EnergyDeviation:
    intervals: int
    covered_hours: float
    eligible_fraction: float
    signed_kwh: float
    gross_shortfall_kwh: float
    gross_surplus_kwh: float


def model_referenced_energy(
    intervals: Iterable[PowerInterval], *, expected_duration: timedelta,
    window_start: datetime, window_end: datetime,
) -> EnergyDeviation:
    """Integrate a single turbine within a declared UTC window without filling gaps."""
    if (window_start.utcoffset() is None or window_end.utcoffset() is None
            or expected_duration <= timedelta(0)):
        raise ValueError("A timezone-aware window and positive duration are required")
    start = window_start.astimezone(timezone.utc)
    end = window_end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("Window must have positive elapsed duration")
    accepted: list[tuple[datetime, datetime, float]] = []
    hours = expected_duration.total_seconds() / 3600
    for row in intervals:
        if row.start.utcoffset() is None or row.end.utcoffset() is None:
            raise ValueError("Interval timestamps must be timezone-aware")
        first, last = row.start.astimezone(timezone.utc), row.end.astimezone(timezone.utc)
        if not start <= first < last <= end or last - first != expected_duration:
            raise ValueError("Interval outside window or inconsistent duration")
        deviation = (row.reference_kw - row.observed_kw) * hours
        if not isfinite(deviation):
            raise ValueError("Power deviation must be finite")
        accepted.append((first, last, deviation))
    accepted.sort()
    if any(a[1] > b[0] for a, b in zip(accepted, accepted[1:])):
        raise ValueError("Duplicate or overlapping intervals")
    covered = len(accepted) * hours
    return EnergyDeviation(
        len(accepted), covered, covered * 3600 / (end - start).total_seconds(),
        fsum(delta for _, _, delta in accepted),
        fsum(delta for _, _, delta in accepted if delta > 0),
        fsum(-delta for _, _, delta in accepted if delta < 0),
    )
