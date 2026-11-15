"""Deterministic physical proxies and retrospective interval evidence."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

CADENCE = timedelta(minutes=10)
HALF = CADENCE / 2
VERSION = "hose-2017-v1"
# Shared exploratory rules for the two 2,050 kW Senvion models.
SITE_MODELS = {"kelmarsh": "MM92", "penmanshiel": "MM82"}
RATED_KW = 2050
PRESETS = {
    "conservative": (0.5, 2.0, 20.0, 100.0, 0.90),
    "sensitivity": (1.0, 1.5, 40.0, 70.0, 0.85),
}
# Maps literal logged categories, independently of code or message.
CATEGORIES = {
    "Full Performance": ("full_performance", "unknown"),
    "Scheduled Maintenance": ("maintenance_event", "maintenance"),
    "Out of Electrical Specification": ("electrical_event", "electrical_unspecified"),
    "Out of Environmental Specification": ("environmental_event", "environment"),
    "Forced outage": ("forced_outage", "unknown"),
    "Technical Standby": ("standby_event", "unknown"),
    "Requested Shutdown": ("requested_shutdown", "unknown"),
}


@dataclass(frozen=True)
class Signals:
    wind_ms: float | None
    rotor_rpm: float | None
    generator_rpm: float | None
    pitch_deg: float | None
    setpoint_kw: float | None


@dataclass(frozen=True)
class Physics:
    regime: str
    restriction: str
    rule: str
    missing: tuple[str, ...]


def physical_proxy(site: str, signals: Signals, preset: str = "conservative") -> Physics:
    """Use only same-row sensors. Control channels may be endogenous."""
    if site not in SITE_MODELS:
        raise ValueError(f"Unsupported farm: {site}")
    stop, rotating, gen_stop, gen_rotating, cap = PRESETS[preset]
    wind, rotor, generator = signals.wind_ms, signals.rotor_rpm, signals.generator_rpm
    setpoint = signals.setpoint_kw
    missing = tuple(name for name, value in vars(signals).items() if value is None)
    restriction = ("possible_setpoint_restriction"
                   if setpoint is not None and isfinite(setpoint) and 0 <= setpoint < cap * RATED_KW
                   and wind is not None and 5 <= wind <= 20 else "unknown")
    if rotor is None or not isfinite(rotor) or not 0 <= rotor <= 35:
        return Physics("unknown", restriction, "missing_or_invalid_rotor", missing)
    valid_generator = generator is not None and isfinite(generator) and 0 <= generator <= 3000
    if rotor <= stop:
        if valid_generator and generator is not None and generator >= gen_rotating:
            return Physics("unknown", restriction, "rotation_channels_conflict", missing)
        return Physics("standstill_like", restriction, "low_rotor_rpm", missing)
    if rotor >= rotating:
        if valid_generator and generator is not None and generator <= gen_stop:
            return Physics("unknown", restriction, "rotation_channels_conflict", missing)
        return Physics("rotating", restriction, "rotor_rpm", missing)
    return Physics("transition", restriction, "rotor_rpm_deadband", missing)


def retrospective_diagnostic(
    proxy: Physics, power_kw: float | None, wind_ms: float | None,
) -> Physics:
    """Describe observed production on the timestamp, retaining the proxy restriction."""
    if power_kw is None or not isfinite(power_kw):
        return Physics(proxy.regime, proxy.restriction, proxy.rule, (*proxy.missing, "power_kw"))
    regime, rule = proxy.regime, proxy.rule
    if regime == "standstill_like" and power_kw > 0.05 * RATED_KW:
        regime, rule = "unknown", "power_rotation_conflict"
    elif regime == "rotating":
        if power_kw <= 0.01 * RATED_KW:
            regime, rule = "rotating_low_export", "rotation_with_low_export"
        elif (wind_ms is not None and 5 <= wind_ms <= 20
              and 0.9 * RATED_KW <= power_kw <= 1.05 * RATED_KW):
            regime, rule = "rated_like", "rotation_near_nameplate"
        elif (wind_ms is not None and 3 <= wind_ms <= 15
              and 0.05 * RATED_KW < power_kw < 0.85 * RATED_KW):
            regime, rule = "partial_load_like", "rotation_and_subrated_export"
    return Physics(regime, proxy.restriction, rule, proxy.missing)


@dataclass(frozen=True)
class Event:
    id: int
    start: datetime
    end: datetime
    status: str
    category: str


@dataclass(frozen=True)
class Window:
    ids: tuple[int, ...]
    union_seconds: float
    stop_seconds: float
    midpoint_ids: tuple[int, ...]
    full_stop_ids: tuple[int, ...]


def union_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    total = 0.0
    end: datetime | None = None
    for a, b in sorted(intervals):
        if end is None or b > end:
            total += (b - (a if end is None else max(a, end))).total_seconds()
            end = b
    return total


def window(events: list[Event], start: datetime) -> Window:
    end = start + CADENCE
    midpoint = start + HALF
    hit = [e for e in events if e.start < end and start < e.end]
    stops = [e for e in hit if e.status == "Stop"]
    return Window(
        tuple(e.id for e in hit),
        union_seconds([(max(start, e.start), min(end, e.end)) for e in hit]),
        union_seconds([(max(start, e.start), min(end, e.end)) for e in stops]),
        tuple(e.id for e in hit if e.start <= midpoint < e.end),
        tuple(e.id for e in stops if e.start <= start and end <= e.end),
    )


def fuse(
    proxy: Physics, before: Window, after: Window, events: dict[int, Event], unresolved: bool,
) -> tuple[str, str, str, tuple[str, ...]]:
    """Return scoped evidence tier, restriction, logged cause, and reasons to abstain."""
    ids = set(before.ids) | set(after.ids)
    shared_stops = set(before.full_stop_ids) & set(after.full_stop_ids)
    categories = {events[i].category for i in ids if events[i].status == "Stop"}
    causes = {CATEGORIES.get(category, ("unmapped", "unknown"))[1] for category in categories}
    flags = []
    if unresolved:
        flags.append("unresolved_event_start")
    if bool(before.stop_seconds) != bool(after.stop_seconds):
        flags.append("stop_anchor_disagreement")
    if shared_stops and (len(categories) > 1 or any(
        CATEGORIES.get(events[i].category, ("unmapped", "unknown"))[0] == "full_performance"
        for i in ids
    )):
        flags.append("concurrent_event_categories")
    if shared_stops and proxy.regime == "rotating":
        flags.append("rotation_during_stop_candidate")
    if flags:
        return "UNKNOWN", "unknown", "unknown", tuple(flags)
    if shared_stops:
        cause = next(iter(causes)) if len(causes) == 1 else "unknown"
        tier = "CORROBORATED_INFERENCE" if proxy.regime == "standstill_like" else "DOCUMENTED"
        return tier, "logged_stop_candidate", cause, ()
    if before.stop_seconds or after.stop_seconds:
        return "UNKNOWN", "unknown", "unknown", ("stop_interval_support_incomplete",)
    if proxy.regime != "unknown" or proxy.restriction != "unknown":
        return "INFERRED", proxy.restriction, "unknown", ()
    return "UNKNOWN", "unknown", "unknown", ()
