"""Event-screened daily power-curve residual monitoring with EWMA and CUSUM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt
from statistics import median

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailySignal:
    day: date
    ewma_kw: float
    cusum_positive: float
    cusum_negative: float
    alert: bool
    onset_left_censored: bool


def monitor(
    residual_days: list[tuple[date, float]],
    calibration: list[float],
    *,
    max_hold_gap_days: int = 1,
) -> list[DailySignal]:
    """Run two-sided EWMA and standardized CUSUM on chronological daily residual means.

    When `max_hold_gap_days > 1` (e.g. 7), isolated calm-wind days without supported
    generation hold the previous EWMA/CUSUM state rather than resetting accumulators to zero.
    Gaps exceeding `max_hold_gap_days` reset the state and flag any immediate alert as
    `onset_left_censored`.
    """
    if len(calibration) < 7 or not all(map(isfinite, calibration)):
        raise ValueError("Seven finite calibration days required")
    if max_hold_gap_days < 1:
        raise ValueError("max_hold_gap_days must be >= 1")
    center = median(calibration)
    sigma = max(5.0, 1.4826 * median(abs(value - center) for value in calibration))
    weight = 0.2
    ewma_limit = 3.0 * sigma * sqrt(weight / (2.0 - weight))
    output: list[DailySignal] = []
    for day, value in residual_days:
        if not isfinite(value) or (output and day <= output[-1].day):
            raise ValueError("Daily values must be finite and strictly chronological")
        new_run = not output or (day - output[-1].day).days > max_hold_gap_days
        previous = None if new_run else output[-1]
        delta = value - center
        ewma = weight * delta + (1.0 - weight) * (previous.ewma_kw if previous else 0.0)
        high = max(0.0, (previous.cusum_positive if previous else 0.0) + delta / sigma - 0.5)
        low = max(0.0, (previous.cusum_negative if previous else 0.0) - delta / sigma - 0.5)
        alert = abs(ewma) > ewma_limit or max(high, low) > 5.0
        output.append(DailySignal(day, ewma, high, low, alert, alert and new_run))
    return output


def build_event_stop_mask(
    timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    turbine_id: str,
) -> tuple[pd.Series, int]:
    """Return a boolean Series marking 10-min intervals overlapping bounded Stop events.

    Uses both start-anchored `[t, t+10m)` and end-anchored `(t-10m, t]` conservatively
    (union) so any interval touching a bounded Stop event is excluded from healthy PC curves.
    Also returns the count of left-censored Stop events starting prior to `timestamps.min()`.
    """
    mask = np.zeros(len(timestamps), dtype=bool)
    if events_df.empty:
        return pd.Series(mask, index=timestamps), 0
    sub = events_df[(events_df["turbine_id"] == turbine_id) & (events_df["status"] == "Stop")]
    if sub.empty:
        return pd.Series(mask, index=timestamps), 0

    first_ts = timestamps.min()
    left_censored_stops = int(
        (sub["timestamp_start"].notna() & (sub["timestamp_start"] < first_ts)).sum()
    )
    bounded = sub[~sub["open_ended"] & sub["timestamp_start"].notna() & sub["timestamp_end"].notna()]
    if bounded.empty or len(timestamps) == 0:
        return pd.Series(mask, index=timestamps), left_censored_stops

    ts_ns = timestamps.as_unit("ns").asi8
    ten_min_ns = 600_000_000_000

    s_idx = pd.DatetimeIndex(bounded["timestamp_start"])
    s_idx = s_idx.tz_localize("UTC") if s_idx.tz is None else s_idx.tz_convert("UTC")
    e_idx = pd.DatetimeIndex(bounded["timestamp_end"])
    e_idx = e_idx.tz_localize("UTC") if e_idx.tz is None else e_idx.tz_convert("UTC")
    s_arr = s_idx.as_unit("ns").asi8
    e_arr = e_idx.as_unit("ns").asi8

    is_sorted = bool(timestamps.is_monotonic_increasing)
    if is_sorted:
        order: np.ndarray | None = None
        ts_sorted = ts_ns
    else:
        order = np.argsort(ts_ns, kind="mergesort")
        ts_sorted = ts_ns[order]

    # Conservative union of start-anchored [t, t+10m) and end-anchored (t-10m, t]:
    # s_ns - ten_min_ns < t < e_ns + ten_min_ns
    lo = np.searchsorted(ts_sorted, s_arr - ten_min_ns, side="right")
    hi = np.searchsorted(ts_sorted, e_arr + ten_min_ns, side="left")
    valid_span = hi > lo
    if np.any(valid_span):
        diff = np.zeros(len(ts_sorted) + 1, dtype=np.int64)
        np.add.at(diff, lo[valid_span], 1)
        np.add.at(diff, hi[valid_span], -1)
        hit_sorted = np.cumsum(diff[:-1]) > 0
        if order is None:
            mask |= hit_sorted
        else:
            mask[order] |= hit_sorted

    return pd.Series(mask, index=timestamps), left_censored_stops


def evaluate_oem_curve(ws: pd.Series, oem_knots: dict[str, float]) -> pd.Series:
    """Linearly interpolate a 1.0 m/s OEM reference curve onto 10-min wind speeds."""
    x_knots = np.array(sorted(float(k) for k in oem_knots.keys()), dtype=float)
    y_knots = np.array([float(oem_knots[str(k)] if str(k) in oem_knots else oem_knots[f"{k:g}"]) for k in x_knots])
    vals = ws.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    ok = np.isfinite(vals) & (vals >= 0.0) & (vals <= x_knots[-1])
    out[ok] = np.interp(vals[ok], x_knots, y_knots)
    return pd.Series(out, index=ws.index)


def evaluate_pc_health_baselines(
    ws_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    stop_mask_wide: pd.DataFrame,
    oem_knots: dict[str, float],
    *,
    train_cutoff: pd.Timestamp,
    cal_end: pd.Timestamp,
    min_daily_rows: int = 24,
    max_hold_gap_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run event-screened PC Health monitoring against 3 baselines for every turbine.

    Baselines:
    - `oem_contractual`: Senvion MM92/MM82 contractual MPC curve
    - `fleet_consensus`: leave-target-out same-farm empirical curve fit on train (< train_cutoff)
    - `self_historical`: target turbine's own pre-cutoff curve
    """
    ids = list(ws_wide.columns)
    signal_rows: list[dict] = []
    summary_rows: list[dict] = []

    for target in ids:
        t_ws = ws_wide[target]
        t_p = power_wide[target]
        t_stop = stop_mask_wide[target]
        healthy = (~t_stop) & (t_ws >= 3.0) & (t_ws <= 20.0) & t_p.notna()
        ws_bin = np.floor(t_ws * 2.0 + 0.5) / 2.0

        # 1. OEM reference
        ref_oem = evaluate_oem_curve(t_ws, oem_knots)

        # 2. Fleet consensus reference (leave-target-out, train only, event-screened)
        others = [k for k in ids if k != target]
        d_ws = ws_wide[others].where((~stop_mask_wide[others]) & (ws_wide[others] >= 3.0) & power_wide[others].notna())
        d_p = power_wide[others].where(d_ws.notna())
        d_stack_ws = d_ws.stack()
        d_stack_p = d_p.stack()
        tr_m = d_stack_ws.index.get_level_values(0) < train_cutoff
        f_bins = np.floor(d_stack_ws[tr_m] * 2.0 + 0.5) / 2.0
        f_grp = d_stack_p[tr_m].groupby(f_bins).agg(["mean", "count"])
        ref_fleet = ws_bin.map(f_grp["mean"].where(f_grp["count"] >= 30))

        # 3. Self historical baseline (target train only, event-screened)
        self_tr = healthy & (t_ws.index < train_cutoff)
        s_grp = t_p[self_tr].groupby(ws_bin[self_tr]).agg(["mean", "count"])
        ref_self = ws_bin.map(s_grp["mean"].where(s_grp["count"] >= 20))

        for baseline_name, ref_series in (
            ("oem_contractual", ref_oem),
            ("fleet_consensus", ref_fleet),
            ("self_historical", ref_self),
        ):
            eligible = healthy & ref_series.gt(50.0)
            dev = (t_p - ref_series).where(eligible)
            daily = dev.groupby(dev.index.normalize()).agg(["mean", "count"])
            daily = daily[daily["count"] >= min_daily_rows]["mean"].dropna()

            cal_vals = daily[(daily.index >= train_cutoff) & (daily.index < cal_end)].tolist()
            eval_daily = daily[daily.index >= cal_end]
            if len(cal_vals) < 7 or eval_daily.empty:
                continue

            series = [(ts.date(), float(val)) for ts, val in eval_daily.items()]
            signals = monitor(series, [float(v) for v in cal_vals], max_hold_gap_days=max_hold_gap_days)
            for sig in signals:
                signal_rows.append(
                    {
                        "turbine_id": target,
                        "baseline": baseline_name,
                        "day": sig.day.isoformat(),
                        "ewma_kw": round(sig.ewma_kw, 2),
                        "cusum_positive": round(sig.cusum_positive, 3),
                        "cusum_negative": round(sig.cusum_negative, 3),
                        "alert": sig.alert,
                        "onset_left_censored": sig.onset_left_censored,
                    }
                )
            alerts = [s for s in signals if s.alert]
            neg_alerts = [s for s in alerts if s.cusum_negative > 5.0 or s.ewma_kw < 0]
            summary_rows.append(
                {
                    "turbine_id": target,
                    "baseline": baseline_name,
                    "calibration_days": int(len(cal_vals)),
                    "calibration_median_bias_kw": round(float(median(cal_vals)), 2),
                    "eval_days": int(len(signals)),
                    "alert_days": int(len(alerts)),
                    "negative_drift_alert_days": int(len(neg_alerts)),
                    "first_alert_day": alerts[0].day.isoformat() if alerts else None,
                    "any_left_censored_onset": any(s.onset_left_censored for s in alerts),
                    "final_cusum_neg": round(signals[-1].cusum_negative, 2),
                    "final_cusum_pos": round(signals[-1].cusum_positive, 2),
                }
            )

    return pd.DataFrame(signal_rows), pd.DataFrame(summary_rows)
