"""Blade pitch asymmetry diagnostics and pre/post maintenance performance recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_KEYWORDS = (
    "pitch",
    "blade",
    "yaw",
    "anemometer",
    "vane",
    "aerodynamic",
    "gearbox",
    "bearing",
    "maintenance",
    "service",
    "repair",
    "inspection",
)


def analyze_blade_pitch_asymmetry(
    df: pd.DataFrame,
    stop_mask: pd.Series | None = None,
    *,
    asymmetry_threshold_deg: float = 0.3,
) -> pd.DataFrame:
    """Detect inter-blade pitch offsets (A-B, B-C, C-A) in partial load (5-11 m/s)."""
    needed = {"turbine_id", "wind_speed_ms", "active_power_kw", "pitch_a_deg", "pitch_b_deg", "pitch_c_deg"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    rows: list[dict] = []
    work = df.copy()
    if stop_mask is not None:
        work = work[~stop_mask.reindex(work.index, fill_value=False)]

    for turbine_id, grp in work.groupby("turbine_id"):
        sub = grp.dropna(subset=["wind_speed_ms", "active_power_kw", "pitch_a_deg", "pitch_b_deg", "pitch_c_deg"])
        sub = sub[sub["wind_speed_ms"].between(5.0, 11.0) & (sub["active_power_kw"] > 50.0)]
        if len(sub) < 100:
            continue
        dab = sub["pitch_a_deg"] - sub["pitch_b_deg"]
        dbc = sub["pitch_b_deg"] - sub["pitch_c_deg"]
        dca = sub["pitch_c_deg"] - sub["pitch_a_deg"]
        max_abs_median = float(max(abs(dab.median()), abs(dbc.median()), abs(dca.median())))
        excursion_frac = float(
            ((dab.abs() > asymmetry_threshold_deg) | (dbc.abs() > asymmetry_threshold_deg) | (dca.abs() > asymmetry_threshold_deg)).mean()
        )
        rows.append(
            {
                "turbine_id": turbine_id,
                "n_partial_load": int(len(sub)),
                "median_pitch_a_deg": round(float(sub["pitch_a_deg"].median()), 3),
                "median_pitch_b_deg": round(float(sub["pitch_b_deg"].median()), 3),
                "median_pitch_c_deg": round(float(sub["pitch_c_deg"].median()), 3),
                "median_a_minus_b_deg": round(float(dab.median()), 3),
                "median_b_minus_c_deg": round(float(dbc.median()), 3),
                "median_c_minus_a_deg": round(float(dca.median()), 3),
                "max_pair_offset_deg": round(max_abs_median, 3),
                "asymmetry_excursion_pct": round(100.0 * excursion_frac, 2),
                "flag_asymmetry": bool(max_abs_median > asymmetry_threshold_deg or excursion_frac > 0.05),
            }
        )
    return pd.DataFrame(rows)


def evaluate_maintenance_pitch_and_ppr_recovery(
    ws_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    pitch_a_wide: pd.DataFrame | None,
    stop_mask_wide: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    window_days: int = 7,
    min_bin_samples: int = 12,
    min_event_hours: float = 0.5,
) -> pd.DataFrame:
    """Evaluate 7-day pre/post repair pitch shifts AND event-screened PPR recovery.

    Inspects bounded Stop episodes in Scheduled Maintenance or Forced outage with
    pitch/maintenance/repair texts, reporting both aerodynamic blade shifts (when pitch
    channels exist) and paired PPR improvement (across all turbines).
    """
    if events_df.empty:
        return pd.DataFrame()
    ids = list(ws_wide.columns)
    times = ws_wide.index
    seven = pd.Timedelta(days=window_days)

    # Precompute leave-target-out healthy farm reference curve across full year
    fleet_refs: dict[str, pd.Series] = {}
    for target in ids:
        others = [k for k in ids if k != target]
        d_ws = ws_wide[others].where((~stop_mask_wide[others]) & (ws_wide[others] >= 4.0) & power_wide[others].notna())
        d_p = power_wide[others].where(d_ws.notna())
        stacked_ws = d_ws.stack()
        stacked_p = d_p.stack()
        d_bins = np.floor(stacked_ws * 2.0 + 0.5) / 2.0
        ref_tbl = stacked_p.groupby(d_bins).agg(["mean", "count"])
        fleet_refs[target] = ref_tbl["mean"].where(ref_tbl["count"] >= 30)

    candidates = events_df[
        (events_df["status"] == "Stop")
        & (~events_df["open_ended"])
        & (events_df["iec_category"].isin(["Scheduled Maintenance", "Forced outage"]))
        & events_df["timestamp_start"].notna()
        & events_df["timestamp_end"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["dur_hours"] = (candidates["timestamp_end"] - candidates["timestamp_start"]).dt.total_seconds() / 3600.0
    rows: list[dict] = []

    for turbine_id in ids:
        t_events = candidates[candidates["turbine_id"] == turbine_id].sort_values("timestamp_start")
        if t_events.empty:
            continue
        t_ws = ws_wide[turbine_id]
        t_p = power_wide[turbine_id]
        t_stop = stop_mask_wide[turbine_id]
        t_pitch = pitch_a_wide[turbine_id] if (pitch_a_wide is not None and turbine_id in pitch_a_wide.columns) else None
        t_bins = np.floor(t_ws * 2.0 + 0.5) / 2.0
        t_ref = t_bins.map(fleet_refs[turbine_id])
        t_ppr = (t_p / t_ref).where((~t_stop) & t_ws.between(5.0, 12.0) & (t_ref > 50.0) & t_p.notna())

        for row in t_events.itertuples(index=False):
            text = f"{row.message} {row.comment} {row.service_contract_category}".lower()
            is_pitch_text = "pitch" in text or "blade" in text
            if not (is_pitch_text or row.dur_hours >= min_event_hours):
                continue
            start_ts = row.timestamp_start
            end_ts = row.timestamp_end
            full_window = (start_ts - seven >= times.min()) and (end_ts + seven <= times.max())
            competing = bool(
                (
                    (t_events["timestamp_start"] != start_ts)
                    & (t_events["iec_category"] == "Scheduled Maintenance")
                    & (t_events["timestamp_start"] < end_ts + seven)
                    & (t_events["timestamp_end"] > start_ts - seven)
                ).any()
            )

            pre_mask = (times >= start_ts - seven) & (times < start_ts) & (~t_stop) & t_ws.between(5.0, 12.0)
            post_mask = (times >= end_ts) & (times < end_ts + seven) & (~t_stop) & t_ws.between(5.0, 12.0)

            # Paired PPR across 0.5 m/s bins
            pre_ppr = t_ppr[pre_mask].groupby(t_bins[pre_mask]).agg(["median", "count"])
            post_ppr = t_ppr[post_mask].groupby(t_bins[post_mask]).agg(["median", "count"])
            joined_ppr = pre_ppr.join(post_ppr, lsuffix="_pre", rsuffix="_post").dropna()
            supp_ppr = joined_ppr[(joined_ppr["count_pre"] >= min_bin_samples) & (joined_ppr["count_post"] >= min_bin_samples)]
            delta_ppr = float((supp_ppr["median_post"] - supp_ppr["median_pre"]).median()) if not supp_ppr.empty else None

            # Paired Blade-A pitch across 0.5 m/s bins (where available)
            delta_pitch = None
            pitch_bins = 0
            if t_pitch is not None and t_pitch.notna().sum() > 100:
                pre_pt = t_pitch[pre_mask].groupby(t_bins[pre_mask]).agg(["median", "count"])
                post_pt = t_pitch[post_mask].groupby(t_bins[post_mask]).agg(["median", "count"])
                joined_pt = pre_pt.join(post_pt, lsuffix="_pre", rsuffix="_post").dropna()
                supp_pt = joined_pt[(joined_pt["count_pre"] >= min_bin_samples) & (joined_pt["count_post"] >= min_bin_samples)]
                pitch_bins = int(len(supp_pt))
                if not supp_pt.empty:
                    delta_pitch = float((supp_pt["median_post"] - supp_pt["median_pre"]).median())

            if not full_window or competing or supp_ppr.empty:
                outcome = "insufficient_or_confounded"
            elif delta_ppr is not None and delta_ppr >= 0.02:
                outcome = "improved_ppr"
            elif delta_ppr is not None and delta_ppr <= -0.02:
                outcome = "degraded_ppr"
            else:
                outcome = "unchanged_ppr"

            rows.append(
                {
                    "turbine_id": turbine_id,
                    "iec_category": row.iec_category,
                    "code": row.code,
                    "message": row.message,
                    "is_pitch_keyword": bool(is_pitch_text),
                    "start_utc": start_ts.isoformat(),
                    "end_utc": end_ts.isoformat(),
                    "duration_hours": round(float(row.dur_hours), 2),
                    "full_followup": bool(full_window),
                    "competing_maintenance": competing,
                    "supported_ppr_bins": int(len(supp_ppr)),
                    "delta_ppr": round(delta_ppr, 4) if delta_ppr is not None else None,
                    "supported_pitch_bins": pitch_bins,
                    "delta_pitch_deg": round(delta_pitch, 3) if delta_pitch is not None else None,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)
