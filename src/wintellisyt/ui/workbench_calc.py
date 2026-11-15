"""Fast interactive diagnostic calculation engine for WINTELLISYT v2 6-tab workbench."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wintellisyt.atmosphere import STANDARD_DENSITY
from wintellisyt.data.workbench_store import (
    WorkbenchStore,
    debounce_events,
    normalize_turbine_id,
    vector_iec_wind_speed,
    vector_svenningsen_fast,
)
from wintellisyt.ui.data_loader import SuiteBundle


def resolve_wind_speed_series(
    df: pd.DataFrame,
    ws_mode: str = "svenningsen",
    rho_ref: float = STANDARD_DENSITY,
) -> pd.Series:
    """Resolve the active wind speed series in m/s according to air-density normalization mode."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    ws_raw = df["wind_speed_ms"].to_numpy(dtype=float)
    mode = ws_mode.lower()
    if mode == "raw":
        return pd.Series(ws_raw, index=df.index)
    if mode == "publisher":
        if "density_adjusted_ws_ms" in df.columns and df["density_adjusted_ws_ms"].notna().any():
            pub = df["density_adjusted_ws_ms"].to_numpy(dtype=float)
            return pd.Series(np.where(np.isfinite(pub), pub, ws_raw), index=df.index)
        return pd.Series(ws_raw, index=df.index)

    rho = (
        df["density_moist_kg_m3"].to_numpy(dtype=float)
        if "density_moist_kg_m3" in df.columns
        else np.full(len(df), STANDARD_DENSITY)
    )
    if mode == "iec":
        if abs(rho_ref - STANDARD_DENSITY) < 1e-4 and "ws_iec_ms" in df.columns:
            vals = df["ws_iec_ms"].to_numpy(dtype=float)
        else:
            vals = vector_iec_wind_speed(ws_raw, rho, rho_ref)
        return pd.Series(np.where(np.isfinite(vals), vals, ws_raw), index=df.index)

    # Default: Svenningsen (WindPRO/EMD-style exponent scaling)
    if abs(rho_ref - STANDARD_DENSITY) < 1e-4 and "ws_svenningsen_ms" in df.columns:
        vals = df["ws_svenningsen_ms"].to_numpy(dtype=float)
    else:
        vals = vector_svenningsen_fast(ws_raw, rho, rho_ref)
    return pd.Series(np.where(np.isfinite(vals), vals, ws_raw), index=df.index)


def resolve_reference_power_series(df: pd.DataFrame, ref_mode: str = "mpc") -> pd.Series:
    """Resolve the baseline expected power column ('mpc', 'learned', or 'peer') with safe MPC fallback."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    mpc = (
        df["potential_power_mpc_kw"]
        if "potential_power_mpc_kw" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    mode = ref_mode.lower()
    if mode == "learned" and "potential_power_learned_kw" in df.columns:
        return df["potential_power_learned_kw"].fillna(mpc)
    if mode == "peer" and "potential_power_ref_kw" in df.columns:
        return df["potential_power_ref_kw"].fillna(mpc)
    return mpc


def compute_method_of_bins(
    df: pd.DataFrame,
    ws_series: pd.Series,
    power_col: str = "active_power_kw",
    ref_series: pd.Series | None = None,
    *,
    bin_width: float = 0.5,
    min_count: int = 3,
) -> pd.DataFrame:
    """Compute exact IEC 61400-12-1 0.5 m/s Method-of-Bins summary (centers 3.0, 3.5, 4.0...)."""
    cols = [
        "ws_bin",
        "count",
        "power_mean_kw",
        "power_std_kw",
        "power_sem_kw",
        "power_median_kw",
        "ref_mean_kw",
        "delta_kw",
        "delta_pct",
        "pitch_mean_deg",
        "rpm_mean",
        "ti_mean",
        "lost_mwh_bin",
    ]
    if df.empty or power_col not in df.columns:
        return pd.DataFrame(columns=cols)

    work = pd.DataFrame(
        {
            "ws": ws_series.to_numpy(dtype=float),
            "power": df[power_col].to_numpy(dtype=float),
            "ref": ref_series.to_numpy(dtype=float) if ref_series is not None else np.nan,
            "pitch": df["pitch_a_deg"].to_numpy(dtype=float) if "pitch_a_deg" in df.columns else np.nan,
            "rpm": df["rotor_speed_rpm"].to_numpy(dtype=float) if "rotor_speed_rpm" in df.columns else np.nan,
            "ti": df["ti"].to_numpy(dtype=float) if "ti" in df.columns else np.nan,
        }
    )
    work = work[np.isfinite(work["ws"]) & np.isfinite(work["power"]) & (work["ws"] >= 0.0) & (work["ws"] <= 30.0)]
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["ws_bin"] = np.round(work["ws"] / bin_width) * bin_width
    grouped = work.groupby("ws_bin", as_index=False).agg(
        count=("power", "count"),
        power_mean_kw=("power", "mean"),
        power_std_kw=("power", "std"),
        power_median_kw=("power", "median"),
        ref_mean_kw=("ref", "mean"),
        pitch_mean_deg=("pitch", "mean"),
        rpm_mean=("rpm", "mean"),
        ti_mean=("ti", "mean"),
    )
    grouped = grouped[grouped["count"] >= min_count].sort_values("ws_bin").reset_index(drop=True)
    if grouped.empty:
        return pd.DataFrame(columns=cols)

    grouped["power_std_kw"] = grouped["power_std_kw"].fillna(0.0)
    grouped["power_sem_kw"] = grouped["power_std_kw"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["delta_kw"] = grouped["power_mean_kw"] - grouped["ref_mean_kw"]
    grouped["delta_pct"] = np.where(
        grouped["ref_mean_kw"] > 20.0,
        100.0 * grouped["delta_kw"] / grouped["ref_mean_kw"],
        np.nan,
    )
    grouped["lost_mwh_bin"] = np.where(
        np.isfinite(grouped["ref_mean_kw"]),
        (grouped["ref_mean_kw"] - grouped["power_mean_kw"]) * grouped["count"] / 6000.0,
        0.0,
    )
    return grouped


def stratified_webgl_sample(
    df: pd.DataFrame,
    *,
    max_clean: int = 5500,
    max_anomaly: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratify normal operating scatter points across wind bins while retaining anomalous rows."""
    if df.empty or len(df) <= (max_clean + max_anomaly):
        return df.copy()

    rng = np.random.default_rng(seed)
    if "state_category" in df.columns:
        clean_mask = df["state_category"] == "Clean Normal"
    else:
        clean_mask = pd.Series(True, index=df.index)

    clean_df = df[clean_mask]
    anom_df = df[~clean_mask]

    # Sample anomalous rows prioritizing Stop outages & Curtailment over Calm points
    if len(anom_df) > max_anomaly:
        if "state_category" in anom_df.columns:
            high_prio = anom_df[anom_df["state_category"].isin(["Event Outage (Stop)", "Curtailment / Derate"])]
            low_prio = anom_df[~anom_df.index.isin(high_prio.index)]
            if len(high_prio) >= max_anomaly:
                idx = rng.choice(high_prio.index.to_numpy(), size=max_anomaly, replace=False)
                anom_sampled = anom_df.loc[idx]
            else:
                rem = max_anomaly - len(high_prio)
                idx_low = rng.choice(low_prio.index.to_numpy(), size=min(rem, len(low_prio)), replace=False)
                anom_sampled = pd.concat([high_prio, low_prio.loc[idx_low]])
        else:
            idx = rng.choice(anom_df.index.to_numpy(), size=max_anomaly, replace=False)
            anom_sampled = anom_df.loc[idx]
    else:
        anom_sampled = anom_df

    # Stratify clean points evenly across 0.5 m/s bins
    if len(clean_df) > max_clean and "wind_speed_ms" in clean_df.columns:
        bins = (np.round(clean_df["wind_speed_ms"].fillna(0.0) * 2.0) / 2.0).to_numpy()
        unique_bins, counts = np.unique(bins, return_counts=True)
        chosen_indices: list[Any] = []
        per_bin_cap = max(25, max_clean // max(1, len(unique_bins)))
        for b, cnt in zip(unique_bins, counts, strict=True):
            sub_idx = clean_df.index[bins == b].to_numpy()
            if cnt <= per_bin_cap:
                chosen_indices.extend(sub_idx.tolist())
            else:
                chosen_indices.extend(rng.choice(sub_idx, size=per_bin_cap, replace=False).tolist())
        if len(chosen_indices) > max_clean:
            chosen_indices = rng.choice(np.array(chosen_indices), size=max_clean, replace=False).tolist()
        clean_sampled = clean_df.loc[chosen_indices]
    elif len(clean_df) > max_clean:
        idx = rng.choice(clean_df.index.to_numpy(), size=max_clean, replace=False)
        clean_sampled = clean_df.loc[idx]
    else:
        clean_sampled = clean_df

    return pd.concat([clean_sampled, anom_sampled]).sort_index()


def _diagnostic_badges_map(bundle: SuiteBundle, farm_id: str) -> dict[str, str]:
    badges: dict[str, list[str]] = {}

    def add_badge(tid: str, tag: str) -> None:
        key = normalize_turbine_id(tid)
        badges.setdefault(key, [])
        if tag not in badges[key]:
            badges[key].append(tag)

    dcon = bundle.get("direction_consensus")
    if not dcon.empty:
        sub = dcon[dcon["farm_id"] == farm_id] if "farm_id" in dcon.columns else dcon
        for r in sub.itertuples(index=False):
            if abs(float(getattr(r, "total_offset_deg", 0.0) or 0.0)) >= 3.0:
                add_badge(str(r.turbine_id), "M4a North")

    yaw = bundle.get("yaw_misalignment_summary")
    if not yaw.empty:
        sub = yaw[yaw["farm_id"] == farm_id] if "farm_id" in yaw.columns else yaw
        for r in sub.itertuples(index=False):
            if abs(float(getattr(r, "estimated_static_yaw_bias_deg", 0.0) or 0.0)) >= 4.0:
                add_badge(str(r.turbine_id), "M4b Yaw")

    asym = bundle.get("pitch_asymmetry")
    if not asym.empty:
        sub = asym[asym["farm_id"] == farm_id] if "farm_id" in asym.columns else asym
        for r in sub.itertuples(index=False):
            if bool(getattr(r, "flag_asymmetry", False)):
                add_badge(str(r.turbine_id), "M3 PitchAsym")

    act = bundle.get("pitch_actuator_health")
    if not act.empty:
        sub = act[act["farm_id"] == farm_id] if "farm_id" in act.columns else act
        for r in sub.itertuples(index=False):
            if bool(getattr(r, "flag_actuator_anomaly", False)):
                add_badge(str(r.turbine_id), "Actuator")

    pc = bundle.get("pc_health_summary")
    if not pc.empty:
        sub = pc[(pc["farm_id"] == farm_id) & (pc["baseline"] == "fleet_consensus")] if "farm_id" in pc.columns else pc
        for r in sub.itertuples(index=False):
            if int(getattr(r, "negative_drift_alert_days", 0) or 0) > 10:
                add_badge(str(r.turbine_id), "M1/PC Drift")

    return {k: (" | ".join(v) if v else "Nominal") for k, v in badges.items()}


def _layout_lookup(store: WorkbenchStore, farm: str) -> dict[str, dict[str, Any]]:
    """Return static turbine layout metadata keyed by normalized two-digit turbine ID."""
    lay = store.get_layout(farm)
    if lay.empty or "turbine_id" not in lay.columns:
        return {}
    return {normalize_turbine_id(row["turbine_id"]): dict(row) for row in lay.to_dict(orient="records")}


def _turbine_rated_kw(layout_map: dict[str, dict[str, Any]], norm_tid: str, default: float = 2050.0) -> float:
    meta = layout_map.get(norm_tid, {})
    val = meta.get("rated_power_kw", default)
    if val is None or not np.isfinite(val) or float(val) <= 0.0:
        return default
    return float(val)


def _classify_perf_status(
    *,
    clean_dev_pct: float,
    clean_ppr_pct: float,
    active_dev_pct: float,
    cf_gap_pp: float,
    avail_prod_b22_pct: float,
    event_mask_avail_pct: float,
    badges: str,
) -> str:
    """Classify turbine health & performance state for spatial map coloring."""
    if np.isfinite(active_dev_pct) and active_dev_pct > 8.0:
        return "Suspect Over-Read (>+8%)"
    if (
        (np.isfinite(clean_dev_pct) and clean_dev_pct <= -1.8)
        or (np.isfinite(clean_ppr_pct) and clean_ppr_pct < 98.2)
        or ("M1/PC Drift" in badges)
    ):
        return "Underperforming"
    if (
        (np.isfinite(clean_dev_pct) and clean_dev_pct < -0.5)
        or (np.isfinite(cf_gap_pp) and cf_gap_pp <= -2.5)
        or (np.isfinite(event_mask_avail_pct) and event_mask_avail_pct < 93.0)
        or (np.isfinite(avail_prod_b22_pct) and avail_prod_b22_pct < 95.5)
    ):
        return "Sub-Optimal / High Downtime"
    return "Healthy / Nominal"


def compute_fleet_cockpit(
    store: WorkbenchStore,
    bundle: SuiteBundle,
    farm_id: str,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    *,
    ws_mode: str = "svenningsen",
    ref_mode: str = "mpc",
    mask_mode: str = "clean_only",
    rho_ref: float = STANDARD_DENSITY,
) -> dict[str, Any]:
    """Compute Tab 1 Fleet Executive KPIs, Spatial Coordinates, Capacity Factor, Deviation Fingerprint & Availability."""
    farm = farm_id.lower()
    df = store.slice_telemetry(farm, None, start_ts=start_ts, end_ts=end_ts)
    if df.empty:
        return {"kpis": {}, "fingerprint_df": pd.DataFrame(), "completeness_matrix": pd.DataFrame(), "curtailment_audit": {}}

    df = df.copy()
    df["ws_active"] = resolve_wind_speed_series(df, ws_mode=ws_mode, rho_ref=rho_ref)
    df["ref_active"] = resolve_reference_power_series(df, ref_mode=ref_mode)
    badge_map = _diagnostic_badges_map(bundle, farm)
    layout_map = _layout_lookup(store, farm)

    fp_rows: list[dict[str, Any]] = []
    for tid, grp in df.groupby("turbine_id"):
        norm_tid = normalize_turbine_id(tid)
        meta = layout_map.get(norm_tid, {})
        rated_kw = _turbine_rated_kw(layout_map, norm_tid)
        period_hours = max(1.0 / 6.0, float(len(grp)) / 6.0)

        export_mwh = (
            float(grp["energy_export_kwh"].fillna(0.0).sum() / 1000.0)
            if grp["energy_export_kwh"].notna().any()
            else float(grp["active_power_kw"].clip(lower=0.0).fillna(0.0).sum() / 6000.0)
        )
        theo_mwh = (
            float(grp["energy_theoretical_kwh"].fillna(0.0).sum() / 1000.0)
            if grp["energy_theoretical_kwh"].notna().any()
            else float(grp["ref_active"].clip(lower=0.0).fillna(0.0).sum() / 6000.0)
        )
        net_cf_pct = 100.0 * export_mwh * 1000.0 / (rated_kw * period_hours)
        theoretical_cf_pct = 100.0 * theo_mwh * 1000.0 / (rated_kw * period_hours)
        cf_gap_pp = net_cf_pct - theoretical_cf_pct

        downtime_mwh = float(grp["lost_downtime_kwh"].fillna(0.0).sum() / 1000.0)
        perf_loss_mwh = float(grp["lost_performance_kwh"].fillna(0.0).sum() / 1000.0)
        curtail_mwh = float(grp["lost_curtailment_total_kwh"].fillna(0.0).sum() / 1000.0)

        # Raw unmasked deviation % (4-15 m/s)
        valid_raw = grp["ws_active"].between(4.0, 15.0) & grp["active_power_kw"].notna() & (grp["ref_active"] > 50.0)
        raw_p_sum = float(grp.loc[valid_raw, "active_power_kw"].sum())
        raw_r_sum = float(grp.loc[valid_raw, "ref_active"].sum())
        raw_dev_pct = (100.0 * (raw_p_sum - raw_r_sum) / raw_r_sum) if raw_r_sum > 100.0 else np.nan

        # Clean event-screened deviation % (excluding stops & derates)
        clean_mask = valid_raw & (~grp["event_stop_mask"]) & (~grp["curtailment_mask"])
        cln_p_sum = float(grp.loc[clean_mask, "active_power_kw"].sum())
        cln_r_sum = float(grp.loc[clean_mask, "ref_active"].sum())
        clean_dev_pct = (100.0 * (cln_p_sum - cln_r_sum) / cln_r_sum) if cln_r_sum > 100.0 else np.nan
        clean_ppr_pct = (100.0 * cln_p_sum / cln_r_sum) if cln_r_sum > 100.0 else np.nan

        active_dev = raw_dev_pct if mask_mode == "unmasked" else clean_dev_pct
        outage_impact_pct = (clean_dev_pct - raw_dev_pct) if (np.isfinite(clean_dev_pct) and np.isfinite(raw_dev_pct)) else 0.0

        b22 = grp["avail_time_iec_b22"].dropna()
        pb22 = grp["avail_prod_iec_b22"].dropna()
        sys_av = grp["avail_time_system"].dropna()
        da = grp["data_availability"].dropna()
        b22_pct = round(100.0 * float(b22.mean()), 2) if not b22.empty else np.nan
        pb22_pct = round(100.0 * float(pb22.mean()), 2) if not pb22.empty else np.nan
        evt_avail = 100.0 * float((~grp["event_stop_mask"]).mean())
        data_comp = 100.0 * float(da.mean()) if not da.empty else 100.0 * float(grp["active_power_kw"].notna().mean())

        cap_curt_hrs = float((grp["available_capacity_kw"].notna() & (grp["available_capacity_kw"] < 1980.0)).sum() / 6.0)
        shelf_curt_hrs = float(grp["curtailment_mask"].sum() / 6.0)
        badges = badge_map.get(norm_tid, "Nominal")
        perf_status = _classify_perf_status(
            clean_dev_pct=float(clean_dev_pct),
            clean_ppr_pct=float(clean_ppr_pct),
            active_dev_pct=float(active_dev),
            cf_gap_pp=float(cf_gap_pp),
            avail_prod_b22_pct=float(pb22_pct),
            event_mask_avail_pct=float(evt_avail),
            badges=badges,
        )

        lat_val = float(meta["latitude"]) if pd.notna(meta.get("latitude")) else np.nan
        lon_val = float(meta["longitude"]) if pd.notna(meta.get("longitude")) else np.nan
        elev_val = float(meta["elevation_m"]) if pd.notna(meta.get("elevation_m")) else np.nan
        rotor_val = float(meta["rotor_diameter_m"]) if pd.notna(meta.get("rotor_diameter_m")) else 92.5

        fp_rows.append(
            {
                "turbine_id": norm_tid,
                "label": f"{farm[:3].upper()}-{norm_tid}",
                "rows": int(len(grp)),
                "latitude": round(lat_val, 5) if np.isfinite(lat_val) else np.nan,
                "longitude": round(lon_val, 5) if np.isfinite(lon_val) else np.nan,
                "elevation_m": round(elev_val, 1) if np.isfinite(elev_val) else np.nan,
                "rotor_diameter_m": round(rotor_val, 1),
                "rated_power_kw": round(rated_kw, 1),
                "perf_status": perf_status,
                "export_mwh": round(export_mwh, 1),
                "theoretical_mwh": round(theo_mwh, 1),
                "net_cf_pct": round(float(net_cf_pct), 2),
                "theoretical_cf_pct": round(float(theoretical_cf_pct), 2),
                "cf_gap_pp": round(float(cf_gap_pp), 2),
                "downtime_mwh": round(downtime_mwh, 1),
                "perf_loss_mwh": round(perf_loss_mwh, 1),
                "curtail_reported_mwh": round(curtail_mwh, 2),
                "raw_dev_pct": round(float(raw_dev_pct), 2) if np.isfinite(raw_dev_pct) else np.nan,
                "clean_dev_pct": round(float(clean_dev_pct), 2) if np.isfinite(clean_dev_pct) else np.nan,
                "active_dev_pct": round(float(active_dev), 2) if np.isfinite(active_dev) else np.nan,
                "outage_mask_shift_pp": round(float(outage_impact_pct), 2),
                "clean_ppr_pct": round(float(clean_ppr_pct), 2) if np.isfinite(clean_ppr_pct) else np.nan,
                "suspect_anemometer_overread": bool(np.isfinite(active_dev) and active_dev > 8.0),
                "avail_time_iec_b22_pct": b22_pct,
                "avail_prod_iec_b22_pct": pb22_pct,
                "avail_system_pct": round(100.0 * float(sys_av.mean()), 2) if not sys_av.empty else np.nan,
                "event_mask_avail_pct": round(evt_avail, 2),
                "data_completeness_pct": round(data_comp, 2),
                "cap_limit_hours": round(cap_curt_hrs, 1),
                "physical_derate_hours": round(shelf_curt_hrs, 1),
                "diagnostic_badges": badges,
            }
        )

    fp_df = pd.DataFrame(fp_rows).sort_values("turbine_id").reset_index(drop=True)
    if fp_df["latitude"].notna().any() and fp_df["longitude"].notna().any():
        lat0 = float(fp_df["latitude"].dropna().mean())
        lon0 = float(fp_df["longitude"].dropna().mean())
        fp_df["east_m"] = ((fp_df["longitude"] - lon0) * np.cos(np.deg2rad(lat0)) * 111320.0).round(1)
        fp_df["north_m"] = ((fp_df["latitude"] - lat0) * 111132.0).round(1)
    else:
        fp_df["east_m"] = np.arange(len(fp_df), dtype=float) * 250.0
        fp_df["north_m"] = 0.0

    # Turbine x Year-Month Data Completeness & Availability matrix
    df["ym"] = df["timestamp_utc"].dt.strftime("%Y-%m")
    comp_grp = (
        df.groupby(["turbine_id", "ym"])
        .agg(
            valid_pct=("active_power_kw", lambda s: round(100.0 * float(s.notna().mean()), 1)),
            iec_b22_pct=("avail_time_iec_b22", lambda s: round(100.0 * float(s.dropna().mean()), 1) if s.notna().any() else np.nan),
        )
        .reset_index()
    )

    kpis = {
        "farm": farm.title(),
        "n_turbines": int(len(fp_df)),
        "export_gwh": round(float(fp_df["export_mwh"].sum() / 1000.0), 2),
        "theoretical_gwh": round(float(fp_df["theoretical_mwh"].sum() / 1000.0), 2),
        "fleet_net_cf_pct": round(float(fp_df["net_cf_pct"].mean()), 2),
        "fleet_theoretical_cf_pct": round(float(fp_df["theoretical_cf_pct"].mean()), 2),
        "fleet_cf_gap_pp": round(float(fp_df["cf_gap_pp"].mean()), 2),
        "underperforming_count": int((fp_df["perf_status"] == "Underperforming").sum()),
        "downtime_mwh": round(float(fp_df["downtime_mwh"].sum()), 1),
        "perf_loss_mwh": round(float(fp_df["perf_loss_mwh"].sum()), 1),
        "fleet_clean_ppr_pct": round(float(fp_df["clean_ppr_pct"].mean()), 2),
        "fleet_iec_b22_pct": round(float(fp_df["avail_time_iec_b22_pct"].mean()), 2),
        "fleet_prod_iec_b22_pct": round(float(fp_df["avail_prod_iec_b22_pct"].mean()), 2),
        "fleet_event_mask_avail_pct": round(float(fp_df["event_mask_avail_pct"].mean()), 2),
        "suspect_overread_count": int(fp_df["suspect_anemometer_overread"].sum()),
    }

    curtailment_audit = {
        "archive_curtail_total_mwh": round(float(fp_df["curtail_reported_mwh"].sum()), 2),
        "archive_bats_birds_note": "Raw Greenbyte CSV columns for Bats/Birds/Grid/Noise curtailment exist in schema but are exported as 0.0 kWh; physical derates below are detected via Available Capacity < 1980 kW & shelf check.",
        "total_cap_limit_hours": round(float(fp_df["cap_limit_hours"].sum()), 1),
        "total_physical_derate_hours": round(float(fp_df["physical_derate_hours"].sum()), 1),
    }

    return {
        "kpis": kpis,
        "fingerprint_df": fp_df,
        "completeness_matrix": comp_grp,
        "curtailment_audit": curtailment_audit,
    }


def compute_turbine_deep_dive(
    store: WorkbenchStore,
    farm_id: str,
    turbine_id: str,
    period_a: tuple[pd.Timestamp | str | None, pd.Timestamp | str | None],
    period_b: tuple[pd.Timestamp | str | None, pd.Timestamp | str | None] | None = None,
    *,
    ws_mode: str = "svenningsen",
    ref_mode: str = "mpc",
    mask_mode: str = "clean_only",
    rho_ref: float = STANDARD_DENSITY,
    full_points: bool = False,
) -> dict[str, Any]:
    """Compute Tab 2 Dual-Panel Scatter, Period A vs B Method-of-Bins, M2 TI Stratification, and M3 Knee Pitch/RPM."""
    farm = farm_id.lower()
    tid = normalize_turbine_id(turbine_id)
    df_a = store.slice_telemetry(farm, tid, start_ts=period_a[0], end_ts=period_a[1])
    if df_a.empty:
        return {"empty": True}

    df_a = df_a.copy()
    df_a["ws_raw"] = df_a["wind_speed_ms"]
    df_a["ws_adj"] = resolve_wind_speed_series(df_a, ws_mode=ws_mode, rho_ref=rho_ref)
    df_a["ref_active"] = resolve_reference_power_series(df_a, ref_mode=ref_mode)

    # Decide which rows feed the measured power curve
    if mask_mode == "clean_only":
        df_a_curve = df_a[(~df_a["event_stop_mask"]) & (~df_a["curtailment_mask"])]
        df_a_points = df_a_curve
    elif mask_mode == "all_data":
        # Show all layers in scatter, fit clean curve excluding Stop/Curtailment outages
        df_a_curve = df_a[(~df_a["event_stop_mask"]) & (~df_a["curtailment_mask"])]
        df_a_points = df_a
    else:
        # Unmasked raw
        df_a_curve = df_a
        df_a_points = df_a

    scatter_a = df_a_points if full_points else stratified_webgl_sample(df_a_points, max_clean=5500, max_anomaly=2500)

    bins_raw_a = compute_method_of_bins(df_a_curve, df_a_curve["ws_raw"], "active_power_kw", df_a_curve["ref_active"])
    bins_adj_a = compute_method_of_bins(df_a_curve, df_a_curve["ws_adj"], "active_power_kw", df_a_curve["ref_active"])

    # Build smooth 0.5 m/s reference curve grid from the telemetry potential power columns
    ref_curve_src = df_a.dropna(subset=["ws_adj"]).copy()
    ref_curve_src["ws_bin"] = np.round(ref_curve_src["ws_adj"] * 2.0) / 2.0
    ref_curves_df = (
        ref_curve_src[ref_curve_src["ws_bin"].between(0.0, 25.0)]
        .groupby("ws_bin", as_index=False)[
            ["potential_power_mpc_kw", "potential_power_learned_kw", "potential_power_ref_kw"]
        ]
        .median()
        .sort_values("ws_bin")
        .reset_index(drop=True)
    )

    # Optional Period B comparison
    bins_adj_b = pd.DataFrame()
    scatter_b = pd.DataFrame()
    delta_ab_df = pd.DataFrame()
    if period_b is not None:
        df_b = store.slice_telemetry(farm, tid, start_ts=period_b[0], end_ts=period_b[1])
        if not df_b.empty:
            df_b = df_b.copy()
            df_b["ws_raw"] = df_b["wind_speed_ms"]
            df_b["ws_adj"] = resolve_wind_speed_series(df_b, ws_mode=ws_mode, rho_ref=rho_ref)
            df_b["ref_active"] = resolve_reference_power_series(df_b, ref_mode=ref_mode)
            df_b_curve = (
                df_b
                if mask_mode == "unmasked"
                else df_b[(~df_b["event_stop_mask"]) & (~df_b["curtailment_mask"])]
            )
            bins_adj_b = compute_method_of_bins(df_b_curve, df_b_curve["ws_adj"], "active_power_kw", df_b_curve["ref_active"])
            scatter_b = df_b_curve if full_points else stratified_webgl_sample(df_b_curve, max_clean=3000, max_anomaly=800)
            if not bins_adj_a.empty and not bins_adj_b.empty:
                m = pd.merge(
                    bins_adj_a[["ws_bin", "power_mean_kw", "count"]],
                    bins_adj_b[["ws_bin", "power_mean_kw", "count"]],
                    on="ws_bin",
                    suffixes=("_a", "_b"),
                )
                m["b_minus_a_kw"] = m["power_mean_kw_b"] - m["power_mean_kw_a"]
                delta_ab_df = m

    # M1 Summary metrics
    partial = df_a_curve[df_a_curve["ws_adj"].between(4.0, 11.0) & (df_a_curve["ref_active"] > 50.0) & df_a_curve["active_power_kw"].notna()]
    overall = df_a_curve[df_a_curve["ws_adj"].between(3.0, 20.0) & (df_a_curve["ref_active"] > 50.0) & df_a_curve["active_power_kw"].notna()]
    partial_ppr = (
        round(100.0 * float(partial["active_power_kw"].sum() / partial["ref_active"].sum()), 2)
        if not partial.empty and partial["ref_active"].sum() > 0
        else np.nan
    )
    overall_ppr = (
        round(100.0 * float(overall["active_power_kw"].sum() / overall["ref_active"].sum()), 2)
        if not overall.empty and overall["ref_active"].sum() > 0
        else np.nan
    )
    energy_gap_mwh = round(float(bins_adj_a["lost_mwh_bin"].sum()), 2) if not bins_adj_a.empty else 0.0
    exp_mwh_a = (
        float(df_a["energy_export_kwh"].fillna(0.0).sum() / 1000.0)
        if "energy_export_kwh" in df_a.columns and df_a["energy_export_kwh"].notna().any()
        else float(df_a["active_power_kw"].clip(lower=0.0).fillna(0.0).sum() / 6000.0)
    )
    theo_mwh_a = (
        float(df_a["energy_theoretical_kwh"].fillna(0.0).sum() / 1000.0)
        if "energy_theoretical_kwh" in df_a.columns and df_a["energy_theoretical_kwh"].notna().any()
        else float(df_a["ref_active"].clip(lower=0.0).fillna(0.0).sum() / 6000.0)
    )
    rated_kw = _turbine_rated_kw(_layout_lookup(store, farm), tid)
    period_hours_a = max(1.0 / 6.0, float(len(df_a)) / 6.0)
    net_cf_pct = round(100.0 * exp_mwh_a * 1000.0 / (rated_kw * period_hours_a), 2)
    theoretical_cf_pct = round(100.0 * theo_mwh_a * 1000.0 / (rated_kw * period_hours_a), 2)

    # M2 TI Stratification (Low / Med / High TI terciles on Clean Curve rows)
    ti_rows: list[pd.DataFrame] = []
    ti_sub = df_a_curve[df_a_curve["ws_adj"].between(3.5, 16.0) & df_a_curve["ti"].notna() & (df_a_curve["ti"] > 0.01) & (df_a_curve["ti"] < 0.8)].copy()
    if len(ti_sub) >= 150:
        q33, q67 = float(ti_sub["ti"].quantile(0.333)), float(ti_sub["ti"].quantile(0.667))
        ti_sub["ti_regime"] = np.where(
            ti_sub["ti"] <= q33,
            f"Low TI (<={q33:.2f})",
            np.where(ti_sub["ti"] <= q67, f"Med TI ({q33:.2f}-{q67:.2f})", f"High TI (>{q67:.2f})"),
        )
        for reg_name, r_grp in ti_sub.groupby("ti_regime"):
            rb = compute_method_of_bins(r_grp, r_grp["ws_adj"], "active_power_kw", r_grp["ref_active"], min_count=5)
            if not rb.empty:
                rb["ti_regime"] = reg_name
                ti_rows.append(rb)
    ti_strat_df = pd.concat(ti_rows, ignore_index=True) if ti_rows else pd.DataFrame()

    # M3 Knee Pitch & Rotor RPM Diagnostic vs Farm Peer Median
    farm_all = store.slice_telemetry(farm, None, start_ts=period_a[0], end_ts=period_a[1])
    peer_df = farm_all[(farm_all["turbine_id"] != tid) & (~farm_all["event_stop_mask"]) & (~farm_all["curtailment_mask"])].copy()
    if not peer_df.empty:
        peer_df["ws_adj"] = resolve_wind_speed_series(peer_df, ws_mode=ws_mode, rho_ref=rho_ref)
        peer_df["ws_bin"] = np.round(peer_df["ws_adj"] * 2.0) / 2.0
        peer_bins = (
            peer_df[peer_df["ws_bin"].between(3.0, 20.0)]
            .groupby("ws_bin", as_index=False)[["pitch_a_deg", "rotor_speed_rpm", "active_power_kw"]]
            .median()
            .rename(
                columns={
                    "pitch_a_deg": "peer_pitch_deg",
                    "rotor_speed_rpm": "peer_rpm",
                    "active_power_kw": "peer_power_kw",
                }
            )
        )
        knee_df = pd.merge(bins_adj_a[["ws_bin", "count", "power_mean_kw", "pitch_mean_deg", "rpm_mean"]], peer_bins, on="ws_bin", how="left")
    else:
        knee_df = bins_adj_a.copy()

    return {
        "empty": False,
        "farm_id": farm,
        "turbine_id": tid,
        "scatter_a": scatter_a,
        "scatter_b": scatter_b,
        "bins_raw_a": bins_raw_a,
        "bins_adj_a": bins_adj_a,
        "bins_adj_b": bins_adj_b,
        "delta_ab_df": delta_ab_df,
        "ref_curves_df": ref_curves_df,
        "ti_strat_df": ti_strat_df,
        "knee_df": knee_df,
        "m1_summary": {
            "total_rows": int(len(df_a)),
            "clean_rows": int((~df_a["event_stop_mask"] & ~df_a["curtailment_mask"]).sum()),
            "stop_masked_rows": int(df_a["event_stop_mask"].sum()),
            "curtail_masked_rows": int(df_a["curtailment_mask"].sum()),
            "wake_rows": int(df_a["in_geometric_wake"].sum()),
            "partial_load_ppr_pct": partial_ppr,
            "overall_ppr_pct": overall_ppr,
            "energy_gap_mwh": energy_gap_mwh,
            "net_cf_pct": net_cf_pct,
            "theoretical_cf_pct": theoretical_cf_pct,
        },
    }


def compute_chronic_trends_and_avail_check(
    store: WorkbenchStore,
    farm_id: str,
    turbine_id: str,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    *,
    ws_mode: str = "svenningsen",
    ref_mode: str = "mpc",
    rho_ref: float = STANDARD_DENSITY,
    ewma_alpha: float = 0.30,
) -> dict[str, Any]:
    """Compute Tab 3 Daily/Monthly PPR EWMA/CUSUM Trends AND Kelmarsh/Penmanshiel IEC Availability Cross-Check."""
    farm = farm_id.lower()
    tid = normalize_turbine_id(turbine_id)
    farm_df = store.slice_telemetry(farm, None, start_ts=start_ts, end_ts=end_ts)
    if farm_df.empty:
        return {"empty": True}

    farm_df = farm_df.copy()
    farm_df["ws_active"] = resolve_wind_speed_series(farm_df, ws_mode=ws_mode, rho_ref=rho_ref)
    farm_df["ref_active"] = resolve_reference_power_series(farm_df, ref_mode=ref_mode)
    farm_df["day"] = farm_df["timestamp_utc"].dt.strftime("%Y-%m-%d")

    eligible = farm_df["ws_active"].between(4.0, 15.0) & farm_df["active_power_kw"].notna() & (farm_df["ref_active"] > 50.0)
    clean_elig = eligible & (~farm_df["event_stop_mask"]) & (~farm_df["curtailment_mask"])

    tgt = farm_df[farm_df["turbine_id"] == tid].copy()
    peers = farm_df[(farm_df["turbine_id"] != tid) & clean_elig].copy()
    if tgt.empty:
        return {"empty": True}

    # Peer daily baseline PPR to isolate turbine-specific chronic drift from seasonal weather
    peer_daily = (
        peers.groupby("day", as_index=False)
        .agg(peer_p=("active_power_kw", "sum"), peer_r=("ref_active", "sum"), peer_n=("active_power_kw", "count"))
    )
    peer_daily["peer_clean_ppr_pct"] = np.where(peer_daily["peer_r"] > 100.0, 100.0 * peer_daily["peer_p"] / peer_daily["peer_r"], np.nan)

    # Target daily raw, clean, capacity factor, and availability signals
    rated_kw = _turbine_rated_kw(_layout_lookup(store, farm), tid)
    tgt["el_raw"] = eligible[tgt.index].to_numpy()
    tgt["el_cln"] = clean_elig[tgt.index].to_numpy()
    tgt["p_raw"] = np.where(tgt["el_raw"], tgt["active_power_kw"], 0.0)
    tgt["r_raw"] = np.where(tgt["el_raw"], tgt["ref_active"], 0.0)
    tgt["p_cln"] = np.where(tgt["el_cln"], tgt["active_power_kw"], 0.0)
    tgt["r_cln"] = np.where(tgt["el_cln"], tgt["ref_active"], 0.0)
    tgt["net_cf_row"] = 100.0 * tgt["active_power_kw"].clip(lower=0.0) / rated_kw
    tgt["theo_cf_row"] = 100.0 * tgt["ref_active"].clip(lower=0.0) / rated_kw
    tgt["mask_avail"] = (~tgt["event_stop_mask"]).astype(float)

    daily = (
        tgt.groupby("day", as_index=False)
        .agg(
            p_raw=("p_raw", "sum"),
            r_raw=("r_raw", "sum"),
            p_cln=("p_cln", "sum"),
            r_cln=("r_cln", "sum"),
            n_cln=("el_cln", "sum"),
            net_cf_pct=("net_cf_row", "mean"),
            theoretical_cf_pct=("theo_cf_row", "mean"),
            mask_avail_pct=("mask_avail", lambda s: 100.0 * float(s.mean())),
            iec_b22_avail_pct=("avail_time_iec_b22", lambda s: 100.0 * float(s.dropna().mean()) if s.notna().any() else np.nan),
            iec_b23_avail_pct=("avail_time_iec_b23", lambda s: 100.0 * float(s.dropna().mean()) if s.notna().any() else np.nan),
            system_avail_pct=("avail_time_system", lambda s: 100.0 * float(s.dropna().mean()) if s.notna().any() else np.nan),
            prod_iec_b22_pct=("avail_prod_iec_b22", lambda s: 100.0 * float(s.dropna().mean()) if s.notna().any() else np.nan),
        )
        .sort_values("day")
        .reset_index(drop=True)
    )
    daily["net_cf_pct"] = daily["net_cf_pct"].round(2)
    daily["theoretical_cf_pct"] = daily["theoretical_cf_pct"].round(2)
    daily = daily.merge(peer_daily[["day", "peer_clean_ppr_pct"]], on="day", how="left")
    daily["raw_ppr_pct"] = np.where(daily["r_raw"] > 100.0, 100.0 * daily["p_raw"] / daily["r_raw"], np.nan)
    daily["clean_ppr_pct"] = np.where((daily["r_cln"] > 100.0) & (daily["n_cln"] >= 6), 100.0 * daily["p_cln"] / daily["r_cln"], np.nan)
    daily["peer_adj_ppr_pct"] = daily["clean_ppr_pct"] - (daily["peer_clean_ppr_pct"] - 100.0)

    # EWMA + Control Limits + Two-Sided CUSUM on peer_adj_ppr_pct (fallback clean_ppr_pct)
    base_series = daily["peer_adj_ppr_pct"].fillna(daily["clean_ppr_pct"])
    valid_vals = base_series.dropna().to_numpy(dtype=float)
    if len(valid_vals) >= 10:
        cal_n = max(10, min(60, len(valid_vals) // 4))
        cal_slice = valid_vals[:cal_n]
        center = float(np.median(cal_slice))
        sigma = max(1.5, 1.4826 * float(np.median(np.abs(cal_slice - center))))
    else:
        center = 100.0
        sigma = 3.0

    ewma_limit = 3.0 * sigma * np.sqrt(ewma_alpha / (2.0 - ewma_alpha))
    ewma_vals: list[float] = []
    c_pos: list[float] = []
    c_neg: list[float] = []
    alerts: list[bool] = []
    prev_e = center
    prev_cp = 0.0
    prev_cn = 0.0
    for val in base_series.to_numpy(dtype=float):
        if not np.isfinite(val):
            ewma_vals.append(prev_e)
            c_pos.append(prev_cp)
            c_neg.append(prev_cn)
            alerts.append(False)
            continue
        cur_e = ewma_alpha * val + (1.0 - ewma_alpha) * prev_e
        delta = val - center
        cur_cp = max(0.0, prev_cp + delta / sigma - 0.5)
        cur_cn = max(0.0, prev_cn - delta / sigma - 0.5)
        is_alert = abs(cur_e - center) > ewma_limit or max(cur_cp, cur_cn) > 5.0
        ewma_vals.append(cur_e)
        c_pos.append(cur_cp)
        c_neg.append(cur_cn)
        alerts.append(is_alert)
        prev_e, prev_cp, prev_cn = cur_e, cur_cp, cur_cn

    daily["ewma_ppr_pct"] = np.round(ewma_vals, 2)
    daily["ewma_center_pct"] = round(center, 2)
    daily["ewma_ucl_pct"] = round(center + ewma_limit, 2)
    daily["ewma_lcl_pct"] = round(center - ewma_limit, 2)
    daily["cusum_pos"] = np.round(c_pos, 2)
    daily["cusum_neg"] = np.round(c_neg, 2)
    daily["alert"] = alerts

    # Left-censored baseline events check
    ev = store.slice_events(farm, tid, start_ts=start_ts, end_ts=end_ts)
    left_censored_count = int(ev["left_censored"].sum()) if (not ev.empty and "left_censored" in ev.columns) else 0

    # Availability agreement metrics
    av_sub = daily[["mask_avail_pct", "iec_b22_avail_pct", "system_avail_pct"]].dropna()
    corr_b22 = float(av_sub["mask_avail_pct"].corr(av_sub["iec_b22_avail_pct"])) if len(av_sub) >= 5 else np.nan
    mae_b22 = float((av_sub["mask_avail_pct"] - av_sub["iec_b22_avail_pct"]).abs().mean()) if not av_sub.empty else np.nan

    daily["ym"] = daily["day"].str.slice(0, 7)
    monthly = (
        daily.groupby("ym", as_index=False)[
            [
                "raw_ppr_pct",
                "clean_ppr_pct",
                "peer_adj_ppr_pct",
                "net_cf_pct",
                "theoretical_cf_pct",
                "mask_avail_pct",
                "iec_b22_avail_pct",
                "iec_b23_avail_pct",
                "system_avail_pct",
                "prod_iec_b22_pct",
            ]
        ]
        .mean()
        .round(2)
    )

    alert_days = daily[daily["alert"]]["day"].tolist()
    return {
        "empty": False,
        "farm_id": farm,
        "turbine_id": tid,
        "daily_df": daily,
        "monthly_df": monthly,
        "summary": {
            "baseline_center_pct": round(center, 2),
            "ewma_limit_band_pct": round(ewma_limit, 2),
            "alert_days": int(len(alert_days)),
            "first_alert_day": alert_days[0] if alert_days else "None",
            "final_cusum_neg": round(float(c_neg[-1]), 2) if c_neg else 0.0,
            "left_censored_stops": left_censored_count,
            "mean_net_cf_pct": round(float(daily["net_cf_pct"].mean()), 2) if daily["net_cf_pct"].notna().any() else None,
            "mean_theoretical_cf_pct": round(float(daily["theoretical_cf_pct"].mean()), 2)
            if daily["theoretical_cf_pct"].notna().any()
            else None,
            "avail_corr_mask_vs_iec_b22": round(corr_b22, 4) if np.isfinite(corr_b22) else None,
            "avail_mae_mask_vs_iec_b22_pp": round(mae_b22, 2) if np.isfinite(mae_b22) else None,
            "mean_mask_avail_pct": round(float(daily["mask_avail_pct"].mean()), 2),
            "mean_iec_b22_avail_pct": round(float(daily["iec_b22_avail_pct"].mean()), 2)
            if daily["iec_b22_avail_pct"].notna().any()
            else None,
        },
    }


def compute_direction_yaw_wake(
    store: WorkbenchStore,
    bundle: SuiteBundle,
    farm_id: str,
    turbine_id: str,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    *,
    ws_mode: str = "svenningsen",
    rho_ref: float = STANDARD_DENSITY,
) -> dict[str, Any]:
    """Compute Tab 4 M4a True-North Sector Spread, M4b Yaw Power-Ratio Parabola & Cable Windings, and Wake Regimes."""
    farm = farm_id.lower()
    tid = normalize_turbine_id(turbine_id)
    farm_df = store.slice_telemetry(farm, None, start_ts=start_ts, end_ts=end_ts)
    if farm_df.empty:
        return {"empty": True}

    farm_df = farm_df.copy()
    farm_df["ws_active"] = resolve_wind_speed_series(farm_df, ws_mode=ws_mode, rho_ref=rho_ref)

    # 1. M4a Sector-wise circular disagreement (raw wind direction vs peer circular median)
    piv_dir = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="wind_direction_deg")
    piv_pwr = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="active_power_kw")
    piv_ws = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="ws_active")
    piv_stop = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="event_stop_mask").fillna(False)

    sector_spread_rows: list[dict[str, Any]] = []
    if tid in piv_dir.columns and piv_dir.shape[1] >= 2:
        peers = [c for c in piv_dir.columns if c != tid]
        rad_p = np.deg2rad(piv_dir[peers])
        peer_med_dir = (np.rad2deg(np.arctan2(np.sin(rad_p).mean(axis=1), np.cos(rad_p).mean(axis=1))) % 360.0)
        diff = ((piv_dir[tid] - peer_med_dir + 180.0) % 360.0) - 180.0
        sec_df = pd.DataFrame({"peer_dir": peer_med_dir, "diff": diff}).dropna()
        if not sec_df.empty:
            sec_df["sector_deg"] = (np.floor(((sec_df["peer_dir"] + 15.0) % 360.0) / 30.0) * 30.0).astype(int)
            grp_sec = (
                sec_df.groupby("sector_deg", as_index=False)["diff"]
                .agg(n="count", median_bias_deg="median", p25=lambda s: float(s.quantile(0.25)), p75=lambda s: float(s.quantile(0.75)))
            )
            sector_spread_rows = grp_sec.to_dict(orient="records")

    # Classify turbines from bundle consensus + sector spread
    dcon = bundle.get("direction_consensus")
    dcon_farm = dcon[dcon["farm_id"] == farm].copy() if not dcon.empty and "farm_id" in dcon.columns else dcon.copy()

    # 2. M4b Yaw Power-Ratio Parabola + Cable Windings (target turbine live window)
    tgt = farm_df[farm_df["turbine_id"] == tid].copy()
    yaw_bins_df = pd.DataFrame()
    yaw_metrics: dict[str, Any] = {}
    if not tgt.empty and {"vane_position_deg", "active_power_kw", "potential_power_mpc_kw"}.issubset(tgt.columns):
        sub_y = tgt[
            (~tgt["event_stop_mask"])
            & (~tgt["curtailment_mask"])
            & tgt["ws_active"].between(5.5, 10.5)
            & (tgt["active_power_kw"] > 50.0)
            & (tgt["potential_power_mpc_kw"] > 100.0)
            & tgt["vane_position_deg"].notna()
        ].copy()
        sub_y["vane_err_deg"] = ((sub_y["vane_position_deg"] + 180.0) % 360.0) - 180.0
        sub_y["p_ratio"] = sub_y["active_power_kw"] / sub_y["potential_power_mpc_kw"]
        band = sub_y[sub_y["vane_err_deg"].between(-22.0, 22.0) & sub_y["p_ratio"].between(0.25, 1.55)].copy()
        if len(band) >= 120:
            band["yaw_bin_deg"] = np.round(band["vane_err_deg"] / 2.0) * 2.0
            yb = (
                band.groupby("yaw_bin_deg", as_index=False)["p_ratio"]
                .agg(n="count", median_p_ratio="median", mean_p_ratio="mean")
            )
            yb = yb[yb["n"] >= 10].sort_values("yaw_bin_deg").reset_index(drop=True)
            if len(yb) >= 5:
                x = yb["yaw_bin_deg"].to_numpy(dtype=float)
                y = yb["median_p_ratio"].to_numpy(dtype=float)
                coeffs = np.polyfit(x, y, 2)
                raw_vertex = -coeffs[1] / (2.0 * coeffs[0]) if coeffs[0] < -1.2e-4 else np.nan
                if np.isfinite(raw_vertex) and -8.0 <= float(raw_vertex) <= 8.0:
                    vertex = float(raw_vertex)
                    yb["fitted_parabola"] = np.polyval(coeffs, x)
                else:
                    vertex = 0.0
                    yb["fitted_parabola"] = np.nan
                med_vane = float(band["vane_err_deg"].median())
                yaw_bias = float(np.clip(med_vane - vertex, -10.0, 10.0))
                cos_loss = max(0.0, 1.0 - float(np.cos(np.deg2rad(yaw_bias)) ** 2))
                partial_mwh = float(band["active_power_kw"].sum() / 6000.0)
                yaw_bins_df = yb
                yaw_metrics = {
                    "median_vane_deg": round(med_vane, 2),
                    "optimal_peak_deg": round(vertex, 2),
                    "estimated_yaw_bias_deg": round(yaw_bias, 2),
                    "est_cos2_loss_pct": round(100.0 * cos_loss, 3),
                    "est_recoverable_mwh": round(partial_mwh * cos_loss, 2),
                    "n_partial_load": int(len(band)),
                }

    # Cable windings daily trajectory
    cable_df = pd.DataFrame()
    if not tgt.empty and "cable_windings" in tgt.columns and tgt["cable_windings"].notna().any():
        tgt["day"] = tgt["timestamp_utc"].dt.strftime("%Y-%m-%d")
        cable_df = (
            tgt.dropna(subset=["cable_windings"])
            .groupby("day", as_index=False)["cable_windings"]
            .agg(mean_turns="mean", max_turns="max", min_turns="min")
            .sort_values("day")
        )

    # 3. Wake Regimes (Freestream vs Active Donor Wake vs Stopped Donor Placebo)
    wake_curves_df = pd.DataFrame()
    if not tgt.empty and tid in piv_ws.columns:
        # Upstream peer operational check: any other turbine running (>50 kW) at that timestamp when geometrically in wake
        peers_running = ((piv_pwr.drop(columns=[tid], errors="ignore") > 50.0) & (~piv_stop.drop(columns=[tid], errors="ignore"))).any(axis=1)
        peers_all_stopped = ~peers_running
        ts_map_run = peers_running.reindex(tgt["timestamp_utc"]).to_numpy(dtype=bool)
        ts_map_stp = peers_all_stopped.reindex(tgt["timestamp_utc"]).to_numpy(dtype=bool)
        tgt_wake = tgt[(~tgt["event_stop_mask"]) & (~tgt["curtailment_mask"])].copy()
        w_run = ts_map_run[(~tgt["event_stop_mask"]) & (~tgt["curtailment_mask"])]
        w_stp = ts_map_stp[(~tgt["event_stop_mask"]) & (~tgt["curtailment_mask"])]
        tgt_wake["wake_regime"] = np.where(
            ~tgt_wake["in_geometric_wake"],
            "Freestream (Unwaked)",
            np.where(w_run, "Active Wake (Donor Operating)", np.where(w_stp, "Wake Sector (All Donors Stopped)", "Freestream (Unwaked)")),
        )
        w_parts: list[pd.DataFrame] = []
        for wname, wg in tgt_wake.groupby("wake_regime"):
            wb = compute_method_of_bins(wg, wg["ws_active"], "active_power_kw", wg["potential_power_mpc_kw"], min_count=4)
            if not wb.empty:
                wb["wake_regime"] = wname
                w_parts.append(wb)
        if w_parts:
            wake_curves_df = pd.concat(w_parts, ignore_index=True)

    # Farm wind rose (30 deg bins x 4 speed bands) from calibrated directions in selected window
    rose_df = pd.DataFrame()
    valid_r = farm_df.dropna(subset=["wd_calibrated_deg", "ws_active"]).copy()
    if not valid_r.empty:
        valid_r["dir_bin"] = (np.floor(((valid_r["wd_calibrated_deg"] + 15.0) % 360.0) / 30.0) * 30.0).astype(int)
        valid_r["speed_band"] = pd.cut(
            valid_r["ws_active"],
            bins=[0.0, 5.0, 10.0, 15.0, np.inf],
            labels=["0-5 m/s", "5-10 m/s", "10-15 m/s", "15+ m/s"],
            right=False,
        ).astype(str)
        rose_df = valid_r.groupby(["dir_bin", "speed_band"], as_index=False).size().rename(columns={"size": "n"})

    return {
        "empty": False,
        "farm_id": farm,
        "turbine_id": tid,
        "direction_consensus_df": dcon_farm,
        "sector_spread_df": pd.DataFrame(sector_spread_rows),
        "wake_troughs_df": bundle.get("direction_wake_troughs"),
        "yaw_bins_df": yaw_bins_df,
        "yaw_metrics": yaw_metrics,
        "cable_df": cable_df,
        "wake_curves_df": wake_curves_df,
        "rose_df": rose_df,
    }


def compute_outage_pareto_workbench(
    store: WorkbenchStore,
    bundle: SuiteBundle,
    farm_id: str,
    turbine_id: str | None = None,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    *,
    debounce_sec: int = 60,
    selected_episode_idx: int = 0,
    code_search: str = "",
    subsystem_filter: str = "All",
    iec_filter: str = "All",
) -> dict[str, Any]:
    """Compute Senvion Event Code Dictionary, Debounced Outage Pareto, Gantt, Alarm Cascade, and Repair Recovery."""
    farm = farm_id.lower()
    raw_events = store.slice_events(farm, None, start_ts=start_ts, end_ts=end_ts)
    dict_df = store.get_event_dictionary(farm)

    if code_search.strip():
        q = code_search.strip().lower()
        mask = (
            dict_df["code"].astype(str).str.lower().str.contains(q, na=False)
            | dict_df["message"].astype(str).str.lower().str.contains(q, na=False)
        )
        dict_df = dict_df[mask]
    if subsystem_filter != "All":
        dict_df = dict_df[dict_df["subsystem"] == subsystem_filter]
    if iec_filter != "All":
        dict_df = dict_df[dict_df["iec_category"] == iec_filter]
    dict_df = dict_df.reset_index(drop=True)

    debounced_farm = debounce_events(raw_events, gap_seconds=debounce_sec, stop_only=True)
    if not debounced_farm.empty and subsystem_filter != "All":
        debounced_farm = debounced_farm[debounced_farm["subsystem"] == subsystem_filter]
    if not debounced_farm.empty and iec_filter != "All":
        debounced_farm = debounced_farm[debounced_farm["iec_category"] == iec_filter]
    debounced_farm = debounced_farm.reset_index(drop=True)

    # Estimate MWh per episode quickly using farm mean theoretical operating kW ~ 650 kW fallback
    if not debounced_farm.empty:
        debounced_farm["est_lost_mwh"] = (debounced_farm["duration_hours"] * 0.65).round(2)

    pareto_sub = (
        debounced_farm.groupby("subsystem", as_index=False)
        .agg(episodes=("root_code", "count"), raw_alarms=("merged_event_count", "sum"), duration_hours=("duration_hours", "sum"), est_lost_mwh=("est_lost_mwh", "sum"))
        .sort_values("duration_hours", ascending=False)
        .reset_index(drop=True)
        if not debounced_farm.empty
        else pd.DataFrame()
    )
    pareto_codes = (
        debounced_farm.groupby(["root_code", "root_message", "subsystem", "iec_category"], as_index=False)
        .agg(episodes=("root_code", "count"), duration_hours=("duration_hours", "sum"), est_lost_mwh=("est_lost_mwh", "sum"))
        .sort_values("duration_hours", ascending=False)
        .head(12)
        .reset_index(drop=True)
        if not debounced_farm.empty
        else pd.DataFrame()
    )
    pareto_turb = (
        debounced_farm.groupby(["turbine_id", "subsystem"], as_index=False)["duration_hours"].sum()
        if not debounced_farm.empty
        else pd.DataFrame()
    )

    if turbine_id is not None and not debounced_farm.empty:
        norm_tid = normalize_turbine_id(turbine_id)
        ep_focus = debounced_farm[debounced_farm["turbine_id"] == norm_tid].reset_index(drop=True)
        if ep_focus.empty:
            ep_focus = debounced_farm
    else:
        ep_focus = debounced_farm

    gantt_df = ep_focus.head(35).copy() if not ep_focus.empty else pd.DataFrame()

    # T0 +- 60 min Alarm Cascade around selected episode
    cascade_df = pd.DataFrame()
    selected_episode_info: dict[str, Any] = {}
    if not gantt_df.empty:
        idx = int(np.clip(selected_episode_idx, 0, len(gantt_df) - 1))
        ep = gantt_df.iloc[idx]
        t0 = pd.Timestamp(ep["timestamp_start"])
        win = store.slice_events(
            farm,
            str(ep["turbine_id"]),
            start_ts=t0 - pd.Timedelta(minutes=60),
            end_ts=t0 + pd.Timedelta(minutes=60),
        )
        if not win.empty:
            cascade_df = win[
                ["timestamp_start", "timestamp_end", "status", "code", "message", "iec_category", "subsystem", "duration_raw"]
            ].sort_values("timestamp_start").reset_index(drop=True)
            cascade_df["offset_min"] = ((pd.to_datetime(cascade_df["timestamp_start"], utc=True) - t0).dt.total_seconds() / 60.0).round(1)
        selected_episode_info = {
            "turbine_id": str(ep["turbine_id"]),
            "t0_utc": t0.strftime("%Y-%m-%d %H:%M UTC"),
            "duration_hours": float(ep["duration_hours"]),
            "root_code": str(ep["root_code"]),
            "root_message": str(ep["root_message"]),
            "merged_alarms": int(ep["merged_event_count"]),
        }

    rec = bundle.get("pitch_ppr_recovery")
    if not rec.empty and "farm_id" in rec.columns:
        rec_farm = rec[rec["farm_id"] == farm].copy()
    else:
        rec_farm = rec.copy()

    return {
        "dictionary_df": dict_df,
        "debounced_farm_df": debounced_farm,
        "pareto_subsystem_df": pareto_sub,
        "pareto_codes_df": pareto_codes,
        "pareto_turbine_df": pareto_turb,
        "gantt_df": gantt_df,
        "cascade_df": cascade_df,
        "selected_episode_info": selected_episode_info,
        "recovery_df": rec_farm,
        "raw_stop_count": int((raw_events["status"] == "Stop").sum()) if not raw_events.empty else 0,
        "debounced_stop_count": int(len(debounced_farm)),
    }


def compute_drivetrain_pitch_mcp(
    store: WorkbenchStore,
    bundle: SuiteBundle,
    farm_id: str,
    turbine_id: str,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    *,
    full_points: bool = False,
) -> dict[str, Any]:
    """Compute Tab 6 Drivetrain Thermal/Vibration CMS, 3-Blade Pitch & Actuator Current, and Peer MCP Reconstruction."""
    farm = farm_id.lower()
    tid = normalize_turbine_id(turbine_id)
    farm_df = store.slice_telemetry(farm, None, start_ts=start_ts, end_ts=end_ts)
    if farm_df.empty:
        return {"empty": True}

    op = farm_df[(~farm_df["event_stop_mask"]) & (farm_df["active_power_kw"] > 50.0)].copy()
    tgt = op[op["turbine_id"] == tid].copy()
    peers = op[op["turbine_id"] != tid].copy()

    # 1. Thermal scatter & peer residual in 250 kW load bins
    temp_channels = [
        ("gear_oil_temp_c", "Gear Oil Temp (C)"),
        ("rotor_bearing_temp_c", "Rotor Bearing Temp (C)"),
        ("gen_bearing_front_temp_c", "Generator Bearing Front (C)"),
        ("stator_temp_1_c", "Stator Temp 1 (C)"),
        ("transformer_temp_c", "Transformer Temp (C)"),
        ("front_bearing_temp_c", "Front Bearing Temp (C)"),
    ]
    available_temps = [(c, lbl) for c, lbl in temp_channels if c in tgt.columns and tgt[c].notna().sum() >= 30]
    thermal_residuals: list[dict[str, Any]] = []
    if not tgt.empty and not peers.empty:
        tgt["p_bin"] = (np.round(tgt["active_power_kw"] / 250.0) * 250.0).astype(float)
        peers["p_bin"] = (np.round(peers["active_power_kw"] / 250.0) * 250.0).astype(float)
        for col, lbl in available_temps:
            p_ref = peers.groupby("p_bin")[col].median()
            t_exp = tgt["p_bin"].map(p_ref)
            res = (tgt[col] - t_exp).dropna()
            if len(res) >= 20:
                thermal_residuals.append(
                    {
                        "channel": col,
                        "label": lbl,
                        "n": int(len(res)),
                        "target_median_c": round(float(tgt[col].median()), 2),
                        "residual_median_c": round(float(res.median()), 2),
                        "residual_p95_c": round(float(res.quantile(0.95)), 2),
                    }
                )

    cms_scatter = tgt if (full_points or len(tgt) <= 5000) else stratified_webgl_sample(tgt, max_clean=4500, max_anomaly=500)
    peer_cms_scatter = peers if len(peers) <= 4500 else peers.sample(n=4500, random_state=42)

    # 2. 3-Blade Pitch Asymmetry & 3-Axis Motor Currents by 0.5 m/s bin
    pitch_motor_df = pd.DataFrame()
    if not tgt.empty and tgt["wind_speed_ms"].notna().any():
        pm = tgt[tgt["wind_speed_ms"].between(3.0, 20.0)].copy()
        pm["ws_bin"] = np.round(pm["wind_speed_ms"] * 2.0) / 2.0
        if {"pitch_a_deg", "pitch_b_deg", "pitch_c_deg"}.issubset(pm.columns):
            pm["dab"] = pm["pitch_a_deg"] - pm["pitch_b_deg"]
            pm["dbc"] = pm["pitch_b_deg"] - pm["pitch_c_deg"]
            pm["dca"] = pm["pitch_c_deg"] - pm["pitch_a_deg"]
        else:
            pm["dab"] = pm["dbc"] = pm["dca"] = np.nan
        pitch_motor_df = (
            pm.groupby("ws_bin", as_index=False)[
                ["dab", "dbc", "dca", "motor_current_1_a", "motor_current_2_a", "motor_current_3_a"]
            ]
            .median()
            .sort_values("ws_bin")
        )

    # 3. Peer MCP Wind Reconstruction & S1/S2 Divergence (Daily series)
    mcp_daily_df = pd.DataFrame()
    tgt_all = farm_df[farm_df["turbine_id"] == tid].copy()
    if not tgt_all.empty:
        piv_ws = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="wind_speed_ms")
        piv_w = farm_df.pivot(index="timestamp_utc", columns="turbine_id", values="in_geometric_wake").fillna(False)
        peer_cols = [c for c in piv_ws.columns if c != tid]
        if peer_cols:
            uw_ws = piv_ws[peer_cols].where(~piv_w[peer_cols])
            peer_uw_med = uw_ws.median(axis=1)
            peer_all_med = piv_ws[peer_cols].median(axis=1)
            donor_ws = peer_uw_med.fillna(peer_all_med)
            # Zero-leakage first-half calibration offset
            obs_ws = piv_ws[tid]
            valid = obs_ws.notna() & donor_ws.notna()
            cal_n = max(1, int(valid.sum() // 2))
            cal_diff = (obs_ws[valid] - donor_ws[valid]).iloc[:cal_n].median()
            pred_ws = (donor_ws + (float(cal_diff) if np.isfinite(cal_diff) else 0.0)).clip(lower=0.0)
            mcp_ts = pd.DataFrame(
                {
                    "timestamp_utc": tgt_all["timestamp_utc"],
                    "obs_ws": obs_ws.reindex(tgt_all["timestamp_utc"]).to_numpy(),
                    "peer_mcp_ws": pred_ws.reindex(tgt_all["timestamp_utc"]).to_numpy(),
                    "s1_ws": tgt_all["wind_speed_s1_ms"].to_numpy(),
                    "s2_ws": tgt_all["wind_speed_s2_ms"].to_numpy(),
                }
            )
            mcp_ts["day"] = mcp_ts["timestamp_utc"].dt.strftime("%Y-%m-%d")
            mcp_ts["mcp_residual_ms"] = mcp_ts["obs_ws"] - mcp_ts["peer_mcp_ws"]
            mcp_ts["s1_minus_s2_ms"] = mcp_ts["s1_ws"] - mcp_ts["s2_ws"]
            mcp_daily_df = (
                mcp_ts.groupby("day", as_index=False)[["obs_ws", "peer_mcp_ws", "mcp_residual_ms", "s1_minus_s2_ms"]]
                .median()
                .sort_values("day")
            )

    gap_b = bundle.get("peer_gap_benchmark")
    drift_b = bundle.get("peer_drift_summary")
    cms_b = bundle.get("drivetrain_cms_summary")
    return {
        "empty": False,
        "farm_id": farm,
        "turbine_id": tid,
        "available_temps": available_temps,
        "thermal_residuals_df": pd.DataFrame(thermal_residuals),
        "cms_scatter": cms_scatter,
        "peer_cms_scatter": peer_cms_scatter,
        "pitch_motor_df": pitch_motor_df,
        "mcp_daily_df": mcp_daily_df,
        "peer_gap_df": gap_b[gap_b["farm_id"] == farm] if not gap_b.empty and "farm_id" in gap_b.columns else gap_b,
        "peer_drift_df": drift_b[drift_b["farm_id"] == farm] if not drift_b.empty and "farm_id" in drift_b.columns else drift_b,
        "cms_summary_df": cms_b[cms_b["farm_id"] == farm] if not cms_b.empty and "farm_id" in cms_b.columns else cms_b,
    }


def synthesize_turbine_root_cause(
    cockpit: dict[str, Any],
    deep: dict[str, Any],
    trend: dict[str, Any],
    dir_data: dict[str, Any],
    ev_data: dict[str, Any],
    cms_data: dict[str, Any],
) -> dict[str, Any]:
    """Synthesize M1-M4, Outage Pareto, and CMS evidence into a single-screen root-cause verdict."""
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    tid = str(deep.get("turbine_id") or trend.get("turbine_id") or "01")
    farm = str(deep.get("farm_id") or trend.get("farm_id") or "").title()
    label = f"{farm[:3].upper()}-{tid}" if farm else f"WT-{tid}"

    fp_row: dict[str, Any] = {}
    if not fp.empty and "turbine_id" in fp.columns:
        hit = fp[fp["turbine_id"] == tid]
        if not hit.empty:
            fp_row = hit.iloc[0].to_dict()

    m1 = deep.get("m1_summary", {})
    sm = trend.get("summary", {})
    ym = dir_data.get("yaw_metrics", {})

    perf_status = str(fp_row.get("perf_status", "Healthy / Nominal"))
    net_cf = fp_row.get("net_cf_pct", m1.get("net_cf_pct", np.nan))
    theo_cf = fp_row.get("theoretical_cf_pct", m1.get("theoretical_cf_pct", np.nan))
    cf_gap = float(fp_row.get("cf_gap_pp", (float(net_cf) - float(theo_cf)) if pd.notna(net_cf) and pd.notna(theo_cf) else 0.0))
    clean_dev = fp_row.get("clean_dev_pct", np.nan)
    badges = str(fp_row.get("diagnostic_badges", "Nominal"))
    downtime_mwh = float(fp_row.get("downtime_mwh", 0.0) or 0.0)
    perf_loss_mwh = float(fp_row.get("perf_loss_mwh", 0.0) or 0.0)
    partial_ppr = m1.get("partial_load_ppr_pct", np.nan)
    energy_gap_mwh = float(m1.get("energy_gap_mwh", 0.0) or 0.0)

    # Worst 0.5 m/s bin energy loss
    worst_bin_txt = "No clean bin shortfall"
    bins_adj = deep.get("bins_adj_a", pd.DataFrame())
    if not bins_adj.empty and "lost_mwh_bin" in bins_adj.columns:
        valid_b = bins_adj[bins_adj["ws_bin"].between(3.5, 18.0)]
        if not valid_b.empty:
            w_idx = int(valid_b["lost_mwh_bin"].idxmax())
            w_row = valid_b.loc[w_idx]
            if float(w_row["lost_mwh_bin"]) > 0.1:
                worst_bin_txt = f"Worst bin at {float(w_row['ws_bin']):.1f} m/s ({-float(w_row['lost_mwh_bin']):+.1f} MWh, {float(w_row['delta_pct']):+.1f}%)"

    # M3 Knee Pitch & RPM offset vs peer median (9.5 - 12.5 m/s)
    knee_dpitch = np.nan
    knee_drpm = np.nan
    knee_df = deep.get("knee_df", pd.DataFrame())
    if not knee_df.empty and "ws_bin" in knee_df.columns:
        ksub = knee_df[knee_df["ws_bin"].between(9.5, 12.5)]
        if not ksub.empty:
            if {"pitch_mean_deg", "peer_pitch_deg"}.issubset(ksub.columns) and ksub["pitch_mean_deg"].notna().any() and ksub["peer_pitch_deg"].notna().any():
                knee_dpitch = float((ksub["pitch_mean_deg"] - ksub["peer_pitch_deg"]).median())
            if {"rpm_mean", "peer_rpm"}.issubset(ksub.columns) and ksub["rpm_mean"].notna().any() and ksub["peer_rpm"].notna().any():
                knee_drpm = float((ksub["rpm_mean"] - ksub["peer_rpm"]).median())

    # 3-blade pitch spread
    pm = cms_data.get("pitch_motor_df", pd.DataFrame())
    max_asym = 0.0
    if not pm.empty and {"dab", "dbc", "dca"}.issubset(pm.columns):
        asym_vals = pm[["dab", "dbc", "dca"]].abs().max().dropna()
        if not asym_vals.empty:
            max_asym = float(asym_vals.max())

    # Yaw & Wake
    yaw_bias = float(ym.get("estimated_yaw_bias_deg", 0.0) or 0.0)
    yaw_mwh = float(ym.get("est_recoverable_mwh", 0.0) or 0.0)
    wake_pct = (
        100.0 * float(m1.get("wake_rows", 0)) / max(1.0, float(m1.get("total_rows", 1)))
        if m1
        else 0.0
    )

    # Top Stop Code for this turbine
    top_stop_txt = "No debounced stops recorded in selected window"
    deb_df = ev_data.get("debounced_farm_df", pd.DataFrame())
    if not deb_df.empty and "turbine_id" in deb_df.columns:
        t_stops = deb_df[deb_df["turbine_id"] == tid]
        if not t_stops.empty:
            grp = (
                t_stops.groupby(["root_code", "root_message"], as_index=False)["duration_hours"]
                .agg(total_h="sum", ep_cnt="count")
                .sort_values("total_h", ascending=False)
            )
            r0 = grp.iloc[0]
            top_stop_txt = (
                f"Code {r0['root_code']} ({str(r0['root_message'])[:36]}): "
                f"{float(r0['total_h']):.1f} h across {int(r0['ep_cnt'])} ep"
            )

    # Anemometer MCP median residual
    mcp_res = np.nan
    mcp_df = cms_data.get("mcp_daily_df", pd.DataFrame())
    if not mcp_df.empty and "mcp_residual_ms" in mcp_df.columns and mcp_df["mcp_residual_ms"].notna().any():
        mcp_res = float(mcp_df["mcp_residual_ms"].median())

    alert_days = int(sm.get("alert_days", 0) or 0)
    first_alert = str(sm.get("first_alert_day", "None"))
    cusum_neg = float(sm.get("final_cusum_neg", 0.0) or 0.0)

    # Primary cause & severity classification
    if perf_status == "Suspect Over-Read (>+8%)" or (pd.notna(clean_dev) and float(clean_dev) > 8.0):
        severity = "error"
        primary_cause = "Anemometer Over-Read / NTF Calibration Suspect"
    elif (pd.notna(mcp_res) and abs(mcp_res) >= 0.45) and (pd.notna(clean_dev) and float(clean_dev) <= -1.8):
        severity = "error"
        primary_cause = "Anemometer Bias + Power Curve Deficit"
    elif perf_status == "Underperforming" and (
        (np.isfinite(knee_dpitch) and abs(knee_dpitch) >= 0.35) or max_asym >= 0.30 or "PitchAsym" in badges
    ):
        severity = "error"
        primary_cause = "Aerodynamic / Knee Pitch Offset or Blade Asymmetry"
    elif abs(yaw_bias) >= 4.0 and perf_status != "Healthy / Nominal":
        severity = "warn" if perf_status != "Underperforming" else "error"
        primary_cause = "Static Yaw Misalignment & Cos^2 Partial-Load Loss"
    elif perf_status == "Sub-Optimal / High Downtime" and downtime_mwh > max(40.0, perf_loss_mwh):
        severity = "warn"
        primary_cause = "Availability & Stop Outage Driven Shortfall"
    elif perf_status == "Underperforming":
        severity = "error"
        primary_cause = "Chronic Power-Curve Degradation (M1 CUSUM/EWMA)"
    elif perf_status == "Sub-Optimal / High Downtime" or alert_days >= 10 or abs(yaw_bias) >= 4.0:
        severity = "warn"
        primary_cause = "Mild Efficiency Drift / Localized Operational Loss"
    else:
        severity = "ok"
        primary_cause = "Healthy Nominal Operation"

    dev_str = f"{float(clean_dev):+.2f}%" if pd.notna(clean_dev) else "N/A"
    cf_str = f"{float(net_cf):.1f}%" if pd.notna(net_cf) else "N/A"
    theo_str = f"{float(theo_cf):.1f}%" if pd.notna(theo_cf) else "N/A"
    headline = f"{label} — {perf_status} ({dev_str} Clean Dev | Net CF {cf_str} vs Theo {theo_str})"

    pitch_str = f"{knee_dpitch:+.2f} deg vs peers" if np.isfinite(knee_dpitch) else "Pitch N/A on farm"
    rpm_str = f"{knee_drpm:+.2f} RPM vs peers" if np.isfinite(knee_drpm) else "RPM N/A"
    mcp_str = f"{mcp_res:+.3f} m/s vs unwaked peers" if np.isfinite(mcp_res) else "MCP N/A"
    ppr_str = f"{float(partial_ppr):.2f}%" if pd.notna(partial_ppr) else "N/A"

    findings = [
        f"M1 Energy & CF: Net CF {cf_str} vs Theo {theo_str} (gap {cf_gap:+.2f} pp) | Partial-Load PPR {ppr_str} | Clean Bin Delta {-energy_gap_mwh:+.1f} MWh | {worst_bin_txt}.",
        f"M1 Chronic Onset: Baseline Peer-Adj PPR {sm.get('baseline_center_pct', 'N/A')}% | Alert Days = {alert_days} (First: {first_alert}) | Neg CUSUM = {cusum_neg:.1f} | Left-censored stops = {sm.get('left_censored_stops', 0)}.",
        f"M2/M3 Knee Control & Rotor: Knee (9.5-12.5 m/s) dPitch = {pitch_str} | dRPM = {rpm_str} | Max 3-blade offset = {max_asym:.2f} deg.",
        f"M4 Yaw & Wake: Static Yaw Bias = {yaw_bias:+.2f} deg (est. recoverable {yaw_mwh:.1f} MWh cos^2) | Geometric wake exposure = {wake_pct:.1f}% of intervals.",
        f"Events & Anemometer MCP: {top_stop_txt} | Downtime {downtime_mwh:.1f} MWh | Daily Obs-Peer MCP wind residual = {mcp_str}.",
    ]
    return {
        "turbine_id": tid,
        "label": label,
        "severity": severity,
        "perf_status": perf_status,
        "primary_cause": primary_cause,
        "headline": headline,
        "findings": findings,
        "flags": badges,
    }
