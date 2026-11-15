"""Plotly figure and KPI builders for the 6-tab WINTELLISYT engineering dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wintellisyt.ui.data_loader import SuiteBundle

PALETTE = {
    "kelmarsh": "#1f77b4",
    "penmanshiel": "#d95f02",
    "accent": "#2ca02c",
    "warn": "#e6ab02",
    "danger": "#d62728",
    "muted": "#6c757d",
    "freestream": "#1b9e77",
    "active_wake": "#d95f02",
    "passive_wake_donor_stopped": "#7570b3",
}


def _empty_figure(title: str, note: str = "No data available for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=note,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#6c757d"},
    )
    fig.update_layout(title=title, template="plotly_white", height=360)
    return fig


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        template="plotly_white",
        height=height,
        margin={"l": 55, "r": 25, "t": 55, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "left", "x": 0.0},
    )
    return fig


def _label(df: pd.DataFrame) -> pd.Series:
    if "farm_id" in df.columns and "turbine_id" in df.columns:
        return df["farm_id"].str[:3].str.upper() + "-" + df["turbine_id"].astype(str)
    if "farm" in df.columns and "turbine_id" in df.columns:
        return df["farm"].str[:3].str.upper() + "-" + df["turbine_id"].astype(str)
    return df["turbine_id"].astype(str)


def compute_kpis(bundle: SuiteBundle) -> dict[str, Any]:
    """Compute top-level executive KPIs from the suite tables."""
    layout = bundle.get("layout")
    wf = bundle.get("energy_waterfall")
    d_con = bundle.get("direction_consensus")
    pc_sum = bundle.get("pc_health_summary")
    asym = bundle.get("pitch_asymmetry")
    yaw = bundle.get("yaw_misalignment_summary")
    act = bundle.get("pitch_actuator_health")

    n_turbines = int(len(layout)) if not layout.empty else 0
    farms = sorted(layout["farm"].unique().tolist()) if not layout.empty and "farm" in layout.columns else []
    export_gwh = round(float(wf["energy_export_mwh"].sum() / 1000.0), 2) if not wf.empty else 0.0
    downtime_mwh = round(float(wf["lost_downtime_mwh"].sum()), 1) if not wf.empty else 0.0
    curtail_mwh = round(float(wf["lost_curtailment_total_mwh"].sum()), 1) if not wf.empty else 0.0

    raw_mae = round(float(d_con["raw_peer_mae_deg"].mean()), 2) if not d_con.empty else float("nan")
    cal_mae = round(float(d_con["relative_peer_mae_deg"].mean()), 2) if not d_con.empty else float("nan")
    pc_alerts = (
        int((pc_sum["negative_drift_alert_days"] > 0).sum())
        if not pc_sum.empty and "negative_drift_alert_days" in pc_sum.columns
        else 0
    )
    pitch_flags = int(asym["flag_asymmetry"].sum()) if not asym.empty and "flag_asymmetry" in asym.columns else 0
    yaw_mwh = (
        round(float(yaw["est_recoverable_mwh"].sum()), 1)
        if not yaw.empty and "est_recoverable_mwh" in yaw.columns
        else 0.0
    )
    act_flags = (
        int(act["flag_actuator_anomaly"].sum())
        if not act.empty and "flag_actuator_anomaly" in act.columns
        else 0
    )
    return {
        "farms": ", ".join(f.title() for f in farms) or "N/A",
        "n_turbines": n_turbines,
        "export_gwh": export_gwh,
        "downtime_mwh": downtime_mwh,
        "curtail_mwh": curtail_mwh,
        "raw_dir_mae_deg": raw_mae,
        "cal_dir_mae_deg": cal_mae,
        "pc_drift_baseline_hits": pc_alerts,
        "pitch_asym_flags": pitch_flags,
        "yaw_recoverable_mwh": yaw_mwh,
        "actuator_flags": act_flags,
    }


def _fig_farm_energy_balance(bundle: SuiteBundle) -> go.Figure:
    wf = bundle.get("energy_waterfall")
    if wf.empty:
        return _empty_figure("Tab 1: Farm IEC 61400-26 Energy Decomposition (2017)")
    grp = wf.groupby("farm_id")[
        ["energy_export_mwh", "lost_downtime_mwh", "lost_curtailment_total_mwh", "lost_performance_mwh"]
    ].sum()
    fig = go.Figure()
    for col, label, color in (
        ("energy_export_mwh", "Exported Generation (MWh)", "#1b9e77"),
        ("lost_downtime_mwh", "Downtime Loss (MWh)", "#d62728"),
        ("lost_curtailment_total_mwh", "Curtailment Loss (MWh)", "#e6ab02"),
        ("lost_performance_mwh", "Performance Gap (MWh)", "#7570b3"),
    ):
        fig.add_bar(name=label, x=[f.title() for f in grp.index], y=grp[col].round(1), marker_color=color)
    fig.update_layout(barmode="stack", yaxis_title="Energy (MWh)")
    return _base_layout(fig, "IEC 61400-26 Farm-Level Energy Balance (2017)", height=400)


def _fig_diagnostic_risk_matrix(bundle: SuiteBundle) -> go.Figure:
    d_con = bundle.get("direction_consensus")
    yaw = bundle.get("yaw_misalignment_summary")
    asym = bundle.get("pitch_asymmetry")
    cms = bundle.get("drivetrain_cms_summary")
    pc = bundle.get("pc_health_summary")
    if d_con.empty:
        return _empty_figure("Turbine Diagnostic Risk Summary")

    base = d_con[["farm_id", "turbine_id", "total_offset_deg"]].copy()
    base["abs_dir_offset_deg"] = base["total_offset_deg"].abs()
    if not yaw.empty:
        base = base.merge(
            yaw[["farm_id", "turbine_id", "estimated_static_yaw_bias_deg"]].assign(
                abs_yaw_bias_deg=lambda d: d["estimated_static_yaw_bias_deg"].abs()
            ),
            on=["farm_id", "turbine_id"],
            how="left",
        )
    if not asym.empty:
        base = base.merge(
            asym[["farm_id", "turbine_id", "max_pair_offset_deg"]],
            on=["farm_id", "turbine_id"],
            how="left",
        )
    if not cms.empty:
        base = base.merge(
            cms[["farm_id", "turbine_id", "max_thermal_excess_c"]],
            on=["farm_id", "turbine_id"],
            how="left",
        )
    if not pc.empty:
        fleet_pc = (
            pc[pc["baseline"] == "fleet_consensus"][["farm_id", "turbine_id", "final_cusum_neg"]]
            if "fleet_consensus" in set(pc["baseline"])
            else pc.groupby(["farm_id", "turbine_id"])["final_cusum_neg"].max().reset_index()
        )
        base = base.merge(fleet_pc, on=["farm_id", "turbine_id"], how="left")

    base["label"] = _label(base)
    cols = [c for c in ("abs_dir_offset_deg", "abs_yaw_bias_deg", "max_pair_offset_deg", "max_thermal_excess_c", "final_cusum_neg") if c in base.columns]
    if not cols:
        return _empty_figure("Turbine Diagnostic Risk Summary")

    z = base[cols].to_numpy(dtype=float)
    # Normalize columns 0-1 purely for color comparability while keeping raw values in hovertext
    col_max = np.nanmax(np.where(np.isfinite(z), z, 0.0), axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    z_norm = z / col_max
    text = [[f"{val:.2f}" if np.isfinite(val) else "N/A" for val in row] for row in z]
    col_labels = {
        "abs_dir_offset_deg": "|Dir Offset| (deg)",
        "abs_yaw_bias_deg": "|Yaw Bias| (deg)",
        "max_pair_offset_deg": "Pitch Asym (deg)",
        "max_thermal_excess_c": "Thermal Excess (C)",
        "final_cusum_neg": "Neg CUSUM (Fleet)",
    }
    fig = go.Figure(
        go.Heatmap(
            z=z_norm.T,
            x=base["label"].tolist(),
            y=[col_labels[c] for c in cols],
            text=np.array(text).T.tolist(),
            texttemplate="%{text}",
            colorscale="YlOrRd",
            showscale=False,
        )
    )
    return _base_layout(fig, "Fleet Diagnostic Risk Matrix (Raw Physical Units Annotated)", height=360)


def _fig_farm_layout(bundle: SuiteBundle) -> go.Figure:
    layout = bundle.get("layout")
    d_con = bundle.get("direction_consensus")
    if layout.empty:
        return _empty_figure("Wind Farm Spatial Layout & Calibrated North Offsets")
    merged = layout.copy()
    if not d_con.empty:
        merged = merged.merge(
            d_con[["farm_id", "turbine_id", "total_offset_deg", "relative_peer_mae_deg"]],
            left_on=["farm", "turbine_id"],
            right_on=["farm_id", "turbine_id"],
            how="left",
        )
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Kelmarsh (6x Senvion MM92)", "Penmanshiel (14x Senvion MM82)"))
    for col_idx, farm in enumerate(("kelmarsh", "penmanshiel"), start=1):
        sub = merged[merged["farm"] == farm]
        if sub.empty:
            continue
        hover = [
            f"WT-{r.turbine_id}<br>Elev: {r.elevation_m:.1f} m | Hub: {r.hub_height_m:.1f} m"
            + (f"<br>Total Offset: {r.total_offset_deg:.2f} deg" if pd.notna(getattr(r, "total_offset_deg", np.nan)) else "")
            for r in sub.itertuples(index=False)
        ]
        fig.add_scatter(
            x=sub["longitude"],
            y=sub["latitude"],
            mode="markers+text",
            text=sub["turbine_id"],
            textposition="top center",
            marker={"size": 12, "color": PALETTE.get(farm, "#1f77b4")},
            hovertext=hover,
            hoverinfo="text",
            name=farm.title(),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Longitude (deg)", row=1, col=col_idx)
        fig.update_yaxes(title_text="Latitude (deg)", row=1, col=col_idx)
    return _base_layout(fig, "Turbine Coordinates & True-North Offset Annotations", height=420)


def _fig_direction_mae(bundle: SuiteBundle) -> go.Figure:
    d_con = bundle.get("direction_consensus")
    if d_con.empty:
        return _empty_figure("NREL Circular Consensus Direction Calibration")
    labels = _label(d_con)
    fig = go.Figure()
    fig.add_bar(name="Raw Peer MAE (deg)", x=labels, y=d_con["raw_peer_mae_deg"], marker_color="#a6cee3")
    fig.add_bar(name="NREL Calibrated Peer MAE (deg)", x=labels, y=d_con["relative_peer_mae_deg"], marker_color="#1f78b4")
    fig.update_layout(barmode="group", yaxis_title="Test MAE vs Peer Consensus (deg)")
    return _base_layout(fig, "NREL 2-Stage Consensus Calibration: Raw vs Calibrated Peer Direction MAE (Nov-Dec 2017)", height=400)


def _fig_wind_rose(bundle: SuiteBundle) -> go.Figure:
    rose = bundle.get("wind_rose")
    if rose.empty:
        return _empty_figure("Calibrated Wind Rose")
    sub = rose[rose["basis"] == "calibrated"].copy()
    if sub.empty:
        sub = rose.copy()
    agg = sub.groupby(["farm_id", "direction_deg", "speed_band"])["n"].sum().reset_index()
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=("Kelmarsh Calibrated Wind Rose", "Penmanshiel Calibrated Wind Rose"),
    )
    band_colors = {
        "[0.0, 5.0)": "#74add1",
        "[5.0, 10.0)": "#abd9e9",
        "[10.0, 15.0)": "#fdae61",
        "[15.0, inf)": "#d73027",
    }
    for col_idx, farm in enumerate(("kelmarsh", "penmanshiel"), start=1):
        f_sub = agg[agg["farm_id"] == farm]
        total = max(1.0, float(f_sub["n"].sum()))
        for sb in ("[0.0, 5.0)", "[5.0, 10.0)", "[10.0, 15.0)", "[15.0, inf)"):
            b_sub = f_sub[f_sub["speed_band"] == sb].sort_values("direction_deg")
            if b_sub.empty:
                continue
            fig.add_trace(
                go.Barpolar(
                    r=(100.0 * b_sub["n"] / total).round(2),
                    theta=b_sub["direction_deg"],
                    name=f"{sb} m/s",
                    marker_color=band_colors.get(sb, "#999999"),
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(barmode="stack")
    return _base_layout(fig, "True-North Calibrated Wind Rose by Speed Band (% Frequency)", height=450)


def _fig_wake_regimes(bundle: SuiteBundle) -> go.Figure:
    w_reg = bundle.get("wake_regime_effects")
    if w_reg.empty:
        return _empty_figure("Active-Donor vs Stopped-Donor Wake Stratification")
    agg = (
        w_reg.groupby(["farm_id", "regime", "exposure_class"])[["ti_median", "ppr_peer_mean", "ws_suppression_median"]]
        .mean()
        .reset_index()
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Turbulence Intensity (TI) by Regime", "Peer Power Performance Ratio (PPR) by Regime"),
    )
    x_labels = agg["farm_id"].str[:3].str.upper() + " | " + agg["regime"].str.replace("_", " ")
    agg["x_lbl"] = x_labels
    for cls_name, color in (
        ("freestream", PALETTE["freestream"]),
        ("passive_wake_donor_stopped", PALETTE["passive_wake_donor_stopped"]),
        ("active_wake", PALETTE["active_wake"]),
    ):
        c_sub = agg[agg["exposure_class"] == cls_name]
        if c_sub.empty:
            continue
        fig.add_bar(name=cls_name, x=c_sub["x_lbl"], y=c_sub["ti_median"].round(4), marker_color=color, row=1, col=1)
        fig.add_bar(
            name=cls_name,
            x=c_sub["x_lbl"],
            y=c_sub["ppr_peer_mean"].round(4),
            marker_color=color,
            showlegend=False,
            row=1,
            col=2,
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Median TI", row=1, col=1)
    fig.update_yaxes(title_text="Mean Peer PPR (1.0 = healthy)", row=1, col=2)
    return _base_layout(fig, "Wake Stratification: Active Operating Donor vs Stopped-Donor Placebo vs Freestream", height=450)


def _fig_pc_cusum(bundle: SuiteBundle) -> go.Figure:
    pc_sum = bundle.get("pc_health_summary")
    if pc_sum.empty:
        return _empty_figure("Event-Screened Power Curve Health (CUSUM/EWMA)")
    pc = pc_sum.copy()
    pc["label"] = _label(pc)
    fig = go.Figure()
    for base_name, color in (
        ("oem_contractual", "#1f77b4"),
        ("fleet_consensus", "#2ca02c"),
        ("self_historical", "#d95f02"),
    ):
        sub = pc[pc["baseline"] == base_name]
        if sub.empty:
            continue
        fig.add_bar(
            name=f"{base_name} (Neg Drift Alert Days)",
            x=sub["label"],
            y=sub["negative_drift_alert_days"],
            marker_color=color,
        )
    fig.update_layout(barmode="group", yaxis_title="Alert Days (Nov-Dec 2017)")
    return _base_layout(fig, "Event-Screened PC Health: Negative Drift Alert Days Across 3 Baselines", height=410)


def _fig_pitch_diagnostics(bundle: SuiteBundle) -> go.Figure:
    asym = bundle.get("pitch_asymmetry")
    rec = bundle.get("pitch_ppr_recovery")
    if asym.empty and rec.empty:
        return _empty_figure("Blade Pitch Asymmetry & Pre/Post Maintenance Recovery")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Inter-Blade Pitch Offset (Partial Load 5-11 m/s)", "7-Day Pre/Post Maintenance PPR Recovery"),
    )
    if not asym.empty:
        lbl = _label(asym)
        for col, name, color in (
            ("median_a_minus_b_deg", "Blade A-B", "#1f77b4"),
            ("median_b_minus_c_deg", "Blade B-C", "#d95f02"),
            ("median_c_minus_a_deg", "Blade C-A", "#2ca02c"),
        ):
            fig.add_bar(name=name, x=lbl, y=asym[col], marker_color=color, row=1, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_yaxes(title_text="Median Offset (deg)", row=1, col=1)
    if not rec.empty:
        clean_rec = rec[rec["outcome"] != "insufficient_or_confounded"].copy()
        if clean_rec.empty:
            clean_rec = rec.copy()
        clean_rec["label"] = _label(clean_rec)
        for out_name, color in (
            ("improved_ppr", "#2ca02c"),
            ("unchanged_ppr", "#6c757d"),
            ("degraded_ppr", "#d62728"),
            ("insufficient_or_confounded", "#e6ab02"),
        ):
            sub = clean_rec[clean_rec["outcome"] == out_name]
            if sub.empty:
                continue
            fig.add_scatter(
                x=sub["duration_hours"],
                y=sub["delta_ppr"],
                mode="markers",
                name=f"Outcome: {out_name}",
                text=sub["label"] + " | " + sub["message"].astype(str),
                marker={"size": 10, "color": color},
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="Outage Duration (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Post - Pre Delta PPR", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base_layout(fig, "Pitch Asymmetry Audit & Post-Maintenance Recovery Diagnostics", height=430)


def _fig_peer_reconstruction(bundle: SuiteBundle) -> go.Figure:
    gaps = bundle.get("peer_gap_benchmark")
    drift = bundle.get("peer_drift_summary")
    if gaps.empty and drift.empty:
        return _empty_figure("Unwaked-Donor MCP Peer Reconstruction & Bias Detection")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Masked Gap MAE vs Persistence (by Horizon)", "Target Bias Detection & +2 m/s Synthetic Injection"),
    )
    if not gaps.empty:
        no_drop = gaps[~gaps["donor_dropout"]].groupby("gap_rows")[["peer_mae_ms", "persistence_mae_ms"]].mean().reset_index()
        drop = gaps[gaps["donor_dropout"]].groupby("gap_rows")["peer_mae_ms"].mean().reset_index()
        horizons = {1: "10 min", 6: "1 hr", 36: "6 hr", 144: "24 hr"}
        fig.add_scatter(
            x=[horizons.get(h, str(h)) for h in no_drop["gap_rows"]],
            y=no_drop["peer_mae_ms"].round(3),
            mode="lines+markers",
            name="Unwaked Peer MCP MAE (Full Donors)",
            line={"color": "#1f77b4", "width": 3},
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=[horizons.get(h, str(h)) for h in drop["gap_rows"]],
            y=drop["peer_mae_ms"].round(3),
            mode="lines+markers",
            name="Unwaked Peer MCP MAE (-1 Donor)",
            line={"color": "#2ca02c", "dash": "dash"},
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=[horizons.get(h, str(h)) for h in no_drop["gap_rows"]],
            y=no_drop["persistence_mae_ms"].round(3),
            mode="lines+markers",
            name="Persistence Baseline MAE",
            line={"color": "#d62728", "dash": "dot"},
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="MAE (m/s)", row=1, col=1)
    if not drift.empty:
        lbl = _label(drift)
        fig.add_bar(
            name="Control q99 Exceedance (%)",
            x=lbl,
            y=drift["control_exceedance_pct"],
            marker_color="#6c757d",
            row=1,
            col=2,
        )
        fig.add_bar(
            name="+2 m/s Injection Exceedance (%)",
            x=lbl,
            y=drift["plus_2ms_injection_exceedance_pct"],
            marker_color="#d95f02",
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Alarm Rate (%)", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base_layout(fig, "All-Turbine Peer MCP Imputation & Anemometer Drift Detection", height=430)


def _fig_density_benchmark(bundle: SuiteBundle) -> go.Figure:
    dens = bundle.get("density_benchmark")
    if dens.empty:
        return _empty_figure("ERA5 Air Density Correction Benchmark (2016 Train -> 2017 Test)")
    agg = dens.groupby(["farm_id", "stratum", "method"])[["mae_kw", "signed_bias_kw"]].mean().reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Kelmarsh Mean MAE (kW)", "Penmanshiel Mean MAE (kW)"))
    method_colors = {
        "ws_raw": "#6c757d",
        "ws_pub_adj": "#e6ab02",
        "ws_iec": "#1f77b4",
        "ws_svenningsen": "#2ca02c",
    }
    for col_idx, farm in enumerate(("kelmarsh", "penmanshiel"), start=1):
        sub = agg[agg["farm_id"] == farm]
        for meth, color in method_colors.items():
            m_sub = sub[sub["method"] == meth]
            if m_sub.empty:
                continue
            fig.add_bar(
                name=meth,
                x=m_sub["stratum"],
                y=m_sub["mae_kw"].round(2),
                marker_color=color,
                showlegend=(col_idx == 1),
                row=1,
                col=col_idx,
            )
        fig.update_yaxes(title_text="Out-of-Sample Power MAE (kW)", row=1, col=col_idx)
    fig.update_layout(barmode="group")
    return _base_layout(fig, "2016->2017 Air Density Normalization: Raw vs Publisher vs IEC vs WindPRO (Svenningsen)", height=430)


def _fig_asset_yaw_and_mpc(bundle: SuiteBundle) -> go.Figure:
    yaw = bundle.get("yaw_misalignment_summary")
    gap = bundle.get("mpc_vs_learned_gap")
    if yaw.empty and gap.empty:
        return _empty_figure("Phase 2.5: Static Yaw Loss & Contractual MPC vs Learned PC")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Static Yaw Misalignment & Recoverable MWh", "Contractual MPC vs Learned PC vs Actual Power (Mean Across Fleet)"),
    )
    if not yaw.empty:
        lbl = _label(yaw)
        fig.add_bar(
            name="Est. Recoverable MWh (cos^2)",
            x=lbl,
            y=yaw["est_recoverable_mwh"],
            text=yaw["estimated_static_yaw_bias_deg"].astype(str) + " deg",
            textposition="auto",
            marker_color="#d95f02",
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Recoverable Energy (MWh)", row=1, col=1)
    if not gap.empty:
        agg = gap.groupby("speed_bin_ms")[["mpc_kw", "learned_pc_kw", "actual_median_kw"]].mean().reset_index()
        for col, name, color, dash in (
            ("mpc_kw", "Contractual MPC (kW)", "#1f77b4", "solid"),
            ("learned_pc_kw", "Publisher Learned PC (kW)", "#2ca02c", "dash"),
            ("actual_median_kw", "Observed Median Power (kW)", "#111111", "dot"),
        ):
            fig.add_scatter(
                x=agg["speed_bin_ms"],
                y=agg[col],
                mode="lines+markers",
                name=name,
                line={"color": color, "dash": dash},
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    return _base_layout(fig, "Actionable Insights: Yaw Misalignment Loss & MPC vs Learned Power Curve Gap", height=430)


def _fig_asset_actuators_and_cms(bundle: SuiteBundle) -> go.Figure:
    act = bundle.get("pitch_actuator_health")
    cms = bundle.get("drivetrain_cms_summary")
    if act.empty and cms.empty:
        return _empty_figure("Phase 2.5: Pitch Actuator Imbalance & Drivetrain/Thermal CMS")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("3-Axis Pitch Motor Current & Temp Spread (p95)", "Drivetrain Thermal Excess (deg C) & Vibration (mm/ss)"),
    )
    if not act.empty:
        lbl = _label(act)
        fig.add_bar(
            name="Motor Current Spread p95 (A)",
            x=lbl,
            y=act["p95_current_imbalance_a"],
            marker_color="#7570b3",
            row=1,
            col=1,
        )
        fig.add_bar(
            name="Motor Temp Spread p95 (C)",
            x=lbl,
            y=act["p95_motor_temp_spread_c"],
            marker_color="#e7298a",
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Axis 1-3 Max-Min Spread (p95)", row=1, col=1)
    if not cms.empty:
        lbl_c = _label(cms)
        if "max_thermal_excess_c" in cms.columns:
            fig.add_bar(
                name="Max Component Thermal Residual (C)",
                x=lbl_c,
                y=cms["max_thermal_excess_c"],
                text=cms["hottest_component"].fillna(""),
                marker_color="#d62728",
                row=1,
                col=2,
            )
        if "drivetrain_accel_p95_mmss" in cms.columns:
            fig.add_scatter(
                name="Drivetrain Accel p95 (mm/ss)",
                x=lbl_c,
                y=cms["drivetrain_accel_p95_mmss"],
                mode="markers+lines",
                marker={"color": "#111111", "size": 8},
                row=1,
                col=2,
            )
        fig.update_yaxes(title_text="deg C / mm/ss", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base_layout(fig, "Predictive Maintenance: Pitch Actuator Stress & Drivetrain/Bearing Thermal Anomalies", height=430)


def _fig_wake_fatigue(bundle: SuiteBundle) -> go.Figure:
    vib = bundle.get("wake_vibration_fatigue")
    if vib.empty:
        return _empty_figure("Wake-Induced Structural & Drivetrain Vibration Uplift (Active Wake vs Freestream)")
    vib = vib.copy()
    vib["label"] = _label(vib)
    fig = go.Figure()
    for ch, color in (
        ("drivetrain_accel_mmss", "#d95f02"),
        ("tower_accel_x_mmss", "#1f77b4"),
        ("tower_accel_y_mmss", "#2ca02c"),
    ):
        sub = vib[vib["vibration_channel"] == ch]
        if sub.empty:
            continue
        fig.add_bar(name=ch, x=sub["label"], y=sub["wake_vibration_uplift_pct"], marker_color=color)
    fig.update_layout(barmode="group", yaxis_title="Active Wake Uplift vs Freestream (%)")
    return _base_layout(fig, "Wake-Induced Mechanical & Structural Vibration Uplift (5-12 m/s Active Wake vs Freestream)", height=390)


def _fig_signal_provenance(bundle: SuiteBundle) -> go.Figure:
    inv = bundle.signal_inventory_summary
    ntf = bundle.get("anemometer_ntf_audit")
    if inv.empty and ntf.empty:
        return _empty_figure("Signal Coverage & Anemometer/NTF Provenance")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Canonical Channel Coverage (%)", "Reported Wind Speed MAE vs Sensor 1/2 (NTF Check)"),
    )
    if not inv.empty:
        canon = inv[inv["canonical_name"].notna()].copy()
        if not canon.empty:
            top_c = canon.groupby(["farm", "canonical_name"])["availability_pct"].mean().reset_index()
            for farm, color in (("kelmarsh", PALETTE["kelmarsh"]), ("penmanshiel", PALETTE["penmanshiel"])):
                sub = top_c[top_c["farm"] == farm].head(18)
                fig.add_bar(
                    name=farm.title(),
                    x=sub["canonical_name"],
                    y=sub["availability_pct"],
                    marker_color=color,
                    row=1,
                    col=1,
                )
        fig.update_yaxes(title_text="Finite Availability (%)", row=1, col=1)
    if not ntf.empty:
        lbl = _label(ntf)
        for col, name, color in (
            ("mae_ws_vs_s1_ms", "MAE vs Sensor 1 (m/s)", "#1f77b4"),
            ("mae_ws_vs_s2_ms", "MAE vs Sensor 2 (m/s)", "#2ca02c"),
            ("mae_ws_vs_mean_s12_ms", "MAE vs Mean(S1,S2) (m/s)", "#d95f02"),
        ):
            fig.add_bar(name=name, x=lbl, y=ntf[col], marker_color=color, row=1, col=2)
        fig.update_yaxes(title_text="Mean Abs Diff (m/s)", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base_layout(fig, "Channel Provenance, Coverage & Anemometer S1/S2 Audit", height=440)


def build_dashboard_figures(bundle: SuiteBundle) -> dict[str, Any]:
    """Build all KPI summaries and tab-grouped Plotly figures for the UI."""
    return {
        "kpis": compute_kpis(bundle),
        "tabs": {
            "overview": {
                "title": "1. Executive Overview",
                "figures": [
                    _fig_farm_energy_balance(bundle),
                    _fig_diagnostic_risk_matrix(bundle),
                ],
            },
            "direction_wake": {
                "title": "2. True North & Wake Physics",
                "figures": [
                    _fig_farm_layout(bundle),
                    _fig_direction_mae(bundle),
                    _fig_wind_rose(bundle),
                    _fig_wake_regimes(bundle),
                ],
            },
            "pc_pitch": {
                "title": "3. PC Health & Pitch",
                "figures": [
                    _fig_pc_cusum(bundle),
                    _fig_pitch_diagnostics(bundle),
                ],
            },
            "peer_density": {
                "title": "4. Peer MCP & Air Density",
                "figures": [
                    _fig_peer_reconstruction(bundle),
                    _fig_density_benchmark(bundle),
                ],
            },
            "asset_intelligence": {
                "title": "5. Actionable Insights (Phase 2.5)",
                "figures": [
                    _fig_asset_yaw_and_mpc(bundle),
                    _fig_asset_actuators_and_cms(bundle),
                    _fig_wake_fatigue(bundle),
                ],
            },
            "provenance": {
                "title": "6. Signal Inventory & Audit",
                "figures": [
                    _fig_signal_provenance(bundle),
                ],
            },
        },
    }
