"""Commercial-grade Plotly WebGL figure factory for the WINTELLISYT v2 6-tab workbench."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SUBSYSTEM_COLORS: dict[str, str] = {
    "Pitch System": "#e67700",
    "Converter & Electrical": "#d62728",
    "Yaw & Cable Twist": "#1f77b4",
    "Drivetrain & Gearbox": "#9467bd",
    "Tower & Vibration": "#8c564b",
    "Grid & Safety Chain": "#e377c2",
    "Manual & Scheduled Maintenance": "#2ca02c",
    "Controller & Comms": "#17becf",
    "Environmental & Operational": "#bcbd22",
    "Other": "#7f7f7f",
}

STATE_STYLE: dict[str, tuple[str, float, int]] = {
    "Clean Normal": ("#1b9e77", 0.42, 4),
    "Event Outage (Stop)": ("#d62728", 0.72, 6),
    "Curtailment / Derate": ("#7570b3", 0.78, 6),
    "Under-Cut-In / Calm": ("#9aa5b1", 0.35, 3),
}


def empty_figure(title: str, note: str = "No telemetry available for the active filter selection.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=note,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#627d98"},
    )
    fig.update_layout(title=title, template="plotly_white", height=360)
    return fig


def _polish(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": "#102a43"}},
        template="plotly_white",
        height=height,
        margin={"l": 58, "r": 28, "t": 58, "b": 62},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0},
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# TAB 1: FLEET & PLANT EXECUTIVE COCKPIT
# ---------------------------------------------------------------------------


_PERF_STATUS_COLORS: dict[str, str] = {
    "Underperforming": "#d62728",
    "Suspect Over-Read (>+8%)": "#e67700",
    "Sub-Optimal / High Downtime": "#d97706",
    "Healthy / Nominal": "#1b9e77",
}


def fig_fleet_spatial_map(cockpit: dict[str, Any]) -> go.Figure:
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty or "east_m" not in fp.columns or "north_m" not in fp.columns:
        return empty_figure("Wind Farm Coordinate Map & Capacity Factor Breakdown")

    kpis = cockpit.get("kpis", {})
    lat_valid = fp["latitude"].dropna() if "latitude" in fp.columns else pd.Series(dtype=float)
    lon_valid = fp["longitude"].dropna() if "longitude" in fp.columns else pd.Series(dtype=float)
    centroid_lbl = (
        f"Centroid ({float(lat_valid.mean()):.4f}, {float(lon_valid.mean()):.4f})"
        if not lat_valid.empty and not lon_valid.empty
        else "Local Layout"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.09,
        specs=[[{"type": "xy"}, {"secondary_y": True}]],
        subplot_titles=(
            f"Turbine Spatial Coordinates & Underperformance Map ({centroid_lbl})",
            "Per-Turbine Net CF (%) vs Theoretical CF (%) & Clean PPR (%)",
        ),
    )

    east = fp["east_m"].to_numpy(dtype=float)
    north = fp["north_m"].to_numpy(dtype=float)

    # Outer rotor-footprint ring overlay for visual spacing scale
    ring_colors = [_PERF_STATUS_COLORS.get(str(s), "#1b9e77") for s in fp["perf_status"]]
    fig.add_scatter(
        x=east,
        y=north,
        mode="markers",
        marker={
            "size": 26,
            "color": "rgba(255, 255, 255, 0.0)",
            "line": {"width": 1.6, "color": ring_colors},
        },
        showlegend=False,
        hoverinfo="skip",
        row=1,
        col=1,
    )

    for status_name in (
        "Underperforming",
        "Suspect Over-Read (>+8%)",
        "Sub-Optimal / High Downtime",
        "Healthy / Nominal",
    ):
        sub = fp[fp["perf_status"] == status_name]
        if sub.empty:
            continue
        clr = _PERF_STATUS_COLORS[status_name]
        labels = [
            f"<b>WT-{r.turbine_id}</b><br>CF {r.net_cf_pct:.1f}% ({r.clean_dev_pct:+.1f}%)"
            if pd.notna(r.clean_dev_pct)
            else f"<b>WT-{r.turbine_id}</b><br>CF {r.net_cf_pct:.1f}%"
            for r in sub.itertuples(index=False)
        ]
        custom = [
            [
                r.label,
                f"{r.latitude:.5f}" if pd.notna(r.latitude) else "N/A",
                f"{r.longitude:.5f}" if pd.notna(r.longitude) else "N/A",
                f"{r.elevation_m:.1f} m" if pd.notna(r.elevation_m) else "N/A",
                f"{r.net_cf_pct:.2f}%",
                f"{r.theoretical_cf_pct:.2f}%",
                f"{r.cf_gap_pp:+.2f} pp",
                f"{r.clean_ppr_pct:.2f}%" if pd.notna(r.clean_ppr_pct) else "N/A",
                f"{r.clean_dev_pct:+.2f}%" if pd.notna(r.clean_dev_pct) else "N/A",
                f"{r.avail_time_iec_b22_pct:.2f}%" if pd.notna(r.avail_time_iec_b22_pct) else "N/A",
                r.diagnostic_badges,
            ]
            for r in sub.itertuples(index=False)
        ]
        fig.add_scatter(
            name=status_name,
            x=sub["east_m"],
            y=sub["north_m"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont={"size": 10, "color": "#102a43"},
            marker={
                "size": np.clip(sub["net_cf_pct"].fillna(20.0).to_numpy(dtype=float) * 0.45 + 6.0, 12.0, 22.0),
                "color": clr,
                "line": {"width": 1.4, "color": "#ffffff"},
            },
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (" + status_name + ")<br>"
                "WGS84 Lat/Lon: (%{customdata[1]}, %{customdata[2]}) | Elev: %{customdata[3]}<br>"
                "Local East / North: (%{x:+.0f} m, %{y:+.0f} m)<br>"
                "Net CF: %{customdata[4]} | Theo CF: %{customdata[5]} | Gap: %{customdata[6]}<br>"
                "Clean PPR: %{customdata[7]} | Clean Dev: %{customdata[8]} | IEC B.2.2: %{customdata[9]}<br>"
                "Badges: %{customdata[10]}<extra></extra>"
            ),
            row=1,
            col=1,
        )

    span_e = float(np.nanmax(east) - np.nanmin(east)) if len(east) > 1 else 600.0
    span_n = float(np.nanmax(north) - np.nanmin(north)) if len(north) > 1 else 600.0
    pad_e = max(180.0, 0.20 * span_e)
    pad_n = max(220.0, 0.25 * span_n)
    fig.update_xaxes(title_text="East Offset from Centroid (m)", range=[float(np.nanmin(east)) - pad_e, float(np.nanmax(east)) + pad_e], row=1, col=1)
    fig.update_yaxes(
        title_text="North Offset from Centroid (m)",
        range=[float(np.nanmin(north)) - pad_n, float(np.nanmax(north)) + pad_n * 1.15],
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )

    # Right panel: Theoretical CF vs Net CF (colored by underperformance status) + Clean PPR overlay
    cf_bar_colors = [_PERF_STATUS_COLORS.get(str(s), "#1b9e77") for s in fp["perf_status"]]
    fig.add_bar(
        name="Theoretical CF (%)",
        x=fp["label"],
        y=fp["theoretical_cf_pct"],
        marker={"color": "rgba(148, 163, 184, 0.45)", "line": {"color": "#64748b", "width": 1}},
        hovertemplate="<b>%{x}</b><br>Theoretical CF: %{y:.2f}%<extra></extra>",
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_bar(
        name="Net CF (%)",
        x=fp["label"],
        y=fp["net_cf_pct"],
        marker={"color": cf_bar_colors, "line": {"color": "#102a43", "width": 0.6}},
        text=fp["net_cf_pct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "N/A"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Net CF: %{y:.2f}%<extra></extra>",
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Clean PPR (%)",
            x=fp["label"],
            y=fp["clean_ppr_pct"],
            mode="lines+markers",
            line={"color": "#102a43", "width": 2.2},
            marker={"size": 6, "color": "#102a43"},
            hovertemplate="<b>%{x}</b><br>Clean PPR: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    cf_max = float(fp[["net_cf_pct", "theoretical_cf_pct"]].max().max()) if not fp.empty else 40.0
    ppr_clean = fp["clean_ppr_pct"].dropna()
    ppr_min = float(ppr_clean.min()) if not ppr_clean.empty else 90.0
    ppr_max = float(ppr_clean.max()) if not ppr_clean.empty else 105.0

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Capacity Factor (%)", range=[0.0, max(50.0, cf_max * 1.20)], row=1, col=2, secondary_y=False)
    fig.update_yaxes(
        title_text="Clean PPR (%)",
        range=[min(88.0, ppr_min - 3.0), max(106.0, ppr_max + 3.0)],
        row=1,
        col=2,
        secondary_y=True,
    )
    return _polish(
        fig,
        f"{kpis.get('farm', '')} — Spatial Performance Map & Capacity Factor Audit "
        f"(Net CF {kpis.get('fleet_net_cf_pct', 0.0)}% vs Theo CF {kpis.get('fleet_theoretical_cf_pct', 0.0)}%)",
        height=470,
    )


def fig_fleet_deviation_fingerprint(cockpit: dict[str, Any]) -> go.Figure:
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty:
        return empty_figure("Fleet Power-Curve Deviation Fingerprint (%)")

    fig = go.Figure()
    fig.add_bar(
        name="Event-Screened Clean Dev (%)",
        x=fp["label"],
        y=fp["clean_dev_pct"],
        marker_color=np.where(fp["clean_dev_pct"] > 8.0, "#e67700", np.where(fp["clean_dev_pct"] < -3.0, "#d62728", "#1f77b4")),
        text=fp["clean_dev_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Clean Dev: %{y:+.2f}%<extra></extra>",
    )
    fig.add_bar(
        name="Unmasked Raw Dev (%)",
        x=fp["label"],
        y=fp["raw_dev_pct"],
        marker_color="rgba(130, 154, 177, 0.55)",
        hovertemplate="<b>%{x}</b><br>Unmasked Raw Dev: %{y:+.2f}%<extra></extra>",
    )
    fig.add_hline(y=0.0, line_color="#334e68", line_width=1.2)
    fig.add_hline(
        y=8.0,
        line_dash="dash",
        line_color="#e67700",
        annotation_text="+8% Suspect Anemometer Over-Read Guard",
        annotation_position="top left",
    )
    fig.update_layout(barmode="group", yaxis_title="Power Deviation vs Reference (%)")
    return _polish(fig, f"{cockpit['kpis'].get('farm', '')} — Fleet Power-Curve Deviation Fingerprint (4-15 m/s)", height=410)


def fig_fleet_energy_and_avail_waterfall(cockpit: dict[str, Any]) -> go.Figure:
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty:
        return empty_figure("IEC 61400-26 Energy Decomposition & Availability Benchmark")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Per-Turbine Energy Balance (Export + Downtime + Perf Loss)",
            "IEC 61400-26 & Event-Mask Availability Comparison (%)",
        ),
    )
    for col, lbl, clr in (
        ("export_mwh", "Exported Generation (MWh)", "#1b9e77"),
        ("downtime_mwh", "Downtime Loss (MWh)", "#d62728"),
        ("perf_loss_mwh", "Performance Gap (MWh)", "#7570b3"),
    ):
        fig.add_bar(name=lbl, x=fp["label"], y=fp[col], marker_color=clr, row=1, col=1)

    for col, lbl, clr in (
        ("avail_time_iec_b22_pct", "Time-Based IEC B.2.2 (%)", "#1f77b4"),
        ("avail_prod_iec_b22_pct", "Prod-Based IEC B.2.2 (%)", "#0ca678"),
        ("avail_system_pct", "System Availability (%)", "#e6ab02"),
        ("event_mask_avail_pct", "Status-Log Mask Avail (%)", "#102a43"),
    ):
        fig.add_scatter(
            name=lbl,
            x=fp["label"],
            y=fp[col],
            mode="lines+markers",
            line={"color": clr, "width": 2},
            row=1,
            col=2,
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Energy (MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Availability (%)", range=[max(0.0, float(fp["event_mask_avail_pct"].min()) - 5.0), 101.0], row=1, col=2)
    return _polish(fig, "IEC 61400-26 Energy Waterfall & Multi-Standard Availability Alignment", height=430)


def fig_fleet_completeness_heatmap(cockpit: dict[str, Any]) -> go.Figure:
    comp = cockpit.get("completeness_matrix", pd.DataFrame())
    if comp.empty:
        return empty_figure("Turbine x Month Telemetry Completeness (%)")

    piv_v = comp.pivot(index="turbine_id", columns="ym", values="valid_pct").sort_index()
    piv_av = comp.pivot(index="turbine_id", columns="ym", values="iec_b22_pct").reindex(index=piv_v.index, columns=piv_v.columns)
    text_grid = [
        [
            f"{v:.0f}%<br>(B2.2:{a:.0f}%)" if (pd.notna(v) and pd.notna(a)) else (f"{v:.0f}%" if pd.notna(v) else "")
            for v, a in zip(row_v, row_a, strict=True)
        ]
        for row_v, row_a in zip(piv_v.to_numpy(dtype=float), piv_av.to_numpy(dtype=float), strict=True)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=piv_v.to_numpy(dtype=float),
            x=piv_v.columns.tolist(),
            y=[f"WT-{t}" for t in piv_v.index],
            colorscale="YlGnBu",
            zmin=50,
            zmax=100,
            text=text_grid,
            hovertemplate="<b>%{y}</b> | %{x}<br>Valid Data & B.2.2: %{text}<extra></extra>",
            colorbar={"title": "Valid %"},
        )
    )
    return _polish(fig, "Data Completeness & IEC B.2.2 Availability Matrix (Turbine x Month)", height=max(320, 26 * len(piv_v) + 140))


# ---------------------------------------------------------------------------
# TAB 2: TURBINE DEEP-DIVE & POWER CURVE WORKBENCH (WEBGL)
# ---------------------------------------------------------------------------


def fig_dual_power_curve_scattergl(
    deep: dict[str, Any],
    *,
    ws_mode_label: str = "ERA5 Svenningsen",
    rho_ref: float = 1.225,
    period_a_label: str = "Period A",
    period_b_label: str | None = None,
) -> go.Figure:
    if deep.get("empty", True):
        return empty_figure("Dual-Panel Power Curve Scatter (Raw vs Density-Normalized)")

    sca_a = deep["scatter_a"]
    sca_b = deep["scatter_b"]
    b_raw = deep["bins_raw_a"]
    b_adj = deep["bins_adj_a"]
    b_adj_b = deep["bins_adj_b"]
    refs = deep["ref_curves_df"]
    tid = deep["turbine_id"]
    farm = deep["farm_id"].title()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Raw Measured Wind Speed (m/s) vs Active Power (kW)",
            f"Density-Normalized ({ws_mode_label}, rho_ref={rho_ref:.3f} kg/m3) vs Power",
        ),
        shared_yaxes=True,
    )

    # Point cloud layers via WebGL go.Scattergl
    if not sca_a.empty:
        ts_lbl = sca_a["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M") if "timestamp_utc" in sca_a.columns else ""
        for state_name in ("Under-Cut-In / Calm", "Event Outage (Stop)", "Curtailment / Derate", "Clean Normal"):
            sub = sca_a[sca_a["state_category"] == state_name]
            if sub.empty:
                continue
            color, opacity, size = STATE_STYLE[state_name]
            hover_t = ts_lbl.loc[sub.index] if isinstance(ts_lbl, pd.Series) else ""
            # Left panel: Raw WS
            fig.add_trace(
                go.Scattergl(
                    x=sub["ws_raw"],
                    y=sub["active_power_kw"],
                    mode="markers",
                    name=f"10-min: {state_name}",
                    legendgroup=state_name,
                    text=hover_t,
                    marker={"color": color, "opacity": opacity, "size": size},
                    hovertemplate=f"{state_name}<br>%{{text}}<br>WS: %{{x:.2f}} m/s | P: %{{y:.1f}} kW<extra></extra>",
                ),
                row=1,
                col=1,
            )
            # Right panel: Density-adjusted WS
            fig.add_trace(
                go.Scattergl(
                    x=sub["ws_adj"],
                    y=sub["active_power_kw"],
                    mode="markers",
                    name=f"10-min: {state_name}",
                    legendgroup=state_name,
                    showlegend=False,
                    text=hover_t,
                    marker={"color": color, "opacity": opacity, "size": size},
                    hovertemplate=f"{state_name}<br>%{{text}}<br>Adj WS: %{{x:.2f}} m/s | P: %{{y:.1f}} kW<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Optional Period B scatter overlay on Right panel
    if not sca_b.empty and period_b_label:
        clean_b = sca_b[sca_b["state_category"] == "Clean Normal"]
        if not clean_b.empty:
            fig.add_trace(
                go.Scattergl(
                    x=clean_b["ws_adj"],
                    y=clean_b["active_power_kw"],
                    mode="markers",
                    name=f"10-min: {period_b_label} Clean",
                    marker={"color": "#2563eb", "opacity": 0.30, "size": 4},
                ),
                row=1,
                col=2,
            )

    # Reference Curves on BOTH panels (and explicitly named!)
    if not refs.empty:
        for ref_col, label, color, dash in (
            ("potential_power_mpc_kw", "OEM Contractual MPC (1.225)", "#102a43", "solid"),
            ("potential_power_learned_kw", "Greenbyte Learned PC", "#0ca678", "dash"),
            ("potential_power_ref_kw", "Peer Reference Turbines PC", "#e67700", "dot"),
        ):
            rsub = refs.dropna(subset=[ref_col])
            if rsub.empty:
                continue
            for c_idx in (1, 2):
                fig.add_scatter(
                    x=rsub["ws_bin"],
                    y=rsub[ref_col],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(c_idx == 1),
                    line={"color": color, "width": 2.3, "dash": dash},
                    row=1,
                    col=c_idx,
                )

    # Measured Method-of-Bins Curves + +/-1 sigma band
    if not b_raw.empty:
        fig.add_scatter(
            x=b_raw["ws_bin"],
            y=b_raw["power_mean_kw"],
            mode="lines+markers",
            name=f"Measured Raw Bins ({period_a_label})",
            line={"color": "#111111", "width": 2.8},
            marker={"size": 6, "color": "#111111"},
            row=1,
            col=1,
        )
    if not b_adj.empty:
        upper = b_adj["power_mean_kw"] + b_adj["power_std_kw"]
        lower = (b_adj["power_mean_kw"] - b_adj["power_std_kw"]).clip(lower=-50.0)
        fig.add_scatter(
            x=pd.concat([b_adj["ws_bin"], b_adj["ws_bin"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(16, 42, 67, 0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name=f"{period_a_label} +/-1 sigma Band",
            row=1,
            col=2,
        )
        fig.add_scatter(
            x=b_adj["ws_bin"],
            y=b_adj["power_mean_kw"],
            mode="lines+markers",
            name=f"Measured {period_a_label} (0.5 m/s bins)",
            line={"color": "#111111", "width": 3.0},
            marker={"size": 6, "color": "#111111"},
            row=1,
            col=2,
        )

    if not b_adj_b.empty and period_b_label:
        fig.add_scatter(
            x=b_adj_b["ws_bin"],
            y=b_adj_b["power_mean_kw"],
            mode="lines+markers",
            name=f"Measured {period_b_label} (0.5 m/s bins)",
            line={"color": "#2563eb", "width": 2.8, "dash": "dashdot"},
            marker={"size": 6, "symbol": "diamond", "color": "#2563eb"},
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Raw Wind Speed (m/s)", range=[0, 23], row=1, col=1)
    fig.update_xaxes(title_text="Normalized Wind Speed (m/s)", range=[0, 23], row=1, col=2)
    fig.update_yaxes(title_text="Active Power (kW)", range=[-100, 2200], row=1, col=1)
    return _polish(fig, f"{farm} WT-{tid} — WebGL Power Curve Workbench ({period_a_label})", height=520)


def fig_ti_stratified_delta_p(deep: dict[str, Any], *, period_b_label: str | None = None) -> go.Figure:
    if deep.get("empty", True):
        return empty_figure("M2 TI-Stratified Delta-P & Bin Energy Loss")

    ti_df = deep.get("ti_strat_df", pd.DataFrame())
    b_adj = deep.get("bins_adj_a", pd.DataFrame())
    d_ab = deep.get("delta_ab_df", pd.DataFrame())

    right_title = (
        f"Period B ({period_b_label}) - Period A Bin Delta-P (kW)"
        if (not d_ab.empty and period_b_label)
        else "0.5 m/s Bin Energy Loss vs Reference (MWh)"
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("M2 Turbulence Intensity (TI) Stratified Delta-P (kW)", right_title),
    )

    if not ti_df.empty:
        ti_colors = ["#1f77b4", "#2ca02c", "#d62728"]
        for idx, (reg, grp) in enumerate(ti_df.groupby("ti_regime")):
            g = grp[grp["ws_bin"].between(3.5, 16.0)].sort_values("ws_bin")
            fig.add_scatter(
                x=g["ws_bin"],
                y=g["delta_kw"].round(1),
                mode="lines+markers",
                name=str(reg),
                line={"color": ti_colors[idx % len(ti_colors)], "width": 2.4},
                row=1,
                col=1,
            )
        fig.add_hline(y=0.0, line_color="#334e68", line_width=1.0, row=1, col=1)
        fig.add_vrect(x0=8.5, x1=12.5, fillcolor="rgba(230,119,0,0.08)", line_width=0, row=1, col=1)
    else:
        fig.add_annotation(
            text="Wind speed StdDev unavailable for this period (e.g., Kelmarsh 2016)",
            xref="x1",
            yref="y1",
            x=8.5,
            y=0,
            showarrow=False,
        )

    if not d_ab.empty and period_b_label:
        sub = d_ab[d_ab["ws_bin"].between(3.0, 18.0)]
        fig.add_bar(
            name=f"{period_b_label} - Period A (kW)",
            x=sub["ws_bin"],
            y=sub["b_minus_a_kw"].round(1),
            marker_color=np.where(sub["b_minus_a_kw"] >= 0, "#1b9e77", "#d62728"),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Power Shift (kW)", row=1, col=2)
    elif not b_adj.empty:
        sub = b_adj[b_adj["ws_bin"].between(3.0, 18.0)]
        fig.add_bar(
            name="Lost / Gained Energy per Bin (MWh)",
            x=sub["ws_bin"],
            y=sub["lost_mwh_bin"].round(2),
            marker_color=np.where(sub["lost_mwh_bin"] > 0, "#d62728", "#1b9e77"),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Energy Loss vs Ref (MWh)", row=1, col=2)

    fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Measured - Reference (kW)", row=1, col=1)
    return _polish(fig, "M2 TI Stratification (Low-Speed Convex Overshoot vs Knee Undershoot) & Bin Energy Impact", height=420)


def fig_knee_pitch_rpm_diagnostic(deep: dict[str, Any]) -> go.Figure:
    knee = deep.get("knee_df", pd.DataFrame())
    if knee.empty:
        return empty_figure("M3 Knee Region Pitch & Rotor RPM Audit")

    sub = knee[knee["ws_bin"].between(3.0, 20.0)].sort_values("ws_bin")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Blade Pitch Angle vs Wind Speed (Target vs Fleet Peer Median)",
            "Rotor Speed RPM vs Wind Speed (Target vs Fleet Peer Median)",
        ),
    )
    if "pitch_mean_deg" in sub.columns and sub["pitch_mean_deg"].notna().any():
        fig.add_scatter(x=sub["ws_bin"], y=sub["pitch_mean_deg"], mode="lines+markers", name="Target Pitch A (deg)", line={"color": "#d62728", "width": 2.6}, row=1, col=1)
        if "peer_pitch_deg" in sub.columns and sub["peer_pitch_deg"].notna().any():
            fig.add_scatter(x=sub["ws_bin"], y=sub["peer_pitch_deg"], mode="lines", name="Peer Median Pitch (deg)", line={"color": "#102a43", "dash": "dash"}, row=1, col=1)
    else:
        fig.add_annotation(text="Blade pitch angle not exported on Penmanshiel", xref="x1", yref="y1", x=10, y=5, showarrow=False)

    if "rpm_mean" in sub.columns and sub["rpm_mean"].notna().any():
        fig.add_scatter(x=sub["ws_bin"], y=sub["rpm_mean"], mode="lines+markers", name="Target Rotor RPM", line={"color": "#1f77b4", "width": 2.6}, row=1, col=2)
        if "peer_rpm" in sub.columns and sub["peer_rpm"].notna().any():
            fig.add_scatter(x=sub["ws_bin"], y=sub["peer_rpm"], mode="lines", name="Peer Median RPM", line={"color": "#102a43", "dash": "dash"}, row=1, col=2)

    for c_idx in (1, 2):
        fig.add_vrect(x0=8.0, x1=13.0, fillcolor="rgba(31,119,180,0.08)", line_width=0, row=1, col=c_idx)
        fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=c_idx)
    fig.update_yaxes(title_text="Pitch Angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Rotor Speed (RPM)", row=1, col=2)
    return _polish(fig, f"WT-{deep['turbine_id']} — M3 Knee-Region Control Law (Pitch & Rotor RPM Tracking)", height=410)


# ---------------------------------------------------------------------------
# TAB 3: CHRONIC TRENDS, EWMA/CUSUM & IEC AVAILABILITY CROSS-CHECK
# ---------------------------------------------------------------------------


def fig_chronic_ppr_ewma_cusum(trend: dict[str, Any]) -> go.Figure:
    if trend.get("empty", True):
        return empty_figure("Chronic Power Curve Health (EWMA & CUSUM)")
    d = trend["daily_df"]
    if d.empty:
        return empty_figure("Chronic Power Curve Health (EWMA & CUSUM)")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            "Daily Performance Ratio (Raw vs Clean vs Peer-Adjusted PPR %) & EWMA Control Chart",
            "Two-Sided Standardized CUSUM Drift Accumulator (Threshold = 5.0)",
        ),
    )

    fig.add_scatter(
        x=d["day"],
        y=d["ewma_ucl_pct"],
        mode="lines",
        line={"color": "rgba(16,42,67,0.25)", "dash": "dot"},
        name="EWMA +3 sigma UCL",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=d["day"],
        y=d["ewma_lcl_pct"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.08)",
        line={"color": "rgba(16,42,67,0.25)", "dash": "dot"},
        name="EWMA -3 sigma LCL",
        row=1,
        col=1,
    )
    fig.add_scatter(x=d["day"], y=d["raw_ppr_pct"], mode="markers", marker={"color": "rgba(214,39,40,0.35)", "size": 5}, name="Raw Unmasked PPR %", row=1, col=1)
    fig.add_scatter(x=d["day"], y=d["peer_adj_ppr_pct"], mode="markers", marker={"color": "rgba(27,158,119,0.55)", "size": 5}, name="Peer-Adjusted Clean PPR %", row=1, col=1)
    fig.add_scatter(x=d["day"], y=d["ewma_ppr_pct"], mode="lines", line={"color": "#102a43", "width": 2.6}, name="EWMA (alpha=0.30)", row=1, col=1)

    al = d[d["alert"]]
    if not al.empty:
        fig.add_scatter(
            x=al["day"],
            y=al["ewma_ppr_pct"],
            mode="markers",
            marker={"color": "#d62728", "size": 7, "symbol": "x"},
            name="EWMA / CUSUM Alert Day",
            row=1,
            col=1,
        )

    fig.add_scatter(x=d["day"], y=d["cusum_pos"], mode="lines", line={"color": "#1b9e77", "width": 2}, name="CUSUM + (Upward)", row=2, col=1)
    fig.add_scatter(x=d["day"], y=d["cusum_neg"], mode="lines", line={"color": "#d62728", "width": 2.3}, name="CUSUM - (Degradation)", row=2, col=1)
    fig.add_hline(y=5.0, line_dash="dash", line_color="#d62728", row=2, col=1)

    fig.update_yaxes(title_text="PPR (%)", row=1, col=1)
    fig.update_yaxes(title_text="CUSUM Score", row=2, col=1)
    return _polish(fig, f"{trend['farm_id'].title()} WT-{trend['turbine_id']} — Chronic Performance Drift (EWMA & CUSUM)", height=520)


def fig_availability_crosscheck(trend: dict[str, Any]) -> go.Figure:
    if trend.get("empty", True):
        return empty_figure("Availability Cross-Check: Status Mask vs Kelmarsh/Penmanshiel IEC B.2.2")
    m = trend["monthly_df"]
    if m.empty:
        return empty_figure("Availability Cross-Check: Status Mask vs IEC B.2.2")

    s = trend["summary"]
    corr_txt = f"Daily Corr r={s['avail_corr_mask_vs_iec_b22']}" if s["avail_corr_mask_vs_iec_b22"] is not None else ""
    mae_txt = f" | MAE={s['avail_mae_mask_vs_iec_b22_pp']} pp" if s["avail_mae_mask_vs_iec_b22_pp"] is not None else ""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Monthly Availability: Status-Log Mask vs Greenbyte IEC B.2.2 / System",
            "Monthly Performance Ratio Trajectory: Raw vs Clean vs Peer-Adjusted (%)",
        ),
    )
    for col, lbl, clr, dash in (
        ("mask_avail_pct", "Status-Log Event Mask Avail (%)", "#102a43", "solid"),
        ("iec_b22_avail_pct", "Time-Based IEC B.2.2 (%)", "#1f77b4", "dash"),
        ("system_avail_pct", "Time-Based System Avail (%)", "#e6ab02", "dot"),
        ("prod_iec_b22_pct", "Production-Based IEC B.2.2 (%)", "#0ca678", "dashdot"),
    ):
        if col in m.columns and m[col].notna().any():
            fig.add_scatter(x=m["ym"], y=m[col], mode="lines+markers", name=lbl, line={"color": clr, "dash": dash, "width": 2.2}, row=1, col=1)

    for col, lbl, clr in (
        ("raw_ppr_pct", "Raw PPR (%)", "#9aa5b1"),
        ("clean_ppr_pct", "Event-Screened Clean PPR (%)", "#1b9e77"),
        ("peer_adj_ppr_pct", "Peer-Adjusted PPR (%)", "#2563eb"),
    ):
        if col in m.columns and m[col].notna().any():
            fig.add_scatter(x=m["ym"], y=m[col], mode="lines+markers", name=lbl, line={"color": clr, "width": 2.4}, row=1, col=2)

    fig.update_yaxes(title_text="Availability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Mean PPR (%)", row=1, col=2)
    return _polish(fig, f"WT-{trend['turbine_id']} Availability Validation ({corr_txt}{mae_txt}) & Monthly PPR Evolution", height=420)


# ---------------------------------------------------------------------------
# TAB 4: TRUE-NORTH (M4A), YAW (M4B) & WAKE PHYSICS
# ---------------------------------------------------------------------------


def fig_direction_northing_and_wake_trough(dir_data: dict[str, Any]) -> go.Figure:
    if dir_data.get("empty", True):
        return empty_figure("M4a True-North Consensus & Sector Bias Profile")
    dcon = dir_data.get("direction_consensus_df", pd.DataFrame())
    sec = dir_data.get("sector_spread_df", pd.DataFrame())
    tid = dir_data.get("turbine_id", "01")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "NREL Total True-North Offset (deg) & Peer Disagreement MAE",
            f"WT-{tid} Sector-by-Sector Wind Direction Bias vs Peers (P25-P75)",
        ),
    )
    if not dcon.empty:
        x_lbl = "WT-" + dcon["turbine_id"].astype(str)
        fig.add_bar(name="Total Offset to True North (deg)", x=x_lbl, y=dcon["total_offset_deg"], marker_color="#1f77b4", row=1, col=1)
        fig.add_scatter(x=x_lbl, y=dcon["raw_peer_mae_deg"], mode="lines+markers", name="Raw Peer MAE (deg)", line={"color": "#d62728"}, row=1, col=1)
        fig.add_scatter(x=x_lbl, y=dcon["relative_peer_mae_deg"], mode="lines+markers", name="Calibrated Peer MAE (deg)", line={"color": "#2ca02c"}, row=1, col=1)
        fig.update_yaxes(title_text="Degrees (deg)", row=1, col=1)

    if not sec.empty:
        fig.add_scatter(x=sec["sector_deg"], y=sec["p75"], mode="lines", line={"color": "rgba(31,119,180,0.2)"}, showlegend=False, row=1, col=2)
        fig.add_scatter(x=sec["sector_deg"], y=sec["p25"], mode="lines", fill="tonexty", fillcolor="rgba(31,119,180,0.18)", line={"color": "rgba(31,119,180,0.2)"}, name="Sector P25-P75 IQR", row=1, col=2)
        fig.add_scatter(x=sec["sector_deg"], y=sec["median_bias_deg"], mode="lines+markers", line={"color": "#102a43", "width": 2.6}, name="Sector Median Bias (deg)", row=1, col=2)
        fig.add_hline(y=0.0, line_dash="dash", line_color="#627d98", row=1, col=2)
        fig.update_xaxes(title_text="Wind Direction Sector (deg)", dtick=60, row=1, col=2)
        fig.update_yaxes(title_text="Target - Peer Dir (deg)", row=1, col=2)
    return _polish(fig, "M4a NREL True-North Calibration & Sectoral Distortion Fingerprint", height=410)


def fig_yaw_parabola_and_cable_windings(dir_data: dict[str, Any]) -> go.Figure:
    if dir_data.get("empty", True):
        return empty_figure("M4b Static Yaw Misalignment Parabola & Cable Windings")
    yb = dir_data.get("yaw_bins_df", pd.DataFrame())
    ym = dir_data.get("yaw_metrics", {})
    cab = dir_data.get("cable_df", pd.DataFrame())
    tid = dir_data.get("turbine_id", "01")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"WT-{tid} Vane Position (1+2) vs Normalized Power Ratio (5.5-10.5 m/s)",
            f"WT-{tid} Nacelle Cable Windings Envelope (Turns from Zero)",
        ),
    )
    if not yb.empty:
        fig.add_scatter(x=yb["yaw_bin_deg"], y=yb["median_p_ratio"], mode="markers+lines", name="2-deg Bin Median P/MPC", line={"color": "#1f77b4"}, marker={"size": 8}, row=1, col=1)
        if "fitted_parabola" in yb.columns and yb["fitted_parabola"].notna().any():
            fig.add_scatter(x=yb["yaw_bin_deg"], y=yb["fitted_parabola"], mode="lines", name=f"Quadratic Fit (Bias={ym.get('estimated_yaw_bias_deg', 0):.2f} deg)", line={"color": "#d62728", "dash": "dash", "width": 2.5}, row=1, col=1)
            fig.add_vline(x=float(ym.get("optimal_peak_deg", 0.0)), line_dash="dot", line_color="#2ca02c", row=1, col=1)
        fig.update_xaxes(title_text="Vane Position 1+2 Error (deg)", row=1, col=1)
        fig.update_yaxes(title_text="Power / Reference Ratio", row=1, col=1)
    else:
        fig.add_annotation(text="Vane position unavailable or <120 partial-load points", xref="x1", yref="y1", x=0, y=1, showarrow=False)

    if not cab.empty:
        fig.add_scatter(x=cab["day"], y=cab["max_turns"], mode="lines", line={"color": "rgba(214,39,40,0.4)", "width": 1}, name="Daily Max Turns", row=1, col=2)
        fig.add_scatter(x=cab["day"], y=cab["min_turns"], mode="lines", fill="tonexty", fillcolor="rgba(31,119,180,0.12)", line={"color": "rgba(27,158,119,0.4)", "width": 1}, name="Daily Min Turns", row=1, col=2)
        fig.add_scatter(x=cab["day"], y=cab["mean_turns"], mode="lines", line={"color": "#102a43", "width": 2}, name="Daily Mean Turns", row=1, col=2)
        fig.add_hline(y=0.0, line_dash="dash", line_color="#627d98", row=1, col=2)
        fig.update_yaxes(title_text="Cable Turns", row=1, col=2)
    else:
        fig.add_annotation(text="Cable windings channel unpopulated on this farm", xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False)

    mwh_txt = f" | Est. Recoverable: {ym['est_recoverable_mwh']} MWh (cos^2)" if ym else ""
    return _polish(fig, f"M4b Static Yaw Alignment Audit{mwh_txt} & Cable Untwist Monitor", height=410)


def fig_wake_donor_active_vs_stopped(dir_data: dict[str, Any]) -> go.Figure:
    if dir_data.get("empty", True):
        return empty_figure("Active-Donor Wake Physics & Calibrated Wind Rose")
    wdf = dir_data.get("wake_curves_df", pd.DataFrame())
    rose = dir_data.get("rose_df", pd.DataFrame())

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=(
            f"WT-{dir_data['turbine_id']} Power Curve by Physical Wake Regime",
            f"{dir_data['farm_id'].title()} Calibrated Wind Rose (% Time)",
        ),
    )
    wake_colors = {
        "Freestream (Unwaked)": "#1b9e77",
        "Active Wake (Donor Operating)": "#d62728",
        "Wake Sector (All Donors Stopped)": "#7570b3",
    }
    if not wdf.empty:
        for wname, grp in wdf.groupby("wake_regime"):
            g = grp[grp["ws_bin"].between(3.0, 18.0)].sort_values("ws_bin")
            fig.add_scatter(
                x=g["ws_bin"],
                y=g["power_mean_kw"],
                mode="lines+markers",
                name=str(wname),
                line={"color": wake_colors.get(str(wname), "#333333"), "width": 2.5},
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=1)
        fig.update_yaxes(title_text="Mean Power (kW)", row=1, col=1)

    if not rose.empty:
        total = max(1.0, float(rose["n"].sum()))
        band_colors = {"0-5 m/s": "#74add1", "5-10 m/s": "#abd9e9", "10-15 m/s": "#fdae61", "15+ m/s": "#d73027"}
        for sb in ("0-5 m/s", "5-10 m/s", "10-15 m/s", "15+ m/s"):
            sub = rose[rose["speed_band"] == sb].sort_values("dir_bin")
            if sub.empty:
                continue
            fig.add_trace(
                go.Barpolar(
                    r=(100.0 * sub["n"] / total).round(2),
                    theta=sub["dir_bin"],
                    name=sb,
                    marker_color=band_colors[sb],
                ),
                row=1,
                col=2,
            )
        fig.update_layout(barmode="stack")
    return _polish(fig, "Active-Donor vs Stopped-Donor Wake Placebo Comparison & Calibrated Wind Rose", height=440)


# ---------------------------------------------------------------------------
# WORKSPACE 3: OUTAGE PARETO, IEC ATTRIBUTION & SENVION EVENT CODE DICTIONARY
# ---------------------------------------------------------------------------


def fig_event_pareto_suite(event_data: dict[str, Any]) -> go.Figure:
    sub_df = event_data.get("pareto_subsystem_df", pd.DataFrame())
    cod_df = event_data.get("pareto_codes_df", pd.DataFrame())
    tur_df = event_data.get("pareto_turbine_df", pd.DataFrame())
    if sub_df.empty and cod_df.empty:
        return empty_figure("Debounced Stop-Outage Pareto Suite")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Downtime Share by RDS-PP Subsystem (hrs)",
            "Top-12 Root Stop Codes (Debounced Hours)",
            "Per-Turbine Stacked Stop Hours by Subsystem",
        ),
        column_widths=[0.28, 0.36, 0.36],
    )

    if not sub_df.empty:
        fig.add_trace(
            go.Pie(
                labels=sub_df["subsystem"],
                values=sub_df["duration_hours"].round(1),
                hole=0.45,
                marker={"colors": [SUBSYSTEM_COLORS.get(str(s), "#7f7f7f") for s in sub_df["subsystem"]]},
                textinfo="percent+label",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if not cod_df.empty:
        top = cod_df.iloc[::-1].copy()
        labels = top["root_code"].astype(str) + ": " + top["root_message"].astype(str).str.slice(0, 28)
        fig.add_bar(
            x=top["duration_hours"].round(1),
            y=labels,
            orientation="h",
            marker_color=[SUBSYSTEM_COLORS.get(str(s), "#7f7f7f") for s in top["subsystem"]],
            text=top["episodes"].astype(str) + " ep",
            textposition="auto",
            showlegend=False,
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Stop Hours", row=1, col=2)

    if not tur_df.empty:
        for ss, grp in tur_df.groupby("subsystem"):
            fig.add_bar(
                name=str(ss),
                x="WT-" + grp["turbine_id"].astype(str),
                y=grp["duration_hours"].round(1),
                marker_color=SUBSYSTEM_COLORS.get(str(ss), "#7f7f7f"),
                row=1,
                col=3,
            )
        fig.update_yaxes(title_text="Stop Hours", row=1, col=3)

    fig.update_layout(barmode="stack")
    raw_c = event_data.get("raw_stop_count", 0)
    deb_c = event_data.get("debounced_stop_count", 0)
    return _polish(fig, f"Debounced Outage Pareto ({raw_c} Raw Stops -> {deb_c} Outage Episodes)", height=440)


def fig_event_gantt_and_recovery(event_data: dict[str, Any]) -> go.Figure:
    gantt = event_data.get("gantt_df", pd.DataFrame())
    rec = event_data.get("recovery_df", pd.DataFrame())
    if gantt.empty and rec.empty:
        return empty_figure("Outage Gantt Timeline & Pre/Post Maintenance PPR Recovery")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Major Debounced Outage Timeline (Duration Hours per Episode)",
            "7-Day Pre vs Post Repair PPR & Pitch Recovery",
        ),
    )

    if not gantt.empty:
        g_show = gantt.head(20).iloc[::-1].copy()
        y_lbl = "WT-" + g_show["turbine_id"].astype(str) + " | " + g_show["timestamp_start"].dt.strftime("%m-%d %H:%M")
        fig.add_bar(
            x=g_show["duration_hours"],
            y=y_lbl,
            orientation="h",
            marker_color=[SUBSYSTEM_COLORS.get(str(s), "#7f7f7f") for s in g_show["subsystem"]],
            text=g_show["root_code"].astype(str) + ": " + g_show["root_message"].astype(str).str.slice(0, 24),
            textposition="inside",
            showlegend=False,
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Outage Episode Duration (hours)", row=1, col=1)

    if not rec.empty:
        out_colors = {
            "improved_ppr": "#1b9e77",
            "unchanged_ppr": "#6c757d",
            "degraded_ppr": "#d62728",
            "insufficient_or_confounded": "#e6ab02",
        }
        for outcome, clr in out_colors.items():
            sub = rec[rec["outcome"] == outcome]
            if sub.empty:
                continue
            fig.add_scatter(
                x=sub["duration_hours"],
                y=sub["delta_ppr"],
                mode="markers",
                name=f"Recovery: {outcome}",
                text="WT-" + sub["turbine_id"].astype(str) + " | " + sub["message"].astype(str),
                marker={"size": 9, "color": clr},
                row=1,
                col=2,
            )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#627d98", row=1, col=2)
        fig.update_xaxes(title_text="Maintenance Duration (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Post - Pre Delta PPR", row=1, col=2)

    return _polish(fig, "Outage Episode Timeline & Pre/Post Repair Performance Verification", height=430)


# ---------------------------------------------------------------------------
# TAB 6: DRIVETRAIN CMS, PITCH ACTUATORS & PEER MCP RECONSTRUCTION
# ---------------------------------------------------------------------------


def fig_cms_thermal_and_vibration(cms_data: dict[str, Any], *, selected_temp_col: str = "gear_oil_temp_c") -> go.Figure:
    if cms_data.get("empty", True):
        return empty_figure("Drivetrain Thermal & Vibration Condition Monitoring")

    sc = cms_data.get("cms_scatter", pd.DataFrame())
    peer_sc = cms_data.get("peer_cms_scatter", pd.DataFrame())
    res_df = cms_data.get("thermal_residuals_df", pd.DataFrame())
    tid = cms_data.get("turbine_id", "01")

    avail = cms_data.get("available_temps", [])
    temp_col = selected_temp_col if (avail and any(c == selected_temp_col for c, _ in avail)) else (avail[0][0] if avail else "")
    temp_lbl = dict(avail).get(temp_col, temp_col)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"WT-{tid} {temp_lbl} vs Active Power (WebGL Scatter)",
            f"WT-{tid} Peer-Matched Component Thermal Residuals (250 kW Load Bins)",
        ),
    )

    if temp_col and not peer_sc.empty and temp_col in peer_sc.columns:
        psub = peer_sc.dropna(subset=[temp_col, "active_power_kw"])
        if not psub.empty:
            fig.add_trace(
                go.Scattergl(
                    x=psub["active_power_kw"],
                    y=psub[temp_col],
                    mode="markers",
                    name="Farm Peer Baseline Envelope",
                    marker={"color": "#9aa5b1", "opacity": 0.25, "size": 4},
                ),
                row=1,
                col=1,
            )
    if temp_col and not sc.empty and temp_col in sc.columns:
        tsub = sc.dropna(subset=[temp_col, "active_power_kw"])
        if not tsub.empty:
            fig.add_trace(
                go.Scattergl(
                    x=tsub["active_power_kw"],
                    y=tsub[temp_col],
                    mode="markers",
                    name=f"WT-{tid} Operating",
                    marker={"color": "#d62728", "opacity": 0.55, "size": 5},
                ),
                row=1,
                col=1,
            )
    fig.update_xaxes(title_text="Active Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text=temp_lbl or "Temperature (C)", row=1, col=1)

    if not res_df.empty:
        fig.add_bar(name="Median Thermal Residual (C)", x=res_df["label"], y=res_df["residual_median_c"], marker_color="#1f77b4", row=1, col=2)
        fig.add_bar(name="p95 Thermal Residual (C)", x=res_df["label"], y=res_df["residual_p95_c"], marker_color="#d62728", row=1, col=2)
        fig.add_hline(y=0.0, line_dash="dash", line_color="#334e68", row=1, col=2)
        fig.update_yaxes(title_text="Target - Peer Load-Matched Residual (C)", row=1, col=2)

    fig.update_layout(barmode="group")
    return _polish(fig, f"{cms_data['farm_id'].title()} WT-{tid} — Thermal CMS & Peer Residual Diagnostics", height=430)


def fig_pitch_asymmetry_and_motors(cms_data: dict[str, Any]) -> go.Figure:
    if cms_data.get("empty", True):
        return empty_figure("3-Blade Pitch Asymmetry & 3-Axis Actuator Currents")
    pm = cms_data.get("pitch_motor_df", pd.DataFrame())
    tid = cms_data.get("turbine_id", "01")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"WT-{tid} Inter-Blade Pitch Offsets by Wind Bin (deg)",
            f"WT-{tid} 3-Axis Pitch Motor Currents by Wind Bin (A)",
        ),
    )
    if not pm.empty:
        if pm["dab"].notna().any():
            for col, lbl, clr in (
                ("dab", "Blade A - B", "#1f77b4"),
                ("dbc", "Blade B - C", "#e67700"),
                ("dca", "Blade C - A", "#2ca02c"),
            ):
                fig.add_scatter(x=pm["ws_bin"], y=pm[col], mode="lines+markers", name=lbl, line={"color": clr, "width": 2.2}, row=1, col=1)
            fig.add_hline(y=0.3, line_dash="dash", line_color="#d62728", row=1, col=1)
            fig.add_hline(y=-0.3, line_dash="dash", line_color="#d62728", row=1, col=1)
        else:
            fig.add_annotation(text="Individual Blade A/B/C pitch channels unavailable", xref="x1", yref="y1", x=10, y=0, showarrow=False)

        if pm["motor_current_1_a"].notna().any():
            for col, lbl, clr in (
                ("motor_current_1_a", "Axis 1 Current (A)", "#102a43"),
                ("motor_current_2_a", "Axis 2 Current (A)", "#7570b3"),
                ("motor_current_3_a", "Axis 3 Current (A)", "#d95f02"),
            ):
                fig.add_scatter(x=pm["ws_bin"], y=pm[col], mode="lines+markers", name=lbl, line={"color": clr, "width": 2.2}, row=1, col=2)
        else:
            fig.add_annotation(text="Pitch motor currents unavailable for this period", xref="x2", yref="y2", x=10, y=0, showarrow=False)

    fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Wind Speed Bin (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Pitch Pair Offset (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Motor Current (A)", row=1, col=2)
    return _polish(fig, f"WT-{tid} — Rotor Blade Alignment & 3-Axis Pitch Actuator Health", height=400)


def fig_peer_mcp_and_cup_drift(cms_data: dict[str, Any]) -> go.Figure:
    if cms_data.get("empty", True):
        return empty_figure("Unwaked Peer MCP Wind Reconstruction & S1/S2 Anemometer Drift")
    mcp = cms_data.get("mcp_daily_df", pd.DataFrame())
    tid = cms_data.get("turbine_id", "01")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"WT-{tid} Daily Observed Wind Speed vs Unwaked Peer MCP Prediction (m/s)",
            f"WT-{tid} Primary vs Secondary Anemometer Divergence (S1 - S2 & MCP Residual)",
        ),
    )
    if not mcp.empty:
        fig.add_scatter(x=mcp["day"], y=mcp["obs_ws"], mode="lines", name="Observed WS (m/s)", line={"color": "#102a43", "width": 2.0}, row=1, col=1)
        fig.add_scatter(x=mcp["day"], y=mcp["peer_mcp_ws"], mode="lines", name="Unwaked Peer MCP WS (m/s)", line={"color": "#1b9e77", "dash": "dash", "width": 2.0}, row=1, col=1)
        fig.add_scatter(x=mcp["day"], y=mcp["mcp_residual_ms"], mode="lines", name="Obs - Peer MCP Residual (m/s)", line={"color": "#1f77b4", "width": 2.0}, row=1, col=2)
        if "s1_minus_s2_ms" in mcp.columns and mcp["s1_minus_s2_ms"].notna().any():
            fig.add_scatter(x=mcp["day"], y=mcp["s1_minus_s2_ms"], mode="lines", name="Sensor 1 - Sensor 2 (m/s)", line={"color": "#d62728", "width": 2.2}, row=1, col=2)
        fig.add_hline(y=0.0, line_dash="dot", line_color="#627d98", row=1, col=2)

    fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Residual / Cup Diff (m/s)", row=1, col=2)
    return _polish(fig, f"{cms_data['farm_id'].title()} WT-{tid} — Virtual Anemometer MCP Reconstruction & Cup Drift", height=400)


# ---------------------------------------------------------------------------
# SYNCHRONIZED PRINCIPAL WORKBENCH FIGURES (SINGLE-SCREEN CHAIN)
# ---------------------------------------------------------------------------


def fig_site_cockpit_companion(cockpit: dict[str, Any]) -> go.Figure:
    """2-row synchronized site companion figure paired next to the Folium geospatial map."""
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty:
        return empty_figure("Site Capacity Factor & Deviation Companion")

    kpis = cockpit.get("kpis", {})
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Per-Turbine Net CF (%) vs Theoretical CF (%) & Clean PPR (%)",
            "Event-Screened Clean Dev (%) vs Unmasked Raw Dev (%)",
        ),
    )

    cf_bar_colors = [_PERF_STATUS_COLORS.get(str(s), "#1b9e77") for s in fp["perf_status"]]
    fig.add_bar(
        name="Theoretical CF (%)",
        x=fp["label"],
        y=fp["theoretical_cf_pct"],
        marker={"color": "rgba(148, 163, 184, 0.42)", "line": {"color": "#64748b", "width": 0.9}},
        hovertemplate="<b>%{x}</b><br>Theoretical CF: %{y:.2f}%<extra></extra>",
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_bar(
        name="Net CF (%)",
        x=fp["label"],
        y=fp["net_cf_pct"],
        marker={"color": cf_bar_colors, "line": {"color": "#0f172a", "width": 0.6}},
        text=fp["net_cf_pct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else ""),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Net CF: %{y:.2f}%<extra></extra>",
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Clean PPR (%)",
            x=fp["label"],
            y=fp["clean_ppr_pct"],
            mode="lines+markers",
            line={"color": "#0f172a", "width": 2.1},
            marker={"size": 6, "color": "#0f172a"},
            hovertemplate="<b>%{x}</b><br>Clean PPR: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    dev_colors = np.where(
        fp["clean_dev_pct"] > 8.0,
        "#f97316",
        np.where(fp["clean_dev_pct"] <= -1.8, "#ef4444", "#2563eb"),
    )
    fig.add_bar(
        name="Event-Screened Clean Dev (%)",
        x=fp["label"],
        y=fp["clean_dev_pct"],
        marker_color=dev_colors,
        text=fp["clean_dev_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Clean Dev: %{y:+.2f}%<extra></extra>",
        row=2,
        col=1,
    )
    fig.add_bar(
        name="Unmasked Raw Dev (%)",
        x=fp["label"],
        y=fp["raw_dev_pct"],
        marker_color="rgba(148, 163, 184, 0.55)",
        hovertemplate="<b>%{x}</b><br>Unmasked Raw Dev: %{y:+.2f}%<extra></extra>",
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_color="#334e68", line_width=1.0, row=2, col=1)
    fig.add_hline(y=-1.8, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=8.0, line_dash="dot", line_color="#f97316", row=2, col=1)

    cf_max = float(fp[["net_cf_pct", "theoretical_cf_pct"]].max().max()) if not fp.empty else 40.0
    ppr_clean = fp["clean_ppr_pct"].dropna()
    ppr_min = float(ppr_clean.min()) if not ppr_clean.empty else 90.0
    ppr_max = float(ppr_clean.max()) if not ppr_clean.empty else 105.0

    fig.update_layout(barmode="group", hovermode="x unified")
    fig.update_yaxes(title_text="CF (%)", range=[0.0, max(45.0, cf_max * 1.24)], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PPR (%)", range=[min(90.0, ppr_min - 3.0), max(106.0, ppr_max + 3.0)], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Dev (%)", row=2, col=1)
    return _polish(
        fig,
        f"{kpis.get('farm', '')} — Net CF ({kpis.get('fleet_net_cf_pct', 0.0)}%) vs Theo CF ({kpis.get('fleet_theoretical_cf_pct', 0.0)}%) & Clean Dev",
        height=520,
    )


def fig_turbine_unified_ws_diagnostics(
    deep: dict[str, Any],
    *,
    ws_mode_label: str = "ERA5 Svenningsen",
    rho_ref: float = 1.225,
    period_a_label: str = "Period A",
    period_b_label: str | None = None,
) -> go.Figure:
    """3-row vertically locked wind-speed diagnostic chain (PC -> TI/Loss -> Knee Pitch & RPM)."""
    if deep.get("empty", True):
        return empty_figure("Unified Wind-Speed Domain Diagnostic Chain (3-20 m/s)")

    sca_a = deep.get("scatter_a", pd.DataFrame())
    b_adj = deep.get("bins_adj_a", pd.DataFrame())
    b_adj_b = deep.get("bins_adj_b", pd.DataFrame())
    refs = deep.get("ref_curves_df", pd.DataFrame())
    ti_df = deep.get("ti_strat_df", pd.DataFrame())
    knee = deep.get("knee_df", pd.DataFrame())
    tid = str(deep.get("turbine_id", "01"))
    farm = str(deep.get("farm_id", "")).title()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_heights=[0.42, 0.29, 0.29],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            f"1. Density-Normalized Power Curve ({ws_mode_label}, rho={rho_ref:.3f}) & WebGL 10-min Cloud",
            "2. M2 TI-Stratified Delta-P (kW) & 0.5 m/s Bin Energy Loss (MWh)",
            "3. M3 Knee Blade Pitch (deg) & Rotor Speed (RPM) vs Peer Median [Knee Zone 9.5-12.5 m/s]",
        ),
    )

    # Row 1: Faint 10-min WebGL cloud + 0.5 m/s Method-of-Bins (+-2 SEM) + references
    if not sca_a.empty:
        for state_name in ("Event Outage (Stop)", "Curtailment / Derate", "Clean Normal"):
            sub = sca_a[sca_a["state_category"] == state_name]
            if sub.empty:
                continue
            clr, op, sz = STATE_STYLE[state_name]
            fig.add_trace(
                go.Scattergl(
                    x=sub["ws_adj"],
                    y=sub["active_power_kw"],
                    mode="markers",
                    name=f"10m {state_name}",
                    marker={"color": clr, "opacity": op * 0.65, "size": max(3, sz - 1)},
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    if not refs.empty:
        for col_n, lbl, clr, dash in (
            ("potential_power_mpc_kw", "OEM Contractual MPC", "#102a43", "solid"),
            ("potential_power_learned_kw", "Greenbyte Learned PC", "#0ca678", "dash"),
        ):
            rsub = refs[refs["ws_bin"].between(3.0, 20.0)].dropna(subset=[col_n])
            if not rsub.empty:
                fig.add_scatter(
                    x=rsub["ws_bin"],
                    y=rsub[col_n],
                    mode="lines",
                    name=lbl,
                    line={"color": clr, "width": 2.2, "dash": dash},
                    row=1,
                    col=1,
                )

    if not b_adj.empty:
        ba = b_adj[b_adj["ws_bin"].between(3.0, 20.0)].sort_values("ws_bin")
        if not ba.empty:
            sem2 = 2.0 * ba["power_sem_kw"].fillna(0.0)
            fig.add_scatter(
                x=pd.concat([ba["ws_bin"], ba["ws_bin"][::-1]]),
                y=pd.concat([ba["power_mean_kw"] + sem2, (ba["power_mean_kw"] - sem2).clip(lower=-50.0)[::-1]]),
                fill="toself",
                fillcolor="rgba(16, 42, 67, 0.14)",
                line={"color": "rgba(0,0,0,0)"},
                name=f"{period_a_label} +/-2 SEM",
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=ba["ws_bin"],
                y=ba["power_mean_kw"],
                mode="lines+markers",
                name=f"Measured {period_a_label} (0.5 m/s bins)",
                line={"color": "#111111", "width": 2.8},
                marker={"size": 5, "color": "#111111"},
                row=1,
                col=1,
            )

            # Row 2 secondary axis: bin energy loss (MWh)
            fig.add_trace(
                go.Bar(
                    name="Bin Energy Gap (MWh)",
                    x=ba["ws_bin"],
                    y=ba["lost_mwh_bin"].round(2),
                    marker_color=np.where(ba["lost_mwh_bin"] > 0, "rgba(239, 68, 68, 0.35)", "rgba(34, 197, 94, 0.30)"),
                    hovertemplate="WS %{x:.1f} m/s | Loss: %{y:+.2f} MWh<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

    if not b_adj_b.empty and period_b_label:
        bb = b_adj_b[b_adj_b["ws_bin"].between(3.0, 20.0)].sort_values("ws_bin")
        if not bb.empty:
            fig.add_scatter(
                x=bb["ws_bin"],
                y=bb["power_mean_kw"],
                mode="lines+markers",
                name=f"Compare {period_b_label} (0.5 m/s bins)",
                line={"color": "#2563eb", "width": 2.4, "dash": "dashdot"},
                marker={"size": 5, "symbol": "diamond", "color": "#2563eb"},
                row=1,
                col=1,
            )

    # Row 2 primary axis: M2 TI stratified delta-P
    if not ti_df.empty:
        ti_colors = ["#1f77b4", "#2ca02c", "#d62728"]
        for idx, (reg, grp) in enumerate(ti_df.groupby("ti_regime")):
            g = grp[grp["ws_bin"].between(3.0, 20.0)].sort_values("ws_bin")
            fig.add_trace(
                go.Scatter(
                    x=g["ws_bin"],
                    y=g["delta_kw"].round(1),
                    mode="lines+markers",
                    name=str(reg),
                    line={"color": ti_colors[idx % len(ti_colors)], "width": 2.1},
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
    fig.add_hline(y=0.0, line_color="#475569", line_width=1.0, row=2, col=1)

    # Row 3: Knee Pitch (primary Y) & Rotor RPM (secondary Y)
    if not knee.empty:
        ksub = knee[knee["ws_bin"].between(3.0, 20.0)].sort_values("ws_bin")
        if "pitch_mean_deg" in ksub.columns and ksub["pitch_mean_deg"].notna().any():
            fig.add_trace(
                go.Scatter(x=ksub["ws_bin"], y=ksub["pitch_mean_deg"], mode="lines+markers", name="Target Pitch (deg)", line={"color": "#d62728", "width": 2.4}),
                row=3,
                col=1,
                secondary_y=False,
            )
            if "peer_pitch_deg" in ksub.columns and ksub["peer_pitch_deg"].notna().any():
                fig.add_trace(
                    go.Scatter(x=ksub["ws_bin"], y=ksub["peer_pitch_deg"], mode="lines", name="Peer Median Pitch (deg)", line={"color": "#0f172a", "dash": "dash", "width": 1.9}),
                    row=3,
                    col=1,
                    secondary_y=False,
                )
        if "rpm_mean" in ksub.columns and ksub["rpm_mean"].notna().any():
            fig.add_trace(
                go.Scatter(x=ksub["ws_bin"], y=ksub["rpm_mean"], mode="lines", name="Target Rotor RPM", line={"color": "#2563eb", "width": 2.0}),
                row=3,
                col=1,
                secondary_y=True,
            )
            if "peer_rpm" in ksub.columns and ksub["peer_rpm"].notna().any():
                fig.add_trace(
                    go.Scatter(x=ksub["ws_bin"], y=ksub["peer_rpm"], mode="lines", name="Peer Median RPM", line={"color": "#64748b", "dash": "dot", "width": 1.8}),
                    row=3,
                    col=1,
                    secondary_y=True,
                )

    for r_idx in (1, 2, 3):
        fig.add_vrect(x0=9.5, x1=12.5, fillcolor="rgba(234, 179, 8, 0.08)", line_width=0, row=r_idx, col=1)
    fig.update_xaxes(range=[3.0, 20.0], dtick=1.0, showspikes=True, spikemode="across")
    fig.update_xaxes(title_text="Normalized Wind Speed Bin (m/s — Vertically Synchronized 3-20 m/s)", row=3, col=1)
    fig.update_yaxes(title_text="Power (kW)", range=[-100, 2200], row=1, col=1)
    fig.update_yaxes(title_text="Delta-P (kW)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Lost MWh", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Pitch (deg)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rotor (RPM)", row=3, col=1, secondary_y=True)
    out = _polish(fig, f"{farm} WT-{tid} — Synchronized Wind-Speed Diagnostic Chain ({period_a_label})", height=820)
    out.update_layout(hovermode="x unified")
    return out


def fig_turbine_unified_timeline(
    trend: dict[str, Any],
    ev_data: dict[str, Any] | None = None,
) -> go.Figure:
    """3-row vertically locked UTC timeline (EWMA/PPR -> Daily Net vs Theo CF -> Availability & Stop Markers)."""
    if trend.get("empty", True):
        return empty_figure("Unified UTC Time-Domain Health & Outage Timeline")
    d = trend.get("daily_df", pd.DataFrame())
    if d.empty:
        return empty_figure("Unified UTC Time-Domain Health & Outage Timeline")

    tid = str(trend.get("turbine_id", "01"))
    farm = str(trend.get("farm_id", "")).title()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_heights=[0.38, 0.31, 0.31],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "1. Daily Peer-Adjusted Clean PPR (%) & EWMA Control Chart (lambda=0.30)",
            "2. Daily Capacity Factor (Net CF % vs Theo CF %) & Standardized Neg CUSUM",
            "3. Availability Cross-Check (Event-Mask % vs IEC B.2.2 %) & Stop Outage Markers",
        ),
    )

    # Row 1: PPR & EWMA
    fig.add_scatter(x=d["day"], y=d["ewma_ucl_pct"], mode="lines", line={"color": "rgba(16,42,67,0.28)", "dash": "dot"}, name="EWMA +3s UCL", row=1, col=1)
    fig.add_scatter(x=d["day"], y=d["ewma_lcl_pct"], mode="lines", fill="tonexty", fillcolor="rgba(37,99,235,0.07)", line={"color": "rgba(16,42,67,0.28)", "dash": "dot"}, name="EWMA -3s LCL", row=1, col=1)
    fig.add_scatter(x=d["day"], y=d["peer_adj_ppr_pct"], mode="markers", marker={"color": "rgba(27,158,119,0.50)", "size": 4}, name="Peer-Adj Clean PPR (%)", row=1, col=1)
    fig.add_scatter(x=d["day"], y=d["ewma_ppr_pct"], mode="lines", line={"color": "#0f172a", "width": 2.4}, name="EWMA PPR (%)", row=1, col=1)

    al = d[d["alert"]]
    if not al.empty:
        fig.add_scatter(
            x=al["day"],
            y=al["ewma_ppr_pct"],
            mode="markers",
            marker={"color": "#ef4444", "size": 6, "symbol": "x"},
            name="M1 Drift Alert Day",
            row=1,
            col=1,
        )
        first_day = str(al["day"].iloc[0])
        for r_idx in (1, 2, 3):
            fig.add_vline(x=first_day, line_dash="dash", line_color="rgba(239, 68, 68, 0.55)", line_width=1.3, row=r_idx, col=1)

    # Row 2: Daily Net CF % vs Theo CF % + Neg CUSUM on secondary Y
    if "theoretical_cf_pct" in d.columns and d["theoretical_cf_pct"].notna().any():
        fig.add_trace(
            go.Scatter(x=d["day"], y=d["theoretical_cf_pct"], mode="lines", name="Theoretical CF (%)", line={"color": "#94a3b8", "width": 1.6, "dash": "dot"}),
            row=2,
            col=1,
            secondary_y=False,
        )
    if "net_cf_pct" in d.columns and d["net_cf_pct"].notna().any():
        fig.add_trace(
            go.Scatter(x=d["day"], y=d["net_cf_pct"], mode="lines", name="Net CF (%)", line={"color": "#0284c7", "width": 2.0}),
            row=2,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(x=d["day"], y=d["cusum_neg"], mode="lines", name="CUSUM - (Drift Accumulator)", line={"color": "#dc2626", "width": 1.8}),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Row 3: Daily Availability & Stop Outage Markers
    fig.add_scatter(x=d["day"], y=d["mask_avail_pct"], mode="lines", name="Event-Mask Avail (%)", line={"color": "#0f172a", "width": 2.0}, row=3, col=1)
    if "iec_b22_avail_pct" in d.columns and d["iec_b22_avail_pct"].notna().any():
        fig.add_scatter(x=d["day"], y=d["iec_b22_avail_pct"], mode="lines", name="IEC B.2.2 Time Avail (%)", line={"color": "#3b82f6", "dash": "dash", "width": 1.7}, row=3, col=1)

    if ev_data:
        deb = ev_data.get("debounced_farm_df", pd.DataFrame())
        if not deb.empty and "turbine_id" in deb.columns:
            t_ev = deb[deb["turbine_id"] == tid].copy()
            if not t_ev.empty:
                t_ev["day"] = pd.to_datetime(t_ev["timestamp_start"], utc=True).dt.strftime("%Y-%m-%d")
                merged_ev = t_ev.merge(d[["day", "mask_avail_pct"]], on="day", how="inner").head(80)
                if not merged_ev.empty:
                    txt = (
                        "Code "
                        + merged_ev["root_code"].astype(str)
                        + ": "
                        + merged_ev["root_message"].astype(str)
                        + " ("
                        + merged_ev["duration_hours"].astype(str)
                        + " h)"
                    )
                    fig.add_scatter(
                        x=merged_ev["day"],
                        y=merged_ev["mask_avail_pct"],
                        mode="markers",
                        text=txt,
                        marker={"color": "#ef4444", "size": 8, "symbol": "triangle-down"},
                        name="Stop Episode",
                        hovertemplate="%{x}<br>%{text}<br>Day Mask Avail: %{y:.1f}%<extra></extra>",
                        row=3,
                        col=1,
                    )

    fig.update_xaxes(showspikes=True, spikemode="across")
    fig.update_xaxes(title_text="Date (UTC — Vertically Synchronized Timeline)", row=3, col=1)
    fig.update_yaxes(title_text="PPR (%)", row=1, col=1)
    fig.update_yaxes(title_text="Daily CF (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Neg CUSUM", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Avail (%)", range=[0.0, 105.0], row=3, col=1)
    out = _polish(fig, f"{farm} WT-{tid} — Synchronized Time-Domain Chronic Drift, CF & Outage Timeline", height=820)
    out.update_layout(hovermode="x unified")
    return out

