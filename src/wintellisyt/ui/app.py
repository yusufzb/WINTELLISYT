"""NiceGUI primary 3-workspace SCADA workbench, Folium geo-map, Starlette fallback, and standalone HTML exporter."""

from __future__ import annotations

import argparse
import html
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wintellisyt.atmosphere import STANDARD_DENSITY
from wintellisyt.data.workbench_store import WorkbenchStore, normalize_turbine_id
from wintellisyt.ui.data_loader import SuiteBundle, load_suite_bundle
from wintellisyt.ui.figures import build_dashboard_figures
from wintellisyt.ui.geo_map import BASEMAP_TILES, build_folium_fleet_map
from wintellisyt.ui.workbench_calc import (
    _diagnostic_badges_map,
    compute_chronic_trends_and_avail_check,
    compute_direction_yaw_wake,
    compute_drivetrain_pitch_mcp,
    compute_fleet_cockpit,
    compute_outage_pareto_workbench,
    compute_turbine_deep_dive,
    synthesize_turbine_root_cause,
)
from wintellisyt.ui.workbench_figures import (
    fig_availability_crosscheck,
    fig_cms_thermal_and_vibration,
    fig_direction_northing_and_wake_trough,
    fig_dual_power_curve_scattergl,
    fig_event_gantt_and_recovery,
    fig_event_pareto_suite,
    fig_fleet_completeness_heatmap,
    fig_fleet_energy_and_avail_waterfall,
    fig_peer_mcp_and_cup_drift,
    fig_pitch_asymmetry_and_motors,
    fig_site_cockpit_companion,
    fig_turbine_unified_timeline,
    fig_turbine_unified_ws_diagnostics,
    fig_wake_donor_active_vs_stopped,
    fig_yaw_parabola_and_cable_windings,
)


def _df_to_html_table(df: Any, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows recorded.</p>"
    sub = df.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in sub.columns)
    rows_html = []
    for row in sub.itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape('' if val is None else str(val))}</td>" for val in row)
        rows_html.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"


def _build_bundle_cockpit_for_map(bundle: SuiteBundle, farm_name: str) -> dict[str, Any]:
    """Assemble a lightweight spatial cockpit dictionary from SuiteBundle summary CSVs for offline HTML export."""
    layout = bundle.get("layout")
    if layout.empty or "farm" not in layout.columns:
        return {"kpis": {"farm": farm_name.title()}, "fingerprint_df": pd.DataFrame()}

    lay = layout[layout["farm"].str.lower() == farm_name.lower()].copy()
    if lay.empty:
        return {"kpis": {"farm": farm_name.title()}, "fingerprint_df": pd.DataFrame()}

    lay["turbine_id"] = lay["turbine_id"].map(normalize_turbine_id)
    wf = bundle.get("energy_waterfall")
    wf_sub = (
        wf[wf["farm_id"].str.lower() == farm_name.lower()].copy()
        if (not wf.empty and "farm_id" in wf.columns)
        else pd.DataFrame()
    )
    if not wf_sub.empty and "turbine_id" in wf_sub.columns:
        wf_sub["turbine_id"] = wf_sub["turbine_id"].map(normalize_turbine_id)
        wf_map = {str(r["turbine_id"]): r for r in wf_sub.to_dict(orient="records")}
    else:
        wf_map = {}

    badges_map = _diagnostic_badges_map(bundle, farm_name.lower())
    rows: list[dict[str, Any]] = []
    for r in lay.to_dict(orient="records"):
        tid = str(r["turbine_id"])
        wf_row = wf_map.get(tid, {})
        rated_kw = float(r.get("rated_power_kw", 2050.0) or 2050.0)
        n_rows = float(wf_row.get("rows", 0) or 0.0)
        exp_mwh = float(wf_row.get("energy_export_mwh", np.nan)) if "energy_export_mwh" in wf_row else np.nan
        theo_mwh = float(wf_row.get("energy_theoretical_mwh", np.nan)) if "energy_theoretical_mwh" in wf_row else np.nan
        dt_mwh = float(wf_row.get("lost_downtime_mwh", np.nan)) if "lost_downtime_mwh" in wf_row else np.nan
        perf_mwh = float(wf_row.get("lost_performance_mwh", np.nan)) if "lost_performance_mwh" in wf_row else np.nan
        mpi = float(wf_row.get("mean_performance_index", np.nan)) if "mean_performance_index" in wf_row else np.nan

        period_hours = (n_rows / 6.0) if n_rows > 0 else 8760.0
        net_cf = round(100.0 * exp_mwh * 1000.0 / (rated_kw * period_hours), 2) if np.isfinite(exp_mwh) else np.nan
        theo_cf = round(100.0 * theo_mwh * 1000.0 / (rated_kw * period_hours), 2) if np.isfinite(theo_mwh) else np.nan
        cf_gap = round(net_cf - theo_cf, 2) if (np.isfinite(net_cf) and np.isfinite(theo_cf)) else np.nan
        ppr_pct = round(100.0 * mpi, 2) if np.isfinite(mpi) else np.nan
        dev_pct = round(ppr_pct - 100.0, 2) if np.isfinite(ppr_pct) else np.nan
        badges = badges_map.get(tid, "Nominal")

        if np.isfinite(dev_pct) and dev_pct > 8.0:
            status = "Suspect Over-Read (>+8%)"
        elif (np.isfinite(dev_pct) and dev_pct <= -1.8) or ("M1/PC Drift" in badges):
            status = "Underperforming"
        elif np.isfinite(cf_gap) and cf_gap <= -2.5:
            status = "Sub-Optimal / High Downtime"
        else:
            status = "Healthy / Nominal"

        rows.append(
            {
                "turbine_id": tid,
                "label": f"{farm_name[:3].upper()}-{tid}",
                "latitude": r.get("latitude", np.nan),
                "longitude": r.get("longitude", np.nan),
                "elevation_m": r.get("elevation_m", np.nan),
                "rotor_diameter_m": r.get("rotor_diameter_m", 92.0),
                "perf_status": status,
                "net_cf_pct": net_cf,
                "theoretical_cf_pct": theo_cf,
                "cf_gap_pp": cf_gap,
                "export_mwh": round(exp_mwh, 1) if np.isfinite(exp_mwh) else np.nan,
                "theoretical_mwh": round(theo_mwh, 1) if np.isfinite(theo_mwh) else np.nan,
                "clean_ppr_pct": ppr_pct,
                "clean_dev_pct": dev_pct,
                "raw_dev_pct": np.nan,
                "downtime_mwh": round(dt_mwh, 1) if np.isfinite(dt_mwh) else np.nan,
                "perf_loss_mwh": round(perf_mwh, 1) if np.isfinite(perf_mwh) else np.nan,
                "avail_time_iec_b22_pct": np.nan,
                "avail_prod_iec_b22_pct": np.nan,
                "event_mask_avail_pct": np.nan,
                "diagnostic_badges": badges,
            }
        )

    fp_df = pd.DataFrame(rows).sort_values("turbine_id").reset_index(drop=True)
    return {"kpis": {"farm": farm_name.title()}, "fingerprint_df": fp_df}


def render_dashboard_html(bundle: SuiteBundle, *, inline_plotlyjs: bool = True) -> str:
    """Render a self-contained multi-tab interactive Folium + Plotly HTML dashboard."""
    data = build_dashboard_figures(bundle)
    kpis = data["kpis"]
    tabs = data["tabs"]

    kpi_cards = [
        ("Asset Scope", f"{kpis['n_turbines']} Turbines", kpis["farms"]),
        ("Exported Energy (2017)", f"{kpis['export_gwh']} GWh", f"Downtime {kpis['downtime_mwh']} MWh"),
        ("Direction Peer MAE", f"{kpis['raw_dir_mae_deg']} -> {kpis['cal_dir_mae_deg']} deg", "NREL 2-stage + wake trough"),
        ("PC Drift Alerts", f"{kpis['pc_drift_baseline_hits']} pairs", "3-baseline CUSUM/EWMA"),
        ("Pitch / Actuator Flags", f"{kpis['pitch_asym_flags']} asym | {kpis['actuator_flags']} motor", "Blade + 3-axis motor check"),
        ("Yaw Loss Opportunity", f"{kpis['yaw_recoverable_mwh']} MWh", "Static cos^2 partial-load estimate"),
    ]
    kpi_html = "".join(
        f"<div class='kpi-card'><div class='kpi-title'>{html.escape(t)}</div>"
        f"<div class='kpi-val'>{html.escape(v)}</div>"
        f"<div class='kpi-sub'>{html.escape(s)}</div></div>"
        for t, v, s in kpi_cards
    )

    tab_tables = {
        "overview": [("IEC Energy Waterfall Summary", bundle.get("energy_waterfall"))],
        "direction_wake": [
            ("NREL Consensus & True North Summary", bundle.get("direction_consensus")),
            ("Wake Regime Summary", bundle.get("wake_regime_effects")),
        ],
        "pc_pitch": [
            ("PC Health Baseline Summary", bundle.get("pc_health_summary")),
            ("Blade Pitch Asymmetry Audit", bundle.get("pitch_asymmetry")),
            ("Pre/Post Maintenance Recovery", bundle.get("pitch_ppr_recovery")),
        ],
        "peer_density": [
            ("Anemometer Drift Summary", bundle.get("peer_drift_summary")),
            ("Density Normalization Benchmark (2016->2017)", bundle.get("density_benchmark")),
        ],
        "asset_intelligence": [
            ("Yaw Misalignment & Cable Windings", bundle.get("yaw_misalignment_summary")),
            ("Pitch Actuator Health", bundle.get("pitch_actuator_health")),
            ("Drivetrain & Thermal CMS", bundle.get("drivetrain_cms_summary")),
        ],
        "provenance": [
            ("Anemometer & NTF Provenance Audit", bundle.get("anemometer_ntf_audit")),
            ("Canonical Signal Inventory Summary", bundle.signal_inventory_summary),
        ],
    }

    layout_df = bundle.get("layout")
    farm_names = (
        sorted(layout_df["farm"].dropna().astype(str).unique().tolist())
        if (not layout_df.empty and "farm" in layout_df.columns)
        else []
    )
    overview_maps_html = ""
    if farm_names:
        map_cards = []
        for fn in farm_names:
            ck = _build_bundle_cockpit_for_map(bundle, fn)
            fmap_iframe = build_folium_fleet_map(ck, default_tile="Satellite (Esri)", height_px=420)
            map_cards.append(
                f"<div class='chart-card'><h3>{html.escape(fn.title())} Interactive Folium Asset Map</h3>{fmap_iframe}</div>"
            )
        overview_maps_html = f"<div class='map-grid'>{''.join(map_cards)}</div>"

    tab_buttons = []
    tab_panes = []
    plotly_included = False
    for idx, (tab_key, tab_obj) in enumerate(tabs.items()):
        active_cls = " active" if idx == 0 else ""
        tab_buttons.append(
            f"<button class='tab-btn{active_cls}' onclick=\"openTab(event, 'tab-{tab_key}')\">"
            f"{html.escape(tab_obj['title'])}</button>"
        )
        blocks = [overview_maps_html] if tab_key == "overview" and overview_maps_html else []
        for fig in tab_obj["figures"]:
            js_mode: Any = (True if not plotly_included and inline_plotlyjs else ("cdn" if not plotly_included else False))
            plotly_included = True
            blocks.append(f"<div class='chart-card'>{fig.to_html(full_html=False, include_plotlyjs=js_mode)}</div>")
        for tbl_title, df_tbl in tab_tables.get(tab_key, []):
            blocks.append(f"<div class='chart-card'><h3>{html.escape(tbl_title)}</h3>{_df_to_html_table(df_tbl)}</div>")
        display_style = "block" if idx == 0 else "none"
        tab_panes.append(
            f"<div id='tab-{tab_key}' class='tab-pane' style='display:{display_style}'>{''.join(blocks)}</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WINTELLISYT Wind Engineering & Asset Intelligence Suite</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f5f7fa; color: #1f2933; }}
    header {{ background: #102a43; color: #fff; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }}
    header h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
    header span {{ font-size: 13px; color: #bcccdc; }}
    .container {{ max-width: 1440px; margin: 0 auto; padding: 20px 24px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .map-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; margin-bottom: 16px; }}
    .kpi-card {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(16,42,67,0.05); }}
    .kpi-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #627d98; font-weight: 600; }}
    .kpi-val {{ font-size: 20px; font-weight: 700; margin: 4px 0; color: #102a43; }}
    .kpi-sub {{ font-size: 12px; color: #486581; }}
    .tab-bar {{ display: flex; flex-wrap: wrap; gap: 6px; border-bottom: 2px solid #d9e2ec; margin-bottom: 18px; padding-bottom: 8px; }}
    .tab-btn {{ background: #e6eef5; border: none; padding: 9px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; color: #243b53; font-size: 13px; }}
    .tab-btn.active {{ background: #102a43; color: #fff; }}
    .chart-card {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
    .chart-card h3 {{ margin: 0 0 10px 0; font-size: 15px; color: #243b53; }}
    .table-wrap {{ overflow-x: auto; max-height: 420px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #e6eef5; text-align: left; white-space: nowrap; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; font-weight: 600; }}
    .muted {{ color: #627d98; font-style: italic; }}
  </style>
</head>
<body>
  <header>
    <h1>WINTELLISYT Wind Engineering & Asset Intelligence Dashboard</h1>
    <span>Schema v{html.escape(str(bundle.manifest.get("schema_version", 2)))} | {html.escape(kpis['farms'])}</span>
  </header>
  <div class="container">
    <div class="kpi-grid">{kpi_html}</div>
    <div class="tab-bar">{''.join(tab_buttons)}</div>
    {''.join(tab_panes)}
  </div>
  <script>
    function openTab(evt, tabId) {{
      document.querySelectorAll('.tab-pane').forEach(el => el.style.display = 'none');
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      const pane = document.getElementById(tabId);
      pane.style.display = 'block';
      evt.currentTarget.classList.add('active');
      if (window.Plotly) {{
        pane.querySelectorAll('.plotly-graph-div').forEach(div => window.Plotly.Plots.resize(div));
      }}
    }}
  </script>
</body>
</html>"""


def export_standalone_html(
    suite_dir: str | Path = "artifacts/engineering_suite_2017",
    output_path: str | Path | None = None,
    *,
    root: str | Path = ".",
) -> Path:
    """Export an offline-capable standalone HTML dashboard from engineering suite outputs."""
    bundle = load_suite_bundle(suite_dir, root=root)
    target = Path(output_path) if output_path is not None else (bundle.suite_dir / "interactive_dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard_html(bundle, inline_plotlyjs=True), encoding="utf-8")
    return target


def create_app(suite_dir: str | Path = "artifacts/engineering_suite_2017", *, root: str | Path = "."):
    """Create the Starlette ASGI fallback app serving the interactive dashboard and JSON APIs."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    bundle = load_suite_bundle(suite_dir, root=root)
    html_page = render_dashboard_html(bundle, inline_plotlyjs=True)

    async def index(_request):
        return HTMLResponse(html_page)

    async def health(_request):
        return JSONResponse({"status": "ok", "tables": sorted(bundle.tables.keys())})

    async def api_manifest(_request):
        return JSONResponse(bundle.manifest)

    async def api_table(request):
        name = request.path_params["name"]
        df = bundle.get(name)
        if df.empty and name not in bundle.tables:
            return JSONResponse({"error": f"Unknown or empty table: {name}"}, status_code=404)
        return JSONResponse(df.where(df.notna(), None).to_dict(orient="records"))

    return Starlette(
        routes=[
            Route("/", index),
            Route("/health", health),
            Route("/api/manifest", api_manifest),
            Route("/api/tables/{name}", api_table),
        ]
    )


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _ensure_loopback_host(host: str) -> str:
    """Reject non-loopback bind addresses so the dashboard never opens an external port."""
    normalized = host.strip().lower()
    if normalized not in _LOOPBACK_HOSTS:
        raise ValueError(
            "Security policy: dashboard server must bind strictly to loopback (127.0.0.1 / localhost)"
        )
    return normalized


_PERIOD_PRESETS: dict[str, tuple[str, str]] = {
    "2017 Full Year": ("2017-01-01", "2017-12-31 23:59:59"),
    "2016 Full Year": ("2016-01-01", "2016-12-31 23:59:59"),
    "2016-2017 All Data": ("2016-01-01", "2017-12-31 23:59:59"),
    "2017 H1 (Jan-Jun)": ("2017-01-01", "2017-06-30 23:59:59"),
    "2017 H2 (Jul-Dec)": ("2017-07-01", "2017-12-31 23:59:59"),
    "2017 Q1 (Jan-Mar)": ("2017-01-01", "2017-03-31 23:59:59"),
    "2017 Q2 (Apr-Jun)": ("2017-04-01", "2017-06-30 23:59:59"),
    "2017 Q3 (Jul-Sep)": ("2017-07-01", "2017-09-30 23:59:59"),
    "2017 Q4 (Oct-Dec)": ("2017-10-01", "2017-12-31 23:59:59"),
}

_COMPARE_PRESETS: dict[str, tuple[str, str] | None] = {
    "None (Single Period)": None,
    "2016 Full Year": ("2016-01-01", "2016-12-31 23:59:59"),
    "2017 H1 (Jan-Jun)": ("2017-01-01", "2017-06-30 23:59:59"),
    "2017 H2 (Jul-Dec)": ("2017-07-01", "2017-12-31 23:59:59"),
    "2017 Q1 (Jan-Mar)": ("2017-01-01", "2017-03-31 23:59:59"),
    "2017 Q4 (Oct-Dec)": ("2017-10-01", "2017-12-31 23:59:59"),
}

_WS_MODE_LABELS: dict[str, str] = {
    "svenningsen": "ERA5 Svenningsen (WindPRO)",
    "iec": "ERA5 IEC 61400-12-1 (1/3)",
    "publisher": "Publisher Density-Adj WS",
    "raw": "Raw Measured WS",
}

_WORKBENCH_TABS: list[tuple[str, str]] = [
    ("tab1", "1. Site Cockpit & Folium Geo-Map"),
    ("tab2", "2. Single-Turbine 360° Diagnostic Chain"),
    ("tab3", "3. Fleet Outage Pareto & Sensor Provenance"),
]

_STATUS_BADGE_ICONS: dict[str, str] = {
    "Underperforming": "🔴",
    "Suspect Over-Read (>+8%)": "🟠",
    "Sub-Optimal / High Downtime": "🟡",
    "Healthy / Nominal": "🟢",
}


def _resolve_period_window(label: str, custom_start: str, custom_end: str) -> tuple[str, str]:
    """Resolve preset label or custom YYYY-MM-DD bounds to normalized UTC date bounds."""
    if label != "Custom Range" and label in _PERIOD_PRESETS:
        return _PERIOD_PRESETS[label]
    s_raw = (custom_start or "2017-01-01").strip()
    e_raw = (custom_end or "2017-12-31").strip()
    if len(e_raw) == 10:
        e_raw = f"{e_raw} 23:59:59"
    try:
        s_ts = pd.Timestamp(s_raw)
        e_ts = pd.Timestamp(e_raw)
        if pd.notna(s_ts) and pd.notna(e_ts) and e_ts >= s_ts:
            return s_raw, e_raw
    except ValueError:
        pass
    return _PERIOD_PRESETS["2017 Full Year"]


def _serialize_cell(value: Any) -> Any:
    """Convert pandas/NumPy scalars into JSON-safe Python values for NiceGUI tables."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return "Yes" if bool(value) else "No"
    if isinstance(value, float) or type(value).__name__.startswith("float"):
        return round(float(value), 3)
    if isinstance(value, int) or type(value).__name__.startswith("int"):
        return int(value)
    return str(value)


def _df_to_table_spec(
    df: pd.DataFrame | None,
    columns: list[str] | None = None,
    max_rows: int = 250,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Quasar column definitions and JSON-safe row dictionaries for `ui.table`."""
    if df is None or df.empty:
        return [], []
    chosen = [c for c in columns if c in df.columns] if columns else list(df.columns[:14])
    if not chosen:
        chosen = list(df.columns[:12])
    col_defs = [
        {"name": c, "label": c.replace("_", " ").title(), "field": c, "sortable": True, "align": "left"}
        for c in chosen
    ]
    sub = df[chosen].head(max_rows)
    rows: list[dict[str, Any]] = []
    for idx, vals in enumerate(sub.itertuples(index=False, name=None)):
        row_dict: dict[str, Any] = {"_row_id": idx}
        for col_name, val in zip(chosen, vals, strict=True):
            row_dict[col_name] = _serialize_cell(val)
        rows.append(row_dict)
    return col_defs, rows


def _render_metric_cards(container: Any, cards: list[tuple[str, str, str]]) -> None:
    """Populate a horizontal strip of uniform diagnostic KPI cards."""
    from nicegui import ui

    container.clear()
    with container:
        for title, value, subtitle in cards:
            with ui.card().classes("flex-1 min-w-[170px] bg-white border border-slate-200 shadow-sm p-3"):
                ui.label(title).classes("text-[11px] font-semibold uppercase tracking-wide text-slate-500")
                ui.label(value).classes("text-lg font-bold text-slate-900 leading-tight mt-0.5")
                if subtitle:
                    ui.label(subtitle).classes("text-xs text-slate-500 mt-0.5")


def _format_turbine_options(cockpit: dict[str, Any], raw_tids: list[str]) -> dict[str, str]:
    """Format turbine dropdown options with status icon, Net CF %, and Clean Dev %."""
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty or "turbine_id" not in fp.columns:
        return {t: f"WT-{t}" for t in raw_tids}
    row_map = {str(r["turbine_id"]): r for r in fp.to_dict(orient="records")}
    opts: dict[str, str] = {}
    for tid in raw_tids:
        r = row_map.get(tid)
        if not r:
            opts[tid] = f"WT-{tid}"
            continue
        icon = _STATUS_BADGE_ICONS.get(str(r.get("perf_status", "")), "🟢")
        cf = r.get("net_cf_pct")
        dev = r.get("clean_dev_pct")
        cf_txt = f"CF {float(cf):.1f}%" if pd.notna(cf) else "CF N/A"
        dev_txt = f"{float(dev):+.1f}%" if pd.notna(dev) else "N/A"
        opts[tid] = f"{icon} WT-{tid} ({cf_txt} | {dev_txt})"
    return opts


def _get_cached_cockpit(
    store: WorkbenchStore,
    bundle: SuiteBundle,
    state: dict[str, Any],
    cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    p_a = _resolve_period_window(state["period_a"], state["custom_start"], state["custom_end"])
    key = (
        state["farm"],
        p_a,
        state["ws_mode"],
        state["ref_mode"],
        state["mask_mode"],
        round(float(state["rho_ref"]), 4),
    )
    if key not in cache:
        cache[key] = compute_fleet_cockpit(
            store,
            bundle,
            state["farm"],
            start_ts=p_a[0],
            end_ts=p_a[1],
            ws_mode=state["ws_mode"],
            ref_mode=state["ref_mode"],
            mask_mode=state["mask_mode"],
            rho_ref=float(state["rho_ref"]),
        )
    return cache[key]


def _render_nicegui_table_section(
    title: str,
    df: pd.DataFrame | None,
    columns: list[str] | None = None,
    *,
    max_rows: int = 250,
    pagination: int = 12,
    empty_note: str = "No records for the selected filters.",
) -> None:
    from nicegui import ui

    with ui.card().classes("w-full bg-white border border-slate-200 shadow-sm p-3"):
        ui.label(title).classes("text-sm font-semibold text-slate-800 mb-2")
        col_defs, rows = _df_to_table_spec(df, columns=columns, max_rows=max_rows)
        if not rows:
            ui.label(empty_note).classes("text-xs text-slate-500 italic")
            return
        ui.table(columns=col_defs, rows=rows, row_key="_row_id", pagination=pagination).classes("w-full text-xs")


def _build_workbench_tab_body(
    tab_key: str,
    body: Any,
    store: WorkbenchStore,
    bundle: SuiteBundle,
    state: dict[str, Any],
    cockpit_cache: dict[tuple[Any, ...], dict[str, Any]],
    on_refresh_tab: Any,
    on_jump_turbine: Any,
    map_slot_ref: dict[str, Any] | None = None,
) -> None:
    """Render the active ergonomic diagnostic workspace inside `body`."""
    from nicegui import ui

    farm = str(state["farm"])
    tid = str(state["turbine_id"])
    p_a = _resolve_period_window(state["period_a"], state["custom_start"], state["custom_end"])
    p_b = _COMPARE_PRESETS.get(state["period_b"])
    p_b_lbl = state["period_b"] if p_b is not None else None
    ws_mode = str(state["ws_mode"])
    ref_mode = str(state["ref_mode"])
    mask_mode = str(state["mask_mode"])
    rho_ref = float(state["rho_ref"])
    full_points = state["webgl_res"] == "full"
    debounce_sec = int(state["debounce_sec"])
    cockpit = _get_cached_cockpit(store, bundle, state, cockpit_cache)
    fp_df = cockpit.get("fingerprint_df", pd.DataFrame())

    body.clear()
    with body:
        # ------------------------------------------------------------------
        # WORKSPACE 1: SITE COCKPIT & INTERACTIVE FOLIUM GEO-MAP
        # ------------------------------------------------------------------
        if tab_key == "tab1":
            # Quick-Jump Flagged Turbine Pills Banner
            with ui.card().classes("w-full bg-white border border-slate-200 shadow-sm p-3"):
                flagged = (
                    fp_df[fp_df["perf_status"] != "Healthy / Nominal"].sort_values("clean_dev_pct")
                    if (not fp_df.empty and "perf_status" in fp_df.columns)
                    else pd.DataFrame()
                )
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    ui.label("Flagged Watchlist (1-Click 360° Chain):").classes("text-xs font-bold text-slate-700")
                    if flagged.empty:
                        ui.badge("All turbines operating within nominal bounds", color="positive")
                    else:
                        for row in flagged.to_dict(orient="records"):
                            t_item = str(row["turbine_id"])
                            icon = _STATUS_BADGE_ICONS.get(str(row["perf_status"]), "🟡")
                            dev_v = row.get("clean_dev_pct")
                            cf_v = row.get("net_cf_pct")
                            btn_lbl = (
                                f"{icon} {row['label']} "
                                f"(CF {float(cf_v):.1f}% | {float(dev_v):+.1f}%)"
                                if (pd.notna(cf_v) and pd.notna(dev_v))
                                else f"{icon} {row['label']}"
                            )
                            btn_clr = "red" if row["perf_status"] == "Underperforming" else "orange"
                            ui.button(
                                btn_lbl,
                                on_click=lambda _e, target=t_item: on_jump_turbine(target, True),
                            ).props(f"dense outline color={btn_clr} size=sm")

            # Hero Row: Interactive Folium Map (Left) + Site CF & Dev Companion (Right)
            with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-12 gap-3 items-start"):
                with ui.card().classes("lg:col-span-7 bg-white border border-slate-200 shadow-sm p-3"):
                    def _render_map_markup() -> str:
                        return build_folium_fleet_map(
                            cockpit,
                            default_tile=str(state["basemap_tile"]),
                            show_wake_corridors=bool(state["show_wake_corridors"]),
                            show_rotor_rings=bool(state["show_rotor_rings"]),
                            selected_turbine_id=tid,
                            height_px=520,
                        )

                    with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap mb-2"):
                        ui.label(
                            f"{farm.title()} Geospatial Asset Map (Folium Multi-Tile + Peer Wake Corridors)"
                        ).classes("text-sm font-bold text-slate-800")
                        with ui.row().classes("items-center gap-3 flex-wrap"):
                            rotor_chk = ui.checkbox(
                                "1D Rotor Rings & Hull",
                                value=bool(state["show_rotor_rings"]),
                            ).props("dense")
                            wake_chk = ui.checkbox(
                                "<=10D Wake Corridors",
                                value=bool(state["show_wake_corridors"]),
                            ).props("dense")

                    map_widget = ui.html(_render_map_markup(), sanitize=False).classes("w-full")
                    if map_slot_ref is not None:
                        map_slot_ref["widget"] = map_widget
                        map_slot_ref["renderer"] = _render_map_markup

                    def _toggle_map_flag(key: str, val: bool) -> None:
                        state[key] = val
                        map_widget.set_content(_render_map_markup())

                    rotor_chk.on_value_change(lambda e: _toggle_map_flag("show_rotor_rings", bool(e.value)))
                    wake_chk.on_value_change(lambda e: _toggle_map_flag("show_wake_corridors", bool(e.value)))

                with ui.card().classes("lg:col-span-5 bg-white border border-slate-200 shadow-sm p-3"):
                    ui.plotly(fig_site_cockpit_companion(cockpit)).classes("w-full")

            audit = cockpit.get("curtailment_audit", {})
            with ui.card().classes("w-full bg-blue-50 border border-blue-200 p-2.5"):
                ui.label(
                    f"Physical Derate & Curtailment Audit: "
                    f"Capacity-limited intervals = {audit.get('total_cap_limit_hours', 0)} h | "
                    f"Shelf-derated intervals = {audit.get('total_physical_derate_hours', 0)} h | "
                    f"{audit.get('archive_bats_birds_note', '')}"
                ).classes("text-xs text-slate-700")

            ui.plotly(fig_fleet_energy_and_avail_waterfall(cockpit)).classes("w-full")

            _render_nicegui_table_section(
                f"{farm.title()} Fleet Diagnostic Scorecard (CF %, Power Curve Dev %, IEC B.2.2)",
                fp_df,
                columns=[
                    "label",
                    "perf_status",
                    "net_cf_pct",
                    "theoretical_cf_pct",
                    "cf_gap_pp",
                    "export_mwh",
                    "theoretical_mwh",
                    "clean_dev_pct",
                    "raw_dev_pct",
                    "outage_mask_shift_pp",
                    "clean_ppr_pct",
                    "avail_time_iec_b22_pct",
                    "avail_prod_iec_b22_pct",
                    "event_mask_avail_pct",
                    "physical_derate_hours",
                    "latitude",
                    "longitude",
                    "diagnostic_badges",
                ],
                pagination=15,
            )

            with ui.expansion("Data Completeness Matrix & M4a NREL True-North Consensus", icon="explore").classes(
                "w-full bg-white border border-slate-200"
            ):
                ui.plotly(fig_fleet_completeness_heatmap(cockpit)).classes("w-full")
                dir_overview = compute_direction_yaw_wake(
                    store,
                    bundle,
                    farm,
                    tid,
                    start_ts=p_a[0],
                    end_ts=p_a[1],
                    ws_mode=ws_mode,
                    rho_ref=rho_ref,
                )
                ui.plotly(fig_direction_northing_and_wake_trough(dir_overview)).classes("w-full")
                _render_nicegui_table_section(
                    f"{farm.title()} NREL True-North Consensus Table (M4a)",
                    dir_overview.get("direction_consensus_df"),
                    pagination=14,
                )
            return

        # ------------------------------------------------------------------
        # WORKSPACE 2: SINGLE-TURBINE 360° DIAGNOSTIC CHAIN (ALL ON 1 SCREEN)
        # ------------------------------------------------------------------
        if tab_key == "tab2":
            deep = compute_turbine_deep_dive(
                store,
                farm,
                tid,
                p_a,
                p_b,
                ws_mode=ws_mode,
                ref_mode=ref_mode,
                mask_mode=mask_mode,
                rho_ref=rho_ref,
                full_points=full_points,
            )
            trend = compute_chronic_trends_and_avail_check(
                store,
                farm,
                tid,
                start_ts=p_a[0],
                end_ts=p_a[1],
                ws_mode=ws_mode,
                ref_mode=ref_mode,
                rho_ref=rho_ref,
            )
            dir_data = compute_direction_yaw_wake(
                store,
                bundle,
                farm,
                tid,
                start_ts=p_a[0],
                end_ts=p_a[1],
                ws_mode=ws_mode,
                rho_ref=rho_ref,
            )
            ev_data = compute_outage_pareto_workbench(
                store,
                bundle,
                farm,
                tid,
                start_ts=p_a[0],
                end_ts=p_a[1],
                debounce_sec=debounce_sec,
            )
            cms_data = compute_drivetrain_pitch_mcp(
                store,
                bundle,
                farm,
                tid,
                start_ts=p_a[0],
                end_ts=p_a[1],
                full_points=full_points,
            )
            verdict = synthesize_turbine_root_cause(cockpit, deep, trend, dir_data, ev_data, cms_data)

            sev = verdict["severity"]
            card_cls = (
                "w-full bg-red-50 border border-red-300 shadow-sm p-3.5"
                if sev == "error"
                else (
                    "w-full bg-amber-50 border border-amber-300 shadow-sm p-3.5"
                    if sev == "warn"
                    else "w-full bg-emerald-50 border border-emerald-300 shadow-sm p-3.5"
                )
            )
            badge_clr = "red" if sev == "error" else ("orange" if sev == "warn" else "green")

            # 1. Automated Root-Cause Verdict Banner + Quick Turbine Switcher
            with ui.card().classes(card_cls):
                with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.label(verdict["headline"]).classes("text-base font-bold text-slate-900")
                        ui.badge(verdict["primary_cause"], color=badge_clr)
                        if verdict["flags"] and verdict["flags"] != "Nominal":
                            ui.badge(f"Flags: {verdict['flags']}", color="blue-grey")
                    if not fp_df.empty and "turbine_id" in fp_df.columns:
                        with ui.row().classes("items-center gap-1 flex-wrap"):
                            ui.label("Switch Turbine:").classes("text-[11px] font-semibold text-slate-600 mr-1")
                            for r in fp_df.to_dict(orient="records"):
                                t_opt = str(r["turbine_id"])
                                icon = _STATUS_BADGE_ICONS.get(str(r.get("perf_status", "")), "🟢")
                                is_cur = t_opt == tid
                                props = "dense color=primary size=sm" if is_cur else "dense outline color=blue-grey size=sm"
                                ui.button(
                                    f"{icon} {t_opt}",
                                    on_click=lambda _e, target=t_opt: on_jump_turbine(target, False),
                                ).props(props)

                with ui.column().classes("w-full gap-1 mt-2"):
                    for line in verdict["findings"]:
                        ui.label(f"• {line}").classes("text-xs text-slate-800 leading-snug")

            # Compact KPI summary strip for active turbine
            m1 = deep.get("m1_summary", {})
            ym = dir_data.get("yaw_metrics", {})
            sm = trend.get("summary", {})
            if m1:
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    for t_lbl, v_txt, s_txt in (
                        (
                            "Net / Theo CF %",
                            f"{m1.get('net_cf_pct', 'N/A')}% / {m1.get('theoretical_cf_pct', 'N/A')}%",
                            f"WT-{tid} in {state['period_a']}",
                        ),
                        (
                            "Partial-Load PPR (4-11 m/s)",
                            f"{m1.get('partial_load_ppr_pct', 'N/A')}%",
                            f"Overall PPR {m1.get('overall_ppr_pct', 'N/A')}% | Delta {-float(m1.get('energy_gap_mwh', 0.0)):+.1f} MWh",
                        ),
                        (
                            "Chronic Onset (M1)",
                            f"{sm.get('alert_days', 0)} alert days",
                            f"First: {sm.get('first_alert_day', 'None')} | CUSUM-: {sm.get('final_cusum_neg', 0.0)}",
                        ),
                        (
                            "Static Yaw Bias (M4b)",
                            f"{ym.get('estimated_yaw_bias_deg', 0.0):+.2f} deg" if ym else "N/A",
                            f"Est. cos^2 recovery: {ym.get('est_recoverable_mwh', 0.0)} MWh" if ym else "Insufficient vane data",
                        ),
                        (
                            "Operating Regimes",
                            f"{m1.get('clean_rows', 0):,} clean",
                            f"Stop={m1.get('stop_masked_rows', 0):,} | Curtail={m1.get('curtail_masked_rows', 0):,} | Wake={m1.get('wake_rows', 0):,}",
                        ),
                    ):
                        with ui.card().classes("flex-1 min-w-[175px] bg-white border border-slate-200 p-2.5"):
                            ui.label(t_lbl).classes("text-[11px] uppercase text-slate-500 font-semibold")
                            ui.label(str(v_txt)).classes("text-base font-bold text-slate-900")
                            ui.label(s_txt).classes("text-xs text-slate-500")

            # 2. Master Synchronized Diagnostic Row (Wind-Speed Locked Left vs Time Locked Right)
            with ui.element("div").classes("w-full grid grid-cols-1 xl:grid-cols-2 gap-3"):
                with ui.card().classes("bg-white border border-slate-200 shadow-sm p-2.5"):
                    ui.plotly(
                        fig_turbine_unified_ws_diagnostics(
                            deep,
                            ws_mode_label=_WS_MODE_LABELS.get(ws_mode, ws_mode),
                            rho_ref=rho_ref,
                            period_a_label=state["period_a"],
                            period_b_label=p_b_lbl,
                        )
                    ).classes("w-full")
                with ui.card().classes("bg-white border border-slate-200 shadow-sm p-2.5"):
                    ui.plotly(fig_turbine_unified_timeline(trend, ev_data)).classes("w-full")

            # 3. Subsystem & Directional Verification Row on the Same Screen
            with ui.element("div").classes("w-full grid grid-cols-1 xl:grid-cols-2 gap-3"):
                with ui.card().classes("bg-white border border-slate-200 shadow-sm p-3"):
                    ui.label(f"WT-{tid} Directional Alignment, Yaw cos^2 Loss & Active Wake Physics (M4)").classes(
                        "text-sm font-bold text-slate-800 mb-2"
                    )
                    ui.plotly(fig_yaw_parabola_and_cable_windings(dir_data)).classes("w-full")
                    ui.plotly(fig_wake_donor_active_vs_stopped(dir_data)).classes("w-full")

                with ui.card().classes("bg-white border border-slate-200 shadow-sm p-3"):
                    ui.label(f"WT-{tid} 3-Blade Pitch Asymmetry, Drivetrain Thermal CMS & Peer MCP").classes(
                        "text-sm font-bold text-slate-800 mb-2"
                    )
                    avail_temps = cms_data.get("available_temps", [])
                    temp_opts = (
                        {col: lbl for col, lbl in avail_temps}
                        if avail_temps
                        else {"gear_oil_temp_c": "Gear Oil Temp (C)"}
                    )
                    if state["cms_temp_col"] not in temp_opts:
                        state["cms_temp_col"] = next(iter(temp_opts))
                    with ui.row().classes("w-full items-center gap-2 mb-1"):
                        ui.label("Thermal Channel:").classes("text-xs font-semibold text-slate-700")
                        cms_sel = ui.select(temp_opts, value=state["cms_temp_col"]).classes("min-w-[240px]")

                    cms_plot = ui.plotly(
                        fig_cms_thermal_and_vibration(cms_data, selected_temp_col=str(state["cms_temp_col"]))
                    ).classes("w-full")

                    def _on_cms_temp_change(e: Any) -> None:
                        state["cms_temp_col"] = str(e.value)
                        cms_plot.update_figure(
                            fig_cms_thermal_and_vibration(cms_data, selected_temp_col=state["cms_temp_col"])
                        )

                    cms_sel.on_value_change(_on_cms_temp_change)
                    ui.plotly(fig_pitch_asymmetry_and_motors(cms_data)).classes("w-full")
                    ui.plotly(fig_peer_mcp_and_cup_drift(cms_data)).classes("w-full")

            # 4. On-Demand Collapsible Detailed Engineering Evidence Tables
            with ui.expansion(
                f"WT-{tid} Raw vs Density-Normalized Dual Curve & 0.5 m/s Method-of-Bins Table",
                icon="table_chart",
            ).classes("w-full bg-white border border-slate-200"):
                ui.plotly(
                    fig_dual_power_curve_scattergl(
                        deep,
                        ws_mode_label=_WS_MODE_LABELS.get(ws_mode, ws_mode),
                        rho_ref=rho_ref,
                        period_a_label=state["period_a"],
                        period_b_label=p_b_lbl,
                    )
                ).classes("w-full")
                _render_nicegui_table_section(
                    f"WT-{tid} Exact 0.5 m/s Method-of-Bins Table",
                    deep.get("bins_adj_a"),
                    columns=[
                        "ws_bin",
                        "count",
                        "power_mean_kw",
                        "power_std_kw",
                        "ref_mean_kw",
                        "delta_kw",
                        "delta_pct",
                        "lost_mwh_bin",
                    ],
                    pagination=14,
                )

            with ui.expansion(
                f"WT-{tid} Stop Outage Episodes, IEC B.2.2 Monthly Check & T0 +/- 60 min Alarm Cascade",
                icon="warning",
            ).classes("w-full bg-white border border-slate-200"):
                _render_nicegui_table_section(
                    f"WT-{tid} Monthly PPR & IEC 61400-26 Availability Cross-Check",
                    trend.get("monthly_df"),
                    pagination=12,
                )
                ui.plotly(fig_availability_crosscheck(trend)).classes("w-full")
                _render_nicegui_table_section(
                    f"WT-{tid} Focused T0 +/- 60 min Alarm Cascade Sequence",
                    ev_data.get("cascade_df"),
                    columns=[
                        "offset_min",
                        "timestamp_start",
                        "status",
                        "code",
                        "message",
                        "iec_category",
                        "subsystem",
                        "duration_raw",
                    ],
                    pagination=10,
                )
            return

        # ------------------------------------------------------------------
        # WORKSPACE 3: FLEET OUTAGE PARETO & SENSOR PROVENANCE
        # ------------------------------------------------------------------
        full_dict = store.get_event_dictionary(farm)
        subsys_opts = (
            ["All", *sorted(full_dict["subsystem"].dropna().astype(str).unique().tolist())]
            if not full_dict.empty
            else ["All"]
        )
        iec_raw = full_dict["iec_category"].fillna("").astype(str).unique().tolist() if not full_dict.empty else []
        iec_opts = ["All", *sorted([c for c in iec_raw if c])]
        if state["subsystem_filter"] not in subsys_opts:
            state["subsystem_filter"] = "All"
        if state["iec_filter"] not in iec_opts:
            state["iec_filter"] = "All"

        ev_data = compute_outage_pareto_workbench(
            store,
            bundle,
            farm,
            tid,
            start_ts=p_a[0],
            end_ts=p_a[1],
            debounce_sec=debounce_sec,
            selected_episode_idx=int(state["selected_episode_idx"]),
            code_search=str(state["code_search"]),
            subsystem_filter=str(state["subsystem_filter"]),
            iec_filter=str(state["iec_filter"]),
        )
        gantt_df = ev_data.get("gantt_df", pd.DataFrame())
        ep_opts: dict[int, str] = {}
        if not gantt_df.empty:
            for idx, row in gantt_df.iterrows():
                t_s = pd.Timestamp(row["timestamp_start"]).strftime("%Y-%m-%d %H:%M")
                ep_opts[int(idx)] = (
                    f"#{idx}: WT-{row['turbine_id']} | {t_s} | "
                    f"Code {row['root_code']} ({row['duration_hours']}h)"
                )
        else:
            ep_opts = {0: "No stop episodes in window"}

        cur_ep = int(state["selected_episode_idx"])
        if cur_ep not in ep_opts:
            cur_ep = next(iter(ep_opts))
            state["selected_episode_idx"] = cur_ep

        with ui.card().classes("w-full bg-white border border-slate-200 p-3"):
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                search_inp = ui.input(
                    label="Search Senvion Code / Message",
                    value=str(state["code_search"]),
                ).classes("min-w-[220px] flex-1")

                def _apply_event_search() -> None:
                    state["code_search"] = str(search_inp.value or "")
                    state["selected_episode_idx"] = 0
                    on_refresh_tab()

                search_inp.on("keydown.enter", lambda: _apply_event_search())
                ui.select(
                    subsys_opts,
                    label="RDS-PP Subsystem",
                    value=state["subsystem_filter"],
                    on_change=lambda e: (
                        state.update({"subsystem_filter": str(e.value), "selected_episode_idx": 0}),
                        on_refresh_tab(),
                    ),
                ).classes("min-w-[180px]")
                ui.select(
                    iec_opts,
                    label="IEC Category",
                    value=state["iec_filter"],
                    on_change=lambda e: (
                        state.update({"iec_filter": str(e.value), "selected_episode_idx": 0}),
                        on_refresh_tab(),
                    ),
                ).classes("min-w-[170px]")
                ui.select(
                    ep_opts,
                    label="Alarm Cascade Episode (T0 +/- 60 min)",
                    value=cur_ep,
                    on_change=lambda e: (
                        state.update({"selected_episode_idx": int(e.value)}),
                        on_refresh_tab(),
                    ),
                ).classes("min-w-[300px] flex-1")
                ui.button("Filter Dict", on_click=_apply_event_search).props("dense outline color=primary")

        ui.plotly(fig_event_pareto_suite(ev_data)).classes("w-full")
        ui.plotly(fig_event_gantt_and_recovery(ev_data)).classes("w-full")

        ep_info = ev_data.get("selected_episode_info", {})
        if ep_info:
            with ui.card().classes("w-full bg-amber-50 border border-amber-200 p-2.5"):
                ui.label(
                    f"Focused Alarm Cascade (T0 +/- 60 min): WT-{ep_info.get('turbine_id')} | "
                    f"T0={ep_info.get('t0_utc')} | Root Code {ep_info.get('root_code')}: "
                    f"{ep_info.get('root_message')} ({ep_info.get('duration_hours')} h, "
                    f"{ep_info.get('merged_alarms')} raw records merged)"
                ).classes("text-xs font-medium text-amber-900")

        _render_nicegui_table_section(
            f"{farm.title()} Senvion Event Code & IEC Category Dictionary",
            ev_data.get("dictionary_df"),
            columns=[
                "code",
                "message",
                "status",
                "iec_category",
                "subsystem",
                "occurrences",
                "n_turbines_affected",
                "total_duration_hours",
                "median_duration_min",
                "estimated_lost_mwh",
            ],
            pagination=12,
        )
        _render_nicegui_table_section(
            "T0 +/- 60 min Pre-Fault Warnings, Comms & Stop Cascade Sequence",
            ev_data.get("cascade_df"),
            columns=[
                "offset_min",
                "timestamp_start",
                "status",
                "code",
                "message",
                "iec_category",
                "subsystem",
                "duration_raw",
            ],
            pagination=10,
        )

        with ui.expansion("Fleet Sensor Provenance, Thermal Residuals & Virtual Anemometer MCP", icon="sensors").classes(
            "w-full bg-white border border-slate-200"
        ):
            cms_data = compute_drivetrain_pitch_mcp(
                store,
                bundle,
                farm,
                tid,
                start_ts=p_a[0],
                end_ts=p_a[1],
                full_points=full_points,
            )
            _render_nicegui_table_section(
                f"WT-{tid} Load-Matched Thermal CMS Residuals vs Fleet Peers",
                cms_data.get("thermal_residuals_df"),
                pagination=10,
            )
            _render_nicegui_table_section(
                f"{farm.title()} Peer Virtual Anemometer MCP Benchmark",
                cms_data.get("peer_gap_df"),
                pagination=12,
            )
            _render_nicegui_table_section(
                "Anemometer & NTF Provenance Audit",
                bundle.get("anemometer_ntf_audit"),
                pagination=12,
            )


def _serve_nicegui(
    bundle: SuiteBundle,
    host: str,
    port: int,
    *,
    show: bool = False,
    root: str | Path = ".",
) -> None:
    """Run the commercial 3-workspace reactive NiceGUI SCADA & Folium Geo-Workbench."""
    safe_host = _ensure_loopback_host(host)
    from nicegui import ui

    store = WorkbenchStore.load_or_build(root=root)
    farms = store.farms() or ["kelmarsh", "penmanshiel"]
    default_farm = "kelmarsh" if "kelmarsh" in farms else farms[0]
    init_turbines = store.turbines(default_farm) or ["01"]

    @ui.page("/")
    def _index() -> None:
        state: dict[str, Any] = {
            "farm": default_farm,
            "turbine_id": init_turbines[0],
            "period_a": "2017 Full Year",
            "custom_start": "2017-01-01",
            "custom_end": "2017-12-31",
            "period_b": "None (Single Period)",
            "basemap_tile": "Satellite (Esri)",
            "show_rotor_rings": True,
            "show_wake_corridors": True,
            "ws_mode": "svenningsen",
            "rho_ref": STANDARD_DENSITY,
            "ref_mode": "mpc",
            "mask_mode": "clean_only",
            "webgl_res": "stratified",
            "debounce_sec": 60,
            "active_tab": "tab1",
            "cms_temp_col": "gear_oil_temp_c",
            "selected_episode_idx": 0,
            "code_search": "",
            "subsystem_filter": "All",
            "iec_filter": "All",
        }
        cockpit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        map_slot_ref: dict[str, Any] = {}
        syncing_controls = False

        ui.query("body").classes("bg-slate-50")
        with ui.header().classes("bg-slate-900 text-white items-center justify-between px-4 py-2"):
            with ui.row().classes("items-center gap-3"):
                ui.label("WINTELLISYT v2").classes("text-base font-bold tracking-wide")
                ui.label("Geospatial Cockpit & 360° Single-Turbine Diagnostic Chain").classes(
                    "text-xs text-slate-300"
                )
            header_scope = ui.label(
                f"{len(store.telemetry_df):,} telemetry rows | {len(store.events_df):,} events | loopback={safe_host}:{port}"
            ).classes("text-xs text-slate-300")

        # Streamlined Command Bar + Collapsible Physics/Masking Settings
        with ui.card().classes("w-full bg-white border-b border-slate-200 rounded-none shadow-none px-4 py-2"):
            with ui.row().classes("w-full items-end gap-2.5 flex-wrap"):
                farm_opts = {
                    f: f"{f.title()} ({'6x MM92' if f == 'kelmarsh' else '14x MM82'})"
                    if f in ("kelmarsh", "penmanshiel")
                    else f.title()
                    for f in farms
                }
                farm_sel = ui.select(farm_opts, label="Wind Farm", value=state["farm"]).classes("min-w-[165px]")
                turb_sel = ui.select(
                    {t: f"WT-{t}" for t in init_turbines},
                    label="Target Turbine",
                    value=state["turbine_id"],
                ).classes("min-w-[210px]")
                period_a_sel = ui.select(
                    [*_PERIOD_PRESETS.keys(), "Custom Range"],
                    label="Primary Period",
                    value=state["period_a"],
                ).classes("min-w-[155px]")
                with ui.row().classes("items-end gap-2") as custom_range_box:
                    start_inp = ui.input(label="Start (UTC)", value=state["custom_start"]).classes("w-[108px]")
                    end_inp = ui.input(label="End (UTC)", value=state["custom_end"]).classes("w-[108px]")
                custom_range_box.set_visibility(state["period_a"] == "Custom Range")

                period_b_sel = ui.select(
                    list(_COMPARE_PRESETS.keys()),
                    label="Compare Period (Delta-t)",
                    value=state["period_b"],
                ).classes("min-w-[160px]")
                basemap_sel = ui.select(
                    list(BASEMAP_TILES.keys()),
                    label="Folium Basemap",
                    value=state["basemap_tile"],
                ).classes("min-w-[190px]")
                apply_btn = ui.button("Apply").props("color=primary dense")

            with ui.expansion("Physics, Air Density & Event Masking Settings", icon="tune").classes(
                "w-full mt-1 bg-slate-50 border border-slate-200 rounded text-xs"
            ):
                with ui.row().classes("w-full items-end gap-2.5 flex-wrap p-1"):
                    ws_sel = ui.select(_WS_MODE_LABELS, label="Air Density Mode", value=state["ws_mode"]).classes(
                        "min-w-[195px]"
                    )
                    rho_num = ui.number(
                        label="rho_ref (kg/m3)",
                        value=float(state["rho_ref"]),
                        min=0.90,
                        max=1.45,
                        step=0.005,
                        format="%.3f",
                    ).classes("w-[110px]")
                    ref_sel = ui.select(
                        {
                            "mpc": "OEM Contractual MPC",
                            "learned": "Greenbyte Learned PC",
                            "peer": "Peer Reference PC",
                        },
                        label="Reference Curve",
                        value=state["ref_mode"],
                    ).classes("min-w-[175px]")
                    mask_sel = ui.select(
                        {
                            "clean_only": "Clean Normal Only",
                            "all_data": "All Points Colored",
                            "unmasked": "Unmasked Raw",
                        },
                        label="Event Masking",
                        value=state["mask_mode"],
                    ).classes("min-w-[160px]")
                    webgl_sel = ui.select(
                        {"stratified": "Stratified WebGL (~8k)", "full": "Full Resolution"},
                        label="WebGL Scatter",
                        value=state["webgl_res"],
                    ).classes("min-w-[160px]")
                    debounce_num = ui.number(
                        label="Debounce (s)",
                        value=float(state["debounce_sec"]),
                        min=0,
                        max=3600,
                        step=30,
                        format="%.0f",
                    ).classes("w-[95px]")

        kpi_strip = ui.row().classes("w-full gap-2 px-4 pt-3 flex-wrap")

        def _jump_to_turbine(target_tid: str, switch_to_chain: bool = True) -> None:
            nonlocal syncing_controls
            syncing_controls = True
            state["turbine_id"] = target_tid
            turb_sel.value = target_tid
            turb_sel.update()
            if switch_to_chain:
                state["active_tab"] = "tab2"
                tab_bar.set_value(tab_widgets["tab2"])
            syncing_controls = False
            _refresh_all(sync_turbines=False)

        def _refresh_tab_only() -> None:
            _build_workbench_tab_body(
                str(state["active_tab"]),
                tab_body,
                store,
                bundle,
                state,
                cockpit_cache,
                _refresh_tab_only,
                _jump_to_turbine,
                map_slot_ref=map_slot_ref,
            )

        def _refresh_all(*, sync_turbines: bool = False) -> None:
            nonlocal syncing_controls
            cockpit = _get_cached_cockpit(store, bundle, state, cockpit_cache)
            tids = store.turbines(str(state["farm"])) or ["01"]

            syncing_controls = True
            turb_sel.options = _format_turbine_options(cockpit, tids)
            if sync_turbines or state["turbine_id"] not in tids:
                if state["turbine_id"] not in tids:
                    state["turbine_id"] = tids[0]
            turb_sel.value = state["turbine_id"]
            turb_sel.update()
            syncing_controls = False

            kpis = cockpit.get("kpis", {})
            header_scope.set_text(
                f"{state['farm'].title()} WT-{state['turbine_id']} | {state['period_a']} | "
                f"{_WS_MODE_LABELS.get(state['ws_mode'], '')} (rho_ref={float(state['rho_ref']):.3f}) | "
                f"loopback={safe_host}:{port}"
            )
            if kpis:
                _render_metric_cards(
                    kpi_strip,
                    [
                        (
                            "Fleet Export Energy",
                            f"{kpis.get('export_gwh', 0.0)} GWh",
                            f"{kpis.get('n_turbines', 0)} turbines | Theo {kpis.get('theoretical_gwh', 0.0)} GWh",
                        ),
                        (
                            "Fleet Capacity Factor (CF)",
                            f"{kpis.get('fleet_net_cf_pct', 0.0)}% Net CF",
                            f"Theo CF: {kpis.get('fleet_theoretical_cf_pct', 0.0)}% ({kpis.get('fleet_cf_gap_pp', 0.0):+.2f} pp) | Underperf: {kpis.get('underperforming_count', 0)}",
                        ),
                        (
                            "Fleet Clean PPR",
                            f"{kpis.get('fleet_clean_ppr_pct', 0.0)}%",
                            f"Downtime {kpis.get('downtime_mwh', 0.0)} MWh | Perf {kpis.get('perf_loss_mwh', 0.0)} MWh",
                        ),
                        (
                            "IEC B.2.2 vs Event Mask",
                            f"{kpis.get('fleet_iec_b22_pct', 0.0)}%",
                            f"Prod B.2.2: {kpis.get('fleet_prod_iec_b22_pct', 0.0)}% | Mask: {kpis.get('fleet_event_mask_avail_pct', 0.0)}%",
                        ),
                        (
                            "Suspect Over-Reads (>8%)",
                            f"{kpis.get('suspect_overread_count', 0)}",
                            f"Active Mask={state['mask_mode']}",
                        ),
                    ],
                )
            else:
                _render_metric_cards(kpi_strip, [("No Telemetry", "0 rows", f"{state['farm']} {state['period_a']}")])

            _refresh_tab_only()

        def _on_tab_change(e: Any) -> None:
            raw_val = getattr(e, "value", e)
            val = (
                str(raw_val._props.get("name", "tab1"))
                if hasattr(raw_val, "_props")
                else str(raw_val or "tab1")
            )
            if val not in dict(_WORKBENCH_TABS) or val == state["active_tab"]:
                return
            state["active_tab"] = val
            _refresh_tab_only()

        with ui.tabs(on_change=_on_tab_change).classes(
            "w-full px-4 mt-2 bg-white border-b border-slate-200"
        ) as tab_bar:
            tab_widgets = {key: ui.tab(key, label=lbl) for key, lbl in _WORKBENCH_TABS}
        tab_bar.set_value(tab_widgets["tab1"])

        tab_body = ui.column().classes("w-full px-4 pb-8 gap-3")

        def _on_basemap_change() -> None:
            if syncing_controls:
                return
            state["basemap_tile"] = str(basemap_sel.value or "Satellite (Esri)")
            if state["active_tab"] == "tab1" and map_slot_ref.get("widget") and map_slot_ref.get("renderer"):
                map_slot_ref["widget"].set_content(map_slot_ref["renderer"]())

        def _commit_inputs_and_refresh(*, sync_turbines: bool = False) -> None:
            state["farm"] = str(farm_sel.value or default_farm)
            if not sync_turbines:
                state["turbine_id"] = str(turb_sel.value or state["turbine_id"])
            state["period_a"] = str(period_a_sel.value or "2017 Full Year")
            custom_range_box.set_visibility(state["period_a"] == "Custom Range")
            state["custom_start"] = str(start_inp.value or "2017-01-01")
            state["custom_end"] = str(end_inp.value or "2017-12-31")
            state["period_b"] = str(period_b_sel.value or "None (Single Period)")
            state["basemap_tile"] = str(basemap_sel.value or "Satellite (Esri)")
            state["ws_mode"] = str(ws_sel.value or "svenningsen")
            state["rho_ref"] = float(rho_num.value) if rho_num.value is not None else STANDARD_DENSITY
            state["ref_mode"] = str(ref_sel.value or "mpc")
            state["mask_mode"] = str(mask_sel.value or "clean_only")
            state["webgl_res"] = str(webgl_sel.value or "stratified")
            state["debounce_sec"] = int(debounce_num.value) if debounce_num.value is not None else 60
            state["selected_episode_idx"] = 0
            _refresh_all(sync_turbines=sync_turbines)

        farm_sel.on_value_change(lambda: None if syncing_controls else _commit_inputs_and_refresh(sync_turbines=True))
        basemap_sel.on_value_change(_on_basemap_change)
        for ctrl in (turb_sel, period_a_sel, period_b_sel, ws_sel, ref_sel, mask_sel, webgl_sel):
            ctrl.on_value_change(lambda: None if syncing_controls else _commit_inputs_and_refresh())
        apply_btn.on_click(lambda: _commit_inputs_and_refresh())

        with tab_body:
            with ui.row().classes("items-center gap-2 p-4"):
                ui.spinner(size="sm")
                ui.label("Loading geospatial asset cockpit & 360° diagnostic chain...").classes("text-xs text-slate-500")

        ui.timer(0.05, lambda: _refresh_all(), once=True)

    ui.run(
        host=safe_host,
        port=port,
        reload=False,
        show=show,
        title="WINTELLISYT v2 Wind Engineering & Asset Intelligence",
    )


def serve_dashboard(
    suite_dir: str | Path = "artifacts/engineering_suite_2017",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    backend: str = "auto",
    show: bool = False,
    root: str | Path = ".",
) -> None:
    """Launch the interactive dashboard via NiceGUI if installed, else Starlette + Uvicorn."""
    safe_host = _ensure_loopback_host(host)
    nicegui_available = importlib.util.find_spec("nicegui") is not None
    if backend == "nicegui" and not nicegui_available:
        raise ImportError(
            "The 'nicegui' package is not installed in the active Python environment. "
            "Install 'nicegui' separately or run with '--backend auto' / '--backend starlette'."
        )
    if backend in ("auto", "nicegui") and nicegui_available:
        bundle = load_suite_bundle(suite_dir, root=root)
        _serve_nicegui(bundle, host=safe_host, port=port, show=show, root=root)
        return

    import uvicorn

    if backend == "auto":
        print(
            "Notice: 'nicegui' is not installed in this environment; starting the built-in "
            f"Starlette + Uvicorn interactive dashboard at http://{safe_host}:{port}/",
            flush=True,
        )
    app = create_app(suite_dir, root=root)
    uvicorn.run(app, host=safe_host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WINTELLISYT Engineering Dashboard & HTML Exporter")
    parser.add_argument("--suite-dir", default="artifacts/engineering_suite_2017")
    parser.add_argument("--export-only", action="store_true", help="Write standalone interactive HTML without starting a server.")
    parser.add_argument("--output", default=None, help="Output HTML path when --export-only is set.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", choices=["auto", "nicegui", "starlette"], default="auto")
    parser.add_argument("--show", action="store_true", help="Automatically open the local dashboard URL in the browser.")
    args = parser.parse_args(argv)

    if args.export_only:
        written = export_standalone_html(args.suite_dir, args.output)
        print(f"Exported standalone interactive dashboard to {written}")
        return 0

    serve_dashboard(args.suite_dir, host=args.host, port=args.port, backend=args.backend, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
