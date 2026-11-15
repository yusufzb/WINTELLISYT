"""Interactive Folium geospatial map builder for WINTELLISYT wind farm cockpits."""

from __future__ import annotations

import html
import math
from typing import Any

import folium
from folium import plugins
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

BASEMAP_TILES: dict[str, dict[str, str]] = {
    "Satellite (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    },
    "Topographic (OpenTopoMap)": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap",
    },
    "Light (CartoDB Positron)": {
        "tiles": "CartoDB positron",
        "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "Dark (CartoDB DarkMatter)": {
        "tiles": "CartoDB dark_matter",
        "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "Street (OpenStreetMap)": {
        "tiles": "OpenStreetMap",
        "attr": "&copy; OpenStreetMap contributors",
    },
}

_STATUS_COLORS: dict[str, str] = {
    "Underperforming": "#ef4444",
    "Suspect Over-Read (>+8%)": "#f97316",
    "Sub-Optimal / High Downtime": "#eab308",
    "Healthy / Nominal": "#22c55e",
}


def _fmt_num(val: Any, *, suffix: str = "", signed: bool = False, decimals: int = 2) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(num):
        return "N/A"
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
    return f"{fmt.format(num)}{suffix}"


def _build_turbine_popup_html(row: pd.Series, status_color: str) -> str:
    label = html.escape(str(row.get("label", f"WT-{row.get('turbine_id', '')}")))
    status = html.escape(str(row.get("perf_status", "Healthy / Nominal")))
    lat_str = _fmt_num(row.get("latitude"), decimals=5)
    lon_str = _fmt_num(row.get("longitude"), decimals=5)
    elev_str = _fmt_num(row.get("elevation_m"), suffix=" m", decimals=1)
    rotor_str = _fmt_num(row.get("rotor_diameter_m"), suffix=" m", decimals=1)
    net_cf = _fmt_num(row.get("net_cf_pct"), suffix="%", decimals=2)
    theo_cf = _fmt_num(row.get("theoretical_cf_pct"), suffix="%", decimals=2)
    cf_gap = _fmt_num(row.get("cf_gap_pp"), suffix=" pp", signed=True, decimals=2)
    exp_mwh = _fmt_num(row.get("export_mwh"), suffix=" MWh", decimals=1)
    theo_mwh = _fmt_num(row.get("theoretical_mwh"), suffix=" MWh", decimals=1)
    ppr_clean = _fmt_num(row.get("clean_ppr_pct"), suffix="%", decimals=2)
    dev_clean = _fmt_num(row.get("clean_dev_pct"), suffix="%", signed=True, decimals=2)
    dev_raw = _fmt_num(row.get("raw_dev_pct"), suffix="%", signed=True, decimals=2)
    dt_mwh = _fmt_num(row.get("downtime_mwh"), suffix=" MWh", decimals=1)
    perf_mwh = _fmt_num(row.get("perf_loss_mwh"), suffix=" MWh", decimals=1)
    b22_time = _fmt_num(row.get("avail_time_iec_b22_pct"), suffix="%", decimals=2)
    b22_prod = _fmt_num(row.get("avail_prod_iec_b22_pct"), suffix="%", decimals=2)
    evt_avail = _fmt_num(row.get("event_mask_avail_pct"), suffix="%", decimals=2)
    badges = html.escape(str(row.get("diagnostic_badges", "Nominal")))

    return (
        "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"
        "width:300px;font-size:12px;color:#0f172a;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"border-bottom:2px solid {status_color};padding-bottom:6px;margin-bottom:6px;'>"
        f"<strong style='font-size:14px;'>{label}</strong>"
        f"<span style='background:{status_color};color:#fff;padding:2px 7px;border-radius:10px;"
        f"font-size:10px;font-weight:700;'>{status}</span></div>"
        f"<div style='font-size:11px;color:#475569;margin-bottom:6px;'>"
        f"Lat/Lon: {lat_str}, {lon_str}<br>Elevation: {elev_str} | Rotor: {rotor_str}</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:11.5px;'>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Net / Theo CF</td>"
        f"<td style='text-align:right;font-weight:600;'>{net_cf} / {theo_cf} ({cf_gap})</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Export / Theo MWh</td>"
        f"<td style='text-align:right;font-weight:600;'>{exp_mwh} / {theo_mwh}</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Clean PPR (4-15 m/s)</td>"
        f"<td style='text-align:right;font-weight:600;'>{ppr_clean}</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Clean / Raw Dev</td>"
        f"<td style='text-align:right;font-weight:600;'>{dev_clean} / {dev_raw}</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Downtime / Perf Loss</td>"
        f"<td style='text-align:right;font-weight:600;'>{dt_mwh} / {perf_mwh}</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>IEC B.2.2 (Time / Prod)</td>"
        f"<td style='text-align:right;font-weight:600;'>{b22_time} / {b22_prod}</td></tr>"
        f"<tr><td style='padding:2px 0;color:#475569;'>Event-Mask Avail</td>"
        f"<td style='text-align:right;font-weight:600;'>{evt_avail}</td></tr>"
        "</table>"
        f"<div style='margin-top:6px;padding-top:5px;border-top:1px solid #e2e8f0;"
        f"font-size:11px;color:#334155;'><b>Flags:</b> {badges}</div></div>"
    )


def _add_site_hull_and_rings(
    fmap: folium.Map,
    geo_df: pd.DataFrame,
    *,
    show_rotor_rings: bool,
) -> None:
    fg = folium.FeatureGroup(name="Site Boundary & 1D Rotor Footprint", show=show_rotor_rings)
    coords = geo_df[["latitude", "longitude"]].to_numpy(dtype=float)
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            hull_pts = [(float(coords[i, 0]), float(coords[i, 1])) for i in hull.vertices]
            folium.Polygon(
                locations=hull_pts,
                color="#38bdf8",
                weight=2,
                dash_array="5, 5",
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.06,
                tooltip=f"Site Envelope ({len(coords)} turbines)",
            ).add_to(fg)
        except QhullError:
            pass

    for _, row in geo_df.iterrows():
        rotor_m = float(row["rotor_diameter_m"]) if pd.notna(row.get("rotor_diameter_m")) else 92.0
        clr = _STATUS_COLORS.get(str(row.get("perf_status", "")), "#22c55e")
        folium.Circle(
            location=(float(row["latitude"]), float(row["longitude"])),
            radius=rotor_m,
            color=clr,
            weight=1.4,
            fill=True,
            fill_color=clr,
            fill_opacity=0.10,
        ).add_to(fg)
    fg.add_to(fmap)


def _add_wake_corridors(
    fmap: folium.Map,
    geo_df: pd.DataFrame,
    *,
    show_wake_corridors: bool,
) -> None:
    fg = folium.FeatureGroup(name="Prevailing SW (225 deg) Wake Corridors", show=show_wake_corridors)
    rows = list(geo_df.itertuples(index=False))
    for i in range(len(rows)):
        r1 = rows[i]
        lat1, lon1 = float(r1.latitude), float(r1.longitude)
        rotor = float(getattr(r1, "rotor_diameter_m", 92.0) or 92.0)
        for j in range(i + 1, len(rows)):
            r2 = rows[j]
            lat2, lon2 = float(r2.latitude), float(r2.longitude)
            lat0 = math.radians(0.5 * (lat1 + lat2))
            d_north = (lat2 - lat1) * 111132.0
            d_east = (lon2 - lon1) * math.cos(lat0) * 111320.0
            dist_m = math.hypot(d_east, d_north)
            if dist_m <= 1.0 or dist_m > 950.0:
                continue
            bearing = (math.degrees(math.atan2(d_east, d_north)) + 360.0) % 360.0
            is_sw_aligned = abs(((bearing - 225.0 + 180.0) % 360.0) - 180.0) <= 35.0 or (
                abs(((bearing - 45.0 + 180.0) % 360.0) - 180.0) <= 35.0
            )
            lbl1 = str(getattr(r1, "label", r1.turbine_id))
            lbl2 = str(getattr(r2, "label", r2.turbine_id))
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color="#f59e0b" if is_sw_aligned else "#94a3b8",
                weight=2.4 if is_sw_aligned else 1.3,
                opacity=0.80 if is_sw_aligned else 0.45,
                dash_array="6, 6",
                tooltip=(
                    f"{lbl1} &lt;-&gt; {lbl2} | {dist_m:.0f} m ({dist_m / max(1.0, rotor):.1f}xD) | "
                    f"Bearing {bearing:.0f} deg" + (" [SW Wake Axis]" if is_sw_aligned else "")
                ),
            ).add_to(fg)
    fg.add_to(fmap)


def _add_turbine_markers(
    fmap: folium.Map,
    geo_df: pd.DataFrame,
    *,
    selected_turbine_id: str | None,
) -> None:
    fg = folium.FeatureGroup(name="Turbine Health & CF Markers", show=True)
    for _, row in geo_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        tid = str(row.get("turbine_id", ""))
        label = str(row.get("label", f"WT-{tid}"))
        status = str(row.get("perf_status", "Healthy / Nominal"))
        clr = _STATUS_COLORS.get(status, "#22c55e")

        cf_gap = abs(float(row["cf_gap_pp"])) if pd.notna(row.get("cf_gap_pp")) else 0.0
        dev_abs = abs(float(row["clean_dev_pct"])) if pd.notna(row.get("clean_dev_pct")) else 0.0
        radius = float(np.clip(max(cf_gap * 2.2, dev_abs * 1.8, 10.0), 11.0, 24.0))

        if selected_turbine_id is not None and tid == str(selected_turbine_id):
            folium.CircleMarker(
                location=(lat, lon),
                radius=radius + 6.0,
                color="#38bdf8",
                weight=3.0,
                fill=False,
                opacity=0.95,
            ).add_to(fg)

        popup_html = _build_turbine_popup_html(row, clr)
        net_cf_txt = _fmt_num(row.get("net_cf_pct"), suffix="%", decimals=1)
        dev_txt = _fmt_num(row.get("clean_dev_pct"), suffix="%", signed=True, decimals=1)
        ppr_txt = _fmt_num(row.get("clean_ppr_pct"), suffix="%", decimals=1)

        folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            color="#ffffff",
            weight=2.0,
            fill=True,
            fill_color=clr,
            fill_opacity=0.90,
            popup=folium.Popup(popup_html, max_width=325),
            tooltip=f"{label} ({status}) | Net CF {net_cf_txt} | Clean Dev {dev_txt}",
        ).add_to(fg)

        badge_html = (
            "<div style='pointer-events:none;transform:translate(-50%,-135%);"
            f"background:rgba(15,23,42,0.88);border:1.5px solid {clr};border-radius:6px;"
            "padding:2px 6px;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "font-size:10.5px;line-height:1.25;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.35);'>"
            f"<b>WT-{html.escape(tid)}</b> &middot; CF {net_cf_txt}<br>"
            f"<span style='color:#cbd5e1;'>Dev {dev_txt} | PPR {ppr_txt}</span>"
            "</div>"
        )
        folium.Marker(
            location=(lat, lon),
            icon=folium.DivIcon(
                icon_size=(0, 0),
                icon_anchor=(0, 0),
                html=badge_html,
            ),
        ).add_to(fg)
    fg.add_to(fmap)


def _legend_overlay_html(farm_title: str) -> str:
    safe_title = html.escape(farm_title)
    return f"""
    <div style="position:fixed;bottom:14px;right:14px;z-index:9999;
                background:rgba(15,23,42,0.90);color:#f8fafc;
                padding:8px 11px;border-radius:8px;border:1px solid #334155;
                font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
                font-size:11px;line-height:1.45;box-shadow:0 4px 12px rgba(0,0,0,0.35);">
      <div style="font-weight:700;margin-bottom:4px;color:#e2e8f0;">{safe_title} Status</div>
      <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#ef4444;margin-right:6px;"></span>Underperforming</div>
      <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#eab308;margin-right:6px;"></span>High Downtime / CF Gap</div>
      <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#f97316;margin-right:6px;"></span>Suspect Over-Read (&gt;+8%)</div>
      <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#22c55e;margin-right:6px;"></span>Healthy / Nominal</div>
      <div style="margin-top:3px;color:#94a3b8;font-size:10px;">Radius ~ |CF Gap / Dev| &bull; Dashed = &le;10D pairs</div>
    </div>
    """


def build_folium_fleet_map(
    cockpit: dict[str, Any],
    *,
    default_tile: str = "Satellite (Esri)",
    show_wake_corridors: bool = True,
    show_rotor_rings: bool = True,
    selected_turbine_id: str | None = None,
    height_px: int = 520,
) -> str:
    """Build an interactive multi-tile Folium geospatial wind farm map wrapped in a responsive iframe."""
    fp = cockpit.get("fingerprint_df", pd.DataFrame())
    if fp.empty or "latitude" not in fp.columns or "longitude" not in fp.columns:
        return (
            f"<div style='height:{height_px}px;display:flex;align-items:center;justify-content:center;"
            "background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;color:#475569;font-size:13px;'>"
            "No WGS84 turbine coordinates available for the active filter selection.</div>"
        )

    geo_df = fp.dropna(subset=["latitude", "longitude"]).copy()
    if geo_df.empty:
        return (
            f"<div style='height:{height_px}px;display:flex;align-items:center;justify-content:center;"
            "background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;color:#475569;font-size:13px;'>"
            "No valid WGS84 coordinates recorded in layout metadata.</div>"
        )

    active_tile = default_tile if default_tile in BASEMAP_TILES else "Satellite (Esri)"
    lat_mean = float(geo_df["latitude"].mean())
    lon_mean = float(geo_df["longitude"].mean())

    fmap = folium.Map(
        location=[lat_mean, lon_mean],
        zoom_start=14,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    for tile_name, tile_cfg in BASEMAP_TILES.items():
        folium.TileLayer(
            tiles=tile_cfg["tiles"],
            attr=tile_cfg["attr"],
            name=tile_name,
            overlay=False,
            control=True,
            show=(tile_name == active_tile),
        ).add_to(fmap)

    plugins.Fullscreen(position="topleft").add_to(fmap)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="meters").add_to(fmap)
    plugins.MiniMap(toggle_display=True, minimized=True).add_to(fmap)

    _add_site_hull_and_rings(fmap, geo_df, show_rotor_rings=show_rotor_rings)
    _add_wake_corridors(fmap, geo_df, show_wake_corridors=show_wake_corridors)
    _add_turbine_markers(fmap, geo_df, selected_turbine_id=selected_turbine_id)

    min_lat, max_lat = float(geo_df["latitude"].min()), float(geo_df["latitude"].max())
    min_lon, max_lon = float(geo_df["longitude"].min()), float(geo_df["longitude"].max())
    lat_pad = max(0.0025, 0.25 * (max_lat - min_lat))
    lon_pad = max(0.0040, 0.25 * (max_lon - min_lon))
    fmap.fit_bounds([[min_lat - lat_pad, min_lon - lon_pad], [max_lat + lat_pad, max_lon + lon_pad]])

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    farm_title = str(cockpit.get("kpis", {}).get("farm", "Wind Farm"))
    root_elem: Any = fmap.get_root()
    root_elem.html.add_child(folium.Element(_legend_overlay_html(farm_title)))

    rendered = fmap.get_root().render()
    escaped_srcdoc = html.escape(rendered, quote=True)
    return (
        f'<iframe srcdoc="{escaped_srcdoc}" '
        f'style="width:100%;height:{height_px}px;border:1px solid #cbd5e1;border-radius:8px;background:#0f172a;" '
        'allowfullscreen webkitallowfullscreen mozallowfullscreen loading="lazy"></iframe>'
    )
