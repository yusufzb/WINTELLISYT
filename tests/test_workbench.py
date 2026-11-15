"""Unit and integration tests for the WINTELLISYT v2 Parquet store, engine, and WebGL figures."""

from __future__ import annotations

import html
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wintellisyt.atmosphere import reference_speed
from wintellisyt.data.workbench_store import (
    WorkbenchStore,
    build_event_code_dictionary,
    classify_rds_pp_subsystem,
    debounce_events,
    normalize_turbine_id,
    vector_iec_wind_speed,
    vector_svenningsen_fast,
)
from wintellisyt.ui.data_loader import SuiteBundle
from wintellisyt.ui.geo_map import build_folium_fleet_map
from wintellisyt.ui.workbench_calc import (
    compute_chronic_trends_and_avail_check,
    compute_direction_yaw_wake,
    compute_drivetrain_pitch_mcp,
    compute_fleet_cockpit,
    compute_method_of_bins,
    compute_outage_pareto_workbench,
    compute_turbine_deep_dive,
    stratified_webgl_sample,
    synthesize_turbine_root_cause,
)
from wintellisyt.ui.workbench_figures import (
    fig_availability_crosscheck,
    fig_chronic_ppr_ewma_cusum,
    fig_cms_thermal_and_vibration,
    fig_direction_northing_and_wake_trough,
    fig_dual_power_curve_scattergl,
    fig_event_gantt_and_recovery,
    fig_event_pareto_suite,
    fig_fleet_completeness_heatmap,
    fig_fleet_deviation_fingerprint,
    fig_fleet_energy_and_avail_waterfall,
    fig_fleet_spatial_map,
    fig_knee_pitch_rpm_diagnostic,
    fig_peer_mcp_and_cup_drift,
    fig_pitch_asymmetry_and_motors,
    fig_site_cockpit_companion,
    fig_ti_stratified_delta_p,
    fig_turbine_unified_timeline,
    fig_turbine_unified_ws_diagnostics,
    fig_wake_donor_active_vs_stopped,
    fig_yaw_parabola_and_cable_windings,
)


def _synthetic_store() -> tuple[WorkbenchStore, SuiteBundle]:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2017-01-01 00:00", periods=2500, freq="10min", tz="UTC")
    rows: list[dict] = []
    for tid in ("01", "02"):
        ws = np.clip(rng.normal(8.2, 2.8, len(idx)), 1.0, 20.0)
        mpc = np.clip(2050.0 * ((ws - 3.0) / 9.5) ** 2.2, 0.0, 2050.0)
        pwr = np.where(ws < 3.0, -5.0, mpc * (0.98 if tid == "01" else 0.90) + rng.normal(0.0, 25.0, len(idx)))
        stop_flag = np.zeros(len(idx), dtype=bool)
        stop_flag[100:112] = True
        pwr[stop_flag] = 0.0
        curt_flag = np.zeros(len(idx), dtype=bool)
        curt_flag[500:520] = True
        ws[500:520] = 11.0
        mpc[500:520] = 2000.0
        pwr[500:520] = 1000.0

        std_ws = ws * 0.12
        ti = std_ws / np.clip(ws, 1.0, None)
        wd = rng.uniform(0.0, 360.0, len(idx))
        vane = rng.normal(1.5 if tid == "01" else -3.5, 4.0, len(idx))
        for i, ts in enumerate(idx):
            state = (
                "Event Outage (Stop)"
                if stop_flag[i]
                else ("Curtailment / Derate" if curt_flag[i] else ("Under-Cut-In / Calm" if ws[i] < 3.0 else "Clean Normal"))
            )
            rows.append(
                {
                    "farm_id": "kelmarsh",
                    "turbine_id": tid,
                    "timestamp_utc": ts,
                    "year": 2017,
                    "month": int(ts.month),
                    "state_category": state,
                    "event_stop_mask": bool(stop_flag[i]),
                    "event_warning_mask": False,
                    "iec_unavail_mask": bool(stop_flag[i]),
                    "curtailment_mask": bool(curt_flag[i]),
                    "in_geometric_wake": bool(210.0 <= wd[i] <= 240.0),
                    "wind_speed_ms": float(ws[i]),
                    "density_adjusted_ws_ms": float(ws[i] * 0.99),
                    "wind_speed_std_ms": float(std_ws[i]),
                    "wind_speed_s1_ms": float(ws[i] + (0.05 if tid == "01" else 0.0)),
                    "wind_speed_s2_ms": float(ws[i] - (0.20 if tid == "01" else 0.0)),
                    "ti": float(ti[i]),
                    "density_moist_kg_m3": 1.205,
                    "density_dry_kg_m3": 1.208,
                    "hub_pressure_hpa": 995.0,
                    "density_publisher_implied_kg_m3": 1.205,
                    "ws_iec_ms": float(ws[i] * (1.205 / 1.225) ** (1.0 / 3.0)),
                    "ws_svenningsen_ms": float(ws[i] * (1.205 / 1.225) ** (1.0 / 3.0)),
                    "active_power_kw": float(pwr[i]),
                    "available_capacity_kw": 1000.0 if curt_flag[i] else 2050.0,
                    "potential_power_mpc_kw": float(mpc[i]),
                    "potential_power_learned_kw": float(mpc[i] * 0.98),
                    "potential_power_ref_kw": float(mpc[i] * 0.97),
                    "potential_power_default_kw": float(mpc[i]),
                    "energy_theoretical_kwh": float(mpc[i] / 6.0),
                    "energy_export_kwh": float(max(0.0, pwr[i]) / 6.0),
                    "energy_import_kwh": 0.2 if stop_flag[i] else 0.0,
                    "lost_contractual_kwh": float(mpc[i] / 6.0) if stop_flag[i] else 0.0,
                    "lost_time_iec_b22_kwh": float(mpc[i] / 6.0) if stop_flag[i] else 0.0,
                    "lost_prod_iec_b22_kwh": float(mpc[i] / 6.0) if stop_flag[i] else 0.0,
                    "lost_downtime_kwh": float(mpc[i] / 6.0) if stop_flag[i] else 0.0,
                    "lost_performance_kwh": float(max(0.0, mpc[i] - pwr[i]) / 6.0) if not stop_flag[i] else 0.0,
                    "lost_curtailment_total_kwh": 0.0,
                    "performance_index": 0.98 if tid == "01" else 0.90,
                    "production_factor": 0.98,
                    "data_availability": 1.0,
                    "avail_time_contractual": 0.0 if stop_flag[i] else 1.0,
                    "avail_time_iec_b22": 0.0 if stop_flag[i] else 1.0,
                    "avail_time_iec_b23": 0.0 if stop_flag[i] else 1.0,
                    "avail_time_iec_b24": 1.0,
                    "avail_time_iec_b32": 0.0 if stop_flag[i] else 1.0,
                    "avail_prod_iec_b22": 0.0 if stop_flag[i] else 1.0,
                    "avail_prod_iec_b23": 0.0 if stop_flag[i] else 1.0,
                    "avail_prod_iec_b32": 0.0 if stop_flag[i] else 1.0,
                    "avail_time_system": 0.0 if stop_flag[i] else 1.0,
                    "avail_prod_system": 0.0 if stop_flag[i] else 1.0,
                    "wind_direction_deg": float(wd[i]),
                    "wd_calibrated_deg": float(wd[i]),
                    "nacelle_position_deg": float(wd[i]),
                    "vane_position_deg": float(vane[i]),
                    "vane_position_std_deg": 2.0,
                    "cable_windings": float(np.sin(i / 200.0)),
                    "rotor_speed_rpm": float(np.clip(ws[i] * 1.3, 0.0, 15.0)),
                    "generator_rpm": float(np.clip(ws[i] * 110.0, 0.0, 1500.0)),
                    "pitch_a_deg": 0.5 if ws[i] < 11.0 else float((ws[i] - 11.0) * 2.2),
                    "pitch_b_deg": 0.5 if ws[i] < 11.0 else float((ws[i] - 11.0) * 2.2),
                    "pitch_c_deg": 0.0 if tid == "02" and ws[i] < 11.0 else (0.5 if ws[i] < 11.0 else float((ws[i] - 11.0) * 2.2)),
                    "pitch_spread_deg": 0.5 if tid == "02" else 0.0,
                    "motor_current_1_a": 1.8,
                    "motor_current_2_a": 1.9,
                    "motor_current_3_a": 4.5 if tid == "02" else 1.8,
                    "motor_temp_1_c": 35.0,
                    "motor_temp_2_c": 35.5,
                    "motor_temp_3_c": 44.0 if tid == "02" else 35.2,
                    "drivetrain_accel_mmss": 1.2,
                    "tower_accel_x_mmss": 2.1,
                    "tower_accel_y_mmss": 2.3,
                    "gear_oil_inlet_bar": 2.1,
                    "gear_oil_pump_bar": 4.5,
                    "metal_particle_count": 1.0,
                    "ambient_temperature_c": 12.0,
                    "rotor_bearing_temp_c": 32.0 + 0.008 * max(0.0, pwr[i]),
                    "front_bearing_temp_c": 30.0 + 0.007 * max(0.0, pwr[i]),
                    "rear_bearing_temp_c": 31.0 + 0.007 * max(0.0, pwr[i]),
                    "gear_oil_temp_c": 45.0 + 0.01 * max(0.0, pwr[i]) + (6.0 if tid == "02" else 0.0),
                    "gen_bearing_front_temp_c": 42.0 + 0.01 * max(0.0, pwr[i]),
                    "gen_bearing_rear_temp_c": 41.0 + 0.01 * max(0.0, pwr[i]),
                    "stator_temp_1_c": 52.0 + 0.015 * max(0.0, pwr[i]),
                    "transformer_temp_c": 48.0 + 0.012 * max(0.0, pwr[i]),
                }
            )

    tel_df = pd.DataFrame(rows)
    ev_df = pd.DataFrame(
        [
            {
                "farm_id": "kelmarsh",
                "turbine_id": "01",
                "timestamp_start": idx[100],
                "timestamp_end": idx[106],
                "duration_raw": "01:00:00",
                "status": "Stop",
                "code": "692",
                "message": "Pitch run-away (hub box v.>=4)",
                "comment": "",
                "service_contract_category": "Forced outage",
                "iec_category": "Forced outage",
                "open_ended": False,
                "left_censored": False,
                "subsystem": "Pitch System",
            },
            {
                "farm_id": "kelmarsh",
                "turbine_id": "01",
                "timestamp_start": idx[106] + pd.Timedelta(seconds=30),
                "timestamp_end": idx[112],
                "duration_raw": "00:59:30",
                "status": "Stop",
                "code": "715",
                "message": "Charging circuit pitch",
                "comment": "",
                "service_contract_category": "Forced outage",
                "iec_category": "Forced outage",
                "open_ended": False,
                "left_censored": False,
                "subsystem": "Pitch System",
            },
            {
                "farm_id": "kelmarsh",
                "turbine_id": "02",
                "timestamp_start": idx[100],
                "timestamp_end": idx[110],
                "duration_raw": "01:40:00",
                "status": "Stop",
                "code": "3110",
                "message": "Frequency converter error",
                "comment": "",
                "service_contract_category": "Forced outage",
                "iec_category": "Forced outage",
                "open_ended": False,
                "left_censored": False,
                "subsystem": "Converter & Electrical",
            },
        ]
    )
    dict_df = build_event_code_dictionary(ev_df, tel_df)
    lay_df = pd.DataFrame(
        [
            {
                "farm": "kelmarsh",
                "turbine_id": "01",
                "title": "Kelmarsh 1",
                "model": "MM92",
                "rated_power_kw": 2050.0,
                "hub_height_m": 78.5,
                "rotor_diameter_m": 92.0,
                "latitude": 52.4006,
                "longitude": -0.9471,
                "elevation_m": 145.6,
            },
            {
                "farm": "kelmarsh",
                "turbine_id": "02",
                "title": "Kelmarsh 2",
                "model": "MM92",
                "rated_power_kw": 2050.0,
                "hub_height_m": 78.5,
                "rotor_diameter_m": 92.0,
                "latitude": 52.4025,
                "longitude": -0.9495,
                "elevation_m": 156.6,
            },
        ]
    )
    store = WorkbenchStore(
        root=Path("."),
        cache_dir=Path("."),
        telemetry_df=tel_df,
        events_df=ev_df,
        event_dictionary_df=dict_df,
        layout_df=lay_df,
        manifest={"schema_version": 2},
    )
    bundle = SuiteBundle(
        suite_dir=Path("."),
        manifest={"schema_version": 2},
        tables={
            "direction_consensus": pd.DataFrame(
                [
                    {"farm_id": "kelmarsh", "turbine_id": "01", "relative_offset_deg": 0.5, "total_offset_deg": -0.9, "raw_peer_mae_deg": 2.8, "relative_peer_mae_deg": 2.7},
                    {"farm_id": "kelmarsh", "turbine_id": "02", "relative_offset_deg": -0.5, "total_offset_deg": -1.9, "raw_peer_mae_deg": 2.4, "relative_peer_mae_deg": 2.3},
                ]
            ),
            "pitch_ppr_recovery": pd.DataFrame(
                [
                    {"farm_id": "kelmarsh", "turbine_id": "01", "duration_hours": 2.0, "delta_ppr": 0.035, "message": "Pitch run-away", "outcome": "improved_ppr"}
                ]
            ),
        },
    )
    return store, bundle


class WorkbenchTests(unittest.TestCase):
    def test_subsystem_classification_and_debouncing(self):
        self.assertEqual(normalize_turbine_id("1"), "01")
        self.assertEqual(classify_rds_pp_subsystem("692", "Pitch run-away (hub box v.>=4)"), "Pitch System")
        self.assertEqual(classify_rds_pp_subsystem("3110", "Frequency converter error"), "Converter & Electrical")
        self.assertEqual(classify_rds_pp_subsystem("4510", "Tower oscillation Y level 1"), "Tower & Vibration")
        self.assertEqual(classify_rds_pp_subsystem("20", "Manual stop - on site"), "Manual & Scheduled Maintenance")

        store, _ = _synthetic_store()
        d = store.get_event_dictionary("kelmarsh")
        self.assertEqual(len(d), 3)
        deb_60 = debounce_events(store.events_df, gap_seconds=60)
        # WT-01's two stops are 30s apart so they merge under 60s debounce -> 2 total episodes across fleet
        self.assertEqual(len(deb_60), 2)
        deb_10 = debounce_events(store.events_df, gap_seconds=10)
        self.assertEqual(len(deb_10), 3)

    def test_vectorized_svenningsen_matches_scalar_reference_speed(self):
        ws = np.array([3.5, 6.0, 9.5, 11.2, 14.0, 19.5])
        rho = np.array([1.18, 1.20, 1.225, 1.25, 1.15, 1.28])
        fast = vector_svenningsen_fast(ws, rho, rho_ref=1.225)
        for v, r, f_val in zip(ws, rho, fast, strict=True):
            exact = reference_speed(float(v), float(r), maximum=35.0, reference_density=1.225)
            self.assertAlmostEqual(float(f_val), float(exact), places=5)
        iec = vector_iec_wind_speed(ws, rho, rho_ref=1.15)
        self.assertAlmostEqual(float(iec[0]), float(ws[0] * (rho[0] / 1.15) ** (1.0 / 3.0)), places=6)

    def test_method_of_bins_and_stratified_webgl_sampling(self):
        store, _ = _synthetic_store()
        df = store.slice_telemetry("kelmarsh", "01")
        bins = compute_method_of_bins(df[df["state_category"] == "Clean Normal"], df.loc[df["state_category"] == "Clean Normal", "ws_svenningsen_ms"], "active_power_kw", df.loc[df["state_category"] == "Clean Normal", "potential_power_mpc_kw"])
        self.assertFalse(bins.empty)
        # Bin centers must align to exact 0.5 m/s multiples
        self.assertTrue(np.allclose((bins["ws_bin"] * 2.0) % 1.0, 0.0))

        sampled = stratified_webgl_sample(df, max_clean=300, max_anomaly=200)
        self.assertLessEqual(len(sampled), 500)
        self.assertTrue((sampled["state_category"] == "Event Outage (Stop)").any())

    def test_all_6_tabs_calc_and_webgl_figures(self):
        store, bundle = _synthetic_store()

        t1 = compute_fleet_cockpit(store, bundle, "kelmarsh")
        self.assertEqual(t1["kpis"]["n_turbines"], 2)
        self.assertIn("fleet_net_cf_pct", t1["kpis"])
        self.assertIn("fleet_theoretical_cf_pct", t1["kpis"])
        fp_df = t1["fingerprint_df"]
        for col in ("net_cf_pct", "theoretical_cf_pct", "cf_gap_pp", "perf_status", "east_m", "north_m"):
            self.assertIn(col, fp_df.columns)
        self.assertEqual(fp_df.loc[fp_df["turbine_id"] == "02", "perf_status"].iloc[0], "Underperforming")
        spatial_fig = fig_fleet_spatial_map(t1)
        self.assertIsInstance(spatial_fig, go.Figure)
        self.assertGreaterEqual(len(spatial_fig.data), 4)
        self.assertIsInstance(fig_fleet_deviation_fingerprint(t1), go.Figure)
        self.assertIsInstance(fig_fleet_energy_and_avail_waterfall(t1), go.Figure)
        self.assertIsInstance(fig_fleet_completeness_heatmap(t1), go.Figure)

        t2 = compute_turbine_deep_dive(
            store,
            "kelmarsh",
            "01",
            ("2017-01-01", "2017-01-10"),
            ("2017-01-11", "2017-01-18"),
            mask_mode="all_data",
        )
        self.assertFalse(t2["empty"])
        self.assertIn("net_cf_pct", t2["m1_summary"])
        self.assertIn("theoretical_cf_pct", t2["m1_summary"])
        dual_fig = fig_dual_power_curve_scattergl(t2, period_a_label="Jan 1-10", period_b_label="Jan 11-18")
        self.assertTrue(any(isinstance(tr, go.Scattergl) for tr in dual_fig.data))
        self.assertIsInstance(fig_ti_stratified_delta_p(t2, period_b_label="Jan 11-18"), go.Figure)
        self.assertIsInstance(fig_knee_pitch_rpm_diagnostic(t2), go.Figure)

        t3 = compute_chronic_trends_and_avail_check(store, "kelmarsh", "02")
        self.assertFalse(t3["empty"])
        self.assertIsNotNone(t3["summary"]["avail_corr_mask_vs_iec_b22"])
        self.assertIsNotNone(t3["summary"]["mean_net_cf_pct"])
        self.assertIsNotNone(t3["summary"]["mean_theoretical_cf_pct"])
        self.assertIsInstance(fig_chronic_ppr_ewma_cusum(t3), go.Figure)
        self.assertIsInstance(fig_availability_crosscheck(t3), go.Figure)

        t4 = compute_direction_yaw_wake(store, bundle, "kelmarsh", "02")
        self.assertFalse(t4["empty"])
        self.assertIsInstance(fig_direction_northing_and_wake_trough(t4), go.Figure)
        self.assertIsInstance(fig_yaw_parabola_and_cable_windings(t4), go.Figure)
        self.assertIsInstance(fig_wake_donor_active_vs_stopped(t4), go.Figure)

        t5 = compute_outage_pareto_workbench(store, bundle, "kelmarsh", "01", debounce_sec=60)
        self.assertEqual(t5["raw_stop_count"], 3)
        self.assertEqual(t5["debounced_stop_count"], 2)
        self.assertIsInstance(fig_event_pareto_suite(t5), go.Figure)
        self.assertIsInstance(fig_event_gantt_and_recovery(t5), go.Figure)

        t6 = compute_drivetrain_pitch_mcp(store, bundle, "kelmarsh", "02")
        self.assertFalse(t6["empty"])
        cms_fig = fig_cms_thermal_and_vibration(t6)
        self.assertTrue(any(isinstance(tr, go.Scattergl) for tr in cms_fig.data))
        self.assertIsInstance(fig_pitch_asymmetry_and_motors(t6), go.Figure)
        self.assertIsInstance(fig_peer_mcp_and_cup_drift(t6), go.Figure)

    def test_folium_geo_map_and_single_screen_diagnostic_chain(self):
        store, bundle = _synthetic_store()
        cockpit = compute_fleet_cockpit(store, bundle, "kelmarsh")
        deep = compute_turbine_deep_dive(store, "kelmarsh", "02", ("2017-01-01", "2017-01-18"))
        trend = compute_chronic_trends_and_avail_check(store, "kelmarsh", "02")
        dir_data = compute_direction_yaw_wake(store, bundle, "kelmarsh", "02")
        ev_data = compute_outage_pareto_workbench(store, bundle, "kelmarsh", "02")
        cms_data = compute_drivetrain_pitch_mcp(store, bundle, "kelmarsh", "02")

        map_iframe = build_folium_fleet_map(
            cockpit,
            default_tile="Satellite (Esri)",
            show_wake_corridors=True,
            show_rotor_rings=True,
            selected_turbine_id="02",
        )
        self.assertIn("<iframe", map_iframe)
        unescaped = html.unescape(map_iframe)
        for token in ("Esri", "OpenTopoMap", "CartoDB", "OpenStreetMap", "WT-01", "WT-02", "Net / Theo CF", "L.control.layers"):
            self.assertIn(token, unescaped)

        verdict = synthesize_turbine_root_cause(cockpit, deep, trend, dir_data, ev_data, cms_data)
        self.assertEqual(verdict["severity"], "error")
        self.assertEqual(len(verdict["findings"]), 5)
        self.assertIn("KEL-02", verdict["headline"])

        comp_fig = fig_site_cockpit_companion(cockpit)
        ws_fig = fig_turbine_unified_ws_diagnostics(deep)
        tl_fig = fig_turbine_unified_timeline(trend, ev_data)
        for fig in (comp_fig, ws_fig, tl_fig):
            self.assertIsInstance(fig, go.Figure)
            self.assertGreater(len(fig.data), 2)
        self.assertEqual(ws_fig.layout.hovermode, "x unified")
        self.assertEqual(tl_fig.layout.hovermode, "x unified")
        # Shared x-axes lock subplot 1 & 2 onto bottom x3 axis
        self.assertEqual(ws_fig.layout.xaxis.matches, "x3")
        self.assertEqual(ws_fig.layout.xaxis2.matches, "x3")
        self.assertEqual(tl_fig.layout.xaxis.matches, "x3")
        self.assertEqual(tl_fig.layout.xaxis2.matches, "x3")

    def test_parquet_roundtrip_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Build minimal raw structure matching Kelmarsh ZIP & static CSV + OEM JSON
            (root / "data/raw/kelmarsh").mkdir(parents=True)
            (root / "data/reference").mkdir(parents=True)
            (root / "data/raw/kelmarsh/Kelmarsh_WT_static.csv").write_text(
                "Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date\n"
                "Kelmarsh,Kelmarsh 1,KWF1,SEN,Senvion,MM92,2050,78.5,92,52.4,-0.94,145.6,UK,15/04/2016\n",
                encoding="utf-8",
            )
            (root / "data/reference/oem_power_curves.json").write_text(
                json.dumps({"curves": {"MM92": {"cut_in_ms": 3.0, "knots_1ms": {"0": 0.0, "3": 20.0, "10": 1794.0, "20": 2050.0}}}}),
                encoding="utf-8",
            )
            scada_csv = (
                "# Format version: 1\n"
                "# Time zone: UTC\n"
                "# Date and time,Wind speed (m/s),Power (kW),Time-based IEC B.2.2 (Users View),Nacelle ambient temperature (°C)\n"
                "2017-05-01 00:00:00,8.0,1000.0,1.0,10.0\n"
                "2017-05-01 00:10:00,8.5,0.0,0.0,10.0\n"
            )
            status_csv = (
                "Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category\n"
                "2017-05-01 00:10:00,2017-05-01 00:20:00,00:10:00,Stop,692,Pitch run-away,,Forced outage,Forced outage\n"
            )
            zpath = root / "data/raw/kelmarsh/Kelmarsh_SCADA_2017_3083.zip"
            with ZipFile(zpath, "w") as zf:
                zf.writestr("Turbine_Data_Kelmarsh_1_2017-05-01_-_2017-05-02_000.csv", scada_csv)
                zf.writestr("Status_Kelmarsh_1_2017-05-01_-_2017-05-02_000.csv", status_csv)

            store1 = WorkbenchStore.load_or_build(root=root, farms=("kelmarsh",), years=(2017,), rebuild=True)
            self.assertEqual(len(store1.telemetry_df), 2)
            self.assertTrue(bool(store1.telemetry_df.iloc[1]["event_stop_mask"]))
            self.assertEqual(store1.event_dictionary_df.iloc[0]["subsystem"], "Pitch System")

            # Reload from disk without rebuild
            store2 = WorkbenchStore.load_or_build(root=root, farms=("kelmarsh",), years=(2017,), rebuild=False)
            self.assertEqual(len(store2.slice_telemetry("kelmarsh", "01")), 2)


if __name__ == "__main__":
    unittest.main()
