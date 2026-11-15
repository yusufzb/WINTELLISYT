"""Contract tests for Greenbyte ZIP ingestion, status event deduplication, UI loader, and Starlette app."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd

from wintellisyt.data import load_rich_telemetry, read_status_events
from wintellisyt.ui import build_dashboard_figures, create_app, export_standalone_html, load_suite_bundle


class ScadaAndSuiteContractsTest(unittest.TestCase):
    def test_rich_scada_quarantines_duplicates_and_status_deduplicates(self):
        scada_csv = (
            "# Format version: 1\n"
            "# Time zone: UTC\n"
            '# Date and time,Wind speed (m/s),Power (kW),"Blade angle (pitch position) A (\u00b0)","Motor current axis 1 (A)"\n'
            "2017-05-01 00:00:00,7.5,820.0,0.5,2.1\n"
            "2017-05-01 00:00:00,7.5,820.0,0.5,2.1\n"
            "2017-05-01 00:10:00,8.1,960.0,0.2,2.2\n"
        )
        status_csv = (
            "# Format version: 1\n"
            "Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category\n"
            "2017-05-01 00:00:00,2017-05-01 01:00:00,01:00:00,Stop,100,Pitch test,,Scheduled Maintenance,Scheduled Maintenance\n"
            "2017-05-01 00:00:00,2017-05-01 01:00:00,01:00:00,Stop,100,Pitch test,,Scheduled Maintenance,Scheduled Maintenance\n"
            "2017-05-02 00:00:00,,00:10:00,Stop,200,Open fault,,Forced outage,Forced outage\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "Kelmarsh_SCADA_2017_test.zip"
            with ZipFile(zpath, "w") as zf:
                zf.writestr("Turbine_Data_Kelmarsh_1_2017-05-01_-_2017-05-03_000.csv", scada_csv)
                zf.writestr("Status_Kelmarsh_1_2017-05-01_-_2017-05-03_000.csv", status_csv)

            df, quarantined = load_rich_telemetry(zpath, "kelmarsh", year=2017)
            self.assertEqual(quarantined, 2)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["turbine_id"], "1")
            self.assertAlmostEqual(float(df.iloc[0]["motor_current_1_a"]), 2.2, places=4)

            events = read_status_events(zpath, "kelmarsh", year=2017)
            self.assertEqual(len(events), 2)
            self.assertTrue(bool(events["open_ended"].iloc[1]))

    def test_ui_bundle_figures_html_export_and_starlette_routes(self):
        from starlette.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp) / "suite"
            suite_dir.mkdir(parents=True)
            (suite_dir / "manifest.json").write_text(json.dumps({"schema_version": 2, "farms": {}}), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "farm": "kelmarsh",
                        "turbine_id": "01",
                        "title": "Kelmarsh 1",
                        "model": "MM92",
                        "rated_power_kw": 2050.0,
                        "hub_height_m": 78.5,
                        "rotor_diameter_m": 92.0,
                        "latitude": 52.4,
                        "longitude": -0.95,
                        "elevation_m": 150.0,
                    }
                ]
            ).to_csv(suite_dir / "layout.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "farm_id": "kelmarsh",
                        "turbine_id": "01",
                        "relative_offset_deg": 2.0,
                        "test_n": 100,
                        "raw_peer_mae_deg": 5.5,
                        "relative_peer_mae_deg": 3.1,
                        "peer_count_mean": 5.0,
                        "resultant_mean": 0.98,
                        "true_north_offset_deg": -10.0,
                        "total_offset_deg": -8.0,
                        "left_censored_stop_events": 1,
                    }
                ]
            ).to_csv(suite_dir / "direction_consensus.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "farm_id": "kelmarsh",
                        "turbine_id": "01",
                        "energy_export_mwh": 5000.0,
                        "lost_downtime_mwh": 120.0,
                        "lost_curtailment_total_mwh": 35.0,
                        "lost_performance_mwh": 80.0,
                    }
                ]
            ).to_csv(suite_dir / "energy_waterfall.csv", index=False)

            bundle = load_suite_bundle(suite_dir, root=tmp)
            dash = build_dashboard_figures(bundle)
            self.assertEqual(dash["kpis"]["n_turbines"], 1)
            self.assertEqual(len(dash["tabs"]), 6)

            out_html = export_standalone_html(suite_dir, suite_dir / "out.html", root=tmp)
            self.assertTrue(out_html.exists())
            self.assertIn("WINTELLISYT", out_html.read_text(encoding="utf-8"))

            app = create_app(suite_dir, root=tmp)
            client = TestClient(app)
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/api/manifest").status_code, 200)
            self.assertEqual(client.get("/api/tables/direction_consensus").status_code, 200)
            self.assertEqual(client.get("/api/tables/missing_tbl").status_code, 404)

    def test_dashboard_rejects_non_loopback_hosts(self):
        from wintellisyt.ui.app import serve_dashboard

        for bad_host in ("0.0.0.0", "::", "192.168.1.50"):
            with self.assertRaisesRegex(ValueError, "loopback"):
                serve_dashboard("artifacts/engineering_suite_2017", host=bad_host, backend="nicegui")


if __name__ == "__main__":
    unittest.main()
