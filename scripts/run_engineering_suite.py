"""Run the end-to-end WINTELLISYT Phase 2 & Phase 2.5 Wind Engineering & Asset Intelligence Suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wintellisyt.asset_intelligence import (
    analyze_drivetrain_and_thermal_cms,
    analyze_pitch_actuator_currents_and_temps,
    analyze_yaw_misalignment_and_cable_windings,
    compute_iec_energy_waterfall,
)
from wintellisyt.atmosphere import (
    benchmark_density_correction,
    compute_hub_density_series,
    load_era5_hourly,
)
from wintellisyt.data import load_rich_telemetry, load_static_layout, read_status_events
from wintellisyt.direction import (
    audit_vane_nacelle_consistency,
    cross_check_era5_direction,
    find_wake_trough_offset,
    fit_relative_consensus,
    wrap_deg,
)
from wintellisyt.pc_health import build_event_stop_mask, evaluate_pc_health_baselines
from wintellisyt.peer_reconstruction import (
    audit_anemometer_and_ntf,
    evaluate_all_turbine_peer_reconstruction,
)
from wintellisyt.pitch_health import (
    analyze_blade_pitch_asymmetry,
    evaluate_maintenance_pitch_and_ppr_recovery,
)
from wintellisyt.wake import classify_wake_exposure, evaluate_wake_regime_effects
from wintellisyt.wind_geometry import LayoutPoint

ROOT = Path(__file__).resolve().parents[1]

ARCHIVES = {
    "kelmarsh": {
        "2016": [ROOT / "data/raw/kelmarsh/Kelmarsh_SCADA_2016_3082.zip"],
        "2017": [ROOT / "data/raw/kelmarsh/Kelmarsh_SCADA_2017_3083.zip"],
        "model": "MM92",
    },
    "penmanshiel": {
        "2016": [ROOT / "data/raw/penmanshiel/Penmanshiel_SCADA_2016_WT11-15_3107.zip"],
        "2017": [
            ROOT / "data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT01-10_3114.zip",
            ROOT / "data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT11-15_3115.zip",
        ],
        "model": "MM82",
    },
}

TRAIN_CUTOFF_2017 = pd.Timestamp("2017-10-01", tz="UTC")
TEST_START_2017 = pd.Timestamp("2017-11-01", tz="UTC")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pivot_grid(df: pd.DataFrame, col: str, grid_idx: pd.DatetimeIndex, turbines: list[str]) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(np.nan, index=grid_idx, columns=turbines)
    wide = df.pivot(index="timestamp_utc", columns="turbine_id", values=col)
    return wide.reindex(index=grid_idx, columns=turbines)


def _attach_hub_density(df: pd.DataFrame, farm: str, meta: dict, era5_df: pd.DataFrame, era5_meta: dict) -> pd.DataFrame:
    """Attach turbine-specific hub air density using ERA5 + SCADA nacelle ambient temperature."""
    era5_elev = float(era5_meta.get("returned_elevation_m") or 0.0)
    parts: list[pd.DataFrame] = []
    for tid, grp in df.groupby("turbine_id"):
        m = meta[tid]
        target_elev = float(m["elevation_m"]) + float(m["hub_height_m"])
        ts_idx = pd.DatetimeIndex(grp["timestamp_utc"])
        temp_s = grp.set_index("timestamp_utc")["ambient_temperature_c"]
        dens = compute_hub_density_series(
            ts_idx,
            temp_s,
            era5_df,
            era5_elevation_m=era5_elev,
            target_elevation_m=target_elev,
        )
        out = grp.copy()
        out["density_moist_kg_m3"] = dens["density_moist_kg_m3"].to_numpy()
        out["density_dry_kg_m3"] = dens["density_dry_kg_m3"].to_numpy()
        out["hub_pressure_hpa"] = dens["hub_pressure_hpa"].to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def run_suite(output_dir: Path, *, overwrite: bool = False) -> dict:
    """Execute full Phase 2 & Phase 2.5 analyses across Kelmarsh and Penmanshiel."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    oem_payload = json.loads((ROOT / "data/reference/oem_power_curves.json").read_text(encoding="utf-8"))
    grid_2017 = pd.date_range("2017-01-01 00:00", "2017-12-31 23:50", freq="10min", tz="UTC")

    tables: dict[str, list[pd.DataFrame]] = {
        "layout": [],
        "anemometer_ntf_audit": [],
        "direction_consensus": [],
        "direction_monthly": [],
        "direction_wake_troughs": [],
        "vane_nacelle_audit": [],
        "wind_rose": [],
        "wake_exposure_counts": [],
        "wake_bin_effects": [],
        "wake_regime_effects": [],
        "pc_health_summary": [],
        "pc_health_daily": [],
        "pitch_asymmetry": [],
        "pitch_ppr_recovery": [],
        "peer_gap_benchmark": [],
        "peer_drift_summary": [],
        "density_benchmark": [],
        "energy_waterfall": [],
        "mpc_vs_learned_gap": [],
        "yaw_misalignment_summary": [],
        "yaw_misalignment_bins": [],
        "pitch_actuator_health": [],
        "drivetrain_cms_summary": [],
        "wake_vibration_fatigue": [],
    }
    farm_summaries: dict[str, dict] = {}

    for farm, cfg in ARCHIVES.items():
        print(f"=== Processing {farm.title()} (2017 + 2016) ===", flush=True)
        meta = load_static_layout(farm, ROOT)
        layout = {
            tid: LayoutPoint(m["latitude"], m["longitude"], m["rotor_diameter_m"])
            for tid, m in meta.items()
        }
        rotor_diam = {tid: m["rotor_diameter_m"] for tid, m in meta.items()}

        df17, dup17 = load_rich_telemetry(cfg["2017"], farm, year=2017)
        df16, dup16 = load_rich_telemetry(cfg["2016"], farm, year=2016)
        events17 = read_status_events(cfg["2017"], farm, year=2017)
        events16 = read_status_events(cfg["2016"], farm, year=2016)

        # Clean invalid wind / direction bounds
        df17["wind_speed_ms"] = df17["wind_speed_ms"].where(df17["wind_speed_ms"] >= 0.0)
        df17["wind_direction_deg"] = (
            df17["wind_direction_deg"].where(df17["wind_direction_deg"].between(0.0, 360.0)) % 360.0
        )

        turbines = sorted(meta.keys(), key=lambda s: int(s))
        for tid in turbines:
            tables["layout"].append(pd.DataFrame([meta[tid]]))

        # Pivoted 10-min grids
        ws_wide = _pivot_grid(df17, "wind_speed_ms", grid_2017, turbines)
        std_wide = _pivot_grid(df17, "wind_speed_std_ms", grid_2017, turbines)
        power_wide = _pivot_grid(df17, "active_power_kw", grid_2017, turbines)
        dir_wide = _pivot_grid(df17, "wind_direction_deg", grid_2017, turbines)
        rpm_wide = _pivot_grid(df17, "rotor_speed_rpm", grid_2017, turbines)
        pitch_a_wide = _pivot_grid(df17, "pitch_a_deg", grid_2017, turbines)

        # Event stop masks
        stop_cols = {}
        left_censored_stops = {}
        for tid in turbines:
            sm, lc_cnt = build_event_stop_mask(grid_2017, events17, tid)
            stop_cols[tid] = sm
            left_censored_stops[tid] = lc_cnt
        stop_mask_wide = pd.DataFrame(stop_cols, index=grid_2017)
        stop_long = stop_mask_wide.stack().rename("is_stop")
        stop_long.index.names = ["timestamp_utc", "turbine_id"]

        # Map stop_mask back to df17 row index
        df17_stop_series = (
            df17.set_index(["timestamp_utc", "turbine_id"])
            .index.map(stop_long)
            .fillna(False)
        )
        stop_series_df17 = pd.Series(df17_stop_series.to_numpy(dtype=bool), index=df17.index)

        # 1. Anemometer & NTF Audit
        print(f"  [{farm}] Auditing anemometers & NTF provenance...", flush=True)
        ntf_df = audit_anemometer_and_ntf(df17)
        ntf_df.insert(0, "farm_id", farm)
        tables["anemometer_ntf_audit"].append(ntf_df)

        # 2. NREL 2-Stage Direction Calibration & True North Anchoring
        print(f"  [{farm}] Running NREL 2-stage direction & wake-trough True North...", flush=True)
        vectors, rel_dir_wide, peer_dir_wide, dir_sum, dir_mon = fit_relative_consensus(
            dir_wide, layout, TRAIN_CUTOFF_2017, TEST_START_2017
        )
        troughs_df, tn_summary = find_wake_trough_offset(
            ws_wide, power_wide, rel_dir_wide, vectors, layout
        )
        true_north_offset = float(tn_summary.get("common_true_north_offset_deg", 0.0))
        calibrated_dir_wide = (rel_dir_wide - true_north_offset) % 360.0
        calibrated_peer_wide = (peer_dir_wide - true_north_offset) % 360.0

        # Load ERA5 & cross-check 100m wind direction
        era5_df, era5_meta = load_era5_hourly(farm, ROOT / "data/external/era5")
        era5_dir_check = cross_check_era5_direction(
            calibrated_dir_wide, ws_wide, era5_df["wind_direction_100m"]
        )
        dir_sum.insert(0, "farm_id", farm)
        dir_sum["true_north_offset_deg"] = true_north_offset
        dir_sum["total_offset_deg"] = [
            round(float(wrap_deg(v + true_north_offset)), 2) for v in dir_sum["relative_offset_deg"]
        ]
        dir_sum["left_censored_stop_events"] = [left_censored_stops[t] for t in dir_sum["turbine_id"]]
        tables["direction_consensus"].append(dir_sum)
        if not dir_mon.empty:
            dir_mon.insert(0, "farm_id", farm)
            tables["direction_monthly"].append(dir_mon)
        if not troughs_df.empty:
            troughs_df.insert(0, "farm_id", farm)
            tables["direction_wake_troughs"].append(troughs_df)

        vane_df = audit_vane_nacelle_consistency(df17)
        if not vane_df.empty:
            vane_df.insert(0, "farm_id", farm)
            tables["vane_nacelle_audit"].append(vane_df)

        # Wind rose summary (30-deg bins x speed bands)
        for tid in turbines:
            for basis, d_ser in (("raw", dir_wide[tid]), ("calibrated", calibrated_dir_wide[tid])):
                rose = pd.DataFrame(
                    {
                        "direction_deg": np.floor(((d_ser + 15.0) % 360.0) / 30.0) * 30.0,
                        "speed_band": pd.cut(
                            ws_wide[tid], [0.0, 5.0, 10.0, 15.0, np.inf], right=False
                        ).astype(str),
                    }
                ).dropna()
                cnts = rose.value_counts().reset_index(name="n")
                cnts.insert(0, "farm_id", farm)
                cnts.insert(1, "turbine_id", tid)
                cnts.insert(2, "basis", basis)
                tables["wind_rose"].append(cnts)

        # 3. Active-Donor Wake Stratification & Knee Physics
        print(f"  [{farm}] Analyzing active-donor wake exposure, TI & knee physics...", flush=True)
        exposure_wide = classify_wake_exposure(
            calibrated_peer_wide, power_wide, rpm_wide, layout
        )
        for tid in turbines:
            vc = exposure_wide[tid].value_counts().reset_index()
            vc.columns = ["exposure_class", "n"]
            vc.insert(0, "farm_id", farm)
            vc.insert(1, "turbine_id", tid)
            tables["wake_exposure_counts"].append(vc)

        w_bins, w_regimes = evaluate_wake_regime_effects(
            ws_wide, std_wide, power_wide, exposure_wide, stop_mask_wide, layout,
            train_cutoff=TRAIN_CUTOFF_2017,
        )
        if not w_bins.empty:
            w_bins.insert(0, "farm_id", farm)
            tables["wake_bin_effects"].append(w_bins)
        if not w_regimes.empty:
            w_regimes.insert(0, "farm_id", farm)
            tables["wake_regime_effects"].append(w_regimes)

        # 4. Event-Screened PC Health (EWMA/CUSUM with 7-day hold & 3 baselines)
        print(f"  [{farm}] Running event-screened PC Health (3 baselines)...", flush=True)
        oem_knots = oem_payload["curves"][cfg["model"]]["knots_1ms"]
        pc_daily, pc_sum = evaluate_pc_health_baselines(
            ws_wide, power_wide, stop_mask_wide, oem_knots,
            train_cutoff=TRAIN_CUTOFF_2017, cal_end=TEST_START_2017,
        )
        if not pc_daily.empty:
            pc_daily.insert(0, "farm_id", farm)
            tables["pc_health_daily"].append(pc_daily)
        if not pc_sum.empty:
            pc_sum.insert(0, "farm_id", farm)
            tables["pc_health_summary"].append(pc_sum)

        # 5. Blade Pitch Asymmetry & Pre/Post Repair Recovery
        print(f"  [{farm}] Evaluating pitch asymmetry & maintenance recovery...", flush=True)
        asym_df = analyze_blade_pitch_asymmetry(df17, stop_series_df17)
        if not asym_df.empty:
            asym_df.insert(0, "farm_id", farm)
            tables["pitch_asymmetry"].append(asym_df)

        rec_df = evaluate_maintenance_pitch_and_ppr_recovery(
            ws_wide, power_wide, pitch_a_wide, stop_mask_wide, events17
        )
        if not rec_df.empty:
            rec_df.insert(0, "farm_id", farm)
            tables["pitch_ppr_recovery"].append(rec_df)

        # 6. All-Turbine Peer Reconstruction
        print(f"  [{farm}] Running all-turbine unwaked-donor MCP reconstruction...", flush=True)
        gaps_df, drift_df = evaluate_all_turbine_peer_reconstruction(
            ws_wide, power_wide, calibrated_dir_wide, exposure_wide,
            train_cutoff=TRAIN_CUTOFF_2017, cal_end=TEST_START_2017,
        )
        if not gaps_df.empty:
            gaps_df.insert(0, "farm_id", farm)
            tables["peer_gap_benchmark"].append(gaps_df)
        if not drift_df.empty:
            drift_df.insert(0, "farm_id", farm)
            tables["peer_drift_summary"].append(drift_df)

        # 7. ERA5 Air Density Benchmark (2016 train -> 2017 test)
        print(f"  [{farm}] Benchmarking IEC vs WindPRO density correction with ERA5...", flush=True)
        df16_d = _attach_hub_density(df16, farm, meta, era5_df, era5_meta)
        df17_d = _attach_hub_density(df17, farm, meta, era5_df, era5_meta)
        grid_2016 = pd.date_range("2016-01-01 00:00", "2016-12-31 23:50", freq="10min", tz="UTC")
        stop16_wide = pd.DataFrame(
            {tid: build_event_stop_mask(grid_2016, events16, tid)[0] for tid in set(df16["turbine_id"])},
            index=grid_2016,
        )
        dens_bench = benchmark_density_correction(df16_d, df17_d, stop16_wide, stop_mask_wide)
        if not dens_bench.empty:
            dens_bench.insert(0, "farm_id", farm)
            tables["density_benchmark"].append(dens_bench)

        # Phase 2.5 Actionable Insights Suite
        print(f"  [{farm}] Running Phase 2.5 Asset Intelligence suite...", flush=True)
        wf_df, gap_pc_df = compute_iec_energy_waterfall(df17)
        wf_df.insert(0, "farm_id", farm)
        tables["energy_waterfall"].append(wf_df)
        if not gap_pc_df.empty:
            gap_pc_df.insert(0, "farm_id", farm)
            tables["mpc_vs_learned_gap"].append(gap_pc_df)

        yaw_sum, yaw_bins = analyze_yaw_misalignment_and_cable_windings(
            df17, rotor_diam, stop_series_df17
        )
        if not yaw_sum.empty:
            yaw_sum.insert(0, "farm_id", farm)
            tables["yaw_misalignment_summary"].append(yaw_sum)
        if not yaw_bins.empty:
            yaw_bins.insert(0, "farm_id", farm)
            tables["yaw_misalignment_bins"].append(yaw_bins)

        act_df = analyze_pitch_actuator_currents_and_temps(df17, stop_series_df17)
        if not act_df.empty:
            act_df.insert(0, "farm_id", farm)
            tables["pitch_actuator_health"].append(act_df)

        cms_df, vib_df = analyze_drivetrain_and_thermal_cms(df17, exposure_wide, stop_series_df17)
        if not cms_df.empty:
            cms_df.insert(0, "farm_id", farm)
            tables["drivetrain_cms_summary"].append(cms_df)
        if not vib_df.empty:
            vib_df.insert(0, "farm_id", farm)
            tables["wake_vibration_fatigue"].append(vib_df)

        farm_summaries[farm] = {
            "model": cfg["model"],
            "turbines": turbines,
            "rows_2017": int(len(df17)),
            "rows_2016": int(len(df16)),
            "quarantined_duplicates_2017": dup17,
            "quarantined_duplicates_2016": dup16,
            "events_2017_unique": int(len(events17)),
            "true_north_anchoring": tn_summary,
            "era5_direction_cross_check": era5_dir_check,
            "era5_manifest_sha256": _file_sha256(ROOT / f"data/external/era5/{farm}.json"),
        }

    # Write CSV tables and manifest
    output_hashes: dict[str, str] = {}
    for name, parts in tables.items():
        non_empty = [p.dropna(axis=1, how="all") for p in parts if not p.empty]
        if not non_empty:
            continue
        merged = pd.concat(non_empty, ignore_index=True)
        csv_path = output_dir / f"{name}.csv"
        merged.to_csv(csv_path, index=False)
        output_hashes[csv_path.name] = _file_sha256(csv_path)

    manifest = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "train_cutoff_2017": TRAIN_CUTOFF_2017.isoformat(),
        "test_start_2017": TEST_START_2017.isoformat(),
        "farms": farm_summaries,
        "output_hashes": output_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Suite completed! Wrote {len(output_hashes)} CSV tables + manifest.json to {output_dir}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "artifacts/engineering_suite_2017")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_suite(args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
