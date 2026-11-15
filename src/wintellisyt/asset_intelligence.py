"""Phase 2.5 Actionable Wind Asset Intelligence Suite (IEC 61400-26, Yaw, Pitch Motor, CMS)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CURTAILMENT_SUBCOLUMNS = {
    "grid": "lost_curtailment_grid_kwh",
    "noise": "lost_curtailment_noise_kwh",
    "shadow": "lost_curtailment_shadow_kwh",
    "bats": "lost_curtailment_bats_kwh",
    "birds": "lost_curtailment_birds_kwh",
    "ice": "lost_curtailment_ice_kwh",
    "sector_management": "lost_curtailment_sector_kwh",
    "technical": "lost_curtailment_technical_kwh",
    "marketing": "lost_curtailment_marketing_kwh",
    "boat_action": "lost_curtailment_boat_kwh",
    "grid_constraint": "lost_curtailment_grid_constraint_kwh",
}


def compute_iec_energy_waterfall(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile IEC 61400-26 theoretical, MPC/contractual, learned PC, export, and lost MWh.

    Returns `(waterfall_summary_df, mpc_vs_learned_gap_df)`.
    """
    summary_rows: list[dict] = []
    gap_rows: list[dict] = []

    for turbine_id, grp in df.groupby("turbine_id"):
        def sum_mwh(col: str) -> float:
            return float(grp[col].fillna(0.0).sum() / 1000.0) if col in grp.columns else 0.0

        export_mwh = sum_mwh("energy_export_kwh")
        import_mwh = sum_mwh("energy_import_kwh")
        theoretical_mwh = sum_mwh("energy_theoretical_kwh")
        mpc_mwh = float(grp["potential_power_mpc_kw"].fillna(0.0).sum() / 6000.0) if "potential_power_mpc_kw" in grp.columns else 0.0
        learned_mwh = float(grp["potential_power_learned_kw"].fillna(0.0).sum() / 6000.0) if "potential_power_learned_kw" in grp.columns else 0.0
        downtime_mwh = sum_mwh("lost_downtime_kwh")
        perf_loss_mwh = sum_mwh("lost_performance_kwh")
        curtail_total_mwh = sum_mwh("lost_curtailment_total_kwh")
        curtail_breakdown = {k: round(sum_mwh(c), 3) for k, c in CURTAILMENT_SUBCOLUMNS.items()}

        row = {
            "turbine_id": turbine_id,
            "rows": int(len(grp)),
            "energy_export_mwh": round(export_mwh, 2),
            "parasitic_import_mwh": round(import_mwh, 2),
            "net_energy_mwh": round(export_mwh - import_mwh, 2),
            "energy_theoretical_mwh": round(theoretical_mwh, 2),
            "potential_mpc_mwh": round(mpc_mwh, 2),
            "potential_learned_mwh": round(learned_mwh, 2),
            "mpc_minus_learned_mwh": round(mpc_mwh - learned_mwh, 2),
            "lost_downtime_mwh": round(downtime_mwh, 2),
            "lost_curtailment_total_mwh": round(curtail_total_mwh, 2),
            "lost_performance_mwh": round(perf_loss_mwh, 2),
            "unexplained_residual_mwh": round(
                theoretical_mwh - (export_mwh + downtime_mwh + curtail_total_mwh + perf_loss_mwh), 2
            ),
            "mean_performance_index": round(float(grp["performance_index"].mean()), 4)
            if "performance_index" in grp.columns and grp["performance_index"].notna().any()
            else None,
            **{f"curtail_{k}_mwh": v for k, v in curtail_breakdown.items()},
        }
        summary_rows.append(row)

        # Bin-by-bin Contractual MPC vs Learned PC vs Actual Power gap
        if {"wind_speed_ms", "potential_power_mpc_kw", "potential_power_learned_kw", "active_power_kw"}.issubset(grp.columns):
            sub = grp.dropna(subset=["wind_speed_ms", "potential_power_mpc_kw", "potential_power_learned_kw", "active_power_kw"]).copy()
            sub = sub[sub["wind_speed_ms"].between(3.0, 16.0) & (sub["active_power_kw"] > 10.0)]
            if not sub.empty:
                sub["speed_bin"] = np.floor(sub["wind_speed_ms"] * 2.0 + 0.5) / 2.0
                binned = sub.groupby("speed_bin")[
                    ["potential_power_mpc_kw", "potential_power_learned_kw", "active_power_kw"]
                ].agg(["median", "count"])
                for sb in binned.index:
                    cnt = int(binned.loc[sb, ("active_power_kw", "count")])
                    if cnt < 20:
                        continue
                    mpc_val = float(binned.loc[sb, ("potential_power_mpc_kw", "median")])
                    lrn_val = float(binned.loc[sb, ("potential_power_learned_kw", "median")])
                    act_val = float(binned.loc[sb, ("active_power_kw", "median")])
                    gap_rows.append(
                        {
                            "turbine_id": turbine_id,
                            "speed_bin_ms": float(sb),
                            "n": cnt,
                            "mpc_kw": round(mpc_val, 1),
                            "learned_pc_kw": round(lrn_val, 1),
                            "actual_median_kw": round(act_val, 1),
                            "learned_minus_mpc_kw": round(lrn_val - mpc_val, 1),
                        }
                    )

    return pd.DataFrame(summary_rows), pd.DataFrame(gap_rows)


def analyze_yaw_misalignment_and_cable_windings(
    df: pd.DataFrame,
    rotor_diameter_by_turbine: dict[str, float],
    stop_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate static yaw misalignment (cos^x loss) via Vane 1+2 and audit cable windings.

    Returns `(yaw_summary_df, yaw_binned_cp_df)`.
    """
    summary_rows: list[dict] = []
    bin_rows: list[dict] = []
    work = df.copy()
    if stop_mask is not None:
        work = work[~stop_mask.reindex(work.index, fill_value=False)]

    for turbine_id, grp in work.groupby("turbine_id"):
        d_m = rotor_diameter_by_turbine.get(turbine_id)
        if not d_m:
            continue
        area = np.pi * (d_m / 2.0) ** 2

        # Cable windings & untwist audit (from raw unfiltered group if possible, else work)
        raw_grp = df[df["turbine_id"] == turbine_id]
        if "cable_windings" in raw_grp.columns and raw_grp["cable_windings"].notna().any():
            cw = raw_grp["cable_windings"].dropna()
            max_abs_turns = float(cw.abs().max())
            untwist_steps = int((cw.diff().abs() > 0.5).sum())
        else:
            max_abs_turns = None
            untwist_steps = 0

        if not {"vane_position_deg", "wind_speed_ms", "active_power_kw"}.issubset(grp.columns):
            continue
        sub = grp.dropna(subset=["vane_position_deg", "wind_speed_ms", "active_power_kw"]).copy()
        sub = sub[sub["wind_speed_ms"].between(5.5, 10.5) & (sub["active_power_kw"] > 50.0)]
        # Wrap vane error into [-180, 180) and focus on operating yaw band [-25, +25]
        vane_err = ((sub["vane_position_deg"] + 180.0) % 360.0) - 180.0
        sub["vane_err_deg"] = vane_err
        sub["cp"] = (sub["active_power_kw"] * 1000.0) / (0.5 * 1.225 * area * sub["wind_speed_ms"] ** 3)
        sub["p_ratio"] = (
            sub["active_power_kw"] / sub["potential_power_mpc_kw"]
            if "potential_power_mpc_kw" in sub.columns
            else sub["cp"] / sub["cp"].median()
        )
        band = sub[sub["vane_err_deg"].between(-22.0, 22.0) & sub["p_ratio"].between(0.2, 1.5)].copy()
        if len(band) < 300:
            continue

        band["yaw_bin"] = np.round(band["vane_err_deg"] / 2.0) * 2.0
        binned = band.groupby("yaw_bin")[["cp", "p_ratio"]].agg(["median", "count"])
        supp = binned[binned[("p_ratio", "count")] >= 20]
        if len(supp) < 5:
            continue

        # Fit quadratic / cos^x peak in [-15, 15] to find optimal zero-offset gamma*
        x_deg = supp.index.to_numpy(dtype=float)
        y_rat = supp[("p_ratio", "median")].to_numpy(dtype=float)
        peak_bin_deg = float(x_deg[int(np.argmax(y_rat))])
        if len(x_deg) >= 5:
            coeffs = np.polyfit(x_deg, y_rat, 2)
            if coeffs[0] < -1e-5:
                vertex = float(np.clip(-coeffs[1] / (2.0 * coeffs[0]), -15.0, 15.0))
            else:
                vertex = peak_bin_deg
        else:
            vertex = peak_bin_deg

        mean_vane_deg = float(band["vane_err_deg"].median())
        effective_misalignment = mean_vane_deg - vertex
        # Cos^2 power-loss approximation over partial-load energy
        cos_loss_frac = max(0.0, 1.0 - float(np.cos(np.deg2rad(effective_misalignment)) ** 2))
        partial_mwh = float(band["active_power_kw"].sum() / 6000.0)
        est_recoverable_mwh = partial_mwh * cos_loss_frac

        summary_rows.append(
            {
                "turbine_id": turbine_id,
                "n_partial_load": int(len(band)),
                "median_vane_angle_deg": round(mean_vane_deg, 2),
                "optimal_cp_yaw_angle_deg": round(vertex, 2),
                "estimated_static_yaw_bias_deg": round(effective_misalignment, 2),
                "est_cos2_loss_pct": round(100.0 * cos_loss_frac, 3),
                "est_recoverable_mwh": round(est_recoverable_mwh, 2),
                "vane_std_median_deg": round(float(band["vane_position_std_deg"].median()), 2)
                if "vane_position_std_deg" in band.columns and band["vane_position_std_deg"].notna().any()
                else None,
                "max_cable_windings_turns": round(max_abs_turns, 2) if max_abs_turns is not None else None,
                "cable_windings_step_changes": untwist_steps,
            }
        )
        for yb in supp.index:
            bin_rows.append(
                {
                    "turbine_id": turbine_id,
                    "yaw_bin_deg": float(yb),
                    "n": int(supp.loc[yb, ("p_ratio", "count")]),
                    "median_cp": round(float(supp.loc[yb, ("cp", "median")]), 4),
                    "median_power_ratio": round(float(supp.loc[yb, ("p_ratio", "median")]), 4),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def analyze_pitch_actuator_currents_and_temps(
    df: pd.DataFrame,
    stop_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Detect pitch bearing friction & motor imbalance via 3-axis motor currents & temperatures."""
    work = df.copy()
    if stop_mask is not None:
        work = work[~stop_mask.reindex(work.index, fill_value=False)]

    rows: list[dict] = []
    for turbine_id, grp in work.groupby("turbine_id"):
        sub = grp[grp.get("active_power_kw", pd.Series(0.0, index=grp.index)) > 50.0].copy()
        if len(sub) < 100:
            continue

        cur_cols = [c for c in ("motor_current_1_a", "motor_current_2_a", "motor_current_3_a") if c in sub.columns]
        tmp_cols = [c for c in ("motor_temp_1_c", "motor_temp_2_c", "motor_temp_3_c") if c in sub.columns]

        cur_sub = sub[cur_cols].dropna() if len(cur_cols) == 3 else pd.DataFrame()
        tmp_sub = sub[tmp_cols].dropna() if len(tmp_cols) == 3 else pd.DataFrame()
        if cur_sub.empty and tmp_sub.empty:
            continue

        cur_spread = (cur_sub.max(axis=1) - cur_sub.min(axis=1)) if not cur_sub.empty else pd.Series(dtype=float)
        tmp_spread = (tmp_sub.max(axis=1) - tmp_sub.min(axis=1)) if not tmp_sub.empty else pd.Series(dtype=float)

        rows.append(
            {
                "turbine_id": turbine_id,
                "n_current": int(len(cur_sub)),
                "mean_axis1_current_a": round(float(cur_sub.iloc[:, 0].mean()), 3) if not cur_sub.empty else None,
                "mean_axis2_current_a": round(float(cur_sub.iloc[:, 1].mean()), 3) if not cur_sub.empty else None,
                "mean_axis3_current_a": round(float(cur_sub.iloc[:, 2].mean()), 3) if not cur_sub.empty else None,
                "p95_current_imbalance_a": round(float(cur_spread.quantile(0.95)), 3) if not cur_spread.empty else None,
                "n_temp": int(len(tmp_sub)),
                "median_axis1_temp_c": round(float(tmp_sub.iloc[:, 0].median()), 2) if not tmp_sub.empty else None,
                "median_axis2_temp_c": round(float(tmp_sub.iloc[:, 1].median()), 2) if not tmp_sub.empty else None,
                "median_axis3_temp_c": round(float(tmp_sub.iloc[:, 2].median()), 2) if not tmp_sub.empty else None,
                "p95_motor_temp_spread_c": round(float(tmp_spread.quantile(0.95)), 2) if not tmp_spread.empty else None,
                "flag_actuator_anomaly": bool(
                    (not cur_spread.empty and cur_spread.quantile(0.95) > 2.5)
                    or (not tmp_spread.empty and tmp_spread.quantile(0.95) > 8.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_drivetrain_and_thermal_cms(
    df: pd.DataFrame,
    exposure_wide: pd.DataFrame | None = None,
    stop_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze Gearbox oil dP, metal particle counts, wake-induced vibration, and thermal residuals.

    Returns `(cms_summary_df, wake_vibration_df)`.
    """
    work = df.copy()
    if stop_mask is not None:
        work = work[~stop_mask.reindex(work.index, fill_value=False)]

    if exposure_wide is not None:
        exp_long = exposure_wide.stack().rename("wake_exposure")
        exp_long.index.names = ["timestamp_utc", "turbine_id"]
        work = work.merge(exp_long.reset_index(), on=["timestamp_utc", "turbine_id"], how="left")

    # Fleet thermal baseline regression: T_comp ~ a * Power + b * T_amb + c
    thermal_cols = [
        c
        for c in (
            "gear_oil_temp_c",
            "rotor_bearing_temp_c",
            "gen_bearing_front_temp_c",
            "gen_bearing_rear_temp_c",
            "stator_temp_1_c",
            "front_bearing_temp_c",
            "rear_bearing_temp_c",
        )
        if c in work.columns and work[c].notna().sum() >= 500
    ]

    op = work[work["active_power_kw"] > 50.0].copy()
    thermal_residuals: dict[str, pd.Series] = {}
    for tcol in thermal_cols:
        sub = op[["active_power_kw", "ambient_temperature_c", tcol]].dropna()
        if len(sub) < 500:
            continue
        x = np.column_stack([sub["active_power_kw"] / 2050.0, sub["ambient_temperature_c"], np.ones(len(sub))])
        beta, *_ = np.linalg.lstsq(x, sub[tcol].to_numpy(dtype=float), rcond=None)
        x_all = np.column_stack([op["active_power_kw"] / 2050.0, op["ambient_temperature_c"], np.ones(len(op))])
        pred = x_all @ beta
        thermal_residuals[tcol] = op[tcol] - pred

    cms_rows: list[dict] = []
    vib_rows: list[dict] = []

    for turbine_id, grp in op.groupby("turbine_id"):
        row: dict = {"turbine_id": turbine_id, "n_operating": int(len(grp))}
        if {"gear_oil_pump_bar", "gear_oil_inlet_bar"}.issubset(grp.columns):
            dp = (grp["gear_oil_pump_bar"] - grp["gear_oil_inlet_bar"]).dropna()
            row["gear_oil_dp_median_bar"] = round(float(dp.median()), 3) if not dp.empty else None
            row["gear_oil_dp_p95_bar"] = round(float(dp.quantile(0.95)), 3) if not dp.empty else None
        if "metal_particle_count" in grp.columns and grp["metal_particle_count"].notna().any():
            mp = grp["metal_particle_count"].dropna()
            row["metal_particles_max"] = round(float(mp.max()), 1)
            row["metal_particles_mean"] = round(float(mp.mean()), 2)
        if "drivetrain_accel_mmss" in grp.columns and grp["drivetrain_accel_mmss"].notna().any():
            row["drivetrain_accel_median_mmss"] = round(float(grp["drivetrain_accel_mmss"].median()), 3)
            row["drivetrain_accel_p95_mmss"] = round(float(grp["drivetrain_accel_mmss"].quantile(0.95)), 3)
        if "tower_accel_x_mmss" in grp.columns and grp["tower_accel_x_mmss"].notna().any():
            row["tower_x_accel_p95_mmss"] = round(float(grp["tower_accel_x_mmss"].quantile(0.95)), 3)
        if "tower_accel_y_mmss" in grp.columns and grp["tower_accel_y_mmss"].notna().any():
            row["tower_y_accel_p95_mmss"] = round(float(grp["tower_accel_y_mmss"].quantile(0.95)), 3)

        max_thermal_excess = 0.0
        hot_component = None
        for tcol, res_s in thermal_residuals.items():
            t_res = res_s.loc[grp.index].dropna()
            if len(t_res) >= 100:
                med_r = float(t_res.median())
                row[f"res_{tcol}_median_c"] = round(med_r, 2)
                if med_r > max_thermal_excess:
                    max_thermal_excess = med_r
                    hot_component = tcol
        row["max_thermal_excess_c"] = round(max_thermal_excess, 2)
        row["hottest_component"] = hot_component
        cms_rows.append(row)

        # Wake-induced structural & drivetrain vibration fatigue comparison (5-12 m/s)
        if "wake_exposure" in grp.columns and "drivetrain_accel_mmss" in grp.columns:
            mid = grp[grp["wind_speed_ms"].between(5.0, 12.0)]
            for vcol in ("drivetrain_accel_mmss", "tower_accel_x_mmss", "tower_accel_y_mmss"):
                if vcol not in mid.columns or mid[vcol].notna().sum() < 200:
                    continue
                aw = mid.loc[mid["wake_exposure"] == "active_wake", vcol].dropna()
                fs = mid.loc[mid["wake_exposure"] == "freestream", vcol].dropna()
                if min(len(aw), len(fs)) >= 50:
                    vib_rows.append(
                        {
                            "turbine_id": turbine_id,
                            "vibration_channel": vcol,
                            "n_active_wake": int(len(aw)),
                            "n_freestream": int(len(fs)),
                            "active_wake_median": round(float(aw.median()), 3),
                            "freestream_median": round(float(fs.median()), 3),
                            "wake_vibration_uplift_pct": round(100.0 * (float(aw.median()) / max(float(fs.median()), 1e-6) - 1.0), 2),
                        }
                    )

    return pd.DataFrame(cms_rows), pd.DataFrame(vib_rows)
