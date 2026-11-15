"""All-turbine unwaked-donor MCP peer reconstruction, anemometer & NTF provenance audit."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_anemometer_and_ntf(df: pd.DataFrame) -> pd.DataFrame:
    """Audit Sensor 1 vs Sensor 2, reported wind speed selection/NTF, and publisher density adjustment."""
    rows: list[dict] = []
    if "turbine_id" not in df.columns:
        return pd.DataFrame()

    for turbine_id, grp in df.groupby("turbine_id"):
        ws = grp.get("wind_speed_ms", pd.Series(dtype=float))
        s1 = grp.get("wind_speed_s1_ms", pd.Series(dtype=float))
        s2 = grp.get("wind_speed_s2_ms", pd.Series(dtype=float))
        den_ws = grp.get("density_adjusted_ws_ms", pd.Series(dtype=float))
        temp_c = grp.get("ambient_temperature_c", pd.Series(dtype=float))

        # 1. Compare reported_ws vs S1 and S2
        s12 = pd.DataFrame({"ws": ws, "s1": s1, "s2": s2}).dropna()
        s12 = s12[(s12["ws"] >= 3.0) & (s12["s1"] >= 3.0) & (s12["s2"] >= 3.0)]
        if not s12.empty:
            mean_s12 = 0.5 * (s12["s1"] + s12["s2"])
            mae_s1 = float((s12["ws"] - s12["s1"]).abs().mean())
            mae_s2 = float((s12["ws"] - s12["s2"]).abs().mean())
            mae_avg = float((s12["ws"] - mean_s12).abs().mean())
            s1_minus_s2 = float((s12["s1"] - s12["s2"]).median())
            ntf_ratio_s1 = float((s12["ws"] / s12["s1"]).median())
            ntf_ratio_avg = float((s12["ws"] / mean_s12).median())
            if min(mae_s1, mae_s2, mae_avg) < 0.02:
                if mae_avg <= min(mae_s1, mae_s2):
                    provenance_guess = "matches_mean_s1_s2"
                elif mae_s1 < mae_s2:
                    provenance_guess = "matches_sensor_1"
                else:
                    provenance_guess = "matches_sensor_2"
            else:
                provenance_guess = "oem_processed_or_ntf_scaled"
        else:
            mae_s1 = mae_s2 = mae_avg = s1_minus_s2 = ntf_ratio_s1 = ntf_ratio_avg = float("nan")
            provenance_guess = "missing_redundant_anemometers"

        # 2. Reverse-engineer implied air density from Density adjusted wind speed
        den_df = pd.DataFrame({"ws": ws, "den": den_ws, "temp": temp_c}).dropna()
        den_df = den_df[(den_df["ws"] >= 4.0) & (den_df["den"] >= 4.0)]
        if len(den_df) >= 100:
            rho_implied = 1.225 * (den_df["den"] / den_df["ws"]) ** 3
            rho_med = float(rho_implied.median())
            temp_corr = float(rho_implied.corr(den_df["temp"])) if den_df["temp"].std() > 0 else float("nan")
        else:
            rho_med = temp_corr = float("nan")

        rows.append(
            {
                "turbine_id": turbine_id,
                "n_s1_s2_valid": int(len(s12)),
                "mae_ws_vs_s1_ms": round(mae_s1, 4) if np.isfinite(mae_s1) else None,
                "mae_ws_vs_s2_ms": round(mae_s2, 4) if np.isfinite(mae_s2) else None,
                "mae_ws_vs_mean_s12_ms": round(mae_avg, 4) if np.isfinite(mae_avg) else None,
                "median_s1_minus_s2_ms": round(s1_minus_s2, 4) if np.isfinite(s1_minus_s2) else None,
                "median_ws_over_s1": round(ntf_ratio_s1, 4) if np.isfinite(ntf_ratio_s1) else None,
                "median_ws_over_mean_s12": round(ntf_ratio_avg, 4) if np.isfinite(ntf_ratio_avg) else None,
                "ws_provenance_inference": provenance_guess,
                "implied_density_median_kg_m3": round(rho_med, 4) if np.isfinite(rho_med) else None,
                "implied_density_temp_corr": round(temp_corr, 3) if np.isfinite(temp_corr) else None,
            }
        )
    return pd.DataFrame(rows)


def _donor_features(
    ws_donors: pd.DataFrame,
    dir_donors: pd.DataFrame,
    unwaked_donors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute contemporaneous donor wind speed, direction sector, and resultant length.

    Prioritizes unwaked (freestream) donors when at least 2 valid unwaked donors exist;
    otherwise falls back to all valid donors while strictly excluding target channels.
    """
    valid = ws_donors.notna() & (ws_donors >= 0.0) & dir_donors.notna()
    ws_valid = ws_donors.where(valid)

    if unwaked_donors is not None:
        uw_valid = valid & unwaked_donors
        uw_count = uw_valid.sum(axis=1)
        uw_ws = ws_donors.where(uw_valid).median(axis=1)
        all_ws = ws_valid.median(axis=1)
        wind = uw_ws.where(uw_count >= 2, all_ws)
    else:
        wind = ws_valid.median(axis=1)

    count = valid.sum(axis=1)
    rad = np.deg2rad(dir_donors.where(valid))
    sin_mean = np.sin(rad).mean(axis=1)
    cos_mean = np.cos(rad).mean(axis=1)
    angle_deg = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0
    resultant = np.hypot(sin_mean, cos_mean)

    sector = (angle_deg // 45.0).fillna(-1).astype(int)
    speed_bin = (wind // 2.0).fillna(-1).astype(int)
    return pd.DataFrame(
        {
            "donor_wind": wind,
            "sector": sector,
            "speed_bin": speed_bin,
            "count": count,
            "resultant": resultant,
        },
        index=ws_donors.index,
    )


def fit_peer_mcp(
    ws_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    dir_wide: pd.DataFrame,
    target_id: str,
    donors: list[str],
    *,
    unwaked_wide: pd.DataFrame | None = None,
    min_cell_count: int = 30,
) -> pd.DataFrame:
    """Fit target-excluded sector x speed-bin median wind offset and power transfer table."""
    uw = unwaked_wide[donors] if unwaked_wide is not None else None
    feat = _donor_features(ws_wide[donors], dir_wide[donors], uw)
    eligible = (feat["count"] >= 2) & (feat["resultant"] >= 0.5) & (ws_wide[target_id] >= 0.0)
    work = feat[eligible].copy()
    work["offset_ws"] = ws_wide.loc[eligible, target_id] - work["donor_wind"]
    work["target_power"] = power_wide.loc[eligible, target_id]
    grouped = work.groupby(["sector", "speed_bin"])
    table = grouped[["offset_ws", "target_power"]].median()
    counts = grouped[["offset_ws", "target_power"]].count()
    return table.where(counts >= min_cell_count)


def predict_peer_mcp(
    ws_wide: pd.DataFrame,
    dir_wide: pd.DataFrame,
    donors: list[str],
    table: pd.DataFrame,
    *,
    unwaked_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict target wind speed and expected power purely from contemporaneous donors."""
    uw = unwaked_wide[donors] if unwaked_wide is not None else None
    feat = _donor_features(ws_wide[donors], dir_wide[donors], uw)
    keys = pd.MultiIndex.from_arrays([feat["sector"], feat["speed_bin"]])
    vals = table.reindex(keys).set_axis(ws_wide.index)
    ok = (feat["count"] >= 2) & (feat["resultant"] >= 0.5)
    return pd.DataFrame(
        {
            "peer_ws": (feat["donor_wind"] + vals["offset_ws"]).clip(lower=0.0).where(ok),
            "peer_power": vals["target_power"].where(ok),
            "donor_median_ws": feat["donor_wind"].where(ok),
            "valid_donors": feat["count"],
            "sector": feat["sector"],
            "speed_bin": feat["speed_bin"],
        },
        index=ws_wide.index,
    )


def evaluate_all_turbine_peer_reconstruction(
    ws_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    dir_wide: pd.DataFrame,
    exposure_wide: pd.DataFrame | None,
    *,
    train_cutoff: pd.Timestamp,
    cal_end: pd.Timestamp,
    gap_lengths: tuple[int, ...] = (1, 6, 36, 144),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-out MCP gap imputation & anemometer drift detection across all turbines.

    Returns `(gap_benchmark_df, drift_summary_df)`.
    """
    ids = list(ws_wide.columns)
    unwaked_wide = exposure_wide.eq("freestream") if exposure_wide is not None else None
    train_mask = ws_wide.index < train_cutoff
    cal_mask = (ws_wide.index >= train_cutoff) & (ws_wide.index < cal_end)
    test_mask = ws_wide.index >= cal_end

    gap_rows: list[dict] = []
    drift_rows: list[dict] = []

    for target in ids:
        donors = [k for k in ids if k != target]
        table = fit_peer_mcp(
            ws_wide.loc[train_mask],
            power_wide.loc[train_mask],
            dir_wide.loc[train_mask],
            target,
            donors,
            unwaked_wide=unwaked_wide.loc[train_mask] if unwaked_wide is not None else None,
        )
        cal_pred = predict_peer_mcp(
            ws_wide.loc[cal_mask],
            dir_wide.loc[cal_mask],
            donors,
            table,
            unwaked_wide=unwaked_wide.loc[cal_mask] if unwaked_wide is not None else None,
        )
        cal_err = (ws_wide.loc[cal_mask, target] - cal_pred["peer_ws"]).dropna()
        if len(cal_err) < 100:
            continue
        q90 = float(cal_err.abs().quantile(0.90))
        q99 = float(cal_err.abs().quantile(0.99))

        test_pred = predict_peer_mcp(
            ws_wide.loc[test_mask],
            dir_wide.loc[test_mask],
            donors,
            table,
            unwaked_wide=unwaked_wide.loc[test_mask] if unwaked_wide is not None else None,
        )
        obs_test = ws_wide.loc[test_mask, target]
        err_test = (obs_test - test_pred["peer_ws"]).dropna()

        # Weekly masked block tests across 4 durations and donor dropout sensitivity
        test_idx = ws_wide.loc[test_mask].index
        week_starts = pd.date_range(test_idx.min().normalize() + pd.Timedelta(days=1), test_idx.max() - pd.Timedelta(days=2), freq="7D", tz="UTC")
        for dropout in (False, True):
            active_donors = donors[1:] if (dropout and len(donors) > 2) else donors
            sub_pred = (
                test_pred
                if not dropout
                else predict_peer_mcp(
                    ws_wide.loc[test_mask],
                    dir_wide.loc[test_mask],
                    active_donors,
                    table,
                    unwaked_wide=unwaked_wide.loc[test_mask] if unwaked_wide is not None else None,
                )
            )
            for length in gap_lengths:
                matched_errs = []
                pers_errs = []
                n_obs = 0
                for start in week_starts:
                    end = start + pd.Timedelta(minutes=10 * (length - 1))
                    block_obs = obs_test.loc[start:end]
                    block_pred = sub_pred.loc[start:end, "peer_ws"]
                    prior = obs_test.loc[: start - pd.Timedelta(minutes=10)].dropna().tail(1)
                    pers_val = float(prior.iloc[0]) if len(prior) else np.nan
                    valid = block_obs.notna() & block_pred.notna() & np.isfinite(pers_val)
                    n_obs += int(block_obs.notna().sum())
                    if valid.any():
                        matched_errs.extend((block_pred[valid] - block_obs[valid]).tolist())
                        pers_errs.extend((pers_val - block_obs[valid]).tolist())
                if matched_errs:
                    arr = np.asarray(matched_errs, dtype=float)
                    parr = np.asarray(pers_errs, dtype=float)
                    gap_rows.append(
                        {
                            "turbine_id": target,
                            "gap_rows": length,
                            "donor_dropout": dropout,
                            "eligible_observed": n_obs,
                            "matched_n": int(len(arr)),
                            "abstention_rate": round(1.0 - len(arr) / max(1, n_obs), 4),
                            "peer_mae_ms": round(float(np.mean(np.abs(arr))), 4),
                            "peer_rmse_ms": round(float(np.sqrt(np.mean(arr**2))), 4),
                            "peer_bias_ms": round(float(np.mean(arr)), 4),
                            "persistence_mae_ms": round(float(np.mean(np.abs(parr))), 4),
                            "interval_90_coverage": round(float(np.mean(np.abs(arr) <= q90)), 4),
                        }
                    )

        drift_rows.append(
            {
                "turbine_id": target,
                "test_n": int(len(err_test)),
                "test_bias_obs_minus_peer_ms": round(float(err_test.median()), 4),
                "test_mae_ms": round(float(err_test.abs().mean()), 4),
                "q90_width_ms": round(2.0 * q90, 4),
                "q99_threshold_ms": round(q99, 4),
                "control_exceedance_pct": round(100.0 * float((err_test.abs() > q99).mean()), 2),
                "plus_2ms_injection_exceedance_pct": round(100.0 * float(((err_test + 2.0).abs() > q99).mean()), 2),
            }
        )

    return pd.DataFrame(gap_rows), pd.DataFrame(drift_rows)
