"""NREL two-stage directional calibration, True North anchoring, and vane audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wintellisyt.wind_geometry import LayoutPoint, pair_vector, source_bearing


def wrap_deg(angle: Any) -> Any:
    """Wrap angular differences into [-180, 180) degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute weighted circular mean in [0, 360) degrees."""
    valid = np.isfinite(angles_deg)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    rad = np.deg2rad(angles_deg[valid])
    w = np.ones_like(rad) if weights is None else weights[valid]
    sin_sum = float(np.sum(w * np.sin(rad)))
    cos_sum = float(np.sum(w * np.cos(rad)))
    if sin_sum == 0.0 and cos_sum == 0.0:
        return float("nan")
    return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360.0)


def fit_relative_consensus(
    direction_wide: pd.DataFrame,
    layout: dict[str, LayoutPoint],
    cutoff_utc: pd.Timestamp,
    test_start_utc: pd.Timestamp,
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float, float]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit target-excluded circular consensus offsets and compute held-out peer disagreement.

    Uses complex phasor synchronization (iterative circular median update) on pairs with
    high circular resultant length so wrapped offsets near +-180 degrees never distort the fit.
    Returns `(vectors, corrected_wide, turbine_summary_df, monthly_drift_df)`.
    """
    ids = list(direction_wide.columns)
    n_turbines = len(ids)
    vectors: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    pair_diffs = np.full((n_turbines, n_turbines), np.nan)
    pair_weights = np.zeros((n_turbines, n_turbines), dtype=float)

    train_dir = direction_wide.loc[direction_wide.index < cutoff_utc]
    for i, target in enumerate(ids):
        for j, donor in enumerate(ids):
            if i == j:
                continue
            east, north = pair_vector(layout[target], layout[donor])
            distance = float(np.hypot(east, north))
            vectors[target, donor] = (east, north, distance, source_bearing(east, north))
            if i < j:
                diff = wrap_deg(train_dir[target] - train_dir[donor]).dropna()
                if len(diff) < 144:
                    continue
                rad = np.deg2rad(diff.to_numpy())
                avg_phasor = np.mean(np.exp(1j * rad))
                res_len = float(np.abs(avg_phasor))
                if res_len >= 0.5:
                    # Center around circular mean, then take robust circular median.
                    center = float(np.rad2deg(np.angle(avg_phasor)))
                    med_diff = float(wrap_deg(center + np.median(wrap_deg(diff.to_numpy() - center))))
                    w = float(np.sqrt(len(diff) * res_len / max(distance, 1.0)))
                    pair_diffs[i, j] = med_diff
                    pair_diffs[j, i] = -med_diff
                    pair_weights[i, j] = w
                    pair_weights[j, i] = w

    if np.any(pair_weights.sum(axis=1) == 0):
        raise ValueError("Relative direction graph contains an isolated turbine")

    # Iterative circular phase synchronization (zero-mean gauge).
    offsets = np.zeros(n_turbines, dtype=float)
    for _ in range(40):
        next_offsets = np.zeros(n_turbines, dtype=float)
        for i in range(n_turbines):
            mask = pair_weights[i] > 0
            candidates = offsets[mask] + pair_diffs[i, mask]
            rad = np.deg2rad(candidates)
            phasor = np.sum(pair_weights[i, mask] * np.exp(1j * rad))
            next_offsets[i] = float(np.rad2deg(np.angle(phasor)))
        # Remove global circular mean to anchor gauge without losing wrap continuity.
        global_mean = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(next_offsets))))))
        next_offsets = wrap_deg(next_offsets - global_mean)
        if np.max(np.abs(wrap_deg(next_offsets - offsets))) < 1e-4:
            offsets = next_offsets
            break
        offsets = next_offsets

    offset_series = pd.Series(offsets, index=ids)
    corrected = (direction_wide - offset_series) % 360.0

    peer_refs: dict[str, pd.Series] = {}
    summaries: list[dict] = []
    monthly_rows: list[dict] = []

    for target in ids:
        donors = [k for k in ids if k != target]
        weights = np.array([1.0 / max(vectors[target, k][2], 1.0) for k in donors])
        refs = []
        for values in (direction_wide, corrected):
            donor_angles = values[donors]
            valid_n = donor_angles.notna().sum(axis=1)
            z = (np.exp(1j * np.deg2rad(donor_angles)) * weights).sum(axis=1)
            denom = (donor_angles.notna() * weights).sum(axis=1)
            res_len = (z.abs() / denom.replace(0, np.nan)).fillna(0.0)
            ref_deg = pd.Series(np.rad2deg(np.angle(z)) % 360.0, index=direction_wide.index).where(
                (valid_n >= 2) & (res_len >= 0.7)
            )
            refs.append((ref_deg, valid_n, res_len))
        (raw_ref, _, _), (adj_ref, valid_n, res_len) = refs
        peer_refs[target] = adj_ref

        before = wrap_deg(direction_wide[target] - raw_ref)
        after = wrap_deg(corrected[target] - adj_ref)
        test_mask = (direction_wide.index >= test_start_utc) & before.notna() & after.notna()
        summaries.append(
            {
                "turbine_id": target,
                "relative_offset_deg": float(offset_series[target]),
                "test_n": int(test_mask.sum()),
                "raw_peer_mae_deg": float(before[test_mask].abs().mean()) if test_mask.any() else float("nan"),
                "relative_peer_mae_deg": float(after[test_mask].abs().mean()) if test_mask.any() else float("nan"),
                "peer_count_mean": float(valid_n[test_mask].mean()) if test_mask.any() else float("nan"),
                "resultant_mean": float(res_len[test_mask].mean()) if test_mask.any() else float("nan"),
            }
        )
        for month, vals in after.groupby(after.index.strftime("%Y-%m")):
            valid = vals.dropna()
            if len(valid) >= 72:
                monthly_rows.append(
                    {
                        "turbine_id": target,
                        "month": month,
                        "n": int(len(valid)),
                        "median_peer_diff_deg": float(valid.median()),
                        "iqr_peer_diff_deg": float(valid.quantile(0.75) - valid.quantile(0.25)),
                    }
                )

    peer_wide = pd.DataFrame(peer_refs, index=direction_wide.index)
    return vectors, corrected, peer_wide, pd.DataFrame(summaries), pd.DataFrame(monthly_rows)


def _circular_gaussian_smooth(bin_centers_deg: np.ndarray, values: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Smooth an angular curve periodically using a wrapped Gaussian kernel."""
    diff = wrap_deg(bin_centers_deg[:, None] - bin_centers_deg[None, :])
    weights = np.exp(-0.5 * (diff / sigma_deg) ** 2)
    valid = np.isfinite(values)
    weights = weights[:, valid]
    w_sum = weights.sum(axis=1)
    out = np.full(len(bin_centers_deg), np.nan)
    ok = w_sum > 0
    out[ok] = (weights[ok] @ values[valid]) / w_sum[ok]
    return out


def find_wake_trough_offset(
    ws_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    corrected_dir_wide: pd.DataFrame,
    vectors: dict[tuple[str, str], tuple[float, float, float, float]],
    layout: dict[str, LayoutPoint],
    *,
    min_diameters: float = 2.0,
    max_diameters: float = 9.0,
    bin_width_deg: float = 2.0,
    sigma_deg: float = 4.0,
    search_halfwidth_deg: float = 40.0,
) -> tuple[pd.DataFrame, dict]:
    """Anchor consensus directions to True North via empirical wake power-ratio troughs.

    For each close turbine pair, evaluates target/donor power ratio across angular bins
    in partial load (5.0-11.5 m/s) when both turbines are actively generating (>50 kW).
    Also tests whether the trough aligns with FROM vs TO (180-deg flipped) convention.
    """
    ids = list(corrected_dir_wide.columns)
    centers = np.arange(bin_width_deg / 2.0, 360.0, bin_width_deg)
    pair_rows: list[dict] = []

    for target in ids:
        for donor in ids:
            if target == donor:
                continue
            _, _, dist_m, geo_bearing_from = vectors[target, donor]
            d_rotor = dist_m / layout[donor].rotor_m
            if not (min_diameters <= d_rotor <= max_diameters):
                continue

            ws_d = ws_wide[donor]
            p_t = power_wide[target]
            p_d = power_wide[donor]
            theta = corrected_dir_wide[donor]

            mask = (
                ws_d.between(5.0, 11.5)
                & (p_d > 100.0)
                & (p_t > 0.0)
                & theta.notna()
            )
            if int(mask.sum()) < 500:
                continue

            ratio = (p_t[mask] / p_d[mask]).clip(lower=0.0, upper=2.0)
            ang_bin = (np.floor(theta[mask] / bin_width_deg) * bin_width_deg + bin_width_deg / 2.0) % 360.0
            binned = ratio.groupby(ang_bin).agg(["median", "count"])
            binned = binned[binned["count"] >= 5]
            if len(binned) < 30:
                continue

            raw_curve = pd.Series(np.nan, index=centers)
            raw_curve.loc[binned.index] = binned["median"]
            smoothed = _circular_gaussian_smooth(centers, raw_curve.to_numpy(), sigma_deg=sigma_deg)
            if np.isnan(smoothed).all():
                continue

            baseline = float(np.nanpercentile(smoothed, 80))
            # Evaluate both FROM (near geo_bearing_from) and TO (near geo_bearing_from + 180)
            best_by_conv: dict[str, tuple[float, float, float, float]] = {}
            for conv, expected_angle in (("FROM", geo_bearing_from), ("TO", (geo_bearing_from + 180.0) % 360.0)):
                in_window = np.abs(wrap_deg(centers - expected_angle)) <= search_halfwidth_deg
                valid_w = in_window & np.isfinite(smoothed)
                if not np.any(valid_w):
                    continue
                idx = int(np.argmin(np.where(valid_w, smoothed, np.inf)))
                trough_angle = float(centers[idx])
                trough_ratio = float(smoothed[idx])
                depth = max(0.0, baseline - trough_ratio)
                offset = float(wrap_deg(trough_angle - expected_angle))
                best_by_conv[conv] = (trough_angle, trough_ratio, depth, offset)

            if "FROM" not in best_by_conv:
                continue
            trough_from, min_ratio_from, depth_from, offset_from = best_by_conv["FROM"]
            depth_to = best_by_conv["TO"][2] if "TO" in best_by_conv else 0.0
            pair_rows.append(
                {
                    "farm_pair": f"{donor}->{target}",
                    "donor": donor,
                    "target": target,
                    "distance_d": round(d_rotor, 2),
                    "geometric_bearing_from_deg": round(geo_bearing_from, 2),
                    "trough_angle_deg": round(trough_from, 2),
                    "offset_from_true_north_deg": round(offset_from, 2),
                    "trough_power_ratio": round(min_ratio_from, 3),
                    "baseline_power_ratio": round(baseline, 3),
                    "wake_depth_from": round(depth_from, 3),
                    "wake_depth_to": round(depth_to, 3),
                    "n_samples": int(mask.sum()),
                }
            )

    pairs_df = pd.DataFrame(pair_rows)
    if pairs_df.empty:
        return pairs_df, {"status": "abstained", "common_true_north_offset_deg": 0.0, "convention": "unverified"}

    strong = pairs_df[pairs_df["wake_depth_from"] >= 0.08].copy()
    if strong.empty:
        strong = pairs_df.copy()

    # Check consistency across strong pairs
    weights = strong["wake_depth_from"].to_numpy() * np.sqrt(strong["n_samples"].to_numpy())
    mean_offset = circular_mean_deg(strong["offset_from_true_north_deg"].to_numpy(), weights)
    signed_mean = float(wrap_deg(mean_offset))
    spread = float(np.median(np.abs(wrap_deg(strong["offset_from_true_north_deg"].to_numpy() - signed_mean))))
    mean_depth_from = float(strong["wake_depth_from"].mean())
    mean_depth_to = float(strong["wake_depth_to"].mean())
    convention = "FROM" if mean_depth_from > mean_depth_to + 0.05 else "ambiguous"

    summary = {
        "status": "anchored_via_wake_trough" if len(strong) >= 2 and spread <= 12.0 else "low_confidence",
        "convention": convention,
        "common_true_north_offset_deg": round(signed_mean, 2),
        "pair_mad_deg": round(spread, 2),
        "strong_pair_count": int(len(strong)),
        "total_pair_count": int(len(pairs_df)),
        "mean_wake_depth_from": round(mean_depth_from, 3),
        "mean_wake_depth_to": round(mean_depth_to, 3),
    }
    return pairs_df, summary


def cross_check_era5_direction(
    calibrated_dir_wide: pd.DataFrame,
    ws_wide: pd.DataFrame,
    era5_dir_100m: pd.Series,
) -> dict:
    """Compare farm-median calibrated SCADA direction against ERA5 100m wind direction."""
    farm_dir_rad = np.arctan2(
        np.sin(np.deg2rad(calibrated_dir_wide)).mean(axis=1),
        np.cos(np.deg2rad(calibrated_dir_wide)).mean(axis=1),
    )
    farm_dir = np.rad2deg(farm_dir_rad) % 360.0
    farm_ws = ws_wide.median(axis=1)
    aligned_era5 = era5_dir_100m.reindex(farm_dir.index).interpolate(limit=6)
    mask = (farm_ws >= 7.0) & farm_dir.notna() & aligned_era5.notna()
    if int(mask.sum()) < 100:
        return {"n": 0, "median_scada_minus_era5_deg": float("nan"), "mae_deg": float("nan")}
    diff = wrap_deg(farm_dir[mask] - aligned_era5[mask])
    return {
        "n": int(mask.sum()),
        "median_scada_minus_era5_deg": round(float(diff.median()), 2),
        "mae_deg": round(float(diff.abs().mean()), 2),
    }


def audit_vane_nacelle_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Separate North-mark encoder offsets from yaw-vane alignment via WindDir vs Nacelle+Vane."""
    rows: list[dict] = []
    required = {"turbine_id", "wind_direction_deg", "nacelle_position_deg", "vane_position_deg"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for turbine_id, group in df.groupby("turbine_id"):
        sub = group.dropna(subset=["wind_direction_deg", "nacelle_position_deg", "vane_position_deg"])
        if "wind_speed_ms" in sub.columns:
            sub = sub[sub["wind_speed_ms"] >= 4.0]
        if len(sub) < 100:
            continue
        # Test both WindDir - (Nacelle + Vane) and WindDir - Nacelle
        diff_sum = wrap_deg(sub["wind_direction_deg"] - (sub["nacelle_position_deg"] + sub["vane_position_deg"]))
        diff_nac = wrap_deg(sub["wind_direction_deg"] - sub["nacelle_position_deg"])
        rows.append(
            {
                "turbine_id": turbine_id,
                "n": int(len(sub)),
                "median_wind_minus_nacelle_plus_vane_deg": round(float(diff_sum.median()), 3),
                "mae_wind_minus_nacelle_plus_vane_deg": round(float(diff_sum.abs().mean()), 3),
                "median_wind_minus_nacelle_deg": round(float(diff_nac.median()), 3),
                "median_vane_deg": round(float(wrap_deg(sub["vane_position_deg"]).median()), 3),
            }
        )
    return pd.DataFrame(rows)
