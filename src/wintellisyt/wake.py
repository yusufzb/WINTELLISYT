"""Active-donor wake sector classification, TI stratification, and knee physics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wintellisyt.direction import wrap_deg
from wintellisyt.wind_geometry import LayoutPoint, pair_vector, source_bearing


def classify_wake_exposure(
    calibrated_dir_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    rpm_wide: pd.DataFrame | None,
    layout: dict[str, LayoutPoint],
    *,
    max_diameters: float = 12.0,
    expansion_k: float = 0.05,
    boundary_deg: float = 25.0,
    min_donor_power_kw: float = 50.0,
    min_donor_rpm: float = 2.0,
) -> pd.DataFrame:
    """Classify each turbine-timestamp into physical wake exposure regimes.

    Unlike geometric-only tagging, `active_wake` requires BOTH:
    1) Target lies inside the Jensen wake expansion cone of an upstream turbine within 12D.
    2) At least one overlapping upstream turbine is actively operating (>50 kW and >2 RPM).
    When inside the geometric cone while all overlapping donors are stopped, tags
    `passive_wake_donor_stopped` as an ambient-flow placebo class.
    """
    ids = list(calibrated_dir_wide.columns)
    index = calibrated_dir_wide.index
    exposure_cols: dict[str, np.ndarray] = {}

    vectors: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for target in ids:
        for donor in ids:
            if target != donor:
                east, north = pair_vector(layout[target], layout[donor])
                vectors[target, donor] = (east, north, float(np.hypot(east, north)), source_bearing(east, north))

    for target in ids:
        direction = calibrated_dir_wide[target].to_numpy()
        active_cone = np.zeros(len(index), dtype=bool)
        passive_cone = np.zeros(len(index), dtype=bool)
        near_boundary = np.zeros(len(index), dtype=bool)

        valid_dir = np.isfinite(direction)
        rad = np.deg2rad(np.where(valid_dir, direction, 0.0))
        sin_a, cos_a = np.sin(rad), np.cos(rad)

        for donor in ids:
            if donor == target:
                continue
            east, north, dist, bearing = vectors[target, donor]
            d_source = layout[donor].rotor_m
            d_target = layout[target].rotor_m
            if dist > max_diameters * d_source:
                continue

            downstream = east * sin_a + north * cos_a
            lateral = east * cos_a - north * sin_a
            in_cone = valid_dir & (downstream > 0.0) & (
                np.abs(lateral) <= (d_source + d_target) / 2.0 + expansion_k * downstream
            )
            near_boundary |= valid_dir & (np.abs(wrap_deg(direction - bearing)) <= boundary_deg)

            donor_p = power_wide[donor].to_numpy()
            donor_active = np.isfinite(donor_p) & (donor_p > min_donor_power_kw)
            if rpm_wide is not None and donor in rpm_wide.columns:
                donor_r = rpm_wide[donor].to_numpy()
                donor_active &= np.isfinite(donor_r) & (donor_r > min_donor_rpm)

            active_cone |= in_cone & donor_active
            passive_cone |= in_cone & (~donor_active)

        labels = np.where(
            ~valid_dir,
            "unknown",
            np.where(
                active_cone,
                "active_wake",
                np.where(
                    passive_cone,
                    "passive_wake_donor_stopped",
                    np.where(near_boundary, "boundary", "freestream"),
                ),
            ),
        )
        exposure_cols[target] = labels

    return pd.DataFrame(exposure_cols, index=index)


def _daily_block_ci(delta: np.ndarray, seed: int = 42, draws: int = 1000) -> tuple[float | None, float | None]:
    if len(delta) < 7:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_wake_regime_effects(
    ws_wide: pd.DataFrame,
    std_wide: pd.DataFrame,
    power_wide: pd.DataFrame,
    exposure_wide: pd.DataFrame,
    stop_mask_wide: pd.DataFrame,
    layout: dict[str, LayoutPoint],
    *,
    train_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify TI stratification, low-speed overshoot vs knee undershoot, and speed suppression.

    Returns `(bin_comparison_df, regime_summary_df)` where regimes are:
    - `low_speed_convex` (4.0-8.0 m/s): high TI / wake often yields positive Delta-P (Jensen's inequality)
    - `knee_concave` (8.5-13.0 m/s): high TI / wake produces sharp negative Delta-P (undershoot)
    - `rated_plateau` (13.5-20.0 m/s)
    """
    ids = list(ws_wide.columns)
    bin_rows: list[dict] = []
    regime_rows: list[dict] = []

    regimes = {
        "low_speed_convex_4_8": (4.0, 8.0),
        "knee_concave_8.5_13": (8.5, 13.0),
        "rated_plateau_13.5_20": (13.5, 20.0),
    }

    for target in ids:
        others = [k for k in ids if k != target]
        # Build leave-target-out operating donor reference on train_cutoff
        donor_ws = ws_wide[others]
        donor_p = power_wide[others]
        donor_stop = stop_mask_wide[others]
        clean_donor = (~donor_stop) & (donor_p > 0.0) & (donor_ws >= 3.0)

        stacked_ws = donor_ws.where(clean_donor).stack()
        stacked_p = donor_p.where(clean_donor).stack()
        train_mask = stacked_ws.index.get_level_values(0) < train_cutoff
        donor_bins = np.floor(stacked_ws[train_mask] * 2.0 + 0.5) / 2.0
        ref_curve = stacked_p[train_mask].groupby(donor_bins).agg(["mean", "count"])
        valid_ref = ref_curve["mean"].where(ref_curve["count"] >= 30)

        # Unwaked peer wind speed index for nacelle wind suppression check
        unwaked_donors = exposure_wide[others].eq("freestream") & (~donor_stop)
        uw_ws = donor_ws.where(unwaked_donors)
        uw_peer_ws = uw_ws.median(axis=1).where(uw_ws.count(axis=1) >= 2)
        all_peer_ws = donor_ws.where(~donor_stop).median(axis=1).where(donor_ws.where(~donor_stop).count(axis=1) >= 2)
        peer_ws = uw_peer_ws.fillna(all_peer_ws)

        t_ws = ws_wide[target]
        t_std = std_wide[target]
        t_p = power_wide[target]
        t_exp = exposure_wide[target]
        t_stop = stop_mask_wide[target]

        ok = (~t_stop) & (t_ws >= 3.0) & t_p.notna()
        peer_bin = np.floor(peer_ws * 2.0 + 0.5) / 2.0
        ws_bin = np.floor(t_ws * 2.0 + 0.5) / 2.0
        p_ref = peer_bin.map(valid_ref)
        p_ref_local = ws_bin.map(valid_ref)

        frame = pd.DataFrame(
            {
                "ws": t_ws,
                "peer_ws": peer_ws,
                "ws_bin": ws_bin,
                "peer_bin": peer_bin,
                "power": t_p,
                "p_ref_peer": p_ref,
                "p_ref_local": p_ref_local,
                "ti": (t_std / t_ws).where((t_ws >= 3.0) & (t_std >= 0.0)),
                "ppr_peer": (t_p / p_ref).where(p_ref > 50.0),
                "delta_kw_local": (t_p - p_ref_local).where(p_ref_local > 50.0),
                "ws_suppression_ratio": (t_ws / peer_ws).where(peer_ws >= 4.0),
                "exposure": t_exp,
            }
        )[ok]

        # 0.5 m/s bin-level active_wake vs freestream paired comparisons
        for metric, bin_col in (("ti", "ws_bin"), ("ppr_peer", "peer_bin"), ("ws_suppression_ratio", "peer_bin")):
            for speed_b, grp in frame.groupby(bin_col):
                a = grp.loc[grp["exposure"] == "active_wake", metric].dropna()
                b = grp.loc[grp["exposure"] == "freestream", metric].dropna()
                if min(len(a), len(b)) < 30:
                    continue
                days = pd.concat(
                    [a.groupby(a.index.normalize()).mean().rename("a"), b.groupby(b.index.normalize()).mean().rename("b")],
                    axis=1,
                ).dropna()
                if len(days) < 7:
                    continue
                delta = (days["a"] - days["b"]).to_numpy()
                lo, hi = _daily_block_ci(delta)
                bin_rows.append(
                    {
                        "turbine_id": target,
                        "metric": metric,
                        "speed_bin_ms": float(speed_b),
                        "n_active_wake": int(len(a)),
                        "n_freestream": int(len(b)),
                        "matched_days": int(len(days)),
                        "active_wake_median": round(float(a.median()), 4),
                        "freestream_median": round(float(b.median()), 4),
                        "paired_daily_diff": round(float(delta.mean()), 4),
                        "ci95_low": round(lo, 4) if lo is not None else None,
                        "ci95_high": round(hi, 4) if hi is not None else None,
                    }
                )

        # Regime-level overshoot vs undershoot test
        for reg_name, (v_lo, v_hi) in regimes.items():
            sub = frame[frame["peer_ws"].between(v_lo, v_hi)]
            for exp_class in ("active_wake", "passive_wake_donor_stopped", "freestream"):
                cls_sub = sub[sub["exposure"] == exp_class]
                if len(cls_sub) < 30:
                    continue
                regime_rows.append(
                    {
                        "turbine_id": target,
                        "regime": reg_name,
                        "exposure_class": exp_class,
                        "n": int(len(cls_sub)),
                        "ti_median": round(float(cls_sub["ti"].median()), 4) if cls_sub["ti"].notna().any() else None,
                        "ppr_peer_mean": round(float(cls_sub["ppr_peer"].mean()), 4) if cls_sub["ppr_peer"].notna().any() else None,
                        "delta_kw_local_mean": round(float(cls_sub["delta_kw_local"].mean()), 2)
                        if cls_sub["delta_kw_local"].notna().any()
                        else None,
                        "ws_suppression_median": round(float(cls_sub["ws_suppression_ratio"].median()), 4)
                        if cls_sub["ws_suppression_ratio"].notna().any()
                        else None,
                    }
                )

    return pd.DataFrame(bin_rows), pd.DataFrame(regime_rows)
