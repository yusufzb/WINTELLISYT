"""Model-based atmospheric density, ERA5 alignment, and IEC vs WindPRO density benchmarks."""

from __future__ import annotations

import json
from math import exp, isfinite, log
from pathlib import Path

import numpy as np
import pandas as pd

RD, RV, G = 287.05, 461.5, 9.80665
STANDARD_DENSITY = 1.225


def moist_density(temperature_c: float, pressure_hpa: float, rh_percent: float) -> float:
    """Magnus water saturation approximation, restricted to -45..60 C."""
    if not all(map(isfinite, (temperature_c, pressure_hpa, rh_percent))):
        raise ValueError("Inputs must be finite")
    if not -45 <= temperature_c <= 60 or pressure_hpa <= 0 or not 0 <= rh_percent <= 100:
        raise ValueError("Inputs outside saturation formula domain")
    vapor_pa = 611.2 * exp(17.67 * temperature_c / (temperature_c + 243.5)) * rh_percent / 100.0
    pressure_pa = pressure_hpa * 100.0
    if vapor_pa >= pressure_pa:
        raise ValueError("Vapor pressure must be below total pressure")
    kelvin = temperature_c + 273.15
    return (pressure_pa - vapor_pa) / (RD * kelvin) + vapor_pa / (RV * kelvin)


def hub_pressure(pressure_hpa: float, delta_z_m: float, mean_virtual_temperature_k: float) -> float:
    """Approximate hydrostatic correction, using an explicitly supplied mean Tv."""
    if (
        not all(map(isfinite, (pressure_hpa, delta_z_m, mean_virtual_temperature_k)))
        or pressure_hpa <= 0
        or mean_virtual_temperature_k <= 0
    ):
        raise ValueError("Invalid pressure, height or temperature")
    return pressure_hpa * exp(-G * delta_z_m / (RD * mean_virtual_temperature_k))


def site_speed(reference_speed: float, density: float, reference_density: float = STANDARD_DENSITY) -> float:
    """Forward generic Svenningsen mapping, evaluated at reference curve speed."""
    if (
        not all(map(isfinite, (reference_speed, density, reference_density)))
        or reference_speed < 0
        or density <= 0
        or reference_density <= 0
    ):
        raise ValueError("Invalid speed or density")
    exponent = 1 / 3 if reference_speed <= 7.5 else (2 / 3 if reference_speed >= 12.5 else reference_speed / 15.0 - 1 / 6)
    return reference_speed * (reference_density / density) ** exponent


def reference_speed(site: float, density: float, maximum: float, reference_density: float = STANDARD_DENSITY) -> float:
    """Invert the forward map on [0, maximum], rejecting nonmonotonic domains."""
    if not isfinite(site) or site < 0 or not isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid inversion domain")
    site_speed(0.0, density, reference_density)
    if 1.0 + min(maximum, 12.5) * log(reference_density / density) / 15.0 <= 0 and maximum > 7.5:
        raise ValueError("Nonmonotonic forward map")
    if site > site_speed(maximum, density, reference_density):
        raise ValueError("Speed outside reference curve support")
    low, high = 0.0, maximum
    for _ in range(60):
        mid = (low + high) / 2.0
        if site_speed(mid, density, reference_density) < site:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def load_era5_hourly(farm_id: str, era5_dir: str | Path = "data/external/era5") -> tuple[pd.DataFrame, dict]:
    """Load cached Open-Meteo ERA5 hourly weather JSON and return DataFrame + metadata."""
    path = Path(era5_dir) / f"{farm_id.lower()}.json"
    manifest_path = Path(era5_dir) / f"{farm_id.lower()}.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    return df, manifest


def compute_hub_density_series(
    timestamps: pd.DatetimeIndex,
    scada_temp_c: pd.Series,
    era5_df: pd.DataFrame,
    *,
    era5_elevation_m: float,
    target_elevation_m: float,
) -> pd.DataFrame:
    """Align hourly ERA5 pressure/RH to 10-min UTC timestamps and compute moist hub density.

    Linearly interpolates hourly ERA5 up to 1-hour gaps (limit=5 10-min intervals).
    Uses SCADA nacelle ambient temperature when valid in [-45, 60] C, falling back
    to ERA5 2m temperature lapsed at -0.0065 K/m.
    """
    union_idx = era5_df.index.union(timestamps).sort_values()
    aligned = era5_df.reindex(union_idx).interpolate(method="time", limit=6).reindex(timestamps)

    t2m = aligned["temperature_2m"].to_numpy(dtype=float)
    rh = aligned["relative_humidity_2m"].to_numpy(dtype=float)
    p_sfc = aligned["surface_pressure"].to_numpy(dtype=float)

    delta_z = target_elevation_m - era5_elevation_m
    t_scada = scada_temp_c.reindex(timestamps).to_numpy(dtype=float)
    valid_scada_t = np.isfinite(t_scada) & (t_scada >= -45.0) & (t_scada <= 60.0)
    t_lapsed = t2m - 0.0065 * delta_z
    t_hub_c = np.where(valid_scada_t, t_scada, t_lapsed)
    used_scada_temp = valid_scada_t

    valid_rh = np.isfinite(rh) & (rh >= 0.0) & (rh <= 100.0)
    rh_clean = np.where(valid_rh, rh, 0.0)

    p_hub = np.full(len(timestamps), np.nan, dtype=float)
    rho_moist = np.full(len(timestamps), np.nan, dtype=float)
    rho_dry = np.full(len(timestamps), np.nan, dtype=float)

    t2m_eff = np.where(np.isfinite(t2m), t2m, t_hub_c)
    t_mean_k = 0.5 * (t2m_eff + t_hub_c) + 273.15
    ok = (
        np.isfinite(p_sfc)
        & (p_sfc > 0.0)
        & np.isfinite(t_hub_c)
        & (t_hub_c >= -45.0)
        & (t_hub_c <= 60.0)
        & np.isfinite(t_mean_k)
        & (t_mean_k > 0.0)
    )
    if np.any(ok):
        t_c_ok = t_hub_c[ok]
        p_z = p_sfc[ok] * np.exp(-G * delta_z / (RD * t_mean_k[ok]))
        p_pa = p_z * 100.0
        kelvin = t_c_ok + 273.15
        p_hub[ok] = p_z
        rho_dry[ok] = p_pa / (RD * kelvin)

        vapor_pa = 611.2 * np.exp(17.67 * t_c_ok / (t_c_ok + 243.5)) * rh_clean[ok] / 100.0
        valid_vap = vapor_pa < p_pa
        moist_vals = np.full(p_z.shape, np.nan, dtype=float)
        moist_vals[valid_vap] = (
            (p_pa[valid_vap] - vapor_pa[valid_vap]) / (RD * kelvin[valid_vap])
            + vapor_pa[valid_vap] / (RV * kelvin[valid_vap])
        )
        rho_moist[ok] = moist_vals

    return pd.DataFrame(
        {
            "era5_temp_2m_c": t2m,
            "era5_rh_2m_pct": rh,
            "era5_surface_pressure_hpa": p_sfc,
            "era5_wind_speed_100m_ms": aligned["wind_speed_100m"].to_numpy(dtype=float),
            "era5_wind_dir_100m_deg": aligned["wind_direction_100m"].to_numpy(dtype=float),
            "hub_temp_c": t_hub_c,
            "used_scada_temp": used_scada_temp,
            "hub_pressure_hpa": p_hub,
            "density_dry_kg_m3": rho_dry,
            "density_moist_kg_m3": rho_moist,
        },
        index=timestamps,
    )


def vector_svenningsen_reference_speed(site_ws: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Vectorized inversion of Svenningsen (WindPRO/EMD-style) density mapping."""
    out = np.full(len(site_ws), np.nan)
    for i, (v, rho) in enumerate(zip(site_ws, density, strict=True)):
        if not (np.isfinite(v) and np.isfinite(rho) and 0.0 <= v <= 30.0 and 0.7 <= rho <= 1.6):
            continue
        try:
            out[i] = reference_speed(float(v), float(rho), maximum=35.0)
        except ValueError:
            out[i] = np.nan
    return out


def benchmark_density_correction(
    df_2016: pd.DataFrame,
    df_2017: pd.DataFrame,
    stop_mask_2016_wide: pd.DataFrame | None = None,
    stop_mask_2017_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Benchmark 4 wind normalization modes (2016 train -> 2017 test) per turbine.

    Expects `density_moist_kg_m3` already joined on both DataFrames. Evaluates matched
    support across:
    - `raw_reported_ws`
    - `publisher_density_adjusted_ws`
    - `iec_cube_root`
    - `windpro_svenningsen`
    across overall, winter (DJF), summer (JJA), and low/high density strata.
    """
    rows: list[dict] = []
    common_turbines = sorted(set(df_2016["turbine_id"]).intersection(df_2017["turbine_id"]))

    for tid in common_turbines:
        tr = df_2016[df_2016["turbine_id"] == tid].copy()
        te = df_2017[df_2017["turbine_id"] == tid].copy()
        if stop_mask_2016_wide is not None and tid in stop_mask_2016_wide.columns:
            tr = tr[~stop_mask_2016_wide[tid].reindex(tr["timestamp_utc"], fill_value=False).to_numpy()]
        if stop_mask_2017_wide is not None and tid in stop_mask_2017_wide.columns:
            te = te[~stop_mask_2017_wide[tid].reindex(te["timestamp_utc"], fill_value=False).to_numpy()]

        for frame in (tr, te):
            rho = frame["density_moist_kg_m3"].to_numpy(dtype=float)
            ws = frame["wind_speed_ms"].to_numpy(dtype=float)
            frame["ws_raw"] = ws
            frame["ws_pub_adj"] = frame["density_adjusted_ws_ms"].to_numpy(dtype=float)
            frame["ws_iec"] = ws * (rho / STANDARD_DENSITY) ** (1.0 / 3.0)
            frame["ws_svenningsen"] = vector_svenningsen_reference_speed(ws, rho)

        methods = ["ws_raw", "ws_pub_adj", "ws_iec", "ws_svenningsen"]
        req = [*methods, "active_power_kw", "density_moist_kg_m3"]
        tr = tr.dropna(subset=req)
        te = te.dropna(subset=req)
        tr = tr[tr["ws_raw"].between(3.0, 15.0)]
        te = te[te["ws_raw"].between(3.0, 15.0)]
        if len(tr) < 500 or len(te) < 500:
            continue

        # Matched support across all 4 methods
        valid_te = pd.Series(True, index=te.index)
        preds: dict[str, pd.Series] = {}
        for m in methods:
            tr_bin = np.floor(tr[m] * 2.0 + 0.5) / 2.0
            grp = tr["active_power_kw"].groupby(tr_bin).agg(["mean", "count"])
            curve = grp["mean"].where(grp["count"] >= 30)
            te_bin = np.floor(te[m] * 2.0 + 0.5) / 2.0
            p_hat = te_bin.map(curve)
            preds[m] = p_hat
            valid_te &= p_hat.notna()

        if int(valid_te.sum()) < 200:
            continue
        te_m = te[valid_te]
        months = te_m["timestamp_utc"].dt.month
        rho_te = te_m["density_moist_kg_m3"]
        strata = {
            "overall": pd.Series(True, index=te_m.index),
            "winter_DJF": months.isin([12, 1, 2]),
            "summer_JJA": months.isin([6, 7, 8]),
            "low_density_lt_1.20": rho_te < 1.20,
            "high_density_gt_1.25": rho_te > 1.25,
        }

        for stratum_name, smask in strata.items():
            if int(smask.sum()) < 50:
                continue
            y_true = te_m.loc[smask, "active_power_kw"]
            for m in methods:
                err = y_true - preds[m].loc[te_m.index][smask]
                rows.append(
                    {
                        "turbine_id": tid,
                        "stratum": stratum_name,
                        "method": m,
                        "matched_n": int(len(err)),
                        "mae_kw": round(float(err.abs().mean()), 2),
                        "rmse_kw": round(float(np.sqrt((err**2).mean())), 2),
                        "signed_bias_kw": round(float(err.mean()), 2),
                    }
                )
    return pd.DataFrame(rows)
