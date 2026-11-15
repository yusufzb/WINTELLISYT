"""High-speed Parquet-backed analytical store and Senvion event dictionary for WINTELLISYT v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from wintellisyt.atmosphere import STANDARD_DENSITY, compute_hub_density_series, load_era5_hourly
from wintellisyt.data.scada import load_rich_telemetry, load_static_layout, read_status_events
from wintellisyt.pc_health import build_event_stop_mask, evaluate_oem_curve
from wintellisyt.wind_geometry import LayoutPoint, pair_vector, source_bearing

FARM_ARCHIVES: dict[str, dict[str, Any]] = {
    "kelmarsh": {
        "years": {
            2016: ["data/raw/kelmarsh/Kelmarsh_SCADA_2016_3082.zip"],
            2017: ["data/raw/kelmarsh/Kelmarsh_SCADA_2017_3083.zip"],
        },
        "model": "MM92",
    },
    "penmanshiel": {
        "years": {
            2016: ["data/raw/penmanshiel/Penmanshiel_SCADA_2016_WT11-15_3107.zip"],
            2017: [
                "data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT01-10_3114.zip",
                "data/raw/penmanshiel/Penmanshiel_SCADA_2017_WT11-15_3115.zip",
            ],
        },
        "model": "MM82",
    },
}

_TELEMETRY_FLOAT_COLS: tuple[str, ...] = (
    "wind_speed_ms",
    "density_adjusted_ws_ms",
    "wind_speed_std_ms",
    "wind_speed_s1_ms",
    "wind_speed_s2_ms",
    "ti",
    "density_moist_kg_m3",
    "density_dry_kg_m3",
    "hub_pressure_hpa",
    "density_publisher_implied_kg_m3",
    "ws_iec_ms",
    "ws_svenningsen_ms",
    "active_power_kw",
    "available_capacity_kw",
    "potential_power_mpc_kw",
    "potential_power_learned_kw",
    "potential_power_ref_kw",
    "potential_power_default_kw",
    "energy_theoretical_kwh",
    "energy_export_kwh",
    "energy_import_kwh",
    "lost_contractual_kwh",
    "lost_time_iec_b22_kwh",
    "lost_prod_iec_b22_kwh",
    "lost_downtime_kwh",
    "lost_performance_kwh",
    "lost_curtailment_total_kwh",
    "performance_index",
    "production_factor",
    "data_availability",
    "avail_time_contractual",
    "avail_time_iec_b22",
    "avail_time_iec_b23",
    "avail_time_iec_b24",
    "avail_time_iec_b32",
    "avail_prod_iec_b22",
    "avail_prod_iec_b23",
    "avail_prod_iec_b32",
    "avail_time_system",
    "avail_prod_system",
    "wind_direction_deg",
    "wd_calibrated_deg",
    "nacelle_position_deg",
    "vane_position_deg",
    "vane_position_std_deg",
    "cable_windings",
    "rotor_speed_rpm",
    "generator_rpm",
    "pitch_a_deg",
    "pitch_b_deg",
    "pitch_c_deg",
    "pitch_spread_deg",
    "motor_current_1_a",
    "motor_current_2_a",
    "motor_current_3_a",
    "motor_temp_1_c",
    "motor_temp_2_c",
    "motor_temp_3_c",
    "drivetrain_accel_mmss",
    "tower_accel_x_mmss",
    "tower_accel_y_mmss",
    "gear_oil_inlet_bar",
    "gear_oil_pump_bar",
    "metal_particle_count",
    "ambient_temperature_c",
    "rotor_bearing_temp_c",
    "front_bearing_temp_c",
    "rear_bearing_temp_c",
    "gear_oil_temp_c",
    "gen_bearing_front_temp_c",
    "gen_bearing_rear_temp_c",
    "stator_temp_1_c",
    "transformer_temp_c",
)

_TELEMETRY_BOOL_COLS: tuple[str, ...] = (
    "event_stop_mask",
    "event_warning_mask",
    "iec_unavail_mask",
    "curtailment_mask",
    "in_geometric_wake",
)


def normalize_turbine_id(value: Any) -> str:
    """Normalize raw turbine numbers such as '1', '01', or 1 into two-digit strings ('01')."""
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(2)
    return text


def vector_iec_wind_speed(
    site_ws: np.ndarray,
    density: np.ndarray,
    rho_ref: float = STANDARD_DENSITY,
) -> np.ndarray:
    """Apply vectorized IEC 61400-12-1 cube-root air-density scaling at `rho_ref`."""
    ws = np.asarray(site_ws, dtype=float)
    rho = np.asarray(density, dtype=float)
    out = np.full(ws.shape, np.nan, dtype=float)
    ok = np.isfinite(ws) & np.isfinite(rho) & (ws >= 0.0) & (rho > 0.0) & (rho_ref > 0.0)
    out[ok] = ws[ok] * (rho[ok] / rho_ref) ** (1.0 / 3.0)
    return out


def vector_svenningsen_fast(
    site_ws: np.ndarray,
    density: np.ndarray,
    rho_ref: float = STANDARD_DENSITY,
    *,
    maximum: float = 35.0,
) -> np.ndarray:
    """Vectorized inversion of the Svenningsen (WindPRO/EMD-style) density mapping."""
    ws = np.asarray(site_ws, dtype=float)
    rho = np.asarray(density, dtype=float)
    out = np.full(ws.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(ws)
        & np.isfinite(rho)
        & (ws >= 0.0)
        & (ws <= 30.0)
        & (rho >= 0.7)
        & (rho <= 1.6)
        & (rho_ref > 0.0)
    )
    if not np.any(ok):
        return out

    v = ws[ok]
    ratio = rho_ref / rho[ok]
    low = np.zeros_like(v)
    high = np.full_like(v, maximum)
    for _ in range(44):
        mid = 0.5 * (low + high)
        exp_m = np.where(mid <= 7.5, 1.0 / 3.0, np.where(mid >= 12.5, 2.0 / 3.0, mid / 15.0 - 1.0 / 6.0))
        mapped = mid * (ratio**exp_m)
        go_right = mapped < v
        low = np.where(go_right, mid, low)
        high = np.where(go_right, high, mid)
    out[ok] = 0.5 * (low + high)
    return out


def classify_rds_pp_subsystem(
    code: str,
    message: str,
    status: str = "",
    iec_category: str = "",
) -> str:
    """Classify Senvion MM82/MM92 event records into RDS-PP engineering subsystems."""
    text = f"{code} {message} {status} {iec_category}".lower()
    if any(k in text for k in ("pitch", "blade", "charging circuit", "axis 1", "axis 2", "axis 3", "hub")):
        return "Pitch System"
    if any(k in text for k in ("yaw", "twist", "cable", "winding", "nacelle position")):
        return "Yaw & Cable Twist"
    if any(k in text for k in ("tower", "oscillation", "vibration", "accel")):
        return "Tower & Vibration"
    if any(k in text for k in ("gear", "bearing", "oil", "rotor", "brake", "lubricat", "shaft")):
        return "Drivetrain & Gearbox"
    if any(k in text for k in ("converter", "generator", "transformer", "inverter", "voltage", "current", "phase", "stator")):
        return "Converter & Electrical"
    if any(k in text for k in ("manual", "maintenance", "service", "inspection", "scheduled")):
        return "Manual & Scheduled Maintenance"
    if any(k in text for k in ("externally stopped", "grid", "safety chain", "emergency", "requested shutdown", "substation")):
        return "Grid & Safety Chain"
    if any(k in text for k in ("communication", "timeout", "anemometer", "vane", "sensor", "watchdog", "plc")):
        return "Controller & Comms"
    if any(k in text for k in ("wind", "temp", "ice", "storm", "out of environmental")):
        return "Environmental & Operational"
    return "Other"


def _assign_subsystem_column(df: pd.DataFrame) -> pd.Series:
    """Classify event rows into RDS-PP subsystems using a deduplicated key lookup."""
    if df.empty:
        return pd.Series(dtype=object, index=df.index)
    keys_df = pd.DataFrame(
        {
            "code": df["code"].fillna("").astype(str),
            "message": df["message"].fillna("").astype(str),
            "status": df["status"].fillna("").astype(str),
            "iec_category": df["iec_category"].fillna("").astype(str),
        },
        index=df.index,
    )
    uniq = keys_df.drop_duplicates()
    mapping = {
        (r.code, r.message, r.status, r.iec_category): classify_rds_pp_subsystem(
            r.code, r.message, r.status, r.iec_category
        )
        for r in uniq.itertuples(index=False)
    }
    return pd.Series(
        [mapping[(r.code, r.message, r.status, r.iec_category)] for r in keys_df.itertuples(index=False)],
        index=df.index,
        dtype=object,
    )


def _parse_duration_hours(duration_raw: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    if pd.notna(start_ts) and pd.notna(end_ts) and end_ts > start_ts:
        return float((end_ts - start_ts).total_seconds() / 3600.0)
    raw = (duration_raw or "").strip()
    m = re.match(r"^(?:(\d+)\s+days?,?\s*)?(\d+):(\d+):(\d+(?:\.\d+)?)$", raw)
    if m:
        days = float(m.group(1) or 0)
        hrs = float(m.group(2))
        mins = float(m.group(3))
        secs = float(m.group(4))
        return days * 24.0 + hrs + mins / 60.0 + secs / 3600.0
    return 0.0


def build_event_code_dictionary(
    events_df: pd.DataFrame,
    telemetry_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical Senvion Event Code & IEC Category dictionary from Status logs."""
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "farm_id",
                "code",
                "message",
                "status",
                "iec_category",
                "service_contract_category",
                "subsystem",
                "occurrences",
                "n_turbines_affected",
                "affected_turbines",
                "total_duration_hours",
                "median_duration_min",
                "open_ended_count",
                "estimated_lost_mwh",
            ]
        )

    work = events_df.copy()
    s_ts = pd.to_datetime(work["timestamp_start"], utc=True, errors="coerce")
    e_ts = pd.to_datetime(work["timestamp_end"], utc=True, errors="coerce")
    has_span = s_ts.notna() & e_ts.notna() & (e_ts > s_ts)
    dur_h = np.zeros(len(work), dtype=float)
    span_mask = has_span.to_numpy(dtype=bool)
    if np.any(span_mask):
        dur_h[span_mask] = (e_ts[has_span] - s_ts[has_span]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    if not bool(np.all(span_mask)) and "duration_raw" in work.columns:
        miss_idx = np.flatnonzero(~span_mask)
        dur_raw = work["duration_raw"].iloc[miss_idx].fillna("").astype(str).tolist()
        for k, pos in enumerate(miss_idx):
            dur_h[pos] = _parse_duration_hours(dur_raw[k], pd.NaT, pd.NaT)
    work["duration_hours"] = dur_h
    if "subsystem" not in work.columns:
        work["subsystem"] = _assign_subsystem_column(work)

    # Estimate mean hourly theoretical power during stops per (farm_id, turbine_id) for fast attribution
    stop_kw_lookup: dict[tuple[str, str], float] = {}
    if telemetry_df is not None and not telemetry_df.empty and "potential_power_mpc_kw" in telemetry_df.columns:
        for (fid, tid), grp in telemetry_df.groupby(["farm_id", "turbine_id"]):
            if "event_stop_mask" in grp.columns:
                stopped = grp[grp["event_stop_mask"].astype(bool)]
                src = stopped if len(stopped) >= 6 else grp
            else:
                src = grp
            valid_pot = src["potential_power_mpc_kw"].dropna()
            mean_kw = float(valid_pot.mean()) if not valid_pot.empty else 650.0
            stop_kw_lookup[(str(fid), str(tid))] = mean_kw

    if stop_kw_lookup:
        pairs = zip(work["farm_id"].astype(str), work["turbine_id"].astype(str), strict=True)
        mean_kw_arr = np.fromiter(
            (stop_kw_lookup.get(pair, 650.0) for pair in pairs),
            dtype=float,
            count=len(work),
        )
    else:
        mean_kw_arr = np.full(len(work), 650.0, dtype=float)
    is_stop = work["status"].fillna("").astype(str).str.lower().to_numpy() == "stop"
    work["est_lost_mwh_row"] = np.where(is_stop, dur_h * mean_kw_arr / 1000.0, 0.0)

    key_cols = ["farm_id", "code", "message", "status", "iec_category", "service_contract_category", "subsystem"]
    grouped = (
        work.groupby(key_cols, dropna=False)
        .agg(
            occurrences=("code", "count"),
            n_turbines_affected=("turbine_id", "nunique"),
            affected_turbines=("turbine_id", lambda s: ", ".join(sorted(set(map(str, s))))),
            total_duration_hours=("duration_hours", "sum"),
            median_duration_min=("duration_hours", lambda s: float(np.median(s) * 60.0)),
            open_ended_count=("open_ended", "sum"),
            estimated_lost_mwh=("est_lost_mwh_row", "sum"),
        )
        .reset_index()
    )
    grouped["total_duration_hours"] = grouped["total_duration_hours"].round(2)
    grouped["median_duration_min"] = grouped["median_duration_min"].round(1)
    grouped["estimated_lost_mwh"] = grouped["estimated_lost_mwh"].round(2)
    return grouped.sort_values(
        ["farm_id", "total_duration_hours", "occurrences"], ascending=[True, False, False]
    ).reset_index(drop=True)


def debounce_events(
    events_df: pd.DataFrame,
    gap_seconds: int = 60,
    *,
    stop_only: bool = True,
) -> pd.DataFrame:
    """Merge consecutive fault/stop episodes within `gap_seconds` per turbine to remove alarm chatter."""
    if events_df.empty:
        return pd.DataFrame()
    work = events_df.copy()
    if stop_only:
        work = work[work["status"] == "Stop"].copy()
    if work.empty:
        return pd.DataFrame()

    if "subsystem" not in work.columns:
        work["subsystem"] = _assign_subsystem_column(work)

    # Ensure valid start/end timestamps
    work = work[work["timestamp_start"].notna()].copy()
    fallback_end = work["timestamp_start"] + pd.Timedelta(minutes=10)
    work["eff_end"] = work["timestamp_end"].where(
        work["timestamp_end"].notna() & (work["timestamp_end"] > work["timestamp_start"]),
        fallback_end,
    )
    work = work.sort_values(["farm_id", "turbine_id", "timestamp_start", "eff_end"]).reset_index(drop=True)

    episodes: list[dict[str, Any]] = []
    for (farm_id, turbine_id), grp in work.groupby(["farm_id", "turbine_id"]):
        cur: dict[str, Any] | None = None
        for row in grp.itertuples(index=False):
            s_ts = row.timestamp_start
            e_ts = row.eff_end
            if cur is None:
                cur = {
                    "farm_id": farm_id,
                    "turbine_id": turbine_id,
                    "timestamp_start": s_ts,
                    "timestamp_end": e_ts,
                    "root_code": str(row.code),
                    "root_message": str(row.message),
                    "status": str(row.status),
                    "iec_category": str(row.iec_category) or "Unclassified",
                    "service_contract_category": str(row.service_contract_category),
                    "subsystem": str(row.subsystem),
                    "merged_event_count": 1,
                    "codes_in_episode": {str(row.code)},
                    "open_ended": bool(row.open_ended),
                }
                continue

            gap = float((s_ts - cur["timestamp_end"]).total_seconds())
            if gap <= float(gap_seconds):
                cur["timestamp_end"] = max(cur["timestamp_end"], e_ts)
                cur["merged_event_count"] += 1
                cur["codes_in_episode"].add(str(row.code))
                cur["open_ended"] = cur["open_ended"] or bool(row.open_ended)
            else:
                episodes.append(cur)
                cur = {
                    "farm_id": farm_id,
                    "turbine_id": turbine_id,
                    "timestamp_start": s_ts,
                    "timestamp_end": e_ts,
                    "root_code": str(row.code),
                    "root_message": str(row.message),
                    "status": str(row.status),
                    "iec_category": str(row.iec_category) or "Unclassified",
                    "service_contract_category": str(row.service_contract_category),
                    "subsystem": str(row.subsystem),
                    "merged_event_count": 1,
                    "codes_in_episode": {str(row.code)},
                    "open_ended": bool(row.open_ended),
                }
        if cur is not None:
            episodes.append(cur)

    for ep in episodes:
        dur_h = max(0.0, float((ep["timestamp_end"] - ep["timestamp_start"]).total_seconds() / 3600.0))
        ep["duration_hours"] = round(dur_h, 3)
        ep["codes_in_episode"] = ", ".join(sorted(ep["codes_in_episode"]))

    out = pd.DataFrame(episodes)
    if not out.empty:
        out = out.sort_values(["farm_id", "duration_hours"], ascending=[True, False]).reset_index(drop=True)
    return out


def _compute_geometric_wake_flags(
    dir_series: pd.Series,
    turbine_id: str,
    layout_points: dict[str, LayoutPoint],
    *,
    max_diameters: float = 10.0,
    expansion_k: float = 0.05,
) -> np.ndarray:
    """Compute geometric Jensen cone wake membership from calibrated wind direction."""
    vals = dir_series.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    in_wake = np.zeros(len(vals), dtype=bool)
    if turbine_id not in layout_points or not np.any(valid):
        return in_wake

    rad = np.deg2rad(np.where(valid, vals, 0.0))
    sin_a, cos_a = np.sin(rad), np.cos(rad)
    target_pt = layout_points[turbine_id]

    for donor_id, donor_pt in layout_points.items():
        if donor_id == turbine_id:
            continue
        east, north = pair_vector(target_pt, donor_pt)
        dist = float(np.hypot(east, north))
        if dist > max_diameters * donor_pt.rotor_m:
            continue
        # Ensure bearing vector helper doesn't fail on valid pairs
        _ = source_bearing(east, north)
        downstream = east * sin_a + north * cos_a
        lateral = east * cos_a - north * sin_a
        cone = valid & (downstream > 0.0) & (
            np.abs(lateral) <= (donor_pt.rotor_m + target_pt.rotor_m) / 2.0 + expansion_k * downstream
        )
        in_wake |= cone
    return in_wake


@dataclass
class WorkbenchStore:
    """Fast in-memory & Parquet-cached multi-year SCADA, event, and availability store."""

    root: Path
    cache_dir: Path
    telemetry_df: pd.DataFrame
    events_df: pd.DataFrame
    event_dictionary_df: pd.DataFrame
    layout_df: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)
    _telemetry_index: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _events_index: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._telemetry_index.clear()
        if not self.telemetry_df.empty:
            for (fid, tid), grp in self.telemetry_df.groupby(["farm_id", "turbine_id"], sort=False):
                self._telemetry_index[(str(fid), str(tid))] = grp.sort_values("timestamp_utc").reset_index(drop=True)
        self._events_index.clear()
        if not self.events_df.empty:
            for (fid, tid), grp in self.events_df.groupby(["farm_id", "turbine_id"], sort=False):
                self._events_index[(str(fid), str(tid))] = grp.sort_values("timestamp_start").reset_index(drop=True)

    def farms(self) -> list[str]:
        if not self.layout_df.empty and "farm" in self.layout_df.columns:
            return sorted(self.layout_df["farm"].astype(str).unique().tolist())
        return sorted({k[0] for k in self._telemetry_index})

    def turbines(self, farm_id: str) -> list[str]:
        farm = farm_id.lower()
        tids = [tid for (fid, tid) in self._telemetry_index if fid == farm]
        if not tids and not self.layout_df.empty:
            tids = self.layout_df[self.layout_df["farm"] == farm]["turbine_id"].astype(str).tolist()
        return sorted(set(tids), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    def slice_telemetry(
        self,
        farm_id: str,
        turbine_id: str | None = None,
        start_ts: pd.Timestamp | str | None = None,
        end_ts: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return telemetry rows filtered by farm, optional turbine, and optional UTC bounds."""
        farm = farm_id.lower()
        if turbine_id is not None:
            norm_tid = normalize_turbine_id(turbine_id)
            df = self._telemetry_index.get((farm, norm_tid), pd.DataFrame())
            if df.empty:
                return df
            sub = df
        else:
            parts = [grp for (fid, _tid), grp in sorted(self._telemetry_index.items()) if fid == farm]
            if not parts:
                return pd.DataFrame()
            sub = pd.concat(parts, ignore_index=True)

        if start_ts is not None:
            s = pd.Timestamp(start_ts)
            if s.tzinfo is None:
                s = s.tz_localize("UTC")
            sub = sub[sub["timestamp_utc"] >= s]
        if end_ts is not None:
            e = pd.Timestamp(end_ts)
            if e.tzinfo is None:
                e = e.tz_localize("UTC")
            sub = sub[sub["timestamp_utc"] <= e]
        return sub.reset_index(drop=True)

    def slice_events(
        self,
        farm_id: str,
        turbine_id: str | None = None,
        start_ts: pd.Timestamp | str | None = None,
        end_ts: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return status events filtered by farm, optional turbine, and optional UTC window."""
        farm = farm_id.lower()
        if turbine_id is not None:
            norm_tid = normalize_turbine_id(turbine_id)
            sub = self._events_index.get((farm, norm_tid), pd.DataFrame())
            if sub.empty:
                return sub
        else:
            parts = [grp for (fid, _tid), grp in sorted(self._events_index.items()) if fid == farm]
            if not parts:
                return pd.DataFrame()
            sub = pd.concat(parts, ignore_index=True)

        if start_ts is not None:
            s = pd.Timestamp(start_ts)
            if s.tzinfo is None:
                s = s.tz_localize("UTC")
            eff_end = sub["timestamp_end"].fillna(sub["timestamp_start"] + pd.Timedelta(minutes=10))
            sub = sub[eff_end >= s]
        if end_ts is not None:
            e = pd.Timestamp(end_ts)
            if e.tzinfo is None:
                e = e.tz_localize("UTC")
            sub = sub[sub["timestamp_start"] <= e]
        return sub.reset_index(drop=True)

    def get_event_dictionary(self, farm_id: str | None = None) -> pd.DataFrame:
        if self.event_dictionary_df.empty:
            return pd.DataFrame()
        if farm_id is None:
            return self.event_dictionary_df.copy()
        return self.event_dictionary_df[self.event_dictionary_df["farm_id"] == farm_id.lower()].reset_index(drop=True)

    def get_layout(self, farm_id: str | None = None) -> pd.DataFrame:
        if self.layout_df.empty:
            return pd.DataFrame()
        if farm_id is None:
            return self.layout_df.copy()
        return self.layout_df[self.layout_df["farm"] == farm_id.lower()].reset_index(drop=True)

    @classmethod
    def load_or_build(
        cls,
        root: str | Path = ".",
        cache_dir: str | Path = "data/processed/workbench_cache",
        *,
        rebuild: bool = False,
        farms: tuple[str, ...] = ("kelmarsh", "penmanshiel"),
        years: tuple[int, ...] = (2016, 2017),
    ) -> WorkbenchStore:
        """Load materialized ZSTD Parquet workbench cache if fresh, otherwise build from raw ZIPs."""
        root_path = Path(root).resolve()
        cdir = Path(cache_dir)
        if not cdir.is_absolute():
            cdir = root_path / cdir

        manifest_path = cdir / "cache_manifest.json"
        has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        ext = "parquet" if has_pyarrow else "pkl.gz"
        tel_path = cdir / f"telemetry_10min.{ext}"
        evt_path = cdir / f"events_raw.{ext}"
        dict_path = cdir / f"event_code_dictionary.{ext}"
        lay_path = cdir / f"layout.{ext}"

        if not rebuild and manifest_path.exists() and tel_path.exists() and evt_path.exists() and dict_path.exists() and lay_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if has_pyarrow:
                tel_df = pd.read_parquet(tel_path)
                evt_df = pd.read_parquet(evt_path)
                dict_df = pd.read_parquet(dict_path)
                lay_df = pd.read_parquet(lay_path)
            else:
                tel_df = pd.read_pickle(tel_path, compression="gzip")
                evt_df = pd.read_pickle(evt_path, compression="gzip")
                dict_df = pd.read_pickle(dict_path, compression="gzip")
                lay_df = pd.read_pickle(lay_path, compression="gzip")

            # Guarantee UTC datetime dtype and normalized IDs after deserialization
            if not tel_df.empty:
                tel_df["timestamp_utc"] = pd.to_datetime(tel_df["timestamp_utc"], utc=True)
                tel_df["turbine_id"] = tel_df["turbine_id"].astype(str).map(normalize_turbine_id)
            if not evt_df.empty:
                evt_df["timestamp_start"] = pd.to_datetime(evt_df["timestamp_start"], utc=True)
                evt_df["timestamp_end"] = pd.to_datetime(evt_df["timestamp_end"], utc=True)
                evt_df["turbine_id"] = evt_df["turbine_id"].astype(str).map(normalize_turbine_id)
            return cls(
                root=root_path,
                cache_dir=cdir,
                telemetry_df=tel_df,
                events_df=evt_df,
                event_dictionary_df=dict_df,
                layout_df=lay_df,
                manifest=manifest,
            )

        return cls.build_from_raw(root=root_path, cache_dir=cdir, farms=farms, years=years)

    @classmethod
    def build_from_raw(
        cls,
        root: str | Path = ".",
        cache_dir: str | Path = "data/processed/workbench_cache",
        *,
        farms: tuple[str, ...] = ("kelmarsh", "penmanshiel"),
        years: tuple[int, ...] = (2016, 2017),
    ) -> WorkbenchStore:
        """Build enriched telemetry, events, and code dictionary from raw archives and persist to Parquet."""
        root_path = Path(root).resolve()
        cdir = Path(cache_dir)
        if not cdir.is_absolute():
            cdir = root_path / cdir
        cdir.mkdir(parents=True, exist_ok=True)

        oem_path = root_path / "data/reference/oem_power_curves.json"
        oem_payload = json.loads(oem_path.read_text(encoding="utf-8")) if oem_path.exists() else {"curves": {}}

        dir_offsets: dict[tuple[str, str], float] = {}
        dcon_path = root_path / "artifacts/engineering_suite_2017/direction_consensus.csv"
        if dcon_path.exists():
            dcon = pd.read_csv(dcon_path, dtype={"turbine_id": str})
            for r in dcon.itertuples(index=False):
                val = getattr(r, "total_offset_deg", np.nan)
                if np.isfinite(val):
                    dir_offsets[(str(r.farm_id).lower(), normalize_turbine_id(r.turbine_id))] = float(val)

        tel_frames: list[pd.DataFrame] = []
        evt_frames: list[pd.DataFrame] = []
        lay_frames: list[pd.DataFrame] = []
        archive_digests: dict[str, str] = {}

        for farm in farms:
            cfg = FARM_ARCHIVES[farm.lower()]
            model = str(cfg["model"])
            oem_knots = oem_payload.get("curves", {}).get(model, {}).get("knots_1ms", {})
            raw_meta = load_static_layout(farm, root_path)
            meta = {normalize_turbine_id(k): {**v, "turbine_id": normalize_turbine_id(k)} for k, v in raw_meta.items()}
            lay_frames.append(pd.DataFrame(list(meta.values())))
            layout_pts = {
                tid: LayoutPoint(m["latitude"], m["longitude"], m["rotor_diameter_m"])
                for tid, m in meta.items()
            }

            era5_dir = root_path / "data/external/era5"
            has_era5 = (era5_dir / f"{farm.lower()}.json").exists()
            era5_df, era5_meta = load_era5_hourly(farm, era5_dir) if has_era5 else (pd.DataFrame(), {})
            era5_elev = float(era5_meta.get("returned_elevation_m") or 0.0)

            for yr in years:
                rel_zips = cfg["years"].get(yr, [])
                zpaths = [root_path / rp for rp in rel_zips if (root_path / rp).exists()]
                if not zpaths:
                    continue
                for zp in zpaths:
                    h = hashlib.sha256()
                    with zp.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(chunk)
                    archive_digests[str(zp.relative_to(root_path))] = h.hexdigest()

                print(f"[WorkbenchStore] Loading {farm} {yr} raw archives...", flush=True)
                df_yr, _dup = load_rich_telemetry(zpaths, farm, year=yr)
                ev_yr = read_status_events(zpaths, farm, year=yr)
                print(
                    f"[WorkbenchStore] {farm} {yr}: {len(df_yr):,} telemetry rows, {len(ev_yr):,} events",
                    flush=True,
                )
                if not df_yr.empty:
                    df_yr["turbine_id"] = df_yr["turbine_id"].astype(str).map(normalize_turbine_id)
                if not ev_yr.empty:
                    ev_yr["turbine_id"] = ev_yr["turbine_id"].astype(str).map(normalize_turbine_id)
                    ev_yr["subsystem"] = _assign_subsystem_column(ev_yr)
                    evt_frames.append(ev_yr)

                if df_yr.empty:
                    continue

                # Ensure every canonical column exists even when farm sensors differ (e.g., pitch on Penmanshiel)
                for col in _TELEMETRY_FLOAT_COLS:
                    if col not in df_yr.columns:
                        df_yr[col] = np.nan

                df_yr["year"] = df_yr["timestamp_utc"].dt.year.astype("int16")
                df_yr["month"] = df_yr["timestamp_utc"].dt.month.astype("int8")
                df_yr["wind_speed_ms"] = df_yr["wind_speed_ms"].where(df_yr["wind_speed_ms"] >= 0.0)
                df_yr["wind_direction_deg"] = (
                    df_yr["wind_direction_deg"].where(df_yr["wind_direction_deg"].between(0.0, 360.0)) % 360.0
                )

                # Derived TI & Pitch spread
                df_yr["ti"] = np.where(
                    (df_yr["wind_speed_ms"] >= 3.0) & df_yr["wind_speed_std_ms"].notna(),
                    df_yr["wind_speed_std_ms"] / df_yr["wind_speed_ms"].clip(lower=1.0),
                    np.nan,
                )
                pitch_stack = df_yr[["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]]
                df_yr["pitch_spread_deg"] = pitch_stack.max(axis=1) - pitch_stack.min(axis=1)

                # Fill contractual MPC from oem_power_curves.json where missing
                if oem_knots:
                    fallback_mpc = evaluate_oem_curve(df_yr["wind_speed_ms"], oem_knots)
                    df_yr["potential_power_mpc_kw"] = df_yr["potential_power_mpc_kw"].fillna(fallback_mpc)

                # Publisher implied density
                ws_arr = df_yr["wind_speed_ms"].to_numpy(dtype=float)
                pub_ws_arr = df_yr["density_adjusted_ws_ms"].to_numpy(dtype=float)
                ok_pub = np.isfinite(ws_arr) & np.isfinite(pub_ws_arr) & (ws_arr >= 3.0) & (pub_ws_arr >= 3.0)
                rho_pub = np.full(len(df_yr), np.nan, dtype=float)
                rho_pub[ok_pub] = STANDARD_DENSITY * (pub_ws_arr[ok_pub] / ws_arr[ok_pub]) ** 3
                df_yr["density_publisher_implied_kg_m3"] = rho_pub

                # Pre-split stop and warning events by turbine once per farm-year
                stop_by_tid: dict[str, pd.DataFrame] = {}
                warn_by_tid: dict[str, pd.DataFrame] = {}
                if not ev_yr.empty:
                    stop_sub = ev_yr[ev_yr["status"] == "Stop"]
                    if not stop_sub.empty:
                        stop_by_tid = {str(k): v for k, v in stop_sub.groupby("turbine_id", sort=False)}
                    warn_sub = ev_yr[ev_yr["status"].isin(["Warning", "Communication"])]
                    if not warn_sub.empty:
                        warn_as_stop = warn_sub.assign(status="Stop")
                        warn_by_tid = {str(k): v for k, v in warn_as_stop.groupby("turbine_id", sort=False)}

                # Per-turbine enrichment: ERA5 density, event stop/warning masks, calibrated direction, wake flag
                enriched_parts: list[pd.DataFrame] = []
                for tid, grp in df_yr.groupby("turbine_id", sort=False):
                    g = grp.sort_values("timestamp_utc").copy()
                    ts_idx = pd.DatetimeIndex(g["timestamp_utc"])

                    if has_era5 and tid in meta:
                        target_elev = float(meta[tid]["elevation_m"]) + float(meta[tid]["hub_height_m"])
                        temp_s = pd.Series(g["ambient_temperature_c"].to_numpy(dtype=float), index=ts_idx)
                        dens = compute_hub_density_series(
                            ts_idx,
                            temp_s,
                            era5_df,
                            era5_elevation_m=era5_elev,
                            target_elevation_m=target_elev,
                        )
                        g["density_moist_kg_m3"] = dens["density_moist_kg_m3"].to_numpy()
                        g["density_dry_kg_m3"] = dens["density_dry_kg_m3"].to_numpy()
                        g["hub_pressure_hpa"] = dens["hub_pressure_hpa"].to_numpy()
                    else:
                        # Fallback hydrostatic dry density when offline ERA5 JSON absent
                        t_k = g["ambient_temperature_c"].fillna(10.0).to_numpy(dtype=float) + 273.15
                        g["density_moist_kg_m3"] = 1013.25 * 100.0 / (287.05 * t_k)
                        g["density_dry_kg_m3"] = g["density_moist_kg_m3"]
                        g["hub_pressure_hpa"] = 1013.25

                    rho_moist = g["density_moist_kg_m3"].to_numpy(dtype=float)
                    g_ws = g["wind_speed_ms"].to_numpy(dtype=float)
                    g["ws_iec_ms"] = vector_iec_wind_speed(g_ws, rho_moist, STANDARD_DENSITY)
                    g["ws_svenningsen_ms"] = vector_svenningsen_fast(g_ws, rho_moist, STANDARD_DENSITY)

                    # Event stop mask and warning mask
                    tid_str = str(tid)
                    if tid_str in stop_by_tid:
                        stop_mask_s, _ = build_event_stop_mask(ts_idx, stop_by_tid[tid_str], tid_str)
                        g["event_stop_mask"] = stop_mask_s.to_numpy(dtype=bool)
                    else:
                        g["event_stop_mask"] = False
                    if tid_str in warn_by_tid:
                        w_mask_s, _ = build_event_stop_mask(ts_idx, warn_by_tid[tid_str], tid_str)
                        g["event_warning_mask"] = w_mask_s.to_numpy(dtype=bool)
                    else:
                        g["event_warning_mask"] = False

                    # Calibrated wind direction & wake sector membership
                    offset = dir_offsets.get((farm.lower(), tid), 0.0)
                    g["wd_calibrated_deg"] = (g["wind_direction_deg"] - offset) % 360.0
                    g["in_geometric_wake"] = _compute_geometric_wake_flags(g["wd_calibrated_deg"], tid, layout_pts)
                    enriched_parts.append(g)

                df_enriched = pd.concat(enriched_parts, ignore_index=True)

                # IEC availability unavail flag & physical curtailment/derate detection
                avail_b22 = df_enriched["avail_time_iec_b22"]
                df_enriched["iec_unavail_mask"] = avail_b22.notna() & (avail_b22 < 0.99)

                cap_lim = df_enriched["available_capacity_kw"].notna() & (df_enriched["available_capacity_kw"] < 1980.0)
                shelf_derate = (
                    (~df_enriched["event_stop_mask"])
                    & (df_enriched["wind_speed_ms"] >= 9.0)
                    & (df_enriched["active_power_kw"] > 50.0)
                    & (df_enriched["potential_power_mpc_kw"] > 500.0)
                    & (df_enriched["active_power_kw"] < 0.80 * df_enriched["potential_power_mpc_kw"])
                )
                df_enriched["curtailment_mask"] = cap_lim | shelf_derate

                cut_in = float(oem_payload.get("curves", {}).get(model, {}).get("cut_in_ms", 3.0))
                df_enriched["state_category"] = np.where(
                    df_enriched["event_stop_mask"],
                    "Event Outage (Stop)",
                    np.where(
                        df_enriched["curtailment_mask"],
                        "Curtailment / Derate",
                        np.where(
                            df_enriched["wind_speed_ms"].isna() | (df_enriched["wind_speed_ms"] < cut_in),
                            "Under-Cut-In / Calm",
                            "Clean Normal",
                        ),
                    ),
                )

                keep_cols = [
                    "farm_id",
                    "turbine_id",
                    "timestamp_utc",
                    "year",
                    "month",
                    "state_category",
                    *_TELEMETRY_BOOL_COLS,
                    *_TELEMETRY_FLOAT_COLS,
                ]
                df_enriched = df_enriched[[c for c in keep_cols if c in df_enriched.columns]]
                tel_frames.append(df_enriched)

        telemetry_df = pd.concat(tel_frames, ignore_index=True) if tel_frames else pd.DataFrame()
        events_df = pd.concat(evt_frames, ignore_index=True) if evt_frames else pd.DataFrame()
        layout_df = pd.concat(lay_frames, ignore_index=True) if lay_frames else pd.DataFrame()
        print(
            f"[WorkbenchStore] Building Event Code Dictionary for {len(events_df):,} events "
            f"and {len(telemetry_df):,} telemetry rows...",
            flush=True,
        )
        dict_df = build_event_code_dictionary(events_df, telemetry_df)

        has_pyarrow = importlib.util.find_spec("pyarrow") is not None
        ext = "parquet" if has_pyarrow else "pkl.gz"
        print(f"[WorkbenchStore] Serializing cache ({ext}) to {cdir}...", flush=True)
        if has_pyarrow:
            telemetry_df.to_parquet(cdir / "telemetry_10min.parquet", index=False, compression="zstd")
            events_df.to_parquet(cdir / "events_raw.parquet", index=False, compression="zstd")
            dict_df.to_parquet(cdir / "event_code_dictionary.parquet", index=False, compression="zstd")
            layout_df.to_parquet(cdir / "layout.parquet", index=False, compression="zstd")
        else:
            telemetry_df.to_pickle(cdir / "telemetry_10min.pkl.gz", compression="gzip")
            events_df.to_pickle(cdir / "events_raw.pkl.gz", compression="gzip")
            dict_df.to_pickle(cdir / "event_code_dictionary.pkl.gz", compression="gzip")
            layout_df.to_pickle(cdir / "layout.pkl.gz", compression="gzip")

        manifest = {
            "schema_version": 2,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "storage_format": ext,
            "default_rho_ref_kg_m3": STANDARD_DENSITY,
            "farms": list(farms),
            "years": list(years),
            "telemetry_rows": int(len(telemetry_df)),
            "events_rows": int(len(events_df)),
            "dictionary_rows": int(len(dict_df)),
            "archive_sha256": archive_digests,
        }
        (cdir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return cls(
            root=root_path,
            cache_dir=cdir,
            telemetry_df=telemetry_df,
            events_df=events_df,
            event_dictionary_df=dict_df,
            layout_df=layout_df,
            manifest=manifest,
        )
