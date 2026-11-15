"""Data loading helpers for the WINTELLISYT engineering & asset intelligence dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

TABLE_NAMES: tuple[str, ...] = (
    "layout",
    "anemometer_ntf_audit",
    "direction_consensus",
    "direction_monthly",
    "direction_wake_troughs",
    "vane_nacelle_audit",
    "wind_rose",
    "wake_exposure_counts",
    "wake_bin_effects",
    "wake_regime_effects",
    "pc_health_summary",
    "pc_health_daily",
    "pitch_asymmetry",
    "pitch_ppr_recovery",
    "peer_gap_benchmark",
    "peer_drift_summary",
    "density_benchmark",
    "energy_waterfall",
    "mpc_vs_learned_gap",
    "yaw_misalignment_summary",
    "yaw_misalignment_bins",
    "pitch_actuator_health",
    "drivetrain_cms_summary",
    "wake_vibration_fatigue",
)


@dataclass
class SuiteBundle:
    """Container for outputs produced by `scripts/run_engineering_suite.py`."""

    suite_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    signal_inventory_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get(self, name: str) -> pd.DataFrame:
        """Return a named output table or an empty DataFrame if absent."""
        return self.tables.get(name, pd.DataFrame())


def _summarize_signal_inventory(inventory_path: Path) -> pd.DataFrame:
    if not inventory_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(
        inventory_path,
        usecols=[
            "farm",
            "year",
            "turbine",
            "source_header",
            "canonical_name",
            "physical_meaning",
            "analysis_eligibility",
            "unit_in_header",
            "rows",
            "finite_count",
            "nan_token_count",
        ],
        low_memory=False,
    )
    if raw.empty:
        return pd.DataFrame()
    grouped = (
        raw.groupby(["farm", "source_header", "canonical_name", "analysis_eligibility", "unit_in_header"], dropna=False)[
            ["rows", "finite_count", "nan_token_count"]
        ]
        .sum()
        .reset_index()
    )
    grouped["availability_pct"] = (
        100.0 * grouped["finite_count"] / grouped["rows"].clip(lower=1)
    ).round(2)
    return grouped.sort_values(["farm", "availability_pct", "source_header"], ascending=[True, False, True]).reset_index(
        drop=True
    )


def load_suite_bundle(
    suite_dir: str | Path = "artifacts/engineering_suite_2017",
    *,
    root: str | Path = ".",
) -> SuiteBundle:
    """Load the engineering suite CSV tables, manifest, and summarized signal inventory."""
    suite_path = Path(suite_dir)
    if not suite_path.is_absolute():
        suite_path = Path(root) / suite_path
    manifest_path = suite_path / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        csv_path = suite_path / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"turbine_id": str})
            if "turbine_id" in df.columns:
                df["turbine_id"] = df["turbine_id"].astype(str).str.zfill(2)
            for col in ("donor", "target"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.zfill(2)
            tables[name] = df

    inv_path = Path(root) / "data" / "reference" / "signal_inventory.csv"
    sig_summary = _summarize_signal_inventory(inv_path)
    return SuiteBundle(
        suite_dir=suite_path,
        manifest=manifest,
        tables=tables,
        signal_inventory_summary=sig_summary,
    )
