"""SCADA ingestion and provenance acquisition helpers."""

from .scada import (
    CADENCE_MINUTES,
    RICH_CHANNEL_MAP,
    ScadaRecord,
    load_rich_telemetry,
    load_static_layout,
    read_scada,
    read_status_events,
)

__all__ = [
    "CADENCE_MINUTES",
    "RICH_CHANNEL_MAP",
    "ScadaRecord",
    "load_rich_telemetry",
    "load_static_layout",
    "read_scada",
    "read_status_events",
]
