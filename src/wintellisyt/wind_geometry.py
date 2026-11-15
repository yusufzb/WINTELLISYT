"""Conditional farm geometry and circular direction primitives."""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, isfinite, radians, sin

from wintellisyt.physics import relative_direction


@dataclass(frozen=True)
class LayoutPoint:
    latitude: float
    longitude: float
    rotor_m: float

    def __post_init__(self):
        if (not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180
                or not isfinite(self.rotor_m) or self.rotor_m <= 0):
            raise ValueError("Invalid layout coordinate or rotor diameter")


def pair_vector(target: LayoutPoint, source: LayoutPoint) -> tuple[float, float]:
    """Local east/north of source from target, in metres."""
    scale = 6_371_008.8
    east = scale * radians(relative_direction(source.longitude, target.longitude)) * cos(
        radians((source.latitude + target.latitude) / 2)
    )
    north = scale * radians(source.latitude - target.latitude)
    if not east and not north:
        raise ValueError("Duplicate turbine position")
    return east, north


def source_bearing(east_m: float, north_m: float) -> float:
    if not all(map(isfinite, (east_m, north_m))) or not (east_m or north_m):
        raise ValueError("Invalid bearing vector")
    return degrees(atan2(east_m, north_m)) % 360


def wake_projection(east_m: float, north_m: float, direction_from_deg: float) -> tuple[float, float]:
    """Positive first component means target is downstream of source."""
    angle = radians(direction_from_deg)
    return east_m * sin(angle) + north_m * cos(angle), east_m * cos(angle) - north_m * sin(angle)


def potential_overlap(
    east_m: float, north_m: float, direction_from_deg: float, source_diameter_m: float,
    target_diameter_m: float, expansion: float = .05, max_diameters: float = 12,
) -> bool:
    downstream, lateral = wake_projection(east_m, north_m, direction_from_deg)
    return (hypot(east_m, north_m) <= max_diameters * source_diameter_m
            and downstream > 0
            and abs(lateral) <= (source_diameter_m + target_diameter_m) / 2 + expansion * downstream)
