"""Wind engineering diagnostics in SI units."""

from math import isfinite, pi

STANDARD_DENSITY = 1.225


def _positive(value: float) -> bool:
    return isfinite(value) and value > 0


def swept_area(radius_m: float) -> float:
    """Rotor area in m²."""
    if not _positive(radius_m):
        raise ValueError("Rotor radius must be positive and finite")
    return pi * radius_m**2


def dry_air_density(pressure_pa: float, temperature_k: float) -> float:
    """Ideal dry air density in kg/m³, without humidity correction."""
    if not _positive(pressure_pa) or not _positive(temperature_k):
        raise ValueError("Pressure and absolute temperature must be positive and finite")
    return pressure_pa / (287.05 * temperature_k)


def wind_power(speed_ms: float, radius_m: float, density: float) -> float:
    """Available wind power in watts for the supplied speed and density."""
    if not isfinite(speed_ms) or speed_ms < 0 or not _positive(density):
        raise ValueError("Wind speed and density outside physical input domain")
    return 0.5 * density * swept_area(radius_m) * speed_ms**3


def cp_electrical(power_kw: float, speed_ms: float, radius_m: float, density: float) -> float | None:
    """Effective electrical ratio using a supplied wind-speed proxy."""
    if not isfinite(power_kw) or not isfinite(speed_ms) or speed_ms < 3:
        return None
    return 1000 * power_kw / wind_power(speed_ms, radius_m, density)


def tip_speed_ratio(rotor_rpm: float, speed_ms: float, radius_m: float) -> float | None:
    """Dimensionless ratio using rotor RPM and a 3 m/s guard."""
    if not all(map(isfinite, (rotor_rpm, speed_ms))) or rotor_rpm < 0 or speed_ms < 3:
        return None
    if not _positive(radius_m):
        raise ValueError("Invalid rotor radius")
    return (2 * pi / 60) * rotor_rpm * radius_m / speed_ms


def turbulence_proxy(std_ms: float, speed_ms: float) -> float | None:
    if not all(map(isfinite, (std_ms, speed_ms))) or std_ms < 0 or speed_ms < 3:
        return None
    return std_ms / speed_ms


def normalized_speed(speed_ms: float, density: float, reference_density: float) -> float:
    """Density-equivalent speed candidate for below-rated operation."""
    if not isfinite(speed_ms) or speed_ms < 0 or not all(map(_positive, (density, reference_density))):
        raise ValueError("Invalid speed or density")
    return speed_ms * (density / reference_density)**(1 / 3)


def relative_direction(first_deg: float, second_deg: float) -> float:
    """Signed angular difference in [-180, 180) degrees."""
    if not all(map(isfinite, (first_deg, second_deg))):
        raise ValueError("Directions must be finite")
    return (first_deg - second_deg + 180) % 360 - 180
