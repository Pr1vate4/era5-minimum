"""Canonical 28-channel WeatherBench2 contract for the hackathon dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Channel:
    index: int
    name: str
    source_name: str
    units: str
    level: int | None = None
    type: Literal["surface", "pressure"] = "surface"

    @property
    def wb2_name(self) -> str:
        """Compatibility alias; pressure level is a coordinate, not a variable name."""
        return self.source_name


_SURFACE = (
    ("t2m", "2m_temperature", "K"),
    ("mslp", "mean_sea_level_pressure", "Pa"),
    ("u10", "10m_u_component_of_wind", "m s-1"),
    ("v10", "10m_v_component_of_wind", "m s-1"),
    ("tp6h", "total_precipitation_6hr", "m"),
    ("sst", "sea_surface_temperature", "K"),
    ("tcwv", "total_column_water_vapour", "kg m-2"),
    ("tcc", "total_cloud_cover", "0-1"),
)
_PRESSURE = (
    ("T", "temperature", "K"),
    ("U", "u_component_of_wind", "m s-1"),
    ("V", "v_component_of_wind", "m s-1"),
    ("Z", "geopotential", "m2 s-2"),
    ("Q", "specific_humidity", "kg kg-1"),
)
PRESSURE_LEVELS: tuple[int, ...] = (1000, 925, 850, 700)

CHANNEL_SPEC: tuple[Channel, ...] = tuple(
    [Channel(i, name, source, unit) for i, (name, source, unit) in enumerate(_SURFACE)]
    + [
        Channel(8 + var_index * len(PRESSURE_LEVELS) + level_index,
                f"{short}{level}", source, unit, level, "pressure")
        for var_index, (short, source, unit) in enumerate(_PRESSURE)
        for level_index, level in enumerate(PRESSURE_LEVELS)
    ]
)
CHANNEL_NAMES: tuple[str, ...] = tuple(channel.name for channel in CHANNEL_SPEC)


def validate_wb2_compatibility(ds: xr.Dataset, channels: tuple[Channel, ...] = CHANNEL_SPEC) -> None:
    """Fail fast when a WeatherBench2-like source changes its required schema."""
    missing = sorted({channel.source_name for channel in channels} - set(ds.data_vars))
    if missing:
        raise ValueError(f"Missing variable(s) in WeatherBench2 dataset: {missing}")
    if "level" not in ds.coords:
        raise ValueError("Missing level coordinate for pressure channels")
    available = {int(value) for value in np.asarray(ds.level.values)}
    missing_levels = sorted(set(PRESSURE_LEVELS) - available)
    if missing_levels:
        raise ValueError(f"Missing pressure level(s): {missing_levels}")
    for channel in channels:
        actual = str(ds[channel.source_name].attrs.get("units", ""))
        normalized_actual = actual.replace("**", "").replace(" ", "")
        normalized_expected = channel.units.replace("**", "").replace(" ", "")
        accepted = {normalized_expected, "0–1", "(0-1)"}
        if actual and normalized_actual not in accepted:
            raise ValueError(f"Unexpected units for {channel.source_name}: {actual!r}")
