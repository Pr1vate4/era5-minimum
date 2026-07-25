"""Explicit CRA5-159 to the canonical ERA5-28 channel mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from era5_minimum.data.channel_spec import CHANNEL_NAMES

CRA5_PRESSURE_VARIABLES: tuple[str, ...] = ("z", "q", "u", "v", "t", "w")
CRA5_PRESSURE_LEVELS: tuple[int, ...] = (
    1000,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
    20,
    10,
    7,
    5,
    3,
    2,
    1,
)
CRA5_SURFACE_VARIABLES: tuple[str, ...] = (
    "v10",
    "u10",
    "v100",
    "u100",
    "t2m",
    "tcc",
    "sp",
    "tp6h",
    "msl",
)
CRA5_PRESSURE_CHANNEL_COUNT = len(CRA5_PRESSURE_VARIABLES) * len(CRA5_PRESSURE_LEVELS)
CRA5_CHANNEL_COUNT = CRA5_PRESSURE_CHANNEL_COUNT + len(CRA5_SURFACE_VARIABLES)


@dataclass(frozen=True, slots=True)
class Cra5ChannelMapping:
    """One canonical output channel and its optional CRA5 source channel."""

    canonical_name: str
    cra5_name: str | None
    cra5_index: int | None
    initialization: Literal["copy", "learned_boundary"]


def _pressure_index(variable: str, level: int) -> int:
    return (
        CRA5_PRESSURE_VARIABLES.index(variable) * len(CRA5_PRESSURE_LEVELS)
        + CRA5_PRESSURE_LEVELS.index(level)
    )


def _pressure_name(variable: str, level: int) -> str:
    return f"{variable}{level}"


def _copy(canonical_name: str, cra5_name: str, cra5_index: int) -> Cra5ChannelMapping:
    return Cra5ChannelMapping(canonical_name, cra5_name, cra5_index, "copy")


def _build_mapping() -> tuple[Cra5ChannelMapping, ...]:
    rows: list[Cra5ChannelMapping] = []

    rows.extend(
        (
            Cra5ChannelMapping("t2m", "t2m", 150 + CRA5_SURFACE_VARIABLES.index("t2m"), "copy"),
            Cra5ChannelMapping("mslp", "msl", 150 + CRA5_SURFACE_VARIABLES.index("msl"), "copy"),
            Cra5ChannelMapping("u10", "u10", 150 + CRA5_SURFACE_VARIABLES.index("u10"), "copy"),
            Cra5ChannelMapping("v10", "v10", 150 + CRA5_SURFACE_VARIABLES.index("v10"), "copy"),
            Cra5ChannelMapping("tp6h", "tp6h", 150 + CRA5_SURFACE_VARIABLES.index("tp6h"), "copy"),
            Cra5ChannelMapping("sst", None, None, "learned_boundary"),
            Cra5ChannelMapping("tcwv", None, None, "learned_boundary"),
            Cra5ChannelMapping("tcc", "tcc", 150 + CRA5_SURFACE_VARIABLES.index("tcc"), "copy"),
        )
    )

    for variable, canonical_prefix in (
        ("t", "T"),
        ("u", "U"),
        ("v", "V"),
        ("z", "Z"),
        ("q", "Q"),
    ):
        for level in (1000, 925, 850, 700):
            rows.append(
                _copy(
                    f"{canonical_prefix}{level}",
                    _pressure_name(variable, level),
                    _pressure_index(variable, level),
                )
            )
    return tuple(rows)


CRA5_CHANNEL_MAPPING = _build_mapping()


def validate_cra5_channel_mapping(
    mapping: tuple[Cra5ChannelMapping, ...] = CRA5_CHANNEL_MAPPING,
) -> None:
    """Validate order, initialization semantics, and unique CRA5 indices."""
    if tuple(row.canonical_name for row in mapping) != CHANNEL_NAMES:
        raise ValueError("CRA5 mapping canonical order does not match CHANNEL_NAMES")
    if len(mapping) != len(CHANNEL_NAMES):
        raise ValueError(f"CRA5 mapping must contain {len(CHANNEL_NAMES)} rows")

    indices = [row.cra5_index for row in mapping if row.cra5_index is not None]
    if len(indices) != len(set(indices)):
        raise ValueError("CRA5 mapping contains duplicate source indices")
    if any(index < 0 or index >= CRA5_CHANNEL_COUNT for index in indices):
        raise ValueError("CRA5 mapping contains an out-of-range source index")

    for row in mapping:
        if row.initialization == "copy" and (row.cra5_name is None or row.cra5_index is None):
            raise ValueError(f"Copied channel {row.canonical_name!r} has no CRA5 source")
        if row.initialization == "learned_boundary" and (
            row.cra5_name is not None or row.cra5_index is not None
        ):
            raise ValueError(f"Learned boundary {row.canonical_name!r} has a CRA5 source")
        if row.initialization not in {"copy", "learned_boundary"}:
            raise ValueError(f"Unknown initialization for {row.canonical_name!r}")


validate_cra5_channel_mapping()
