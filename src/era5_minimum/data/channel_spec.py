from dataclasses import dataclass
from typing import List, Literal
import xarray as xr

@dataclass(frozen=True)
class Channel:
    index: int
    name: str
    wb2_name: str
    units: str
    level: int | None = None
    type: Literal["surface", "pressure"] = "surface"

SURFACE_CHANNELS = [
    Channel(0, "t2m", "2m_temperature", "K"),
    Channel(1, "mslp", "mean_sea_level_pressure", "Pa"),
    Channel(2, "u10", "10m_u_component_of_wind", "m s-1"),
    Channel(3, "v10", "10m_v_component_of_wind", "m s-1"),
    Channel(4, "tp6h", "total_precipitation_6hr", "m"),
    Channel(5, "sst", "sea_surface_temperature", "K"),
    Channel(6, "tcwv", "total_column_water_vapour", "kg m-2"),
    Channel(7, "tcc", "total_cloud_cover", "0-1"),
]

PRESSURE_VARS = [
    ("t", "temperature", "K"),
    ("u", "u_component_of_wind", "m s-1"),
    ("v", "v_component_of_wind", "m s-1"),
    ("z", "geopotential", "m2 s-2"),
    ("q", "specific_humidity", "kg kg-1"),
]
LEVELS = [1000, 925, 850, 700]

CHANNEL_SPEC: List[Channel] = list(SURFACE_CHANNELS)
_idx = 8
for var_short, wb_name, unit in PRESSURE_VARS:
    for level in LEVELS:
        CHANNEL_SPEC.append(Channel(
            index=_idx,
            name=f"{var_short}{level}",
            wb2_name=f"{wb_name}_{level}hPa",
            units=unit,
            level=level,
            type="pressure"
        ))
        _idx += 1

assert len(CHANNEL_SPEC) == 28, "Должно быть ровно 28 каналов"

def validate_wb2_compatibility(ds: xr.Dataset, channels: list = None) -> None:
    """
    Проверяет, что открытый WeatherBench2 датасет содержит все необходимые
    переменные и уровни для формирования 28 каналов.

    Args:
        ds: Открытый xarray.Dataset из WeatherBench2.
        channels: Список каналов для проверки (по умолчанию CHANNEL_SPEC).

    Raises:
        ValueError: Если отсутствует переменная или уровень давления.
    """
    if channels is None:
        channels = CHANNEL_SPEC

    available_vars = set(ds.data_vars) | set(ds.coords)

    # Группируем каналы по wb2_name для оптимизации проверок
    required_wb2_names = set(ch.wb2_name for ch in channels)

    # 1. Проверка наличия всех переменных
    missing_vars = required_wb2_names - available_vars
    if missing_vars:
        raise ValueError(
            f"Missing variable(s) in WeatherBench2 dataset: {missing_vars}. "
            f"Available: {available_vars}"
        )

    # 2. Проверка наличия всех уровней давления для pressure-каналов
    pressure_channels = [ch for ch in channels if ch.type == "pressure"]
    if pressure_channels:
        if "level" not in ds.coords:
            raise ValueError("Missing 'level' coordinate for pressure variables")

        available_levels = set(ds.coords["level"].values)
        required_levels = set(ch.level for ch in pressure_channels)

        missing_levels = required_levels - available_levels
        if missing_levels:
            raise ValueError(
                f"Missing pressure level(s): {missing_levels}. "
                f"Available levels: {available_levels}"
            )

    print(f" Dataset validation passed: {len(channels)} channels compatible")