"""Lazy WeatherBench2 Zarr provider used by the real-layer API routes."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import xarray as xr

from era5_minimum.data.channel_spec import CHANNEL_SPEC


class WeatherDataProvider(Protocol):
    def dataset_metadata(self) -> dict[str, Any]: ...
    def variables(self) -> list[dict[str, Any]]: ...
    def timestamps(self) -> list[str]: ...
    def layer(self, variable: str, timestamp: str, level: int | None,
              target_width: int, target_height: int) -> dict[str, Any]: ...


class WeatherProviderError(ValueError):
    pass


class ZarrWeatherDataProvider:
    """One opened native-variable Zarr store; each call computes one 2-D layer."""

    def __init__(self, root: str | Path, split: str) -> None:
        self.root, self.split = Path(root), split
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise WeatherProviderError(f"dataset manifest does not exist: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = self.root / f"{split}.zarr"
        if not store.is_dir():
            raise WeatherProviderError(f"dataset split does not exist: {store}")
        self.ds = xr.open_zarr(store, consolidated=True)
        self.static = xr.open_zarr(self.root / "static.zarr", consolidated=True)
        self.by_name = {channel.name: channel for channel in CHANNEL_SPEC}

    def dataset_metadata(self) -> dict[str, Any]:
        times = self.timestamps()
        return {"id": self.manifest.get("dataset_id", self.root.name), "source": self.manifest["source_uri"],
                "split": self.split, "grid": "721x1440", "resolution": self.manifest.get("resolution"),
                "time_start": times[0], "time_end": times[-1], "timestamp_count": len(times),
                "channel_count": len(CHANNEL_SPEC), "is_mock": False}

    def variables(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for channel in CHANNEL_SPEC:
            levels = [channel.level] if channel.level is not None else []
            result.append({"logical_name": channel.name, "display_name": channel.name,
                           "physical_source_name": channel.source_name, "unit": channel.units,
                           "kind": channel.type, "allowed_levels": levels,
                           "default_level": channel.level, "has_mask": channel.name == "sst"})
        return result

    def timestamps(self) -> list[str]:
        return [np.datetime_as_string(value, unit="s") + "Z" for value in self.ds.time.values]

    def layer(self, variable: str, timestamp: str, level: int | None,
              target_width: int = 360, target_height: int = 180) -> dict[str, Any]:
        channel = self.by_name.get(variable)
        if channel is None: raise WeatherProviderError(f"unknown variable: {variable}")
        if channel.level is not None and level != channel.level:
            raise WeatherProviderError(f"{variable} requires level={channel.level}")
        if channel.level is None and level is not None:
            raise WeatherProviderError(f"surface variable {variable} does not accept level")
        if timestamp not in self.timestamps(): raise WeatherProviderError(f"unknown timestamp: {timestamp}")
        field = self.ds[channel.source_name].sel(time=np.datetime64(timestamp.removesuffix("Z")))
        if channel.level is not None: field = field.sel(level=channel.level)
        h, w = field.sizes["latitude"], field.sizes["longitude"]
        if not (1 <= target_width <= w and 1 <= target_height <= h):
            raise WeatherProviderError("target dimensions must be within source grid")
        rows = np.linspace(0, h - 1, target_height, dtype=int)
        cols = np.linspace(0, w - 1, target_width, dtype=int)
        # This is display sampling, explicitly not scientific/remapping output.
        view = field.isel(latitude=rows, longitude=cols).load()
        values = view.values.astype(np.float32, copy=False)
        valid = np.isfinite(values)
        safe_values = np.where(valid, values, np.nan)
        return {"dataset_id": self.dataset_metadata()["id"], "variable": variable, "timestamp": timestamp,
                "level": channel.level, "mode": "original", "unit": channel.units,
                "latitude": view.latitude.values.tolist(), "longitude": view.longitude.values.tolist(),
                "values": [[None if not finite else float(value) for value, finite in zip(row, mask)] for row, mask in zip(safe_values, valid)],
                "mask": valid.tolist(), "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
                "shape": [target_height, target_width], "is_mock": False,
                "visualization_sampling": "index sampling only; not conservative remapping"}


@lru_cache(maxsize=1)
def get_weather_provider() -> WeatherDataProvider:
    provider = os.getenv("ERA5_WEATHER_PROVIDER", "zarr")
    if provider != "zarr":
        raise WeatherProviderError(f"weather provider {provider!r} is unavailable; set ERA5_WEATHER_PROVIDER=zarr")
    return ZarrWeatherDataProvider(os.getenv("ERA5_DATASET_ROOT", "data/era5_28ch_demo"),
                                   os.getenv("ERA5_DATASET_SPLIT", "validation"))
