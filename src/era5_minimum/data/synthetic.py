from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_CHANNELS = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")


def make_synthetic_era5(
    samples: int,
    height: int,
    width: int,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth, correlated ERA5-like fields for pipeline validation.

    Returns:
        data: float32 array with shape [time, channel, latitude, longitude].
        latitudes: float32 array in degrees, north to south.
    """
    if samples < 1 or height < 8 or width < 8:
        raise ValueError("samples must be positive and spatial dimensions must be at least 8")

    rng = np.random.default_rng(seed)
    latitudes = np.linspace(90.0, -90.0, height, dtype=np.float32)
    longitudes = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
    lat = np.deg2rad(latitudes)[:, None]
    lon = np.deg2rad(longitudes)[None, :]

    data = np.empty((samples, len(channels), height, width), dtype=np.float32)
    for t in range(samples):
        phase = 2.0 * math.pi * t / max(samples, 1)
        synoptic = np.sin(2 * lon + phase) * np.cos(lat)
        planetary = np.cos(lon - phase / 2) * np.cos(2 * lat)
        storm = np.exp(-((lat - 0.45 * np.sin(phase)) ** 2 + (lon - math.pi - phase) ** 2) / 0.45)
        noise = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)

        fields: dict[str, np.ndarray] = {
            "t2m": 273.15 + 28 * np.cos(lat) + 5 * synoptic + 0.5 * noise,
            "mslp": 101325 + 1700 * planetary - 2500 * storm + 60 * noise,
            "u10": 8 * np.sin(lat) + 5 * synoptic + 0.3 * noise,
            "v10": 5 * np.cos(2 * lat) + 4 * planetary + 0.3 * noise,
            "tp6h": np.clip(0.002 * storm + 0.0002 * np.maximum(synoptic, 0) + 0.00002 * noise, 0, None),
            "sst": 271 + 25 * np.cos(lat) + 2 * synoptic + 0.3 * noise,
            "tcwv": np.clip(8 + 42 * np.cos(lat) ** 2 + 5 * synoptic + 0.4 * noise, 0, None),
            "tcc": np.clip(0.45 + 0.35 * storm + 0.18 * synoptic + 0.04 * noise, 0, 1),
        }
        for index, name in enumerate(channels):
            if name not in fields:
                raise ValueError(f"Unsupported synthetic channel: {name}")
            data[t, index] = fields[name]

    return data, latitudes


class SyntheticERA5Dataset(Dataset[torch.Tensor]):
    """In-memory normalized synthetic dataset with train-fitted statistics."""

    def __init__(
        self,
        data: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        if data.ndim != 4:
            raise ValueError("data must have shape [time, channel, latitude, longitude]")
        if mean is None:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
        if std is None:
            std = data.std(axis=(0, 2, 3), keepdims=True)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, 1e-6).astype(np.float32)
        normalized = (data - self.mean) / self.std
        self.tensor = torch.from_numpy(normalized.astype(np.float32))

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]
