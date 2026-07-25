"""Tests for the dependency-light conservative WeatherBench2 remapper."""
from __future__ import annotations

import numpy as np
import xarray as xr

from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.remapping import _WEIGHT_CACHE, conservative_remap


def _small_global_dataset(values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"field": (("latitude", "longitude"), values)},
        coords={
            # ERA5 order is north-to-south, not the ascending order many
            # regridding examples assume.
            "latitude": [90.0, 30.0, -30.0, -90.0],
            "longitude": np.arange(0.0, 360.0, 45.0),
        },
    )


def _small_target() -> xr.Dataset:
    return xr.Dataset(
        coords={
            "latitude": [-67.5, -22.5, 22.5, 67.5],
            "longitude": np.arange(22.5, 360.0, 45.0),
        }
    )


def test_target_grid_exposes_named_coordinate_axes() -> None:
    target = get_target_grid_05()
    assert target.sizes == {"latitude": 360, "longitude": 720}
    assert target.latitude.values[0] == -89.75
    assert target.longitude.values[-1] == 359.75


def test_conservative_remap_handles_descending_era5_latitude() -> None:
    remapped = conservative_remap(_small_global_dataset(np.full((4, 8), 5.0)), _small_target())
    np.testing.assert_allclose(remapped.field.values, 5.0, rtol=0.0, atol=1e-6)


def test_conservative_remap_renormalises_missing_source_values() -> None:
    values = np.full((4, 8), 10.0, dtype=np.float32)
    values[0, 0] = np.nan
    remapped = conservative_remap(_small_global_dataset(values), _small_target())
    # The affected target cell remains a valid 10 K cell rather than being
    # diluted by a zero-filled missing value.
    assert remapped.field.isel(latitude=-1, longitude=0).item() == 10.0


def test_conservative_remap_reuses_coordinate_weights() -> None:
    _WEIGHT_CACHE.clear()
    source = _small_global_dataset(np.full((4, 8), 3.0))
    target = _small_target()
    conservative_remap(source, target)
    conservative_remap(source, target)
    assert len(_WEIGHT_CACHE) == 1
