from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from era5_minimum.data.channel_spec import CHANNEL_NAMES, CHANNEL_SPEC
from era5_minimum.data.era5_28ch import (
    PRESSURE_SOURCES,
    SURFACE_SOURCES,
    assemble_model_tensor,
    build_static,
    expected_times,
    select_dynamic,
)


def _source() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=2, freq="6h")
    coords = {"time": time, "level": [1000, 925, 850, 700], "latitude": [-90.0, 90.0], "longitude": [0.0, 180.0]}
    values = {}
    for name in SURFACE_SOURCES:
        data = np.ones((2, 2, 2), dtype=np.float32)
        if name == "sea_surface_temperature": data[:, 0, 0] = np.nan
        values[name] = (("time", "latitude", "longitude"), data)
    for name in PRESSURE_SOURCES:
        values[name] = (("time", "level", "latitude", "longitude"), np.ones((2, 4, 2, 2), dtype=np.float32))
    values["land_sea_mask"] = (("latitude", "longitude"), np.array([[0, 1], [0, 1]], dtype=np.float32))
    return xr.Dataset(values, coords=coords)


def test_model_adapter_preserves_official_channel_order_lazily() -> None:
    tensor = assemble_model_tensor(_source())
    assert tensor.dims == ("time", "channel", "latitude", "longitude")
    assert tensor.channel.values.tolist() == list(CHANNEL_NAMES)
    assert tensor.shape == (2, 28, 2, 2)
    assert [channel.name for channel in CHANNEL_SPEC] == list(CHANNEL_NAMES)


def test_static_ocean_mask_rule_and_sst_nan_are_physical() -> None:
    source = _source()
    static = build_static(source)
    assert static.ocean_mask.values.tolist() == [[1, 0], [1, 0]]
    assert np.isnan(source.sea_surface_temperature.values[:, 0, 0]).all()


def test_split_timestamps_are_sorted_unique_and_six_hourly() -> None:
    train = expected_times("train")
    validation = expected_times("validation")
    test = expected_times("test")
    assert train.is_monotonic_increasing and train.is_unique
    assert (train[1:] - train[:-1] == pd.Timedelta(hours=6)).all()
    assert train[-1] < validation[0] < test[0]
    assert len(train) == len(set(train))


def test_selection_rejects_outside_split() -> None:
    with pytest.raises(ValueError, match="Empty requested range"):
        expected_times("validation", "2019-01-01", "2019-01-02")
