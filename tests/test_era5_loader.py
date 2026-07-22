from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from era5_minimum.data.era5_loader import (
    CANONICAL_CHANNEL_ORDER,
    ERA5SchemaError,
    build_loader_report,
    load_era5_sample,
    open_era5_pair,
    to_channel_tensor,
    validate_era5_pair,
)


def _write_pair(
    directory: Path,
    *,
    accumulated_times: np.ndarray | None = None,
    t2m_units: str = "K",
    t2m_nan: bool = False,
    expver: str = "0001",
) -> tuple[Path, Path]:
    times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"],
        dtype="datetime64[ns]",
    )
    if accumulated_times is None:
        accumulated_times = times
    latitude = np.array([90.0, 89.75, 89.5, 89.25, 89.0], dtype=np.float32)
    longitude = np.arange(6, dtype=np.float32) * 0.25
    shape = (len(times), len(latitude), len(longitude))
    values = np.ones(shape, dtype=np.float32)
    sst = values.copy()
    sst[:, 0, 0] = np.nan
    t2m = values.copy()
    if t2m_nan:
        t2m[0, 1, 1] = np.nan
    instant = xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), values, {"units": "m s**-1", "GRIB_stepType": "instant"}),
            "v10": (("valid_time", "latitude", "longitude"), values, {"units": "m s**-1", "GRIB_stepType": "instant"}),
            "t2m": (("valid_time", "latitude", "longitude"), t2m, {"units": t2m_units, "GRIB_stepType": "instant"}),
            "msl": (("valid_time", "latitude", "longitude"), values, {"units": "Pa", "GRIB_stepType": "instant"}),
            "sst": (("valid_time", "latitude", "longitude"), sst, {"units": "K", "GRIB_stepType": "instant"}),
            "tcc": (("valid_time", "latitude", "longitude"), values, {"units": "(0 - 1)", "GRIB_stepType": "instant"}),
            "tcwv": (("valid_time", "latitude", "longitude"), values, {"units": "kg m**-2", "GRIB_stepType": "instant"}),
        },
        coords={"valid_time": times, "latitude": latitude, "longitude": longitude, "expver": ("valid_time", [expver] * len(times))},
    )
    accum = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), np.ones((len(accumulated_times), len(latitude), len(longitude)), dtype=np.float32), {"units": "m", "GRIB_stepType": "accum"})},
        coords={"valid_time": accumulated_times, "latitude": latitude, "longitude": longitude, "expver": ("valid_time", [expver] * len(accumulated_times))},
    )
    instant_path = directory / "instant.nc"
    accum_path = directory / "accum.nc"
    instant.to_netcdf(instant_path)
    accum.to_netcdf(accum_path)
    return instant_path, accum_path


def test_load_pair_uses_tp1h_and_returns_sst_mask(tmp_path) -> None:
    instant, accumulated = _write_pair(tmp_path)
    sample = load_era5_sample(instant, accumulated, half_degree=True)
    assert sample.data.shape == (3, 8, 3, 3)
    assert sample.data.dtype == np.float32
    assert sample.channel_names == CANONICAL_CHANNEL_ORDER
    assert sample.units["tp1h"] == "m"
    assert sample.sst_mask.shape == (3, 3)
    assert not sample.sst_mask[0, 0]
    assert sample.data[:, 4, 0, 0].tolist() == [0.0, 0.0, 0.0]
    report = build_loader_report(sample)
    assert report["downsampling_method"] == "subsampling_every_second_grid_point"
    assert "not tp6h" in report["warnings"][0]


def test_mismatching_timestamps_fail_validation(tmp_path) -> None:
    times = np.array(["2024-01-01T01:00:00", "2024-01-01T07:00:00", "2024-01-01T13:00:00"], dtype="datetime64[ns]")
    instant, accumulated = _write_pair(tmp_path, accumulated_times=times)
    with pytest.raises(ERA5SchemaError, match="valid_time.*does not match"):
        load_era5_sample(instant, accumulated)


def test_wrong_units_fail_validation(tmp_path) -> None:
    instant, accumulated = _write_pair(tmp_path, t2m_units="C")
    with pytest.raises(ERA5SchemaError, match="t2m.*expected units"):
        load_era5_sample(instant, accumulated)


def test_nan_outside_sst_fails_validation(tmp_path) -> None:
    instant, accumulated = _write_pair(tmp_path, t2m_nan=True)
    with pytest.raises(ERA5SchemaError, match="t2m.*contains NaN"):
        load_era5_sample(instant, accumulated)


def test_expver_other_than_0001_fails_validation(tmp_path) -> None:
    instant, accumulated = _write_pair(tmp_path, expver="0005")
    with pytest.raises(ERA5SchemaError, match="expver.*0001"):
        load_era5_sample(instant, accumulated)


def test_tp6h_is_never_silently_created(tmp_path) -> None:
    instant, accumulated = _write_pair(tmp_path)
    instant_ds, accumulated_ds = open_era5_pair(instant, accumulated)
    try:
        validate_era5_pair(instant_ds, accumulated_ds)
        merged = xr.merge(
            (instant_ds.rename({"valid_time": "time"}), accumulated_ds.rename({"valid_time": "time", "tp": "tp1h"})),
            join="exact",
        )
        with pytest.raises(ValueError, match="tp6h cannot be created"):
            to_channel_tensor(merged, ["tp6h"])
    finally:
        instant_ds.close()
        accumulated_ds.close()
