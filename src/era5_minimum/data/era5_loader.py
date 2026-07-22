"""Strict loader for the confirmed ERA5 single-level NetCDF pair.

This module preserves the units and raw physical values from ERA5.  The
accumulated ``tp`` field is named ``tp1h`` internally because this sample is a
one-hour accumulation; it is never inferred or converted to ``tp6h``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr


CANONICAL_CHANNEL_ORDER = ("u10", "v10", "t2m", "msl", "sst", "tcc", "tcwv", "tp1h")
INSTANT_VARIABLES = CANONICAL_CHANNEL_ORDER[:-1]
EXPECTED_UNITS = {
    "u10": {"m s**-1"},
    "v10": {"m s**-1"},
    "t2m": {"K"},
    "msl": {"Pa"},
    "sst": {"K"},
    "tcc": {"(0 - 1)", "0-1", "0–1"},
    "tcwv": {"kg m**-2"},
    "tp": {"m"},
}


class ERA5SchemaError(ValueError):
    """Raised when an input file differs from the confirmed ERA5 schema."""


@dataclass(frozen=True)
class ERA5Sample:
    """Raw ERA5 tensor and metadata; no normalization or training statistics."""

    data: np.ndarray
    sst_mask: np.ndarray
    timestamps: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    channel_names: tuple[str, ...]
    units: dict[str, str]
    source_shape: tuple[int, int, int, int]
    downsampling_method: str
    sst_nan_count_before_fill: int


def _source(dataset: xr.Dataset) -> str:
    return str(dataset.encoding.get("source", "<in-memory dataset>"))


def _error(dataset: xr.Dataset, variable: str, message: str) -> ERA5SchemaError:
    return ERA5SchemaError(f"{_source(dataset)}: variable '{variable}': {message}")


def _require_coordinate(dataset: xr.Dataset, name: str) -> xr.DataArray:
    if name not in dataset.coords:
        raise _error(dataset, name, "coordinate is missing")
    return dataset[name]


def _validate_expected_variables(dataset: xr.Dataset, expected: Sequence[str]) -> None:
    missing = [name for name in expected if name not in dataset.data_vars]
    if missing:
        raise _error(dataset, ", ".join(missing), f"missing; found {list(dataset.data_vars)}")


def _validate_units(dataset: xr.Dataset, variables: Sequence[str]) -> None:
    for name in variables:
        actual = dataset[name].attrs.get("units")
        if actual not in EXPECTED_UNITS[name]:
            raise _error(
                dataset,
                name,
                f"expected units {sorted(EXPECTED_UNITS[name])}, got {actual!r}",
            )


def _validate_step_type(dataset: xr.Dataset, variables: Sequence[str], expected: str) -> None:
    for name in variables:
        actual = dataset[name].attrs.get("GRIB_stepType")
        if actual != expected:
            raise _error(dataset, name, f"expected GRIB_stepType={expected!r}, got {actual!r}")


def _validate_coordinates(dataset: xr.Dataset) -> None:
    latitude = np.asarray(_require_coordinate(dataset, "latitude").values)
    longitude = np.asarray(_require_coordinate(dataset, "longitude").values)
    if latitude.ndim != 1 or latitude.size < 2:
        raise _error(dataset, "latitude", f"expected at least two 1-D values, got shape {latitude.shape}")
    if longitude.ndim != 1 or longitude.size < 2:
        raise _error(dataset, "longitude", f"expected at least two 1-D values, got shape {longitude.shape}")
    lat_steps = np.diff(latitude)
    lon_steps = np.diff(longitude)
    if not np.all(lat_steps < 0):
        raise _error(dataset, "latitude", "expected strictly descending values")
    if not np.all(lon_steps > 0):
        raise _error(dataset, "longitude", "expected strictly ascending values")
    if not np.allclose(lat_steps, -0.25, rtol=0.0, atol=1e-6):
        raise _error(dataset, "latitude", f"expected 0.25 degree spacing, got steps {np.unique(lat_steps)}")
    if not np.allclose(lon_steps, 0.25, rtol=0.0, atol=1e-6):
        raise _error(dataset, "longitude", f"expected 0.25 degree spacing, got steps {np.unique(lon_steps)}")


def _validate_expver(dataset: xr.Dataset) -> None:
    expver = _require_coordinate(dataset, "expver")
    values = {str(value) for value in np.asarray(expver.values).reshape(-1)}
    if values != {"0001"}:
        raise _error(dataset, "expver", f"expected only '0001', got {sorted(values)}")


def _validate_no_nans(dataset: xr.Dataset, variables: Sequence[str]) -> None:
    """Check each field independently, avoiding a whole-dataset in-memory load."""
    for name in variables:
        if bool(dataset[name].isnull().any().item()):
            raise _error(dataset, name, "contains NaN values")


def open_era5_pair(
    instant_path: str | Path, accumulated_path: str | Path
) -> tuple[xr.Dataset, xr.Dataset]:
    """Open the instant and accumulated files lazily without loading their fields."""
    instant_file = Path(instant_path)
    accumulated_file = Path(accumulated_path)
    if not instant_file.is_file():
        raise FileNotFoundError(f"Instant ERA5 NetCDF file does not exist: {instant_file}")
    if not accumulated_file.is_file():
        raise FileNotFoundError(f"Accumulated ERA5 NetCDF file does not exist: {accumulated_file}")
    try:
        return xr.open_dataset(instant_file, cache=False), xr.open_dataset(accumulated_file, cache=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not open ERA5 NetCDF pair: {error}") from error


def validate_era5_pair(instant_ds: xr.Dataset, accumulated_ds: xr.Dataset) -> None:
    """Validate the confirmed raw schema before any rename or subsampling."""
    _validate_expected_variables(instant_ds, INSTANT_VARIABLES)
    _validate_expected_variables(accumulated_ds, ("tp",))
    for dataset in (instant_ds, accumulated_ds):
        _require_coordinate(dataset, "valid_time")
        _validate_coordinates(dataset)
        _validate_expver(dataset)
    instant_time = np.asarray(instant_ds["valid_time"].values)
    accumulated_time = np.asarray(accumulated_ds["valid_time"].values)
    if not np.array_equal(instant_time, accumulated_time):
        raise _error(
            accumulated_ds,
            "valid_time",
            f"does not match instant file {_source(instant_ds)}; got {accumulated_time!r}",
        )
    for coordinate in ("latitude", "longitude"):
        if not np.array_equal(instant_ds[coordinate].values, accumulated_ds[coordinate].values):
            raise _error(
                accumulated_ds,
                coordinate,
                f"does not match instant file {_source(instant_ds)}",
            )
    _validate_units(instant_ds, INSTANT_VARIABLES)
    _validate_units(accumulated_ds, ("tp",))
    _validate_step_type(instant_ds, INSTANT_VARIABLES, "instant")
    _validate_step_type(accumulated_ds, ("tp",), "accum")
    _validate_no_nans(instant_ds, tuple(name for name in INSTANT_VARIABLES if name != "sst"))
    sst = instant_ds["sst"]
    if not bool(sst.notnull().any().item()):
        raise _error(instant_ds, "sst", "has no valid values")
    _validate_no_nans(accumulated_ds, ("tp",))
    build_sst_mask(instant_ds)


def downsample_to_half_degree(ds: xr.Dataset) -> xr.Dataset:
    """Return deterministic 0.5 degree subsampling (every second grid point)."""
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise _error(ds, "coordinates", "latitude and longitude are required for subsampling")
    return ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))


def build_sst_mask(ds: xr.Dataset) -> np.ndarray:
    """Return a static [latitude, longitude] SST-valid mask after consistency validation."""
    if "sst" not in ds.data_vars:
        raise _error(ds, "sst", "missing; cannot build SST mask")
    if "valid_time" not in ds["sst"].dims and "time" not in ds["sst"].dims:
        raise _error(ds, "sst", f"expected time dimension, got {ds['sst'].dims}")
    time_dimension = "valid_time" if "valid_time" in ds["sst"].dims else "time"
    valid = np.asarray(ds["sst"].notnull().values, dtype=bool)
    reference = valid[0]
    if not np.array_equal(valid, np.broadcast_to(reference, valid.shape)):
        raise _error(ds, "sst", "valid-value mask is inconsistent between timestamps")
    return reference


def to_channel_tensor(ds: xr.Dataset, channel_order: Sequence[str]) -> np.ndarray:
    """Create a float32 [time, channel, latitude, longitude] tensor.

    Only SST NaNs are replaced by zero in this derived tensor; its mask must be
    carried separately by the caller.
    """
    requested = tuple(channel_order)
    if "tp6h" in requested:
        raise ValueError("tp6h cannot be created from this sparse tp1h ERA5 sample")
    missing = [name for name in requested if name not in ds.data_vars]
    if missing:
        raise _error(ds, ", ".join(missing), f"missing; found {list(ds.data_vars)}")
    arrays: list[np.ndarray] = []
    for name in requested:
        values = np.asarray(ds[name].transpose("time", "latitude", "longitude").values, dtype=np.float32)
        if name == "sst":
            values = np.nan_to_num(values, nan=0.0)
        elif np.isnan(values).any():
            raise _error(ds, name, "contains NaN values")
        arrays.append(values)
    return np.stack(arrays, axis=1).astype(np.float32, copy=False)


def load_era5_sample(
    instant_path: str | Path,
    accumulated_path: str | Path,
    *,
    half_degree: bool = False,
    channel_order: Sequence[str] = CANONICAL_CHANNEL_ORDER,
) -> ERA5Sample:
    """Validate, rename, optionally subsample, and materialise one ERA5 sample."""
    instant_ds, accumulated_ds = open_era5_pair(instant_path, accumulated_path)
    try:
        validate_era5_pair(instant_ds, accumulated_ds)
        source_shape = tuple(int(instant_ds.sizes[name]) for name in ("valid_time", "latitude", "longitude"))
        sst_nan_count = int(instant_ds["sst"].isnull().sum().item())
        instant_renamed = instant_ds.rename({"valid_time": "time"})
        accumulated_renamed = accumulated_ds.rename({"valid_time": "time", "tp": "tp1h"})
        merged = xr.merge((instant_renamed, accumulated_renamed), compat="equals", join="exact")
        method = "subsampling_every_second_grid_point" if half_degree else "none"
        if half_degree:
            merged = downsample_to_half_degree(merged)
        sst_mask = build_sst_mask(merged)
        tensor = to_channel_tensor(merged, channel_order)
        units = {name: str(merged[name].attrs["units"]) for name in channel_order}
        return ERA5Sample(
            data=tensor,
            sst_mask=sst_mask,
            timestamps=np.asarray(merged["time"].values),
            latitude=np.asarray(merged["latitude"].values, dtype=np.float32),
            longitude=np.asarray(merged["longitude"].values, dtype=np.float32),
            channel_names=tuple(channel_order),
            units=units,
            source_shape=(source_shape[0], len(CANONICAL_CHANNEL_ORDER), source_shape[1], source_shape[2]),
            downsampling_method=method,
            sst_nan_count_before_fill=sst_nan_count,
        )
    finally:
        instant_ds.close()
        accumulated_ds.close()


def _timestamps_for_json(values: np.ndarray) -> list[str]:
    return [np.datetime_as_string(value, unit="s") + "Z" for value in values]


def build_loader_report(sample: ERA5Sample) -> dict[str, Any]:
    """Build the JSON-serialisable report emitted by the loader CLI."""
    return {
        "source_shape": list(sample.source_shape),
        "output_shape": list(sample.data.shape),
        "channel_order": list(sample.channel_names),
        "units": sample.units,
        "timestamps": _timestamps_for_json(sample.timestamps),
        "latitude_range": {"minimum": float(sample.latitude.min()), "maximum": float(sample.latitude.max())},
        "longitude_range": {"minimum": float(sample.longitude.min()), "maximum": float(sample.longitude.max())},
        "nan_count_before_fill": {"sst": sample.sst_nan_count_before_fill},
        "sst_valid_fraction": float(sample.sst_mask.mean()),
        "sst_mask_consistent_across_time": True,
        "downsampling_method": sample.downsampling_method,
        "warnings": ["tp1h is a one-hour accumulation and is not tp6h; no tp6h was created."],
    }


def _write_outputs(sample: ERA5Sample, output: Path, report_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        data=sample.data,
        sst_mask=sample.sst_mask,
        timestamps=sample.timestamps,
        latitude=sample.latitude,
        longitude=sample.longitude,
        channel_names=np.asarray(sample.channel_names),
        units=np.asarray([sample.units[name] for name in sample.channel_names]),
    )
    report = build_loader_report(sample)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the production ERA5-pair loader command-line interface."""
    parser = argparse.ArgumentParser(description="Validate and load the confirmed ERA5 NetCDF pair.")
    parser.add_argument("--instant", required=True, help="Instant-field NetCDF path.")
    parser.add_argument("--accumulated", required=True, help="Accumulated-field NetCDF path.")
    parser.add_argument("--half-degree", action="store_true", help="Apply deterministic 0.5° subsampling.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    args = parser.parse_args(argv)
    try:
        sample = load_era5_sample(args.instant, args.accumulated, half_degree=args.half_degree)
        report = _write_outputs(sample, Path(args.output), Path(args.report))
    except (FileNotFoundError, RuntimeError, ERA5SchemaError, ValueError) as error:
        parser.error(str(error))
    print(f"Loaded ERA5 tensor with shape {tuple(report['output_shape'])}.")
    print(f"Saved sample to {args.output} and report to {args.report}.")
    print("WARNING: tp1h is not tp6h; no precipitation conversion was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
