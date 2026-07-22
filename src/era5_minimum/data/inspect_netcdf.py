"""Inspect an ERA5 NetCDF file without changing its variable semantics.

The inspector deliberately reports raw variable names.  In particular, it never
renames or derives precipitation variables: deciding whether an accumulation is
equivalent to ``tp6h`` requires confirmation from the data provider.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


TIME_COORDINATE_NAMES = ("time", "valid_time")
LATITUDE_COORDINATE_NAMES = ("latitude", "lat")
LONGITUDE_COORDINATE_NAMES = ("longitude", "lon")
UNKNOWN_UNIT_VALUES = {"", "-", "n/a", "na", "none", "unknown", "undefined"}


def _json_value(value: Any) -> Any:
    """Convert common NetCDF attribute values to JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attributes(array: xr.DataArray) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in array.attrs.items()}


def _find_coordinate(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in dataset.coords), None)


def _timestamp(value: Any) -> str:
    """Return a stable ISO representation for numpy, pandas, or cftime values."""
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s") + "Z"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _time_summary(time: xr.DataArray | None, warnings: list[str]) -> dict[str, Any] | None:
    if time is None:
        warnings.append("No recognised time coordinate (expected 'time' or 'valid_time').")
        return None

    values = np.asarray(time.values)
    if values.size == 0:
        warnings.append(f"Time coordinate '{time.name}' is empty.")
        return {"coordinate": str(time.name), "minimum": None, "maximum": None, "cadence": None}

    cadence: dict[str, Any] | None = None
    if values.size > 1:
        differences = np.diff(values)
        unique_differences = np.unique(differences)
        first = unique_differences[0]
        try:
            seconds = float(first / np.timedelta64(1, "s"))
            cadence = {"seconds": seconds, "uniform": len(unique_differences) == 1}
        except (TypeError, ValueError):
            cadence = {"value": str(first), "uniform": len(unique_differences) == 1}
        if not cadence["uniform"]:
            warnings.append("Time cadence is not uniform across adjacent timestamps.")

    return {
        "coordinate": str(time.name),
        "minimum": _timestamp(values.min()),
        "maximum": _timestamp(values.max()),
        "cadence": cadence,
    }


def _latitude_direction(latitude: xr.DataArray | None, warnings: list[str]) -> str | None:
    if latitude is None:
        warnings.append("No recognised latitude coordinate (expected 'latitude' or 'lat').")
        return None
    values = np.asarray(latitude.values)
    if values.size < 2:
        warnings.append(f"Latitude coordinate '{latitude.name}' has fewer than two values.")
        return "single_value"
    differences = np.diff(values)
    if np.all(differences > 0):
        return "ascending"
    if np.all(differences < 0):
        return "descending"
    warnings.append(f"Latitude coordinate '{latitude.name}' is not monotonic.")
    return "non_monotonic"


def _numeric_statistics(array: xr.DataArray, time_name: str | None) -> dict[str, Any] | None:
    """Calculate statistics one time slice at a time to bound peak memory use."""
    if not np.issubdtype(array.dtype, np.number):
        return None

    if time_name is not None and time_name in array.dims:
        chunks = (array.isel({time_name: index}) for index in range(array.sizes[time_name]))
    else:
        chunks = (array,)

    count = 0
    nan_count = 0
    total = 0.0
    total_squares = 0.0
    minimum = math.inf
    maximum = -math.inf
    for chunk in chunks:
        values = np.asarray(chunk.values)
        if np.issubdtype(values.dtype, np.inexact):
            finite = values[np.isfinite(values)]
            nan_count += int(np.isnan(values).sum())
        else:
            finite = values.reshape(-1)
        if finite.size == 0:
            continue
        float_values = finite.astype(np.float64, copy=False)
        count += int(float_values.size)
        total += float(float_values.sum())
        total_squares += float(np.square(float_values).sum())
        minimum = min(minimum, float(float_values.min()))
        maximum = max(maximum, float(float_values.max()))

    total_values = int(array.size)
    if count == 0:
        return {
            "nan_count": nan_count,
            "nan_fraction": nan_count / total_values if total_values else None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    mean = total / count
    variance = max(total_squares / count - mean * mean, 0.0)
    return {
        "nan_count": nan_count,
        "nan_fraction": nan_count / total_values if total_values else None,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": math.sqrt(variance),
    }


def _variable_summary(array: xr.DataArray, time_name: str | None) -> dict[str, Any]:
    return {
        "dimensions": list(array.dims),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "attributes": _attributes(array),
        "statistics": _numeric_statistics(array, time_name),
    }


def _slice_size_bytes(dataset: xr.Dataset, time_name: str | None) -> int | None:
    """Estimate bytes for one timestamp across time-dependent data variables."""
    if time_name is None or dataset.sizes.get(time_name, 0) == 0:
        return None
    total = 0
    for array in dataset.data_vars.values():
        if time_name in array.dims:
            total += int(array.nbytes // array.sizes[time_name])
    return total


def inspect_netcdf(path: str | Path) -> dict[str, Any]:
    """Inspect ``path`` and return a JSON-serialisable report of its raw schema."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"NetCDF file does not exist: {input_path}")

    warnings: list[str] = []
    try:
        with xr.open_dataset(input_path, cache=False) as dataset:
            time_name = _find_coordinate(dataset, TIME_COORDINATE_NAMES)
            latitude_name = _find_coordinate(dataset, LATITUDE_COORDINATE_NAMES)
            longitude_name = _find_coordinate(dataset, LONGITUDE_COORDINATE_NAMES)
            variable_reports = {
                name: _variable_summary(array, time_name) for name, array in dataset.data_vars.items()
            }

            for name, summary in variable_reports.items():
                units = summary["attributes"].get("units")
                if units is None:
                    warnings.append(f"Variable '{name}' has no units attribute.")
                elif str(units).strip().lower() in UNKNOWN_UNIT_VALUES:
                    warnings.append(f"Variable '{name}' has unknown units: {units!r}.")

            if "total_precipitation" in dataset.data_vars and "tp6h" not in dataset.data_vars:
                warnings.append(
                    "Found 'total_precipitation' but no confirmed 'tp6h'; no precipitation conversion was applied."
                )

            longitude_range: dict[str, float] | None = None
            if longitude_name is None:
                warnings.append("No recognised longitude coordinate (expected 'longitude' or 'lon').")
            else:
                longitude_values = np.asarray(dataset[longitude_name].values)
                if longitude_values.size:
                    longitude_range = {
                        "minimum": float(np.nanmin(longitude_values)),
                        "maximum": float(np.nanmax(longitude_values)),
                    }

            sst_report: dict[str, Any] | None = None
            if "sea_surface_temperature" in dataset.data_vars:
                sst = dataset["sea_surface_temperature"]
                sst_statistics = variable_reports["sea_surface_temperature"]["statistics"]
                fill_value = sst.encoding.get("_FillValue", sst.attrs.get("_FillValue"))
                sst_report = {
                    "variable": "sea_surface_temperature",
                    "fill_value": _json_value(fill_value),
                    "has_fill_value": fill_value is not None,
                    "nan_count": None if sst_statistics is None else sst_statistics["nan_count"],
                    "nan_fraction": None if sst_statistics is None else sst_statistics["nan_fraction"],
                }
            else:
                warnings.append("Variable 'sea_surface_temperature' was not found; SST mask cannot be inspected.")

            report: dict[str, Any] = {
                "file": {"path": str(input_path), "size_bytes": input_path.stat().st_size},
                "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
                "coordinates": list(dataset.coords),
                "data_variables": list(dataset.data_vars),
                "variables": variable_reports,
                "time": _time_summary(dataset[time_name] if time_name else None, warnings),
                "latitude": {
                    "coordinate": latitude_name,
                    "direction": _latitude_direction(
                        dataset[latitude_name] if latitude_name else None, warnings
                    ),
                },
                "longitude": {"coordinate": longitude_name, "range": longitude_range},
                "estimated_time_slice_size_bytes": _slice_size_bytes(dataset, time_name),
                "sea_surface_temperature": sst_report,
                "warnings": warnings,
            }
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not inspect NetCDF file '{input_path}': {error}") from error
    return report


def write_report(path: str | Path, output: str | Path) -> dict[str, Any]:
    """Inspect a NetCDF file and write its report to ``output``."""
    report = inspect_netcdf(path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a compact terminal summary of the report."""
    file_info = report["file"]
    print(f"NetCDF: {file_info['path']} ({file_info['size_bytes']} bytes)")
    print(f"Dimensions: {report['dimensions']}")
    print("Data variables: " + ", ".join(report["data_variables"]))
    time = report["time"]
    if time is not None:
        print(f"Time: {time['minimum']} to {time['maximum']}; cadence={time['cadence']}")
    print(f"Latitude direction: {report['latitude']['direction']}")
    print(f"Longitude range: {report['longitude']['range']}")
    for name, variable in report["variables"].items():
        statistics = variable["statistics"]
        units = variable["attributes"].get("units", "missing")
        nan_count = None if statistics is None else statistics["nan_count"]
        print(f"  {name}: shape={variable['shape']}, dtype={variable['dtype']}, units={units}, NaN={nan_count}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def main(argv: list[str] | None = None) -> int:
    """Run the NetCDF inspector command-line interface."""
    parser = argparse.ArgumentParser(description="Inspect a raw ERA5 NetCDF schema without transforming data.")
    parser.add_argument("--path", required=True, help="Path to the input NetCDF file.")
    parser.add_argument("--output", required=True, help="Path for the JSON schema report.")
    args = parser.parse_args(argv)
    try:
        report = write_report(args.path, args.output)
    except (FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    _print_summary(report)
    print(f"JSON report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
