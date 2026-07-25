"""Lazy, native-variable WeatherBench2 preparation utilities.

The physical Zarr stores retain source variables. ``assemble_model_tensor`` is
the sole logical adapter and never writes a second 28-channel copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .channel_spec import CHANNEL_NAMES, CHANNEL_SPEC, PRESSURE_LEVELS, validate_wb2_compatibility
from .weatherbench2 import WB2_URL, open_wb2_era5

Split = Literal["train", "validation", "test"]
SPLIT_RANGES: dict[Split, tuple[str, str]] = {
    "train": ("2014-01-01T00:00:00", "2019-12-31T18:00:00"),
    "validation": ("2020-01-01T00:00:00", "2020-12-31T18:00:00"),
    "test": ("2021-01-01T00:00:00", "2021-12-31T18:00:00"),
}
SURFACE_SOURCES = tuple(c.source_name for c in CHANNEL_SPEC if c.type == "surface")
PRESSURE_SOURCES = tuple(dict.fromkeys(c.source_name for c in CHANNEL_SPEC if c.type == "pressure"))
STATIC_CANDIDATES = ("land_sea_mask", "geopotential_at_surface")
DEFAULT_CHUNKS = {"time": 4, "latitude": 180, "longitude": 180, "level": 4}


def expected_times(split: Split, start: str | None = None, end: str | None = None) -> pd.DatetimeIndex:
    """Return authoritative six-hour timestamps, optionally bounded within a split."""
    lower, upper = SPLIT_RANGES[split]
    start = max(start or lower, lower)
    end = min(end or upper, upper)
    values = pd.date_range(start, end, freq="6h")
    if values.empty:
        raise ValueError(f"Empty requested range for {split}: {start}..{end}")
    return values


def inspect_source(ds: xr.Dataset) -> dict[str, Any]:
    """Validate source metadata without reading global field values."""
    validate_wb2_compatibility(ds)
    required_sizes = {"latitude": 721, "longitude": 1440}
    for name, size in required_sizes.items():
        if ds.sizes.get(name) != size:
            raise ValueError(f"WeatherBench2 {name} size changed: {ds.sizes.get(name)} != {size}")
    sample_time = pd.DatetimeIndex(ds.time.values[: min(8, ds.sizes["time"])])
    if len(sample_time) > 1 and not np.all(np.diff(sample_time.values) == np.timedelta64(6, "h")):
        raise ValueError("WeatherBench2 time cadence is not six hours")
    return {
        "source_uri": WB2_URL,
        "sizes": {key: int(value) for key, value in ds.sizes.items()},
        "coordinates": list(ds.coords),
        "data_variables": list(ds.data_vars),
        "time_range": [str(ds.time.values[0]), str(ds.time.values[-1])],
        "pressure_levels": [int(value) for value in ds.level.values],
        "variables": {
            name: {"dims": list(ds[name].dims), "dtype": str(ds[name].dtype),
                   "units": ds[name].attrs.get("units"),
                   "chunks": [{"chunk_size": int(axis[0]), "chunk_count": len(axis),
                               "last_chunk_size": int(axis[-1])} for axis in (ds[name].chunks or ())]}
            for name in sorted(set(SURFACE_SOURCES + PRESSURE_SOURCES + STATIC_CANDIDATES) & set(ds.data_vars))
        },
        "consolidated_metadata": True,
    }


def select_dynamic(ds: xr.Dataset, timestamps: pd.DatetimeIndex) -> xr.Dataset:
    """Lazily select physical source variables, pressure levels and timestamps."""
    names = list(SURFACE_SOURCES + PRESSURE_SOURCES)
    selected = ds[names].sel(time=timestamps, level=list(PRESSURE_LEVELS))
    for name in names:
        selected[name] = selected[name].astype(np.float32)
        selected[name].attrs = dict(ds[name].attrs)
    return selected.chunk(DEFAULT_CHUNKS)


def build_static(ds: xr.Dataset) -> xr.Dataset:
    """Build static conditioning fields; ocean_mask is exactly land_sea_mask <= 0.5."""
    present = [name for name in STATIC_CANDIDATES if name in ds.data_vars]
    static = ds[present].astype(np.float32) if present else xr.Dataset()
    latitude = ds["latitude"]
    static["sin_latitude"] = xr.DataArray(np.sin(np.deg2rad(latitude.values)).astype(np.float32), dims=("latitude",), coords={"latitude": latitude})
    static["cos_latitude"] = xr.DataArray(np.cos(np.deg2rad(latitude.values)).astype(np.float32), dims=("latitude",), coords={"latitude": latitude})
    if "land_sea_mask" in static:
        static["ocean_mask"] = (static["land_sea_mask"] <= 0.5).astype(np.uint8)
        static["ocean_mask"].attrs["policy"] = "land_sea_mask <= 0.5 (1=ocean, 0=land)"
    return static.chunk({"latitude": 180, "longitude": 180})


def assemble_model_tensor(ds: xr.Dataset) -> xr.DataArray:
    """Return lazy ``[time, channel, latitude, longitude]`` in official order."""
    arrays: list[xr.DataArray] = []
    for channel in CHANNEL_SPEC:
        field = ds[channel.source_name]
        if channel.level is not None:
            field = field.sel(level=channel.level, drop=True)
        arrays.append(field.expand_dims(channel=[channel.name]))
    tensor = xr.concat(arrays, dim="channel").transpose("time", "channel", "latitude", "longitude")
    tensor.name = "data"
    tensor.attrs["channel_order"] = list(CHANNEL_NAMES)
    return tensor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def write_manifest(root: Path, split_entries: dict[str, Any], *, chunks: dict[str, int] = DEFAULT_CHUNKS) -> dict[str, Any]:
    """Write a deterministic manifest plus a hash of the manifest file itself."""
    try:
        commit = os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None
    except OSError:
        commit = None
    manifest = {
        "format_version": "2.0.0", "source_uri": WB2_URL,
        "prepared_at": datetime.now(UTC).isoformat(), "git_commit": commit,
        "software": {"python": platform.python_version(), "xarray": __import__("xarray").__version__},
        "resolution": "0.25_degree", "grid_shape": [721, 1440], "cadence_hours": 6,
        "dimension_order": ["time", "latitude", "longitude"], "dtype": "float32", "chunks": chunks,
        "compressor": "Zarr v2 default compressor selected by xarray", "pressure_levels_hpa": list(PRESSURE_LEVELS),
        "channel_order": list(CHANNEL_NAMES),
        "channels": [{"name": c.name, "source_name": c.source_name, "type": c.type,
                      "pressure_level_hpa": c.level, "units": c.units} for c in CHANNEL_SPEC],
        "sst_missing_value_policy": "NaN is retained in physical sea_surface_temperature; only ML adapter may fill invalid cells after normalization.",
        "ocean_mask_policy": "land_sea_mask <= 0.5 (1=ocean, 0=land), stored in static.zarr when source field exists.",
        "static_fields": ["land_sea_mask", "geopotential_at_surface", "ocean_mask", "sin_latitude", "cos_latitude"],
        "statistics": {"status": "not_computed"}, "splits": split_entries, "completion": "complete",
    }
    path = _manifest_path(root)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "manifest.json.sha256").write_text(f"{_sha256(path)}  manifest.json\n", encoding="utf-8")
    return manifest


def _load_entries(root: Path) -> dict[str, Any]:
    path = _manifest_path(root)
    return json.loads(path.read_text())["splits"] if path.exists() else {}


def write_split(ds: xr.Dataset, root: str | Path, split: Split, *, start: str | None = None,
                end: str | None = None, overwrite: bool = False) -> Path:
    """Atomically write one split, never deleting other completed splits."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root / f"{split}.zarr"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite to replace this split")
    values = expected_times(split, start, end)
    dynamic = select_dynamic(ds, values)
    temporary = Path(tempfile.mkdtemp(prefix=f".{split}.zarr.", dir=root))
    shutil.rmtree(temporary)
    try:
        dynamic.to_zarr(temporary, mode="w", consolidated=True, zarr_format=2)
        if target.exists(): shutil.rmtree(target)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    entries = _load_entries(root)
    entries[split] = {"path": target.name, "range": [str(values[0]), str(values[-1])], "timestamps": len(values)}
    write_manifest(root, entries)
    if not (root / "static.zarr").exists():
        build_static(ds).to_zarr(root / "static.zarr", mode="w", consolidated=True, zarr_format=2)
    if set(entries) == set(SPLIT_RANGES): (root / "_SUCCESS").touch()
    return target


def get_layer(root: str | Path, split: Split, variable: str, timestamp: str,
              pressure_level_hpa: int | None = None) -> dict[str, Any]:
    """Read exactly one physical 2-D layer for a future backend consumer."""
    ds = xr.open_zarr(Path(root) / f"{split}.zarr", consolidated=True)
    channel = next((item for item in CHANNEL_SPEC if item.name == variable), None)
    if channel is None: raise KeyError(f"Unknown channel {variable!r}")
    if channel.level != pressure_level_hpa and channel.level is not None:
        raise ValueError(f"{variable} requires pressure_level_hpa={channel.level}")
    field = ds[channel.source_name].sel(time=np.datetime64(timestamp))
    if channel.level is not None: field = field.sel(level=channel.level)
    values = field.load().values
    mask = np.isfinite(values)
    return {"variable": variable, "source_name": channel.source_name, "timestamp": str(field.time.values),
            "pressure_level_hpa": channel.level, "units": channel.units,
            "latitude": ds.latitude.values, "longitude": ds.longitude.values, "values": values,
            "mask": mask, "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
            "valid_count": int(mask.sum())}


def validate_prepared(root: str | Path) -> None:
    """Metadata-first validator; reads one bounded frame per dynamic variable."""
    root = Path(root); manifest_path = _manifest_path(root)
    if not manifest_path.exists() or not (root / "manifest.json.sha256").exists(): raise ValueError("manifest and SHA256 are required")
    expected_hash = (root / "manifest.json.sha256").read_text().split()[0]
    if expected_hash != _sha256(manifest_path): raise ValueError("manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["channel_order"] != list(CHANNEL_NAMES): raise ValueError("channel order mismatch")
    entries = manifest["splits"]
    all_times: list[np.ndarray] = []
    for split, entry in entries.items():
        store = root / entry["path"]
        ds = xr.open_zarr(store, consolidated=True)
        validate_wb2_compatibility(ds)
        if ds.sizes.get("latitude") != 721 or ds.sizes.get("longitude") != 1440: raise ValueError("unexpected grid shape")
        times = ds.time.values
        if len(times) != entry["timestamps"] or not np.all(times[:-1] < times[1:]): raise ValueError(f"invalid timestamps in {split}")
        if len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(6, "h")): raise ValueError(f"non-6h cadence in {split}")
        sample = ds[list(SURFACE_SOURCES + PRESSURE_SOURCES)].isel(time=0).load()
        if any(np.isinf(value.values).any() for value in sample.data_vars.values()): raise ValueError(f"Infinity in {split}")
        all_times.append(times)
    if all_times and len(np.unique(np.concatenate(all_times))) != sum(len(x) for x in all_times): raise ValueError("split overlap")
    if entries and not (root / "static.zarr").exists(): raise ValueError("static.zarr is required")


def compute_train_statistics(root: str | Path) -> Path:
    """Compute lazy, train-only per-channel mean/std without materialising frames."""
    root = Path(root)
    ds = xr.open_zarr(root / "train.zarr", consolidated=True)
    tensor = assemble_model_tensor(ds)
    reduced = xr.Dataset({"mean": tensor.mean(("time", "latitude", "longitude"), skipna=True),
                          "std": tensor.std(("time", "latitude", "longitude"), skipna=True),
                          "valid_count": tensor.count(("time", "latitude", "longitude"))}).compute()
    result = {name: {key: float(reduced[key].sel(channel=name).item()) for key in ("mean", "std", "valid_count")}
              for name in CHANNEL_NAMES}
    path = root / "statistics_train.json"
    path.write_text(json.dumps({"scope": "train_only", "channels": result}, indent=2), encoding="utf-8")
    return path
