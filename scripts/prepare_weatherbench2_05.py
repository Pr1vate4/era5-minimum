#!/usr/bin/env python3
"""Prepare a bounded, real 28-channel WeatherBench2 sample on the 0.5° grid.

This is the data entry point for the first small-sample experiment.  It reads
only explicit timestamps, performs first-order conservative remapping, and
writes each frame incrementally.  It never downloads a multi-year native
0.25° copy by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.data.channel_spec import CHANNEL_NAMES, CHANNEL_SPEC
from era5_minimum.data.era5_28ch import assemble_model_tensor, expected_times, select_dynamic
from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.remapping import conservative_remap
from era5_minimum.data.seasonal_subsets import generate_nested_subsets, month_to_season
from era5_minimum.data.weatherbench2 import WB2_URL, open_wb2_era5


_WORKER_REMOTE: xr.Dataset | None = None
_WORKER_TARGET: xr.Dataset | None = None


def _training_timestamps(*, population_size: int, start_index: int, count: int, seed: int) -> list[str]:
    """Select a contiguous deterministic slice of a nested training subset."""

    if population_size <= 0 or population_size % 16:
        raise ValueError("training population size must be a positive multiple of 16")
    if start_index < 0 or count <= 0 or start_index + count > population_size:
        raise ValueError("requested training slice is outside the deterministic population")
    return generate_nested_subsets([population_size], seed=seed)[population_size][start_index : start_index + count]


def _evaluation_timestamps(split: str, count: int, seed: int) -> list[str]:
    """Return a deterministic season/hour-balanced holdout selection."""

    if count < 0 or count % 16:
        raise ValueError("validation/test sample count must be a non-negative multiple of 16")
    if count == 0:
        return []
    candidates = expected_times(split)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)
    pools: dict[tuple[str, int], list[pd.Timestamp]] = {}
    for timestamp in candidates:
        pools.setdefault((month_to_season(timestamp.month), timestamp.hour), []).append(timestamp)
    for pool in pools.values():
        rng.shuffle(pool)
    selected: list[pd.Timestamp] = []
    cells = [(season, hour) for season in ("DJF", "MAM", "JJA", "SON") for hour in (0, 6, 12, 18)]
    for block in range(count // 16):
        for cell in cells:
            try:
                selected.append(pools[cell].pop())
            except (KeyError, IndexError) as error:  # pragma: no cover - source has complete years.
                raise ValueError(f"cannot build balanced {split} selection") from error
    return [value.strftime("%Y-%m-%dT%H:%M:%S") for value in selected]


def _write_frame(store: Path, tensor: xr.DataArray, *, first: bool) -> None:
    """Append one physical, remapped frame without accumulating a split in RAM."""

    frame = xr.Dataset({"data": tensor.astype(np.float32)})
    encoding = {"data": {"chunks": (1, len(CHANNEL_NAMES), 180, 180)}}
    if first:
        frame.to_zarr(store, mode="w", consolidated=False, encoding=encoding, zarr_format=2)
    else:
        frame.to_zarr(store, mode="a", append_dim="time", consolidated=False)


def _init_frame_worker() -> None:
    """Open one independent WeatherBench2 connection per worker process."""

    global _WORKER_REMOTE, _WORKER_TARGET
    _WORKER_REMOTE = open_wb2_era5(validate=True)
    _WORKER_TARGET = get_target_grid_05()


def _load_and_remap_frame(timestamp: str) -> xr.DataArray:
    """Load and remap one frame in an isolated worker process."""

    if _WORKER_REMOTE is None or _WORKER_TARGET is None:  # pragma: no cover - executor invariant.
        raise RuntimeError("WeatherBench2 worker was not initialized")
    source = select_dynamic(_WORKER_REMOTE, pd.DatetimeIndex([timestamp])).load()
    return assemble_model_tensor(conservative_remap(source, _WORKER_TARGET))


def _write_split(
    ds: xr.Dataset,
    timestamps: list[str],
    store: Path,
    target_grid: xr.Dataset,
    *,
    workers: int,
) -> None:
    existing: set[np.datetime64] = set()
    if store.exists():
        existing = set(np.asarray(xr.open_zarr(store, consolidated=False).time.values, dtype="datetime64[ns]"))
    pending = [timestamp for timestamp in sorted(timestamps) if np.datetime64(timestamp) not in existing]
    if workers == 1:
        tensors = (
            assemble_model_tensor(conservative_remap(select_dynamic(ds, pd.DatetimeIndex([timestamp])).load(), target_grid))
            for timestamp in pending
        )
        for index, tensor in enumerate(tensors):
            _write_frame(store, tensor, first=not store.exists() and index == 0)
            print(f"[{store.stem}] {len(existing) + index + 1}/{len(timestamps)} {pending[index]}", flush=True)
    elif pending:
        # Every worker has an independent remote handle; output remains in the
        # sorted ``pending`` order, so parallelism cannot change timestamps or
        # Zarr layout semantics.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_frame_worker,
        ) as executor:
            for index, tensor in enumerate(executor.map(_load_and_remap_frame, pending, chunksize=1)):
                _write_frame(store, tensor, first=not store.exists() and index == 0)
                print(f"[{store.stem}] {len(existing) + index + 1}/{len(timestamps)} {pending[index]}", flush=True)
    zarr.consolidate_metadata(store)


def _write_static(ds: xr.Dataset, store: Path, target_grid: xr.Dataset) -> None:
    native = ds[["land_sea_mask"]].load()
    remapped = conservative_remap(native, target_grid)["land_sea_mask"]
    static = xr.Dataset(
        {
            "ocean_mask": (remapped <= 0.5).astype(np.uint8),
            "land_sea_mask": remapped.astype(np.float32),
        }
    )
    static["ocean_mask"].attrs["policy"] = "conservative land_sea_mask <= 0.5 (1=ocean)"
    static.to_zarr(store, mode="w", consolidated=True, zarr_format=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/weatherbench2_28ch_05_n128"))
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument(
        "--train-population-size",
        type=int,
        default=None,
        help="Size of the deterministic nested subset from which --train-size frames are sliced.",
    )
    parser.add_argument(
        "--train-start-index",
        type=int,
        default=0,
        help="Zero-based starting index within --train-population-size; useful for non-overlapping shards.",
    )
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="Independent frame preparation processes (default: 1).")
    parser.add_argument("--resume-dir", type=Path, default=None, help="Resume an interrupted temporary dataset directory.")
    args = parser.parse_args(argv)
    population_size = args.train_population_size or args.train_size
    if args.train_size <= 0:
        parser.error("--train-size must be positive")
    if population_size <= 0 or population_size % 16:
        parser.error("--train-population-size must be a positive multiple of 16")
    if args.train_start_index < 0 or args.train_start_index + args.train_size > population_size:
        parser.error("requested training slice is outside --train-population-size")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    output = args.output_dir
    if output.exists():
        parser.error(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_dir is None:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        shutil.rmtree(temporary)
        owns_temporary = True
    else:
        temporary = args.resume_dir
        if not temporary.is_dir() or temporary.parent != output.parent:
            parser.error("--resume-dir must be an existing temporary directory beside --output-dir")
        owns_temporary = False
    try:
        train = _training_timestamps(
            population_size=population_size,
            start_index=args.train_start_index,
            count=args.train_size,
            seed=args.seed,
        )
        validation = _evaluation_timestamps("validation", args.validation_size, args.seed + 1)
        test = _evaluation_timestamps("test", args.test_size, args.seed + 2)
        remote = open_wb2_era5(validate=True)
        try:
            target = get_target_grid_05()
            if not (temporary / "static.zarr").exists():
                _write_static(remote, temporary / "static.zarr", target)
            _write_split(remote, train, temporary / "train.zarr", target, workers=args.workers)
            if validation:
                _write_split(remote, validation, temporary / "validation.zarr", target, workers=args.workers)
            if test:
                _write_split(remote, test, temporary / "test.zarr", target, workers=args.workers)
        finally:
            remote.close()
        manifest = {
            "schema_version": "weatherbench2-28ch-0p5-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "source_uri": WB2_URL,
            "grid": {"resolution_degrees": 0.5, "shape": [360, 720], "method": "first_order_conservative", "periodic_longitude": True},
            "channel_order": list(CHANNEL_NAMES),
            "channels": [{"name": channel.name, "units": channel.units, "source_name": channel.source_name, "level_hpa": channel.level} for channel in CHANNEL_SPEC],
            "seed": args.seed,
            "preparation": {"workers": args.workers},
            "train_selection": {
                "population_size": population_size,
                "start_index": args.train_start_index,
                "count": args.train_size,
            },
            "splits": {"train": train, "validation": validation, "test": test},
            "normalization": {"status": "not_fitted", "required_scope": "train_only"},
            "sst_policy": "NaN over land is retained in data; static.ocean_mask identifies valid ocean cells.",
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (temporary / "manifest.json.sha256").write_text(f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        if owns_temporary:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Prepared real WeatherBench2 dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
