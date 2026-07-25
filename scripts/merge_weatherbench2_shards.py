#!/usr/bin/env python3
"""Strictly assemble locally prepared WeatherBench2 train shards.

The command is deliberately a merge-and-validate step, not a downloader.  It
refuses overlaps, missing timestamps, incompatible grids and accidental use of
validation/test data as training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.data.seasonal_subsets import generate_nested_subsets


def _open_train(root: Path) -> xr.Dataset:
    return _open_split(root, "train")


def _open_split(root: Path, split: str) -> xr.Dataset:
    path = root / f"{split}.zarr"
    if not path.exists():
        raise ValueError(f"missing {path}")
    dataset = xr.open_zarr(path, consolidated=True)
    if tuple(dataset.data.dims) != ("time", "channel", "latitude", "longitude"):
        raise ValueError(f"{path} has invalid dimensions {dataset.data.dims}")
    if dataset.data.shape[1:] != (28, 360, 720):
        raise ValueError(f"{path} has invalid data shape {dataset.data.shape}")
    if tuple(str(value) for value in dataset.channel.values) != CHANNEL_NAMES:
        raise ValueError(f"{path} has a non-canonical channel order")
    return dataset


def _timestamps(dataset: xr.Dataset) -> set[np.datetime64]:
    values = np.asarray(dataset.time.values, dtype="datetime64[ns]")
    result = set(values)
    if len(result) != len(values):
        raise ValueError("a shard contains duplicate timestamps")
    return result


def _copy_store(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Prepared local pilot with static/validation/test stores.")
    parser.add_argument("--train-shard", type=Path, action="append", required=True, help="Additional directory containing train.zarr; repeat once per teammate.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/weatherbench2_28ch_05_n128"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    if not (args.base / "static.zarr").exists() or not (args.base / "validation.zarr").exists() or not (args.base / "test.zarr").exists():
        parser.error("--base must contain static.zarr, validation.zarr and test.zarr")

    expected = {
        np.datetime64(value)
        for value in generate_nested_subsets([args.train_size], seed=args.seed)[args.train_size]
    }
    roots = [args.base, *args.train_shard]
    datasets = [_open_train(root) for root in roots]
    validation = _open_split(args.base, "validation")
    test = _open_split(args.base, "test")
    if validation.sizes["time"] != 16 or test.sizes["time"] != 16:
        raise ValueError("base must contain exactly 16 fixed validation and 16 fixed test timestamps")
    seen: set[np.datetime64] = set()
    for root, dataset in zip(roots, datasets, strict=True):
        shard_times = _timestamps(dataset)
        overlap = seen & shard_times
        if overlap:
            raise ValueError(f"duplicate timestamps between shards, first seen in {root}: {sorted(map(str, overlap))[:3]}")
        seen |= shard_times
    if seen != expected:
        missing = sorted(map(str, expected - seen))
        unexpected = sorted(map(str, seen - expected))
        raise ValueError(f"train timestamp set differs from deterministic N={args.train_size}; missing={missing[:3]}, unexpected={unexpected[:3]}")
    holdout_times = _timestamps(validation) | _timestamps(test)
    if seen & holdout_times:
        raise ValueError("training timestamps overlap validation/test timestamps")
    static = xr.open_zarr(args.base / "static.zarr", consolidated=True)
    if "ocean_mask" not in static or tuple(static.ocean_mask.shape) != (360, 720):
        raise ValueError("base static.zarr must contain a 360x720 ocean_mask")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    shutil.rmtree(temporary)
    try:
        merged = xr.concat(datasets, dim="time").sortby("time")
        # xarray opens the original Zarr string coordinate as ``object``.
        # Re-materialise it as a fixed-width Unicode coordinate before
        # writing a new store; otherwise numcodecs may receive the channel
        # dimension length (28) as an object value during serialisation.
        merged = merged.assign_coords(channel=np.asarray(CHANNEL_NAMES, dtype="U5"))
        merged.to_zarr(temporary / "train.zarr", mode="w", consolidated=True, zarr_format=2)
        _copy_store(args.base / "validation.zarr", temporary / "validation.zarr")
        _copy_store(args.base / "test.zarr", temporary / "test.zarr")
        _copy_store(args.base / "static.zarr", temporary / "static.zarr")
        manifest = {
            "schema_version": "weatherbench2-28ch-0p5-merged-v1",
            "source": "local conservative-remapped shards",
            "seed": args.seed,
            "train_size": args.train_size,
            "channel_order": list(CHANNEL_NAMES),
            "grid_shape": [360, 720],
            "train_timestamps": [str(value) for value in merged.time.values],
            "base": str(args.base),
            "train_shards": [str(value) for value in args.train_shard],
            "validation_source": str(args.base / "validation.zarr"),
            "test_source": str(args.base / "test.zarr"),
            "normalization": {"status": "not_fitted", "required_scope": "train_only"},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (temporary / "manifest.json.sha256").write_text(f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
        temporary.replace(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Merged {len(roots)} shards into {args.output_dir} with {args.train_size} exact train timestamps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
