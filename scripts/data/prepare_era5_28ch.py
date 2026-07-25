#!/usr/bin/env python3
"""Prepare the native-variable 28-channel WeatherBench2 dataset safely."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import numpy as np
import xarray as xr

from era5_minimum.data.era5_28ch import (
    SPLIT_RANGES, assemble_model_tensor, expected_times, get_layer, inspect_source,
    validate_prepared, write_split, compute_train_statistics,
)
from era5_minimum.data.weatherbench2 import open_wb2_era5
from era5_minimum.data.manifest import generate_manifest
from era5_minimum.data.seasonal_subsets import generate_nested_subsets
from era5_minimum.data.channel_spec import CHANNEL_NAMES


@click.group()
def cli() -> None:
    """Metadata-first ERA5 28-channel preparation; downloads are split-scoped."""


@cli.command()
def inspect() -> None:
    """Open remote consolidated metadata and validate the expected source contract."""
    ds = open_wb2_era5(validate=False)
    try:
        click.echo(json.dumps(inspect_source(ds), indent=2, default=str))
    finally:
        ds.close()


@cli.command()
def smoke() -> None:
    """Offline compatibility smoke fixture; it never contacts WeatherBench2."""
    root = Path(os.environ.get("SMOKE_OUTPUT_DIR", "smoke_test_output"))
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = rng.random((2, 28, 4, 4), dtype=np.float32)
    data[:, 5, 0, 0] = np.nan
    ds = xr.Dataset({"data": (("time", "channel", "latitude", "longitude"), data)}, coords={
        "time": np.array(["2014-01-01T00:00", "2014-01-01T06:00"], dtype="datetime64[ns]"),
        "channel": list(CHANNEL_NAMES), "latitude": [-90.0, -30.0, 30.0, 90.0], "longitude": [0.0, 90.0, 180.0, 270.0],
    })
    ds.to_zarr(root / "data.zarr", mode="w")
    generate_manifest(str(root), {}, generate_nested_subsets(n_list=[16, 32, 48], seed=42))
    click.echo(f"Offline smoke fixture written to {root}")


@cli.command("download")
@click.option("--split", type=click.Choice(list(SPLIT_RANGES)), required=True)
@click.option("--start", default=None, help="Optional inclusive ISO timestamp within the split.")
@click.option("--end", default=None, help="Optional inclusive ISO timestamp within the split.")
@click.option("--output-dir", default="data/era5_28ch_0p25_6h", type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Replace only the named completed split.")
def download(split: str, start: str | None, end: str | None, output_dir: Path, overwrite: bool) -> None:
    """Lazily download one split or an explicitly small range; never full default accidentally."""
    values = expected_times(split, start, end)
    click.echo(f"Preparing {split}: {values[0]} .. {values[-1]} ({len(values)} timestamps)")
    ds = open_wb2_era5(time_slice=slice(str(values[0]), str(values[-1])), validate=True)
    try:
        target = write_split(ds, output_dir, split, start=str(values[0]), end=str(values[-1]), overwrite=overwrite)
    finally:
        ds.close()
    click.echo(f"Completed {target}; run validate after all desired splits are prepared.")


@cli.command()
@click.option("--dataset-dir", default="data/era5_28ch_0p25_6h", type=click.Path(exists=True, path_type=Path))
def validate(dataset_dir: Path) -> None:
    validate_prepared(dataset_dir)
    click.echo("Prepared ERA5 28ch dataset validation passed.")


@cli.command()
@click.option("--dataset-dir", default="data/era5_28ch_0p25_6h", type=click.Path(exists=True, path_type=Path))
@click.option("--split", type=click.Choice(list(SPLIT_RANGES)), default="validation")
@click.option("--variable", required=True)
@click.option("--timestamp", required=True)
@click.option("--pressure-level-hpa", type=int, default=None)
def layer(dataset_dir: Path, split: str, variable: str, timestamp: str, pressure_level_hpa: int | None) -> None:
    """Inspect one 2-D layer metadata; values are deliberately not printed as JSON."""
    result = get_layer(dataset_dir, split, variable, timestamp, pressure_level_hpa)
    summary = {key: value for key, value in result.items() if key not in {"values", "mask", "latitude", "longitude"}}
    summary["shape"] = list(result["values"].shape)
    click.echo(json.dumps(summary, indent=2, default=str))


@cli.command()
@click.option("--source", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def remap(source: Path, output: Path) -> None:
    """Reserve the required conservative 0.5° operation without a false substitute."""
    raise click.ClickException(
        "BLOCKED: no bounded, validated first-order conservative remapper/weights artifact is implemented. "
        "This command intentionally refuses nearest, subsampling, coarsen mean and bilinear substitutes."
    )


@cli.command()
@click.option("--dataset-dir", default="data/era5_28ch_0p25_6h", type=click.Path(exists=True, path_type=Path))
def statistics(dataset_dir: Path) -> None:
    """Run explicit, lazy Dask reductions on train only (never validation/test)."""
    click.echo(f"Writing {compute_train_statistics(dataset_dir)}")


if __name__ == "__main__":
    cli()
