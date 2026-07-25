#!/usr/bin/env python3
"""Measure a checkpoint-backed ConvAE bitstream on one real Zarr frame."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec.acceptance import WeatherCodec, compression_metrics
from era5_minimum.codec.evaluation import evaluate_reconstruction
from era5_minimum.data.channel_spec import CHANNEL_NAMES


def _train_bounds(dataset: xr.Dataset) -> np.ndarray:
    lower = np.full(28, np.inf, dtype=np.float64)
    upper = np.full(28, -np.inf, dtype=np.float64)
    for index in range(dataset.sizes["time"]):
        frame = dataset.data.isel(time=slice(index, index + 1)).load().values
        for channel in range(28):
            values = frame[:, channel][np.isfinite(frame[:, channel])]
            lower[channel] = min(lower[channel], float(values.min()))
            upper[channel] = max(upper[channel], float(values.max()))
    return np.stack((lower, upper), axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {args.output_dir}")
    train = xr.open_zarr(args.dataset_dir / "train.zarr", consolidated=True)
    evaluation = xr.open_zarr(args.dataset_dir / f"{args.split}.zarr", consolidated=True)
    if not 0 <= args.index < evaluation.sizes["time"]:
        parser.error("--index is outside the evaluation split")
    ocean = xr.open_zarr(args.dataset_dir / "static.zarr", consolidated=True).ocean_mask.load().values.astype(bool)
    original = evaluation.data.isel(time=slice(args.index, args.index + 1)).load().values.astype(np.float32)
    codec = WeatherCodec.load(args.checkpoint, device=args.device)
    model_input, validity, _ = codec.preprocess(original, ocean_mask=ocean)
    result = codec.reconstruct(model_input)
    reconstructed = codec.postprocess(result.reconstruction_normalized, ocean_mask=ocean)
    bounds = _train_bounds(train)
    metrics = evaluate_reconstruction(
        original, reconstructed, latitudes=evaluation.latitude.values,
        channel_order=CHANNEL_NAMES, train_std=codec.normalization.std,
        train_ranges=bounds, validity_mask=validity.astype(bool),
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "frame.bitstream").write_bytes(result.bitstream)
    report = {
        "status": "technical_codec_pilot",
        "timestamp": str(evaluation.time.values[args.index]),
        "exact_quantized_symbol_roundtrip": True,
        "compression": compression_metrics(original_shape=tuple(original.shape), bitstream=result.bitstream, header=result.header),
        "tensor_element_ratio": float(np.prod(original.shape) / np.prod(result.latent_shape)),
        "metrics": metrics,
        "limitations": ["Result is tied to the supplied checkpoint and evaluation frame.", "A pilot N=16 checkpoint is not a final N=128 scientific result."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"bitstream_bytes": len(result.bitstream), "serialized_ratio": report["compression"]["compression_ratio"], "overall_nrmse": metrics["overall_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
