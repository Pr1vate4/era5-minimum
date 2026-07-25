#!/usr/bin/env python3
"""Fit and evaluate the real-data Patch-PCA reference from prepared Zarr data.

This runner is deliberately a linear, small-sample reference suggested by the
organizers' hint.  It is not presented as a neural model or a serialized
codec: payload reduction and actual bitstream compression remain separate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.baselines.normalization import ChannelNormalizer, denormalize, normalize
from era5_minimum.baselines.patch_pca import PatchPCABaseline
from era5_minimum.baselines.patches import extract_patches, patch_grid_dims
from era5_minimum.codec.evaluation import evaluate_reconstruction
from era5_minimum.data.channel_spec import CHANNEL_NAMES


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _open_split(root: Path, split: str) -> xr.Dataset:
    dataset = xr.open_zarr(root / f"{split}.zarr", consolidated=True)
    if tuple(dataset.data.dims) != ("time", "channel", "latitude", "longitude"):
        raise ValueError(f"{split}.zarr has invalid data dims: {dataset.data.dims}")
    if tuple(str(item) for item in dataset.channel.values) != CHANNEL_NAMES:
        raise ValueError("prepared channel order does not match the official 28-channel contract")
    if dataset.data.shape[1:] != (28, 360, 720):
        raise ValueError(f"expected [N, 28, 360, 720], got {tuple(dataset.data.shape)}")
    return dataset


def _iter_frames(dataset: xr.Dataset) -> Iterator[np.ndarray]:
    for index in range(dataset.sizes["time"]):
        yield dataset.data.isel(time=slice(index, index + 1)).load().values.astype(np.float32, copy=False)


def _fit_normalization_and_ranges(dataset: xr.Dataset) -> tuple[ChannelNormalizer, np.ndarray, np.ndarray]:
    normalizer = ChannelNormalizer(list(CHANNEL_NAMES))
    lower = np.full(len(CHANNEL_NAMES), np.inf, dtype=np.float64)
    upper = np.full(len(CHANNEL_NAMES), -np.inf, dtype=np.float64)
    for frame in _iter_frames(dataset):
        valid = np.isfinite(frame)
        normalizer.update(frame, valid_mask=valid)
        for channel in range(len(CHANNEL_NAMES)):
            values = frame[:, channel][valid[:, channel]]
            if values.size:
                lower[channel] = min(lower[channel], float(values.min()))
                upper[channel] = max(upper[channel], float(values.max()))
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("at least one training channel has no finite values")
    return normalizer, lower, upper


def _patch_batches(
    dataset: xr.Dataset,
    *,
    stats,
    patch_height: int,
    patch_width: int,
) -> Iterator[np.ndarray]:
    for frame in _iter_frames(dataset):
        # SST land cells are NaN in physical data.  Zero is introduced only in
        # normalised model space and remains excluded from scientific metrics.
        normalised = normalize(frame, stats).astype(np.float32)
        normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)
        patches, _ = extract_patches(normalised, patch_height, patch_width)
        yield patches


def _evaluate(model: PatchPCABaseline, dataset: xr.Dataset, stats, train_bounds: np.ndarray) -> dict:
    originals: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame in _iter_frames(dataset):
        normalised = normalize(frame, stats).astype(np.float32)
        reconstruction_norm, _ = model.reconstruct(np.nan_to_num(normalised, nan=0.0))
        reconstruction = denormalize(reconstruction_norm, stats).astype(np.float32)
        originals.append(frame)
        reconstructions.append(reconstruction)
        masks.append(np.isfinite(frame))
    return evaluate_reconstruction(
        np.concatenate(originals),
        np.concatenate(reconstructions),
        latitudes=np.asarray(dataset.latitude.values),
        channel_order=CHANNEL_NAMES,
        train_std=stats.std,
        train_ranges=train_bounds,
        validity_mask=np.concatenate(masks),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default=None)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    root = args.dataset_dir or Path(data_cfg["dataset_dir"])
    output = args.output_dir or Path("outputs") / config["experiment"]["name"]
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = _open_split(root, "train")
    split_name = args.evaluation_split or eval_cfg["split"]
    evaluation = _open_split(root, split_name)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        data_manifest = json.load(handle)

    normalizer, lower, upper = _fit_normalization_and_ranges(train)
    stats = normalizer.finalize(source_split="train")
    stats.save(output / "normalization_train_only.json")
    patch_height, patch_width = (int(value) for value in model_cfg["patch_shape"])
    per_frame = np.prod(patch_grid_dims(360, 720, patch_height, patch_width))
    model = PatchPCABaseline(
        patch_height=patch_height,
        patch_width=patch_width,
        channel_names=list(CHANNEL_NAMES),
        target_compression_ratio=float(model_cfg["target_payload_ratio"]),
        incremental_batch_size=int(model_cfg["incremental_batch_size"]),
    )
    model.fit(
        _patch_batches(train, stats=stats, patch_height=patch_height, patch_width=patch_width),
        n_train_patches=int(train.sizes["time"] * per_frame),
    )
    model.save(output)
    metrics = _evaluate(model, evaluation, stats, np.stack((lower, upper), axis=1))
    compression = model.compression_info(
        original_shape=(28, 360, 720),
        n_patches=int(train.sizes["time"] * per_frame),
        normalization_bytes=int(stats.mean.nbytes + stats.std.nbytes + stats.valid_count.nbytes),
        metadata_bytes=0,
    )
    result = {
        "experiment": config["experiment"],
        "model": {"name": "patch_pca", "n_components": model.n_components, "patch_shape": [patch_height, patch_width]},
        "data": {
            "manifest": str(root / "manifest.json"),
            "source_uri": data_manifest["source_uri"],
            "train_timestamps": int(train.sizes["time"]),
            "evaluation_split": split_name,
            "evaluation_timestamps": int(evaluation.sizes["time"]),
            "channel_order": list(CHANNEL_NAMES),
        },
        "normalization": {"scope": "train_only", "file": "normalization_train_only.json", "range_bounds": np.stack((lower, upper), axis=1).tolist()},
        "metrics": metrics,
        "compression": {
            **compression.to_json_dict(),
            "classification": "tensor/payload reduction baseline only",
            "serialized_compression_ratio": None,
            "note": "No entropy-coded bitstream is produced by this Patch-PCA reference.",
        },
        "reproducibility": {"config": str(args.config), "git_commit": _git_commit(), "seed": config["experiment"]["seed"], "runtime_seconds": time.perf_counter() - started},
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_components": model.n_components, "overall_nrmse": metrics["overall_score"], "runtime_seconds": result["reproducibility"]["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
