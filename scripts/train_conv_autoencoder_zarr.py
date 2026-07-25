#!/usr/bin/env python3
"""Train the compact real-data ConvAE baseline on prepared WeatherBench2 Zarr.

This runner intentionally shares the PCA data contract: fixed temporal splits,
train-only statistics, canonical channels and latitude-weighted physical
evaluation.  It does not claim a serialized codec ratio until a bitstream is
actually produced and decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.baselines.normalization import ChannelNormalizer
from era5_minimum.codec.evaluation import evaluate_reconstruction
from era5_minimum.codec.normalization import NormalizationSpec, denormalize_reconstruction, normalize_physical_tensor
from era5_minimum.codec.rate_distortion import grouped_latitude_distortion
from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.models import ConvAutoencoder


def _open(root: Path, split: str) -> xr.Dataset:
    data = xr.open_zarr(root / f"{split}.zarr", consolidated=True)
    if tuple(data.data.dims) != ("time", "channel", "latitude", "longitude"):
        raise ValueError(f"invalid {split} dimensions: {data.data.dims}")
    if data.data.shape[1:] != (28, 360, 720) or tuple(data.channel.values) != CHANNEL_NAMES:
        raise ValueError(f"{split} is not canonical 28-channel 0.5° WeatherBench2 data")
    return data


def _frame(data: xr.Dataset, index: int) -> np.ndarray:
    return data.data.isel(time=slice(index, index + 1)).load().values.astype(np.float32, copy=False)


def _fit_spec(train: xr.Dataset, manifest: Path) -> tuple[NormalizationSpec, np.ndarray]:
    normalizer = ChannelNormalizer(list(CHANNEL_NAMES))
    lower = np.full(28, np.inf, dtype=np.float64)
    upper = np.full(28, -np.inf, dtype=np.float64)
    for index in range(train.sizes["time"]):
        values = _frame(train, index)
        valid = np.isfinite(values)
        normalizer.update(values, valid_mask=valid)
        for channel in range(28):
            finite = values[:, channel][valid[:, channel]]
            lower[channel] = min(lower[channel], float(finite.min()))
            upper[channel] = max(upper[channel], float(finite.max()))
    stats = normalizer.finalize(source_split="train")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return NormalizationSpec(CHANNEL_NAMES, stats.mean, stats.std, digest, train_only=True), np.stack((lower, upper), axis=1)


def _normalise(values: np.ndarray, spec: NormalizationSpec, ocean_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    normalised, valid, _ = normalize_physical_tensor(
        values, spec=spec, ocean_mask=ocean_mask, sst_index=CHANNEL_NAMES.index("sst")
    )
    return torch.from_numpy(normalised), torch.from_numpy(valid)


@torch.no_grad()
def _evaluate(model: ConvAutoencoder, data: xr.Dataset, spec: NormalizationSpec, bounds: np.ndarray, ocean_mask: np.ndarray, device: torch.device) -> dict:
    originals: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in range(data.sizes["time"]):
        original = _frame(data, index)
        normalised, valid = _normalise(original, spec, ocean_mask)
        reconstruction = model(normalised.to(device)).cpu().numpy()
        physical = denormalize_reconstruction(reconstruction, spec=spec)
        originals.append(original)
        reconstructions.append(physical)
        masks.append(valid.numpy().astype(bool))
    return evaluate_reconstruction(
        np.concatenate(originals), np.concatenate(reconstructions),
        latitudes=data.latitude.values, channel_order=CHANNEL_NAMES,
        train_std=spec.std, train_ranges=bounds, validity_mask=np.concatenate(masks),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count; useful when continuing a small pilot.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-from", type=Path, default=None, help="Compatible prior model.ckpt; --max-steps then means additional steps.")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text()) or {}
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = args.dataset_dir or Path(config["data"]["dataset_dir"])
    output = args.output_dir or Path("outputs") / config["experiment"]["name"]
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train, validation = _open(root, "train"), _open(root, "validation")
    manifest = root / "manifest.json"
    spec, bounds = _fit_spec(train, manifest)
    (output / "normalization_train_only.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    ocean_mask = xr.open_zarr(root / "static.zarr", consolidated=True).ocean_mask.load().values.astype(bool)
    model_cfg, train_cfg = config["model"], config["training"]
    latent_channels = int(model_cfg["latent_channels"])
    model = ConvAutoencoder(28, latent_channels=latent_channels)
    resumed_from: str | None = None
    prior_steps = 0
    if args.resume_from is not None:
        prior = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if dict(prior.get("model_config", {})) != {"in_channels": 28, "latent_channels": latent_channels}:
            raise ValueError("--resume-from model configuration is incompatible")
        model.load_state_dict(prior["state_dict"], strict=True)
        prior_steps = int(prior.get("training", {}).get("steps", 0))
        resumed_from = str(args.resume_from)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable > 20_000_000:
        raise ValueError(f"trainable parameter limit exceeded: {trainable}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    if max_steps < 1 or max_steps > 50_000:
        raise ValueError("max_steps must be in [1, 50000]")
    latitude = torch.from_numpy(train.latitude.values.astype(np.float32)).to(device)
    steps = 0
    history: list[float] = []
    started = time.perf_counter()
    model.train()
    epochs = int(args.epochs or train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    for _epoch in range(epochs):
        for index in torch.randperm(train.sizes["time"]).tolist():
            inputs, valid = _normalise(_frame(train, index), spec, ocean_mask)
            inputs, valid = inputs.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            distortion = grouped_latitude_distortion(
                prediction, inputs, valid, latitudes=latitude, loss_type="mse", surface_weight=0.5, pressure_weight=0.5
            )
            distortion.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            history.append(float(distortion.total.detach().cpu()))
            steps += 1
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    metrics = _evaluate(model.eval(), validation, spec, bounds, ocean_mask, device)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {"in_channels": 28, "latent_channels": latent_channels},
        "codec_config": {"version": "conv-ae-real-v1", "quantization_step": float(config["codec"]["quantization_step"])},
        "normalization": spec.to_dict(), "channel_order": list(CHANNEL_NAMES),
        "preprocessing": {"sst_index": CHANNEL_NAMES.index("sst"), "ocean_mask_policy": "static.zarr ocean_mask"},
        "training": {"seed": seed, "steps": steps + prior_steps, "additional_steps": steps, "device": str(device), "trainable_params": trainable, "runtime_seconds": time.perf_counter() - started, "resumed_from": resumed_from},
    }
    torch.save(checkpoint, output / "model.ckpt")
    result = {"experiment": config["experiment"], "data": {"dataset_dir": str(root), "train_timestamps": int(train.sizes["time"]), "validation_timestamps": int(validation.sizes["time"])}, "model": checkpoint["model_config"], "training": checkpoint["training"], "metrics": metrics, "codec": {"serialized_compression_ratio": None, "status": "checkpoint ready; bitstream evaluation pending"}, "loss_history": history}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"steps": steps + prior_steps, "additional_steps": steps, "overall_nrmse": metrics["overall_score"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
