from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec.resources import measure_runtime_resources, write_resource_usage
from era5_minimum.codec.workflow import SMOKE_CHANNELS, build_smoke_tensor
from era5_minimum.data.synthetic import SyntheticERA5Dataset, make_synthetic_era5
from era5_minimum.latent_probe import (
    ProbeConfig,
    build_latent_forecast_pairs,
    select_pair_subset,
    train_latent_probe,
)
from era5_minimum.metrics import mae, rmse
from era5_minimum.models import ConvAutoencoder, PCAAutoencoder


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    seed = int(config["seed"])
    data_config = config["data"]
    model_config = config["model"]
    probe_config = config["probe"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_config.get("source") != "synthetic":
        raise NotImplementedError("train_latent_probe currently supports source=synthetic only")

    raw, _ = make_synthetic_era5(
        samples=int(data_config["samples"]),
        height=int(data_config["height"]),
        width=int(data_config["width"]),
        channels=tuple(data_config["channels"]),
        seed=seed,
    )
    train_samples = int(data_config["train_samples"])
    validation_samples = int(data_config["validation_samples"])
    if train_samples + validation_samples > len(raw):
        raise ValueError("train_samples + validation_samples exceeds available samples")

    train_raw = raw[:train_samples]
    validation_raw = raw[train_samples : train_samples + validation_samples]
    train_dataset = SyntheticERA5Dataset(train_raw)
    validation_dataset = SyntheticERA5Dataset(validation_raw, mean=train_dataset.mean, std=train_dataset.std)

    pca = PCAAutoencoder(latent_dim=int(model_config["latent_dim"]))
    pca.fit(train_dataset.tensor.numpy())

    train_latents = pca.encode(train_dataset.tensor.numpy())
    validation_latents = pca.encode(validation_dataset.tensor.numpy())
    train_pairs = build_latent_forecast_pairs(train_latents)
    validation_pairs = build_latent_forecast_pairs(validation_latents)

    result = train_latent_probe(
        train_inputs=train_pairs.inputs,
        train_targets=train_pairs.targets,
        validation_inputs=validation_pairs.inputs,
        validation_targets=validation_pairs.targets,
        config=ProbeConfig(
            latent_dim=int(model_config["latent_dim"]),
            hidden_dim=int(probe_config["hidden_dim"]),
            batch_size=int(probe_config["batch_size"]),
            learning_rate=float(probe_config["learning_rate"]),
            max_steps=int(probe_config["max_steps"]),
            parameter_limit=int(probe_config.get("parameter_limit", 2_000_000)),
            seed=seed,
        ),
        output_dir=output_dir,
        train_pair_indices=train_pairs.pair_indices,
        validation_pair_indices=validation_pairs.pair_indices,
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    probe = _build_probe_from_config(checkpoint["config"])
    probe.load_state_dict(checkpoint["state_dict"])
    probe.eval()
    with torch.no_grad():
        forecast_latents = probe(torch.from_numpy(validation_pairs.inputs.astype(np.float32))).cpu().numpy()

    forecast_reconstruction = pca.decode(forecast_latents)
    target_reconstruction = pca.decode(validation_pairs.targets)
    persistence_reconstruction = pca.decode(validation_pairs.inputs)

    forecast_payload = {
        "normalized": {
            "rmse": rmse(
                torch.from_numpy(forecast_reconstruction),
                torch.from_numpy(target_reconstruction),
            ),
            "mae": mae(
                torch.from_numpy(forecast_reconstruction),
                torch.from_numpy(target_reconstruction),
            ),
        },
        "persistence_normalized": {
            "rmse": rmse(
                torch.from_numpy(persistence_reconstruction),
                torch.from_numpy(target_reconstruction),
            ),
            "mae": mae(
                torch.from_numpy(persistence_reconstruction),
                torch.from_numpy(target_reconstruction),
            ),
        },
    }

    np.savez_compressed(
        output_dir / "forecast_reconstruction.npz",
        predicted_latent=forecast_latents.astype(np.float32),
        target_latent=validation_pairs.targets.astype(np.float32),
        predicted_normalized=forecast_reconstruction.astype(np.float32),
        target_normalized=target_reconstruction.astype(np.float32),
    )

    metrics_path = result.metrics_path
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["forecast"] = forecast_payload
    metrics["forecast_reconstruction_path"] = "forecast_reconstruction.npz"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_probe_resource_usage(
        output_dir=output_dir,
        run_id=f"latent-probe-config-{seed}",
        wall_clock_seconds=time.perf_counter() - started,
        parameter_count=int(metrics["parameter_count"]),
        optimizer_steps=int(metrics["optimizer_steps"]),
        train_pair_count=int(metrics["train_pair_count"]),
        validation_pair_count=int(metrics["validation_pair_count"]),
    )

    return metrics


def run_from_codec_checkpoint(
    *,
    codec_checkpoint: str | Path,
    pair_count: int,
    max_steps: int,
    output_dir: str | Path,
    smoke_test: bool,
    seed: int = 7,
    hidden_dim: int = 64,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    parameter_limit: int = 2_000_000,
    height: int = 8,
    width: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not smoke_test:
        raise NotImplementedError("codec-checkpoint probe currently supports --smoke-test only")
    checkpoint = torch.load(Path(codec_checkpoint), map_location="cpu", weights_only=False)
    channel_order = tuple(checkpoint["channel_order"])
    if channel_order != SMOKE_CHANNELS:
        raise NotImplementedError("codec-checkpoint probe currently supports smoke codec channel order only")

    model_cfg = checkpoint["model_config"]
    model = ConvAutoencoder(
        in_channels=int(model_cfg["in_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    validation_pair_count = max(1, min(pair_count // 2, pair_count))
    total_samples = pair_count + validation_pair_count + 2
    raw, _latitudes, _ocean_mask = build_smoke_tensor(samples=total_samples, height=height, width=width, seed=seed)

    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, -1, 1, 1)
    normalized = ((raw - mean) / std).astype(np.float32)
    train_samples = pair_count + 1
    validation_samples = validation_pair_count + 1
    train_tensor = normalized[:train_samples]
    validation_tensor = normalized[train_samples : train_samples + validation_samples]

    with torch.no_grad():
        train_latents = model.encode(torch.from_numpy(train_tensor)).cpu().numpy()
        validation_latents = model.encode(torch.from_numpy(validation_tensor)).cpu().numpy()

    train_flat = train_latents.reshape(train_latents.shape[0], -1)
    validation_flat = validation_latents.reshape(validation_latents.shape[0], -1)
    train_pairs = select_pair_subset(build_latent_forecast_pairs(train_flat), pair_count=pair_count)
    validation_pairs = build_latent_forecast_pairs(validation_flat)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_config = ProbeConfig(
        latent_dim=int(train_pairs.inputs.shape[1]),
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        parameter_limit=parameter_limit,
        seed=seed,
        forecast_horizon_hours=6,
    )
    result = train_latent_probe(
        train_inputs=train_pairs.inputs,
        train_targets=train_pairs.targets,
        validation_inputs=validation_pairs.inputs,
        validation_targets=validation_pairs.targets,
        config=probe_config,
        output_dir=output_dir,
        train_pair_indices=train_pairs.pair_indices,
        validation_pair_indices=validation_pairs.pair_indices,
        extra_metadata={
            "encoder_frozen": True,
            "decoder_frozen": True,
            "codec_checkpoint": str(codec_checkpoint),
        },
    )

    checkpoint_payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    probe = _build_probe_from_config(checkpoint_payload["config"])
    probe.load_state_dict(checkpoint_payload["state_dict"])
    probe.eval()
    with torch.no_grad():
        forecast_latents = probe(torch.from_numpy(validation_pairs.inputs.astype(np.float32))).cpu().numpy()

    latent_shape = train_latents.shape[1:]
    forecast_latents_reshaped = forecast_latents.reshape((-1, *latent_shape))
    target_latents_reshaped = validation_pairs.targets.reshape((-1, *latent_shape))
    persistence_latents_reshaped = validation_pairs.inputs.reshape((-1, *latent_shape))
    with torch.no_grad():
        forecast_reconstruction = model.decode(torch.from_numpy(forecast_latents_reshaped)).cpu().numpy()
        target_reconstruction = model.decode(torch.from_numpy(target_latents_reshaped)).cpu().numpy()
        persistence_reconstruction = model.decode(torch.from_numpy(persistence_latents_reshaped)).cpu().numpy()

    forecast_payload = {
        "normalized": {
            "rmse": rmse(torch.from_numpy(forecast_reconstruction), torch.from_numpy(target_reconstruction)),
            "mae": mae(torch.from_numpy(forecast_reconstruction), torch.from_numpy(target_reconstruction)),
        },
        "persistence_normalized": {
            "rmse": rmse(torch.from_numpy(persistence_reconstruction), torch.from_numpy(target_reconstruction)),
            "mae": mae(torch.from_numpy(persistence_reconstruction), torch.from_numpy(target_reconstruction)),
        },
    }

    np.savez_compressed(
        output_dir / "forecast_reconstruction.npz",
        predicted_latent=forecast_latents.astype(np.float32),
        target_latent=validation_pairs.targets.astype(np.float32),
        predicted_normalized=forecast_reconstruction.astype(np.float32),
        target_normalized=target_reconstruction.astype(np.float32),
    )

    metrics_path = result.metrics_path
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["forecast"] = forecast_payload
    metrics["forecast_reconstruction_path"] = "forecast_reconstruction.npz"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_probe_resource_usage(
        output_dir=output_dir,
        run_id=f"latent-probe-checkpoint-{seed}",
        wall_clock_seconds=time.perf_counter() - started,
        parameter_count=int(metrics["parameter_count"]),
        optimizer_steps=int(metrics["optimizer_steps"]),
        train_pair_count=int(metrics["train_pair_count"]),
        validation_pair_count=int(metrics["validation_pair_count"]),
    )
    return metrics


def _build_probe_from_config(config: dict[str, Any]) -> torch.nn.Module:
    from era5_minimum.latent_probe import LatentProbeMLP

    return LatentProbeMLP(latent_dim=int(config["latent_dim"]), hidden_dim=int(config["hidden_dim"]))


def _write_probe_resource_usage(
    *,
    output_dir: Path,
    run_id: str,
    wall_clock_seconds: float,
    parameter_count: int,
    optimizer_steps: int,
    train_pair_count: int,
    validation_pair_count: int,
) -> None:
    record = measure_runtime_resources(
        run_id=run_id,
        operation="train_latent_probe",
        grid_resolution="smoke",
        dataset_size=train_pair_count + validation_pair_count,
        started_at=None,
        finished_at=None,
        wall_clock_seconds=wall_clock_seconds,
        trainable_parameter_count=parameter_count,
        total_parameter_count=parameter_count,
        optimizer_steps=optimizer_steps,
        examples_seen=train_pair_count,
        unique_train_timestamps=train_pair_count,
        patches_seen=train_pair_count,
        encode_seconds=None,
        decode_seconds=None,
        bitstream_bytes=None,
        parameter_limit=2_000_000,
        max_vram_gb=24.0,
        max_gpu_hours=None,
        max_optimizer_steps=5_000,
    )
    write_resource_usage(output_dir / "resource_usage.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a latent forecast probe")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--codec-checkpoint", help="Path to codec checkpoint")
    parser.add_argument("--pair-count", type=int, help="Exact number of train pairs to use")
    parser.add_argument("--max-steps", type=int, help="Override optimizer step limit")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic smoke data")
    args = parser.parse_args()
    if args.config:
        run(args.config)
        return
    if args.codec_checkpoint:
        if args.pair_count is None or args.max_steps is None or args.output_dir is None:
            raise SystemExit("--codec-checkpoint requires --pair-count, --max-steps and --output-dir")
        run_from_codec_checkpoint(
            codec_checkpoint=args.codec_checkpoint,
            pair_count=args.pair_count,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            smoke_test=args.smoke_test,
        )
        return
    raise SystemExit("either --config or --codec-checkpoint must be provided")


if __name__ == "__main__":
    main()
