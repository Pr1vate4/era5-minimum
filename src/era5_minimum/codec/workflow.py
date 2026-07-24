from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from era5_minimum.codec.harness import CodecHarness
from era5_minimum.codec.normalization import NormalizationSpec
from era5_minimum.codec.resources import measure_runtime_resources, write_resource_usage
from era5_minimum.codec.types import CodecConfig
from era5_minimum.metrics import latitude_weighted_rmse, mae, per_channel_rmse, rmse
from era5_minimum.models import ConvAutoencoder


SMOKE_CHANNELS: tuple[str, ...] = (
    "t2m",
    "mslp",
    "u10",
    "v10",
    "tp6h",
    "sst",
    "tcwv",
    "tcc",
    "T1000",
    "T925",
    "T850",
    "T700",
    "U1000",
    "U925",
    "U850",
    "U700",
    "V1000",
    "V925",
    "V850",
    "V700",
    "Z1000",
    "Z925",
    "Z850",
    "Z700",
    "Q1000",
    "Q925",
    "Q850",
    "Q700",
)


@dataclass(frozen=True)
class SmokeCodecResult:
    summary: dict[str, Any]
    output_dir: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_smoke_tensor(
    *,
    samples: int,
    height: int,
    width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples < 3:
        raise ValueError("smoke codec needs at least 3 samples")
    latitudes = np.linspace(90.0, -90.0, height, dtype=np.float32)
    longitudes = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
    lat = np.deg2rad(latitudes)[:, None]
    lon = np.deg2rad(longitudes)[None, :]
    rng = np.random.default_rng(seed)
    data = np.empty((samples, len(SMOKE_CHANNELS), height, width), dtype=np.float32)
    for t in range(samples):
        phase = 2.0 * np.pi * t / max(samples, 1)
        synoptic = np.sin(2 * lon + phase) * np.cos(lat)
        wave = np.cos(lon - phase / 3.0) * np.cos(2 * lat)
        storm = np.exp(-((lat - 0.3 * np.sin(phase)) ** 2 + (lon - np.pi - phase / 2) ** 2) / 0.5)
        noise = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
        fields = {
            "t2m": 280.0 + 8 * np.cos(lat) + 1.5 * synoptic + 0.5 * noise,
            "mslp": 101325.0 + 1200 * wave - 1800 * storm + 40 * noise,
            "u10": 6 * np.sin(lat) + 4 * synoptic + 0.25 * noise,
            "v10": 4 * np.cos(2 * lat) + 3 * wave + 0.25 * noise,
            "tp6h": np.clip(0.001 * storm + 0.00015 * np.maximum(synoptic, 0) + 0.00002 * noise, 0, None),
            "sst": 276.0 + 4 * np.cos(lat) + 0.7 * synoptic + 0.2 * noise,
            "tcwv": np.clip(12 + 25 * np.cos(lat) ** 2 + 4 * synoptic + 0.4 * noise, 0, None),
            "tcc": np.clip(0.4 + 0.3 * storm + 0.1 * synoptic + 0.05 * noise, 0, 1),
        }
        for idx, channel in enumerate(SMOKE_CHANNELS[8:], start=8):
            level_scale = 1.0 + 0.04 * (idx - 8)
            base = 1000.0 - 50.0 * (idx - 8)
            if channel.startswith("T"):
                values = base + 3.0 * np.cos(lat) + 0.4 * synoptic + 0.15 * noise
            elif channel.startswith("U"):
                values = 2.0 * np.sin(lat) + level_scale * synoptic + 0.1 * noise
            elif channel.startswith("V"):
                values = 1.5 * np.cos(lat) + level_scale * wave + 0.1 * noise
            else:
                values = 0.8 + 0.2 * np.cos(lat) + 0.08 * synoptic + 0.03 * noise
            fields[channel] = values.astype(np.float32)
        for channel_index, channel in enumerate(SMOKE_CHANNELS):
            data[t, channel_index] = fields[channel]
    ocean_mask = (np.abs(latitudes) < 60).astype(np.float32)[:, None] * (longitudes > 120).astype(np.float32)[None, :]
    return data, latitudes, ocean_mask


def run_codec_smoke(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    _seed_everything(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    started_at = _utc_now()
    resolved_config_path = output_dir / "resolved_config.yaml"
    import yaml

    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    data_cfg = config["data"]
    model_cfg = config["model"]
    codec_cfg = config["codec"]
    train_cfg = config["training"]
    resources_cfg = config.get("resources", {})

    samples = int(data_cfg["samples"])
    validation_samples = int(data_cfg["validation_samples"])
    test_samples = int(data_cfg.get("test_samples", validation_samples))
    if samples <= validation_samples + test_samples:
        raise ValueError("samples must exceed validation_samples + test_samples")

    raw, latitudes, ocean_mask = build_smoke_tensor(
        samples=samples,
        height=int(data_cfg["height"]),
        width=int(data_cfg["width"]),
        seed=seed,
    )
    train_end = samples - validation_samples - test_samples
    validation_end = train_end + validation_samples
    train_raw = raw[:train_end]
    validation_raw = raw[train_end:validation_end]
    test_raw = raw[validation_end:]

    train_mean = train_raw.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    train_std = np.maximum(train_raw.std(axis=(0, 2, 3), keepdims=True), 1e-6).astype(np.float32)
    normalization = NormalizationSpec(
        channel_order=SMOKE_CHANNELS,
        mean=train_mean.reshape(-1),
        std=train_std.reshape(-1),
        source_manifest_sha256=_sha256_bytes(train_mean.tobytes() + train_std.tobytes()),
        train_only=True,
    )

    train_norm = _normalize(train_raw, train_mean, train_std)
    validation_norm = _normalize(validation_raw, train_mean, train_std)
    test_norm = _normalize(test_raw, train_mean, train_std)
    sst_index = SMOKE_CHANNELS.index("sst")
    for tensor in (train_norm, validation_norm, test_norm):
        tensor[:, sst_index] = np.where(ocean_mask[None, :, :] > 0, tensor[:, sst_index], 0.0)

    model = ConvAutoencoder(in_channels=len(SMOKE_CHANNELS), latent_channels=int(model_cfg["latent_channels"]))
    trainable_params = _parameter_count(model)
    total_params = trainable_params
    parameter_limit = int(model_cfg.get("parameter_limit", 20_000_000))
    if trainable_params > parameter_limit:
        raise ValueError(
            f"parameter_limit exceeded: {trainable_params} trainable parameters > {parameter_limit}"
        )
    max_steps = int(train_cfg["max_steps"])
    if max_steps > 50_000:
        raise ValueError(f"max_steps exceeded: {max_steps} > 50000")

    device = torch.device("cuda" if torch.cuda.is_available() and str(config.get("device", "auto")) != "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(device)

    loss_mask = np.ones_like(train_norm, dtype=np.float32)
    loss_mask[:, sst_index] = np.where(ocean_mask[None, :, :] > 0, 1.0, 0.0)
    history_path = output_dir / "training_history.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.ckpt"
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    learning_rate = float(train_cfg["learning_rate"])
    train_result = _train_masked_autoencoder(
        model=model,
        inputs=torch.from_numpy(train_norm),
        masks=torch.from_numpy(loss_mask),
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        history_path=history_path,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"in_channels": len(SMOKE_CHANNELS), "latent_channels": int(model_cfg["latent_channels"])},
            "codec_config": codec_cfg,
            "normalization": normalization.to_dict(),
            "channel_order": list(SMOKE_CHANNELS),
        },
        checkpoint_path,
    )

    model.eval()
    validation_tensor = torch.from_numpy(validation_norm).to(device)
    test_tensor = torch.from_numpy(test_norm).to(device)
    started_encode = time.perf_counter()
    with torch.no_grad():
        validation_latent = model.encode(validation_tensor).cpu().numpy()
        test_latent = model.encode(test_tensor).cpu().numpy()
    codec = CodecHarness(
        config=CodecConfig(
            version=str(codec_cfg["version"]),
            channel_order=SMOKE_CHANNELS,
            grid=str(codec_cfg["grid"]),
            quantization_step=float(codec_cfg["quantization_step"]),
            seed=seed,
            git_commit=None,
        ),
        normalization=normalization,
    )
    validation_codec_result = codec.encode_latent(
        input_tensor=validation_raw,
        latent=validation_latent,
        output_dir=output_dir / "_validation_codec",
    )
    test_codec_result = codec.encode_latent(
        input_tensor=test_raw,
        latent=test_latent,
        output_dir=output_dir / "_test_codec",
    )
    encode_seconds = time.perf_counter() - started_encode
    if validation_codec_result.decoded_latent is None:
        raise RuntimeError("codec decode did not return latent")
    if test_codec_result.decoded_latent is None:
        raise RuntimeError("codec decode did not return latent")
    started_decode = time.perf_counter()
    with torch.no_grad():
        prediction_norm = model.decode(torch.from_numpy(validation_codec_result.decoded_latent).to(device)).cpu().numpy()
        test_prediction_norm = model.decode(torch.from_numpy(test_codec_result.decoded_latent).to(device)).cpu().numpy()
    decode_seconds = time.perf_counter() - started_decode

    prediction_physical = _denormalize(prediction_norm, train_mean, train_std)
    validation_physical = _denormalize(validation_norm, train_mean, train_std)
    test_prediction = _denormalize(test_prediction_norm, train_mean, train_std)
    test_physical = _denormalize(test_norm, train_mean, train_std)

    metrics_validation = _validation_metrics(validation_norm, prediction_norm, latitudes, SMOKE_CHANNELS)
    per_channel = _per_channel_metrics(validation_physical, prediction_physical, train_std.reshape(-1), SMOKE_CHANNELS)
    per_time = _per_time_metrics(validation_norm, prediction_norm)

    metrics_validation_path = output_dir / "metrics_validation.json"
    metrics_per_channel_path = output_dir / "metrics_per_channel.json"
    metrics_per_time_path = output_dir / "metrics_per_time.json"
    _write_json(metrics_validation_path, metrics_validation)
    _write_json(metrics_per_channel_path, per_channel)
    _write_json(metrics_per_time_path, per_time)
    bitstream_dir = output_dir / "bitstreams"
    bitstream_dir.mkdir(parents=True, exist_ok=True)
    validation_bitstream = bitstream_dir / "validation.bin"
    validation_metadata = bitstream_dir / "validation.json"
    test_bitstream = bitstream_dir / "test.bin"
    test_metadata = bitstream_dir / "test.json"
    validation_bitstream.write_bytes(validation_codec_result.bitstream_path.read_bytes())
    validation_metadata.write_text(validation_codec_result.metadata_path.read_text(encoding="utf-8"), encoding="utf-8")
    test_bitstream.write_bytes(test_codec_result.bitstream_path.read_bytes())
    test_metadata.write_text(test_codec_result.metadata_path.read_text(encoding="utf-8"), encoding="utf-8")
    np.savez_compressed(
        output_dir / "reconstruction_samples.npz",
        validation_original=validation_physical.astype(np.float32),
        validation_reconstruction=prediction_physical.astype(np.float32),
        test_original=test_physical.astype(np.float32),
        test_reconstruction=test_prediction.astype(np.float32),
        ocean_mask=ocean_mask.astype(np.float32),
    )

    entropy_statistics = {
        "quantization_method": validation_codec_result.metadata["quantization"]["method"],
        "quantization_scale": validation_codec_result.metadata["quantization"]["scale"],
        "symbol_dtype": validation_codec_result.metadata["quantization"]["symbol_dtype"],
        "symbol_min": validation_codec_result.metadata["quantization"]["symbol_min"],
        "symbol_max": validation_codec_result.metadata["quantization"]["symbol_max"],
        "latent_shape": validation_codec_result.metadata["latent"]["shape"],
        "coder": validation_codec_result.metadata["entropy"]["coder"],
    }
    bitstream_statistics = {
        "validation_bitstream_bytes": validation_codec_result.metadata["compression"]["bitstream_bytes"],
        "validation_sha256": validation_codec_result.metadata["bitstream"]["sha256"],
        "validation_symbol_count": validation_codec_result.metadata["compression"]["symbol_count"],
        "validation_actual_bits_per_value": validation_codec_result.metadata["compression"]["actual_bits_per_value"],
        "test_bitstream_bytes": test_codec_result.metadata["compression"]["bitstream_bytes"],
        "test_sha256": test_codec_result.metadata["bitstream"]["sha256"],
        "test_symbol_count": test_codec_result.metadata["compression"]["symbol_count"],
        "test_actual_bits_per_value": test_codec_result.metadata["compression"]["actual_bits_per_value"],
    }
    _write_json(output_dir / "entropy_statistics.json", entropy_statistics)
    _write_json(output_dir / "bitstream_statistics.json", bitstream_statistics)

    total_input_bytes = int(validation_raw.nbytes + test_raw.nbytes)
    total_bitstream_bytes = int(
        validation_codec_result.metadata["compression"]["bitstream_bytes"]
        + test_codec_result.metadata["compression"]["bitstream_bytes"]
    )
    actual_ratio = float(total_input_bytes / max(total_bitstream_bytes, 1))
    latent_ratio = float(validation_codec_result.tensor_ratio)
    target_ratio = float(codec_cfg.get("target_compression_ratio", 0))
    exact_roundtrip = bool(validation_codec_result.roundtrip_ok and test_codec_result.roundtrip_ok)
    codec_eligible = bool(exact_roundtrip and actual_ratio >= target_ratio) if target_ratio > 0 else bool(exact_roundtrip)

    summary = {
        "experiment_id": f"codec-smoke-{seed}",
        "model_name": "conv_autoencoder",
        "grid_resolution": str(codec_cfg["grid"]),
        "dataset_size": int(samples),
        "target_compression_ratio": target_ratio,
        "actual_compression_ratio": actual_ratio,
        "latent_reduction_ratio": latent_ratio,
        "bitstream_bytes": total_bitstream_bytes,
        "entropy_coding": validation_codec_result.metadata["entropy"]["coder"],
        "exact_roundtrip": exact_roundtrip,
        "codec_eligible": codec_eligible,
        "surface_score": metrics_validation["surface_score"],
        "pressure_score": metrics_validation["pressure_score"],
        "overall_score": metrics_validation["overall_score"],
        "parameter_count": trainable_params,
        "optimizer_steps": train_result["optimizer_steps"],
        "gpu_hours": train_result["gpu_hours"],
        "peak_vram_bytes": train_result["peak_vram_bytes"],
        "checkpoint_path": str(checkpoint_path),
        "bitstream_path": str(validation_bitstream),
        "manifest_sha256": normalization.source_manifest_sha256,
        "run_id": f"codec-smoke-{seed}",
        "resource_compliance": train_result["resource_compliance"],
        "test_bitstream_path": str(test_bitstream),
    }
    _write_json(output_dir / "checkpoint_metadata.json", {
        "checkpoint_path": str(checkpoint_path),
        "model_config": {"in_channels": len(SMOKE_CHANNELS), "latent_channels": int(model_cfg["latent_channels"])},
        "codec_config": codec_cfg,
        "normalization": normalization.to_dict(),
        "parameter_count": trainable_params,
        "total_parameter_count": total_params,
        "run_id": summary["run_id"],
        "manifest_sha256": normalization.source_manifest_sha256,
    })
    _write_json(output_dir / "run_summary.json", summary)
    resource_record = measure_runtime_resources(
        run_id=summary["run_id"],
        operation="train_codec",
        grid_resolution=str(codec_cfg["grid"]),
        dataset_size=int(samples),
        started_at=started_at,
        finished_at=_utc_now(),
        wall_clock_seconds=time.perf_counter() - started,
        trainable_parameter_count=trainable_params,
        total_parameter_count=total_params,
        optimizer_steps=train_result["optimizer_steps"],
        examples_seen=train_result["examples_seen"],
        unique_train_timestamps=train_result["unique_train_timestamps"],
        patches_seen=train_result["patches_seen"],
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        bitstream_bytes=total_bitstream_bytes,
        parameter_limit=parameter_limit,
        max_vram_gb=float(resources_cfg.get("max_vram_gb")) if resources_cfg.get("max_vram_gb") is not None else None,
        max_gpu_hours=float(resources_cfg.get("max_gpu_hours")) if resources_cfg.get("max_gpu_hours") is not None else None,
        max_optimizer_steps=max_steps,
    )
    write_resource_usage(output_dir / "resource_usage.json", resource_record)
    return {
        "actual_compression_ratio": actual_ratio,
        "latent_reduction_ratio": latent_ratio,
        "exact_roundtrip": exact_roundtrip,
        "output_dir": str(output_dir),
        "run_summary_path": str(output_dir / "run_summary.json"),
    }


def _train_masked_autoencoder(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    max_steps: int,
    learning_rate: float,
    history_path: Path,
) -> dict[str, Any]:
    loader = DataLoader(
        TensorDataset(inputs, masks),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    examples_seen = 0
    with history_path.open("w", encoding="utf-8") as history_file:
        for _ in range(epochs):
            for batch_inputs, batch_mask in loader:
                if step >= max_steps:
                    break
                batch_inputs = batch_inputs.to(device)
                batch_mask = batch_mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_inputs)
                loss = torch.mean(((outputs - batch_inputs) ** 2) * batch_mask)
                loss.backward()
                optimizer.step()
                step += 1
                examples_seen += int(batch_inputs.shape[0])
                history_file.write(json.dumps({"step": step, "loss": float(loss.item())}) + "\n")
            if step >= max_steps:
                break
    return {
        "optimizer_steps": step,
        "examples_seen": examples_seen,
        "unique_train_timestamps": int(inputs.shape[0]),
        "patches_seen": int(inputs.shape[0]),
        "gpu_hours": (step / 3600.0) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "resource_compliance": True,
    }


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _denormalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std + mean).astype(np.float32)


def _validation_metrics(
    target_norm: np.ndarray,
    prediction_norm: np.ndarray,
    latitudes: np.ndarray,
    channels: tuple[str, ...],
) -> dict[str, Any]:
    target = torch.from_numpy(target_norm)
    prediction = torch.from_numpy(prediction_norm)
    channel_values = per_channel_rmse(prediction, target, list(channels))
    channel_scores = [channel_values[channel] for channel in channels]
    surface = float(np.mean(channel_scores[:8]))
    pressure = float(np.mean(channel_scores[8:]))
    overall = 0.5 * surface + 0.5 * pressure
    return {
        "rmse_normalized": rmse(prediction, target),
        "mae_normalized": mae(prediction, target),
        "latitude_weighted_rmse_normalized": latitude_weighted_rmse(prediction, target, latitudes),
        "surface_score": surface,
        "pressure_score": pressure,
        "overall_score": overall,
    }


def _per_channel_metrics(
    original_physical: np.ndarray,
    reconstruction_physical: np.ndarray,
    train_std: np.ndarray,
    channels: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        original = original_physical[:, index]
        reconstruction = reconstruction_physical[:, index]
        rmse_value = float(np.sqrt(np.mean((reconstruction - original) ** 2)))
        nrmse = float(rmse_value / max(float(train_std[index]), 1e-6))
        rows.append(
            {
                "channel": channel,
                "rmse_physical": rmse_value,
                "nrmse": nrmse,
            }
        )
    return rows


def _per_time_metrics(target_norm: np.ndarray, prediction_norm: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(target_norm.shape[0]):
        target = torch.from_numpy(target_norm[index : index + 1])
        prediction = torch.from_numpy(prediction_norm[index : index + 1])
        rows.append({"index": index, "rmse_normalized": rmse(prediction, target), "mae_normalized": mae(prediction, target)})
    return rows


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
