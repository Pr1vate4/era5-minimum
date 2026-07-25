from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from era5_minimum.codec import CodecConfig, CodecHarness, NormalizationSpec
from era5_minimum.data.synthetic import SyntheticERA5Dataset, make_synthetic_era5
from era5_minimum.metrics import latitude_weighted_rmse, mae, per_channel_rmse, rmse
from era5_minimum.models import ConvAutoencoder, PCAAutoencoder
from era5_minimum.training import resolve_device, seed_everything, train_autoencoder


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def inverse_transform(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_tensor = torch.from_numpy(mean.squeeze(0)).to(tensor.device).view(1, tensor.shape[1], 1, 1)
    std_tensor = torch.from_numpy(std.squeeze(0)).to(tensor.device).view(1, tensor.shape[1], 1, 1)
    return tensor * std_tensor + mean_tensor


def _common_row(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    channels: list[str],
    latitudes: np.ndarray,
) -> dict[str, Any]:
    # Aggregate metrics are meaningful in normalized space because physical
    # channels use incompatible units and scales. Per-channel values are
    # reported after inverse normalization in their original units.
    row: dict[str, Any] = {
        "rmse_normalized": rmse(prediction_norm, target_norm),
        "mae_normalized": mae(prediction_norm, target_norm),
        "latitude_weighted_rmse_normalized": latitude_weighted_rmse(
            prediction_norm, target_norm, latitudes
        ),
    }
    row.update(
        {
            f"rmse_{key}_physical": value
            for key, value in per_channel_rmse(
                prediction_physical, target_physical, channels
            ).items()
        }
    )
    return row


def _write_normalization_artifact(
    output_dir: Path,
    channels: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[Path, str]:
    payload = {
        "channel_order": channels,
        "mean": mean.reshape(-1).astype(np.float32).tolist(),
        "std": std.reshape(-1).astype(np.float32).tolist(),
        "train_only": True,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path = output_dir / "train_normalization.json"
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    if data_config.get("source") != "synthetic":
        raise NotImplementedError("Day-1 runner currently supports source=synthetic only")

    channels = list(data_config["channels"])
    raw, latitudes = make_synthetic_era5(
        samples=int(data_config["samples"]),
        height=int(data_config["height"]),
        width=int(data_config["width"]),
        channels=tuple(channels),
        seed=seed,
    )
    validation_samples = int(data_config["validation_samples"])
    train_raw = raw[:-validation_samples]
    validation_raw = raw[-validation_samples:]
    train_dataset = SyntheticERA5Dataset(train_raw)
    validation_dataset = SyntheticERA5Dataset(validation_raw, mean=train_dataset.mean, std=train_dataset.std)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    _, normalization_sha256 = _write_normalization_artifact(
        output_dir=output_dir,
        channels=channels,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    normalization_spec = NormalizationSpec(
        channel_order=tuple(channels),
        mean=train_dataset.mean.reshape(-1),
        std=train_dataset.std.reshape(-1),
        source_manifest_sha256=normalization_sha256,
        train_only=True,
    )

    rows: list[dict[str, Any]] = []
    model_type = str(model_config.get("type", "conv"))
    for train_size_value in training_config["train_sizes"]:
        train_size = int(train_size_value)
        if train_size > len(train_dataset):
            raise ValueError(f"train_size={train_size} exceeds available samples={len(train_dataset)}")
        seed_everything(seed + train_size)

        if model_type == "pca":
            model = PCAAutoencoder(latent_dim=int(model_config["latent_dim"]))
            fit_result = model.fit(train_dataset.tensor[:train_size].numpy())
            validation_norm_np = validation_dataset.tensor.numpy()
            prediction_norm_np = model.reconstruct(validation_norm_np)
            prediction_norm = torch.from_numpy(prediction_norm_np)
            target_norm = validation_dataset.tensor
            prediction = inverse_transform(prediction_norm, train_dataset.mean, train_dataset.std)
            target = inverse_transform(target_norm, train_dataset.mean, train_dataset.std)
            row: dict[str, Any] = {
                "model_type": "pca",
                "train_size": train_size,
                "tensor_compression_ratio": model.tensor_compression_ratio(),
                "retained_variance_fraction": fit_result.retained_variance_fraction,
                "runtime_seconds": fit_result.runtime_seconds,
            }
            codec_config = model_config.get("codec")
            if codec_config is not None:
                latent = model.encode(validation_norm_np)
                codec_result = CodecHarness(
                    config=CodecConfig(
                        version=str(codec_config.get("version", "ml-001")),
                        channel_order=tuple(channels),
                        grid=str(codec_config.get("grid", f"synthetic-{data_config['height']}x{data_config['width']}")),
                        quantization_step=float(codec_config["quantization_step"]),
                        seed=seed + train_size,
                        git_commit=None,
                    ),
                    normalization=normalization_spec,
                ).encode_latent(
                    input_tensor=validation_raw,
                    latent=latent,
                    output_dir=output_dir / f"pca_train_{train_size}_codec",
                )
                if codec_result.decoded_latent is None:
                    raise RuntimeError("codec result is missing decoded_latent")
                prediction_norm = torch.from_numpy(model.decode(codec_result.decoded_latent))
                prediction = inverse_transform(prediction_norm, train_dataset.mean, train_dataset.std)
                row.update(
                    {
                        "latent_reduction_ratio": codec_result.tensor_ratio,
                        "actual_compression_ratio": codec_result.serialized_ratio,
                        "bitstream_bytes": codec_result.metadata["compression"]["bitstream_bytes"],
                        "codec_roundtrip_exact": codec_result.roundtrip_ok,
                        "codec_metadata_path": str(codec_result.metadata_path),
                    }
                )
            row.update(_common_row(prediction_norm, target_norm, prediction, target, channels, latitudes))
            np.savez_compressed(
                output_dir / f"pca_train_{train_size}.npz",
                mean=model.mean_,
                components=model.components_,
                input_shape=np.asarray(model.input_shape_),
            )
        elif model_type == "conv":
            subset = Subset(train_dataset, list(range(train_size)))
            model = ConvAutoencoder(len(channels), int(model_config["latent_channels"]))
            result = train_autoencoder(
                model=model,
                dataset=subset,
                device=device,
                epochs=int(training_config["epochs"]),
                batch_size=int(training_config["batch_size"]),
                learning_rate=float(training_config["learning_rate"]),
                num_workers=int(training_config.get("num_workers", 0)),
            )
            model.eval()
            validation_tensor = validation_dataset.tensor.to(device)
            with torch.no_grad():
                prediction_norm = model(validation_tensor)
                prediction = inverse_transform(prediction_norm, train_dataset.mean, train_dataset.std)
                target = inverse_transform(validation_tensor, train_dataset.mean, train_dataset.std)
                ratio = model.tensor_compression_ratio(validation_tensor[:1])
            row = {
                "model_type": "conv",
                "train_size": train_size,
                "tensor_compression_ratio": ratio,
                "train_loss_normalized": result.train_loss,
                "runtime_seconds": result.runtime_seconds,
            }
            row.update(_common_row(prediction_norm, validation_tensor, prediction, target, channels, latitudes))
            torch.save({"state_dict": model.state_dict(), "channels": channels}, output_dir / f"conv_train_{train_size}.pt")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ERA5-Minimum sample-size experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
