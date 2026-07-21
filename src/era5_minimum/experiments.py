from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

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
            prediction_norm_np = model.reconstruct(validation_dataset.tensor.numpy())
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
