from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.data.synthetic import SyntheticERA5Dataset, make_synthetic_era5
from era5_minimum.latent_probe import ProbeConfig, build_latent_forecast_pairs, train_latent_probe
from era5_minimum.metrics import mae, rmse
from era5_minimum.models import PCAAutoencoder


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run(config_path: str | Path) -> dict[str, Any]:
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

    return metrics


def _build_probe_from_config(config: dict[str, Any]) -> torch.nn.Module:
    from era5_minimum.latent_probe import LatentProbeMLP

    return LatentProbeMLP(latent_dim=int(config["latent_dim"]), hidden_dim=int(config["hidden_dim"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a latent forecast probe")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
