from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ProbeConfig:
    latent_dim: int
    hidden_dim: int
    batch_size: int
    learning_rate: float
    max_steps: int
    parameter_limit: int = 2_000_000
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "max_steps": self.max_steps,
            "parameter_limit": self.parameter_limit,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class LatentForecastPairs:
    inputs: np.ndarray
    targets: np.ndarray
    pair_indices: np.ndarray


@dataclass(frozen=True)
class ProbeResult:
    checkpoint_path: Path
    metrics_path: Path
    parameter_count: int
    optimizer_steps: int
    train_pair_count: int
    validation_pair_count: int
    train_latent_mse: float
    validation_latent_mse: float
    persistence_latent_mse: float
    relative_improvement_vs_persistence_pct: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "parameter_count": self.parameter_count,
            "optimizer_steps": self.optimizer_steps,
            "train_pair_count": self.train_pair_count,
            "validation_pair_count": self.validation_pair_count,
            "train_latent_mse": self.train_latent_mse,
            "validation_latent_mse": self.validation_latent_mse,
            "persistence_latent_mse": self.persistence_latent_mse,
            "relative_improvement_vs_persistence_pct": self.relative_improvement_vs_persistence_pct,
            "metadata": self.metadata,
        }


class LatentProbeMLP(nn.Module):
    """Compact latent predictor for +6h forecasting."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


def build_latent_forecast_pairs(latents: np.ndarray) -> LatentForecastPairs:
    values = np.asarray(latents, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"latents must have shape [time, latent_dim], got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("latents must contain at least two time steps")
    indices = np.stack(
        [np.arange(values.shape[0] - 1, dtype=np.int32), np.arange(1, values.shape[0], dtype=np.int32)],
        axis=1,
    )
    return LatentForecastPairs(inputs=values[:-1], targets=values[1:], pair_indices=indices)


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def persistence_forecast(inputs: np.ndarray) -> np.ndarray:
    return np.asarray(inputs, dtype=np.float32).copy()


def train_latent_probe(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    config: ProbeConfig,
    output_dir: Path | None = None,
    train_pair_indices: np.ndarray | None = None,
    validation_pair_indices: np.ndarray | None = None,
) -> ProbeResult:
    train_x = np.asarray(train_inputs, dtype=np.float32)
    train_y = np.asarray(train_targets, dtype=np.float32)
    val_x = np.asarray(validation_inputs, dtype=np.float32)
    val_y = np.asarray(validation_targets, dtype=np.float32)

    _validate_pair_shapes(train_x, train_y, config.latent_dim)
    _validate_pair_shapes(val_x, val_y, config.latent_dim)

    _seed_everything(config.seed)
    model = LatentProbeMLP(config.latent_dim, config.hidden_dim)
    parameter_count = count_trainable_parameters(model)
    if parameter_count > config.parameter_limit:
        raise ValueError(
            f"parameter_limit exceeded: {parameter_count} trainable parameters > {config.parameter_limit}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    optimizer_steps = 0
    while optimizer_steps < config.max_steps:
        for batch_inputs, batch_targets in loader:
            if optimizer_steps >= config.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

    model.eval()
    with torch.no_grad():
        train_predictions = model(torch.from_numpy(train_x)).cpu().numpy()
        validation_predictions = model(torch.from_numpy(val_x)).cpu().numpy()

    train_latent_mse = float(np.mean((train_predictions - train_y) ** 2))
    validation_latent_mse = float(np.mean((validation_predictions - val_y) ** 2))
    persistence_validation = persistence_forecast(val_x)
    persistence_latent_mse = float(np.mean((persistence_validation - val_y) ** 2))
    relative_improvement = 100.0 * (
        1.0 - validation_latent_mse / max(persistence_latent_mse, 1e-12)
    )

    metrics = {
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "optimizer_steps": optimizer_steps,
        "train_pair_count": int(train_x.shape[0]),
        "validation_pair_count": int(val_x.shape[0]),
        "train_latent_mse": train_latent_mse,
        "validation_latent_mse": validation_latent_mse,
        "persistence_latent_mse": persistence_latent_mse,
        "relative_improvement_vs_persistence_pct": relative_improvement,
        "train_pair_indices": _indices_to_list(train_pair_indices, train_x.shape[0]),
        "validation_pair_indices": _indices_to_list(validation_pair_indices, val_x.shape[0]),
    }

    checkpoint_path = Path("probe.pt")
    metrics_path = Path("probe_metrics.json")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "probe.pt"
        metrics_path = output_dir / "probe_metrics.json"
        torch.save({"state_dict": model.state_dict(), "config": config.to_dict(), "metrics": metrics}, checkpoint_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return ProbeResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        parameter_count=parameter_count,
        optimizer_steps=optimizer_steps,
        train_pair_count=int(train_x.shape[0]),
        validation_pair_count=int(val_x.shape[0]),
        train_latent_mse=train_latent_mse,
        validation_latent_mse=validation_latent_mse,
        persistence_latent_mse=persistence_latent_mse,
        relative_improvement_vs_persistence_pct=relative_improvement,
        metadata=metrics,
    )


def _indices_to_list(indices: np.ndarray | None, length: int) -> list[list[int]]:
    if indices is None:
        return [[int(i), int(i + 1)] for i in range(length)]
    values = np.asarray(indices, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("pair indices must have shape [n_pairs, 2]")
    return values.tolist()


def _validate_pair_shapes(inputs: np.ndarray, targets: np.ndarray, latent_dim: int) -> None:
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("latent pairs must have shape [n_pairs, latent_dim]")
    if inputs.shape != targets.shape:
        raise ValueError(f"inputs.shape {inputs.shape} != targets.shape {targets.shape}")
    if inputs.shape[1] != latent_dim:
        raise ValueError(f"latent_dim mismatch: got {inputs.shape[1]}, expected {latent_dim}")
    if inputs.shape[0] < 1:
        raise ValueError("latent probe needs at least one pair")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
