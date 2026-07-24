from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from era5_minimum.latent_probe import (
    ProbeConfig,
    build_latent_forecast_pairs,
    train_latent_probe,
)


def test_build_latent_forecast_pairs_uses_consecutive_steps() -> None:
    latents = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )

    pairs = build_latent_forecast_pairs(latents)

    np.testing.assert_array_equal(pairs.inputs, latents[:-1])
    np.testing.assert_array_equal(pairs.targets, latents[1:])
    np.testing.assert_array_equal(pairs.pair_indices, np.array([[0, 1], [1, 2]], dtype=np.int32))


def test_train_latent_probe_enforces_parameter_limit_before_training() -> None:
    inputs = np.ones((8, 4), dtype=np.float32)
    targets = np.ones((8, 4), dtype=np.float32) * 2.0
    config = ProbeConfig(
        latent_dim=4,
        hidden_dim=16,
        batch_size=4,
        learning_rate=1e-2,
        max_steps=8,
        parameter_limit=8,
        seed=7,
    )

    with pytest.raises(ValueError, match="parameter_limit"):
        train_latent_probe(
            train_inputs=inputs,
            train_targets=targets,
            validation_inputs=inputs,
            validation_targets=targets,
            config=config,
        )


def test_train_latent_probe_writes_metrics_and_beats_persistence(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    train_inputs = rng.uniform(0.0, 1.0, size=(64, 4)).astype(np.float32)
    train_targets = (train_inputs * 1.5 + 0.25).astype(np.float32)
    validation_inputs = rng.uniform(0.0, 1.0, size=(16, 4)).astype(np.float32)
    validation_targets = (validation_inputs * 1.5 + 0.25).astype(np.float32)
    config = ProbeConfig(
        latent_dim=4,
        hidden_dim=32,
        batch_size=8,
        learning_rate=5e-2,
        max_steps=120,
        parameter_limit=2_000_000,
        seed=7,
    )

    result = train_latent_probe(
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        config=config,
        output_dir=tmp_path,
    )

    assert result.parameter_count <= config.parameter_limit
    assert result.optimizer_steps == config.max_steps
    assert result.validation_latent_mse < result.persistence_latent_mse
    assert result.relative_improvement_vs_persistence_pct > 0.0

    payload = json.loads((tmp_path / "probe_metrics.json").read_text(encoding="utf-8"))
    assert payload["optimizer_steps"] == config.max_steps
    assert payload["validation_latent_mse"] == pytest.approx(result.validation_latent_mse)
    assert payload["relative_improvement_vs_persistence_pct"] == pytest.approx(
        result.relative_improvement_vs_persistence_pct
    )
