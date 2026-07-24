from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from era5_minimum.codec.workflow import run_codec_smoke


def test_codec_smoke_writes_full_artifact_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke"
    config = {
        "seed": 7,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 8,
            "width": 8,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "model": {
            "latent_channels": 8,
            "parameter_limit": 2_000_000,
        },
        "codec": {
            "version": "ml-001",
            "grid": "smoke-8x8",
            "quantization_step": 0.25,
            "target_compression_ratio": 32,
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 1e-3,
            "max_steps": 4,
        },
        "resources": {
            "max_vram_gb": 24,
            "max_gpu_hours": 48,
        },
    }

    summary = run_codec_smoke(config)

    assert summary["exact_roundtrip"] is True
    assert summary["actual_compression_ratio"] > 0
    assert summary["latent_reduction_ratio"] > 0

    expected = [
        "resolved_config.yaml",
        "checkpoint_metadata.json",
        "metrics_validation.json",
        "metrics_per_channel.json",
        "metrics_per_time.json",
        "resource_usage.json",
        "training_history.jsonl",
        "reconstruction_samples.npz",
        "entropy_statistics.json",
        "bitstream_statistics.json",
        "run_summary.json",
    ]
    for name in expected:
        assert (output_dir / name).exists()

    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["codec_eligible"] in {True, False}
    assert run_summary["exact_roundtrip"] is True
    assert run_summary["bitstream_bytes"] > 0

    resource_usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource_usage["operation"] == "train_codec"
    assert resource_usage["visible_gpu_count"] >= 0


def test_codec_smoke_preserves_non_multiple_of_eight_grid_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_odd_grid"
    config = {
        "seed": 11,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 9,
            "width": 16,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "model": {
            "latent_channels": 8,
            "parameter_limit": 2_000_000,
        },
        "codec": {
            "version": "ml-001",
            "grid": "smoke-9x16",
            "quantization_step": 0.25,
            "target_compression_ratio": 32,
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 1e-3,
            "max_steps": 4,
        },
        "resources": {
            "max_vram_gb": 24,
            "max_gpu_hours": 48,
        },
    }

    summary = run_codec_smoke(config)

    reconstruction = np.load(output_dir / "reconstruction_samples.npz")
    assert reconstruction["validation_reconstruction"].shape[-2:] == (9, 16)
    assert reconstruction["test_reconstruction"].shape[-2:] == (9, 16)
    assert summary["exact_roundtrip"] is True
