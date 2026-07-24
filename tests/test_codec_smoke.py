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


def test_codec_smoke_masks_nan_values_after_normalization(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_nan"
    config = {
        "seed": 17,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 8,
            "width": 8,
            "validation_samples": 4,
            "test_samples": 4,
            "inject_nan_fraction": 0.05,
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

    metrics_validation = json.loads((output_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    reconstruction = np.load(output_dir / "reconstruction_samples.npz")

    assert summary["invalid_value_count"] > 0
    assert metrics_validation["invalid_value_count"] == summary["invalid_value_count"]
    assert not np.isnan(reconstruction["validation_reconstruction"]).any()
    assert not np.isnan(reconstruction["test_reconstruction"]).any()


def test_codec_smoke_writes_tiled_inference_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_tiled"
    config = {
        "seed": 19,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 16,
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
            "grid": "smoke-16x16",
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
        "inference": {
            "mode": "tiled",
            "tile_height": 8,
            "tile_width": 8,
            "halo": 8,
            "boundary_width": 1,
        },
    }

    summary = run_codec_smoke(config)

    tile_report = json.loads((output_dir / "tile_inference.json").read_text(encoding="utf-8"))

    assert summary["tile_inference_enabled"] is True
    assert summary["reconstruction_inference_mode"] == "tiled"
    assert summary["tile_fullframe_rmse_normalized"] >= 0.0
    assert summary["tile_seam_rmse_normalized"] >= 0.0
    assert tile_report["mode"] == "tiled"
    assert tile_report["tile_height"] == 8
    assert tile_report["tile_width"] == 8
    assert tile_report["halo"] == 8
    assert tile_report["reference"] == "full_frame_quantized_latent_decode"
    assert tile_report["validation_fullframe_rmse_normalized"] >= 0.0
    assert tile_report["validation_seam_rmse_normalized"] >= 0.0
    assert tile_report["validation_internal_seam_rmse_normalized"] >= 0.0
    assert tile_report["validation_longitude_wrap_rmse_normalized"] >= 0.0

    reconstruction = np.load(output_dir / "reconstruction_samples.npz")
    assert reconstruction["inference_mode"].item() == "tiled"
