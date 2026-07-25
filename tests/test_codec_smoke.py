from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from era5_minimum.codec import workflow
from era5_minimum.codec.workflow import (
    SMOKE_CHANNELS,
    build_smoke_tensor,
    run_codec_smoke,
)
from era5_minimum.models import ConvAutoencoder


def _smoke_config(output_dir: Path, *, latent_channels: int = 4) -> dict[str, object]:
    return {
        "seed": 31,
        "output_dir": str(output_dir),
        "data": {
            "samples": 12,
            "height": 8,
            "width": 8,
            "validation_samples": 2,
            "test_samples": 2,
        },
        "model": {
            "latent_channels": latent_channels,
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
            "epochs": 1,
            "learning_rate": 1e-3,
            "max_steps": 1,
        },
        "resources": {
            "max_vram_gb": 24,
            "max_gpu_hours": 48,
        },
    }


def test_codec_smoke_records_rate_distortion_training(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_rate_distortion"
    entropy_config = {
        "model_type": "factorized_logistic",
        "min_probability": 1e-9,
    }
    loss_config = {
        "type": "mse",
        "latitude_weighting": True,
        "surface_weight": 0.5,
        "pressure_weight": 0.5,
        "rate_lambda": 0.001,
    }
    config = {
        "seed": 23,
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
        "entropy": entropy_config,
        "loss": loss_config,
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

    result = run_codec_smoke(config)

    history = [
        json.loads(line)
        for line in (output_dir / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in history] == [1, 2, 3, 4]
    component_names = (
        "loss",
        "distortion",
        "surface_distortion",
        "pressure_distortion",
        "estimated_rate_bits",
        "estimated_rate_bits_per_input_value",
    )
    for row in history:
        assert all(math.isfinite(row[name]) for name in component_names)
        assert row["loss"] == pytest.approx(
            row["distortion"] + loss_config["rate_lambda"] * row["estimated_rate_bits_per_input_value"]
        )

    final = history[-1]
    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint_metadata = json.loads(
        (output_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(output_dir / "checkpoints" / "model.ckpt", weights_only=False)

    for payload in (run_summary, checkpoint_metadata):
        assert payload["loss_config"] == loss_config
        assert payload["entropy_model_config"] == entropy_config
        assert payload["training_loss"] == pytest.approx(final["loss"])
        assert payload["training_distortion"] == pytest.approx(final["distortion"])
        assert payload["training_surface_distortion"] == pytest.approx(final["surface_distortion"])
        assert payload["training_pressure_distortion"] == pytest.approx(final["pressure_distortion"])
        assert payload["training_estimated_rate_bits"] == pytest.approx(final["estimated_rate_bits"])
        assert payload["training_estimated_rate_bits_per_input_value"] == pytest.approx(
            final["estimated_rate_bits_per_input_value"]
        )
        assert payload["rate_lambda"] == loss_config["rate_lambda"]

    assert result["training_loss"] == pytest.approx(final["loss"])
    assert result["training_distortion"] == pytest.approx(final["distortion"])
    assert result["training_surface_distortion"] == pytest.approx(final["surface_distortion"])
    assert result["training_pressure_distortion"] == pytest.approx(final["pressure_distortion"])
    assert result["training_estimated_rate_bits_per_input_value"] == pytest.approx(
        final["estimated_rate_bits_per_input_value"]
    )
    assert result["rate_lambda"] == loss_config["rate_lambda"]
    assert checkpoint["entropy_model_config"] == entropy_config
    assert checkpoint["loss_config"] == loss_config
    assert set(checkpoint["entropy_model_state_dict"]) == {"log_scale"}


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
        "local_evaluation.json",
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
    assert run_summary["loss_config"]["rate_lambda"] > 0.0
    assert run_summary["rate_lambda"] == run_summary["loss_config"]["rate_lambda"]
    assert math.isfinite(run_summary["training_estimated_rate_bits_per_input_value"])

    resource_usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource_usage["operation"] == "train_codec"
    assert resource_usage["visible_gpu_count"] >= 0


def test_codec_smoke_writes_scientific_local_evaluation_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_local_evaluation"
    summary = run_codec_smoke(_smoke_config(output_dir))

    local_evaluation_path = output_dir / "local_evaluation.json"
    assert local_evaluation_path.exists()
    local_evaluation = json.loads(local_evaluation_path.read_text(encoding="utf-8"))
    assert local_evaluation["evaluator_version"].startswith("local-scientific-")
    assert local_evaluation["train_only"] is True
    assert local_evaluation["channel_order"] == list(SMOKE_CHANNELS)
    assert len(local_evaluation["per_channel"]) == 28
    assert all(
        {
            "rmse_physical",
            "nrmse",
            "psnr_db",
            "psnr_status",
        }
        <= row.keys()
        for row in local_evaluation["per_channel"]
    )
    assert local_evaluation["groups"]["surface"]["channel_count"] == 8
    assert local_evaluation["groups"]["pressure"]["channel_count"] == 20
    assert local_evaluation["overall_score"] == pytest.approx(
        0.5 * local_evaluation["surface_score"]
        + 0.5 * local_evaluation["pressure_score"]
    )
    assert local_evaluation["physical_diagnostics"]["mslp"]["rmse_hpa"] >= 0.0
    assert local_evaluation["physical_diagnostics"]["tp6h"]["rmse_mm_per_6h"] >= 0.0
    assert local_evaluation["physical_diagnostics"]["wind_speed"]["rmse_m_s"] >= 0.0
    assert local_evaluation["physical_diagnostics"]["geopotential"]["Z1000"][
        "rmse_height_m"
    ] >= 0.0

    train_statistics = local_evaluation["train_statistics"]
    assert train_statistics["train_only"] is True
    assert len(train_statistics["std"]) == 28
    assert len(train_statistics["ranges"]) == 28
    assert len(train_statistics["checksum"]) == 64
    assert local_evaluation["compression"]["actual_serialized_compression_ratio"] > 0.0
    assert local_evaluation["compression"]["tensor_compression_ratio"] > 0.0
    assert local_evaluation["exact_roundtrip"] is True
    assert local_evaluation["timings"]["encode_seconds"] >= 0.0
    assert local_evaluation["timings"]["decode_seconds"] >= 0.0
    assert local_evaluation["provenance"]["run_id"] == summary["run_id"]
    assert local_evaluation["provenance"]["seed"] == 31
    assert local_evaluation["provenance"]["git_commit"]

    legacy_validation = json.loads(
        (output_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    legacy_per_channel = json.loads(
        (output_dir / "metrics_per_channel.json").read_text(encoding="utf-8")
    )
    assert legacy_validation["overall_score"] == local_evaluation["overall_score"]
    assert legacy_per_channel == local_evaluation["per_channel"]

    checkpoint = torch.load(output_dir / "checkpoints" / "model.ckpt", weights_only=False)
    checkpoint_metadata = json.loads(
        (output_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert checkpoint["train_statistics"]["checksum"] == train_statistics["checksum"]
    assert checkpoint_metadata["train_statistics"]["checksum"] == train_statistics["checksum"]
    assert summary["local_evaluation_path"] == str(local_evaluation_path)


def test_codec_smoke_fits_sst_normalization_on_train_ocean_only(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "codec_smoke_ocean_normalization"
    config = _smoke_config(output_dir)

    run_codec_smoke(config)

    data_config = config["data"]
    assert isinstance(data_config, dict)
    raw, _, ocean_mask = build_smoke_tensor(
        samples=int(data_config["samples"]),
        height=int(data_config["height"]),
        width=int(data_config["width"]),
        seed=int(config["seed"]),
    )
    train_end = (
        int(data_config["samples"])
        - int(data_config["validation_samples"])
        - int(data_config["test_samples"])
    )
    train_raw = raw[:train_end]
    sst_index = SMOKE_CHANNELS.index("sst")
    ocean_values = train_raw[:, sst_index][:, ocean_mask > 0]
    expected_mean = np.mean(ocean_values, dtype=np.float64)
    expected_std = np.std(ocean_values, dtype=np.float64)
    all_grid_mean = np.mean(train_raw[:, sst_index], dtype=np.float64)
    all_grid_std = np.std(train_raw[:, sst_index], dtype=np.float64)

    checkpoint = torch.load(
        output_dir / "checkpoints" / "model.ckpt",
        weights_only=False,
    )
    normalization = checkpoint["normalization"]
    saved_mean = normalization["mean"][sst_index]
    saved_std = normalization["std"][sst_index]
    assert saved_mean == pytest.approx(expected_mean, rel=1e-6)
    assert saved_std == pytest.approx(expected_std, rel=1e-6)
    assert saved_mean != pytest.approx(all_grid_mean, rel=1e-6)
    assert saved_std != pytest.approx(all_grid_std, rel=1e-6)
    non_sst = np.arange(len(SMOKE_CHANNELS)) != sst_index
    expected_other_mean = np.nanmean(train_raw, axis=(0, 2, 3))
    expected_other_std = np.nanstd(train_raw, axis=(0, 2, 3))
    assert np.allclose(
        np.asarray(normalization["mean"])[non_sst],
        expected_other_mean[non_sst],
    )
    assert np.allclose(
        np.asarray(normalization["std"])[non_sst],
        expected_other_std[non_sst],
    )

    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, -1, 1, 1)
    normalized, validity_mask, _, _ = workflow._normalize_with_validity_mask(
        train_raw,
        mean=mean,
        std=std,
        ocean_mask=ocean_mask,
        sst_index=sst_index,
    )
    assert np.all(normalized[:, sst_index, ocean_mask == 0] == 0.0)
    expected_sst_mask = np.broadcast_to(
        ocean_mask > 0,
        validity_mask[:, sst_index].shape,
    )
    assert np.array_equal(validity_mask[:, sst_index] > 0, expected_sst_mask)
    assert np.all(validity_mask[:, :sst_index] == 1.0)
    assert np.all(validity_mask[:, sst_index + 1 :] == 1.0)


def test_codec_smoke_records_current_git_commit_in_all_artifacts(
    tmp_path: Path,
) -> None:
    resolver = getattr(workflow, "_resolve_git_commit", None)
    assert callable(resolver)
    git_commit = resolver()
    if git_commit is None:
        pytest.skip("git provenance is unavailable outside a git checkout")
    assert len(git_commit) == 40

    output_dir = tmp_path / "codec_smoke_git_provenance"
    result = run_codec_smoke(_smoke_config(output_dir))

    checkpoint = torch.load(
        output_dir / "checkpoints" / "model.ckpt",
        weights_only=False,
    )
    checkpoint_metadata = json.loads(
        (output_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    run_summary = json.loads(
        (output_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    bitstream_metadata = json.loads(
        (output_dir / "bitstreams" / "validation.json").read_text(encoding="utf-8")
    )

    assert checkpoint["git_commit"] == git_commit
    assert checkpoint_metadata["git_commit"] == git_commit
    assert run_summary["git_commit"] == git_commit
    assert result["git_commit"] == git_commit
    assert bitstream_metadata["config"]["git_commit"] == git_commit


def test_git_commit_resolver_returns_none_outside_git(tmp_path: Path) -> None:
    resolver = getattr(workflow, "_resolve_git_commit", None)
    assert callable(resolver)
    assert resolver(tmp_path) is None


def test_codec_parameter_limit_includes_entropy_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_smoke_parameter_limit"
    latent_channels = 4
    config = _smoke_config(output_dir, latent_channels=latent_channels)
    model = ConvAutoencoder(
        in_channels=len(SMOKE_CHANNELS),
        latent_channels=latent_channels,
    )
    model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_config = config["model"]
    assert isinstance(model_config, dict)
    model_config["parameter_limit"] = model_parameter_count

    with pytest.raises(ValueError, match="parameter_limit exceeded"):
        run_codec_smoke(config)

    history_path = output_dir / "training_history.jsonl"
    assert not history_path.exists() or history_path.read_text(encoding="utf-8") == ""


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
