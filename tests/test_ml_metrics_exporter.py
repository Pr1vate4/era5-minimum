from __future__ import annotations

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.monitoring.ml_artifacts import (
    MlArtifactContractError,
    load_ml_artifact_snapshot,
)
from era5_minimum.monitoring.ml_exporter import MlArtifactCollector


def _write_valid_bundle(
    root: Path,
    *,
    channel_order: list[str] | None = None,
    metric_space: str = "physical",
    latitude_weighting: bool = True,
    train_only: bool = True,
    exact_roundtrip: object = True,
) -> Path:
    root.mkdir(parents=True)
    channels = channel_order or list(CHANNEL_NAMES)
    per_channel = [
        {
            "channel": channel,
            "index": index,
            "rmse_physical": float(index + 1),
            "nrmse": 0.1 + index / 1000,
            "psnr_db": 30.0 + index / 10,
        }
        for index, channel in enumerate(channels)
    ]
    metrics = {
        "experiment": {"name": "conv_ae_weatherbench2_05_n32", "seed": 42},
        "data": {"train_timestamps": 32, "validation_timestamps": 16},
        "training": {
            "steps": 2560,
            "trainable_params": 163476,
            "runtime_seconds": 304.1,
            "device": "cpu",
        },
        "metrics": {
            "metric_space": metric_space,
            "latitude_weighting": latitude_weighting,
            "train_only": train_only,
            "channel_order": channels,
            "per_channel": per_channel,
            "surface_score": 0.19,
            "pressure_score": 0.18,
            "overall_score": 0.185,
            "mean_finite_psnr_db": 33.5,
            "physical_diagnostics": {
                "mslp_rmse_hpa": 1.14,
                "tp6h_rmse_mm_per_6h": 0.83,
                "wind_speed_rmse_m_s": 0.99,
            },
        },
    }
    report = {
        "status": "test_checkpoint_codec_evaluation",
        "split": "test",
        "frame_count": 16,
        "tensor_element_ratio": 32.0,
        "exact_quantized_symbol_roundtrip_all_frames": exact_roundtrip,
        "compression": {
            "total_bitstream_bytes": 3423714,
            "bits_per_value": 0.235,
            "compression_ratio": 135.6,
        },
        "timing_seconds": {
            "encode_total": 5.0,
            "decode_total": 8.8,
            "encode_per_frame": 0.31,
            "decode_per_frame": 0.55,
        },
        "metrics": metrics["metrics"],
    }
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def test_load_snapshot_keeps_serialized_and_tensor_ratios_separate(
    tmp_path: Path,
) -> None:
    snapshot = load_ml_artifact_snapshot(_write_valid_bundle(tmp_path / "model"))

    assert snapshot.labels == {"grid": "0.5", "target_cr": "32", "split": "test"}
    assert snapshot.values["actual_compression_ratio"] == 135.6
    assert snapshot.values["tensor_compression_ratio"] == 32.0
    assert snapshot.values["bitstream_bytes"] == 3423714
    assert snapshot.values["exact_roundtrip"] == 1.0
    assert snapshot.values["unique_train_timestamps"] == 32
    assert snapshot.values["unique_validation_timestamps"] == 16
    assert len(snapshot.channel_nrmse) == 28


def test_load_snapshot_rejects_noncanonical_channels(tmp_path: Path) -> None:
    root = _write_valid_bundle(
        tmp_path / "model", channel_order=list(reversed(CHANNEL_NAMES))
    )

    with pytest.raises(MlArtifactContractError, match="channel order"):
        load_ml_artifact_snapshot(root)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"metric_space": "normalized"}, "physical"),
        ({"latitude_weighting": False}, "latitude"),
        ({"train_only": False}, "train_only"),
    ],
)
def test_load_snapshot_rejects_invalid_scientific_contract(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    root = _write_valid_bundle(tmp_path / "model", **override)

    with pytest.raises(MlArtifactContractError, match=message):
        load_ml_artifact_snapshot(root)


def test_load_snapshot_rejects_non_boolean_roundtrip(tmp_path: Path) -> None:
    root = _write_valid_bundle(tmp_path / "model", exact_roundtrip=1)

    with pytest.raises(MlArtifactContractError, match="roundtrip"):
        load_ml_artifact_snapshot(root)


def test_optional_resource_metrics_are_omitted_not_zero_filled(
    tmp_path: Path,
) -> None:
    snapshot = load_ml_artifact_snapshot(_write_valid_bundle(tmp_path / "model"))

    assert "peak_vram_bytes" not in snapshot.values
    assert "gpu_hours" not in snapshot.values


def test_collector_exposes_real_codec_metrics(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(_write_valid_bundle(tmp_path / "model")))

    exposition = generate_latest(registry).decode()

    assert "era5_codec_artifact_ready 1.0" in exposition
    assert "era5_codec_actual_compression_ratio" in exposition
    assert "era5_codec_tensor_compression_ratio" in exposition
    assert 'era5_codec_channel_nrmse{channel="t2m"}' in exposition
    assert 'era5_codec_channel_psnr_db{channel="t2m"}' in exposition


def test_invalid_bundle_exposes_readiness_only(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(tmp_path / "missing"))

    exposition = generate_latest(registry).decode()

    assert "era5_codec_artifact_ready 0.0" in exposition
    assert "era5_codec_artifact_validation_failures_total 1.0" in exposition
    assert "era5_codec_actual_compression_ratio" not in exposition


def test_collector_does_not_zero_fill_missing_resource_metrics(
    tmp_path: Path,
) -> None:
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(_write_valid_bundle(tmp_path / "model")))

    exposition = generate_latest(registry).decode()

    assert "era5_codec_peak_vram_bytes" not in exposition
    assert "era5_codec_gpu_hours" not in exposition
