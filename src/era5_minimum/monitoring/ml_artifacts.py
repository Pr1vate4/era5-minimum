"""Validate confirmed ML artifacts before exporting bounded Prometheus metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from era5_minimum.data.channel_spec import CHANNEL_NAMES


class MlArtifactContractError(ValueError):
    """Raised when an artifact bundle cannot support honest ML metrics."""


@dataclass(frozen=True)
class MlArtifactSnapshot:
    """Validated scalar and per-channel values from one model artifact bundle."""

    labels: dict[str, str]
    values: dict[str, float]
    channel_nrmse: dict[str, float]
    channel_psnr_db: dict[str, float]
    last_modified_timestamp: float


def load_ml_artifact_snapshot(root: Path) -> MlArtifactSnapshot:
    """Load one strict metrics/report pair without deriving missing evidence."""
    metrics_path = root / "metrics.json"
    report_path = root / "report.json"
    metrics_document = _read_object(metrics_path)
    report = _read_object(report_path)
    metrics = _object(metrics_document.get("metrics"), "metrics")
    compression = _object(report.get("compression"), "report.compression")
    timing = _object(report.get("timing_seconds"), "report.timing_seconds")
    data = _object(metrics_document.get("data"), "data")
    training = _object(metrics_document.get("training"), "training")

    channel_order = metrics.get("channel_order")
    if not isinstance(channel_order, list) or tuple(channel_order) != CHANNEL_NAMES:
        raise MlArtifactContractError("metrics channel order is not canonical")
    if metrics.get("metric_space") != "physical":
        raise MlArtifactContractError("metrics must be reported in physical space")
    if metrics.get("latitude_weighting") is not True:
        raise MlArtifactContractError("metrics must use latitude weighting")
    if metrics.get("train_only") is not True:
        raise MlArtifactContractError("normalization metrics must be train_only")

    exact_roundtrip = report.get("exact_quantized_symbol_roundtrip_all_frames")
    if type(exact_roundtrip) is not bool:
        raise MlArtifactContractError("exact roundtrip must be a boolean")

    tensor_ratio = _positive_number(
        report.get("tensor_element_ratio"), "tensor_element_ratio"
    )
    actual_ratio = _positive_number(
        compression.get("compression_ratio"), "compression.compression_ratio"
    )
    bitstream_bytes = _positive_number(
        compression.get("total_bitstream_bytes"),
        "compression.total_bitstream_bytes",
    )
    frame_count = _positive_number(report.get("frame_count"), "frame_count")
    split = report.get("split")
    if not isinstance(split, str) or not split.strip():
        raise MlArtifactContractError("report split must be a non-empty string")

    per_channel = metrics.get("per_channel")
    if not isinstance(per_channel, list) or len(per_channel) != len(CHANNEL_NAMES):
        raise MlArtifactContractError("metrics.per_channel must contain 28 channels")
    channel_nrmse: dict[str, float] = {}
    channel_psnr_db: dict[str, float] = {}
    for index, expected_channel in enumerate(CHANNEL_NAMES):
        row = _object(per_channel[index], f"metrics.per_channel[{index}]")
        if row.get("channel") != expected_channel or row.get("index") != index:
            raise MlArtifactContractError(
                f"per-channel order mismatch at {expected_channel}"
            )
        channel_nrmse[expected_channel] = _finite_number(
            row.get("nrmse"), f"{expected_channel}.nrmse"
        )
        channel_psnr_db[expected_channel] = _finite_number(
            row.get("psnr_db"), f"{expected_channel}.psnr_db"
        )

    values = {
        "actual_compression_ratio": actual_ratio,
        "tensor_compression_ratio": tensor_ratio,
        "bitstream_bytes": bitstream_bytes,
        "bits_per_value": _positive_number(
            compression.get("bits_per_value"), "compression.bits_per_value"
        ),
        "exact_roundtrip": float(exact_roundtrip),
        "frame_count": frame_count,
        "overall_score": _finite_number(
            metrics.get("overall_score"), "metrics.overall_score"
        ),
        "surface_score": _finite_number(
            metrics.get("surface_score"), "metrics.surface_score"
        ),
        "pressure_score": _finite_number(
            metrics.get("pressure_score"), "metrics.pressure_score"
        ),
        "mean_psnr_db": _finite_number(
            metrics.get("mean_finite_psnr_db"), "metrics.mean_finite_psnr_db"
        ),
        "encode_duration_seconds": _nonnegative_number(
            timing.get("encode_total"), "timing_seconds.encode_total"
        ),
        "decode_duration_seconds": _nonnegative_number(
            timing.get("decode_total"), "timing_seconds.decode_total"
        ),
        "encode_per_frame_seconds": _nonnegative_number(
            timing.get("encode_per_frame"), "timing_seconds.encode_per_frame"
        ),
        "decode_per_frame_seconds": _nonnegative_number(
            timing.get("decode_per_frame"), "timing_seconds.decode_per_frame"
        ),
        "unique_train_timestamps": _positive_number(
            data.get("train_timestamps"), "data.train_timestamps"
        ),
        "unique_validation_timestamps": _positive_number(
            data.get("validation_timestamps"), "data.validation_timestamps"
        ),
        "unique_test_timestamps": frame_count,
        "optimizer_steps": _nonnegative_number(
            training.get("steps"), "training.steps"
        ),
        "trainable_parameters": _positive_number(
            training.get("trainable_params"), "training.trainable_params"
        ),
        "training_duration_seconds": _nonnegative_number(
            training.get("runtime_seconds"), "training.runtime_seconds"
        ),
    }
    diagnostics = _object(
        metrics.get("physical_diagnostics"), "metrics.physical_diagnostics"
    )
    _add_optional(
        values,
        "mslp_rmse_hpa",
        diagnostics.get("mslp_rmse_hpa"),
    )
    _add_optional(
        values,
        "tp6h_rmse_mm_per_6h",
        diagnostics.get("tp6h_rmse_mm_per_6h"),
    )
    _add_optional(
        values,
        "wind_speed_rmse_m_per_s",
        diagnostics.get("wind_speed_rmse_m_s"),
    )
    _add_optional(values, "peak_vram_bytes", training.get("peak_vram_bytes"))
    _add_optional(values, "gpu_hours", training.get("gpu_hours"))

    return MlArtifactSnapshot(
        labels={
            "grid": _infer_grid(metrics_document),
            "target_cr": f"{tensor_ratio:g}",
            "split": split.strip(),
        },
        values=values,
        channel_nrmse=channel_nrmse,
        channel_psnr_db=channel_psnr_db,
        last_modified_timestamp=max(
            metrics_path.stat().st_mtime, report_path.stat().st_mtime
        ),
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MlArtifactContractError(f"cannot read required artifact {path.name}") from exc
    return dict(_object(value, path.name))


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MlArtifactContractError(f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MlArtifactContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MlArtifactContractError(f"{label} must be a finite number")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0:
        raise MlArtifactContractError(f"{label} must be positive")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0:
        raise MlArtifactContractError(f"{label} must be nonnegative")
    return result


def _add_optional(values: dict[str, float], name: str, value: object) -> None:
    if value is None:
        return
    values[name] = _nonnegative_number(value, name)


def _infer_grid(document: Mapping[str, Any]) -> str:
    data = _object(document.get("data"), "data")
    experiment = _object(document.get("experiment"), "experiment")
    text = " ".join(
        str(value)
        for value in (
            data.get("grid"),
            data.get("dataset_dir"),
            experiment.get("name"),
        )
        if value is not None
    ).lower()
    if any(marker in text for marker in ("0.25", "0p25", "_025_")):
        return "0.25"
    if any(marker in text for marker in ("0.5", "0p5", "_05_")):
        return "0.5"
    raise MlArtifactContractError("cannot determine 0.25 or 0.5 grid")
