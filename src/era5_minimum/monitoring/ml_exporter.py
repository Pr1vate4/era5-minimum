"""Expose validated ERA5 model artifacts as bounded Prometheus metrics."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

from era5_minimum.monitoring.ml_artifacts import (
    MlArtifactContractError,
    MlArtifactSnapshot,
    load_ml_artifact_snapshot,
)

LOGGER = logging.getLogger(__name__)

_METRIC_DESCRIPTIONS = {
    "actual_compression_ratio": (
        "Measured raw tensor bytes divided by serialized bitstream bytes."
    ),
    "tensor_compression_ratio": (
        "Latent tensor element ratio, kept separate from serialized compression."
    ),
    "bitstream_bytes": "Measured serialized bitstream size in bytes.",
    "bits_per_value": "Measured serialized bits per input tensor value.",
    "exact_roundtrip": "Whether every tested quantized symbol round-tripped exactly.",
    "frame_count": "Number of frames in the evaluated split.",
    "overall_score": "Latitude-weighted overall NRMSE in physical space.",
    "surface_score": "Latitude-weighted surface-variable NRMSE in physical space.",
    "pressure_score": "Latitude-weighted pressure-level NRMSE in physical space.",
    "mean_psnr_db": "Mean finite per-channel PSNR in decibels.",
    "encode_duration_seconds": "Total measured encoding duration in seconds.",
    "decode_duration_seconds": "Total measured decoding duration in seconds.",
    "encode_per_frame_seconds": "Measured encoding duration per frame in seconds.",
    "decode_per_frame_seconds": "Measured decoding duration per frame in seconds.",
    "unique_train_timestamps": "Unique timestamps used for training.",
    "unique_validation_timestamps": "Unique timestamps used for validation.",
    "unique_test_timestamps": "Unique timestamps evaluated in the test split.",
    "optimizer_steps": "Completed optimizer steps.",
    "trainable_parameters": "Number of trainable model parameters.",
    "training_duration_seconds": "Measured training duration in seconds.",
    "mslp_rmse_hpa": "Latitude-weighted mean sea-level pressure RMSE in hPa.",
    "tp6h_rmse_mm_per_6h": (
        "Latitude-weighted six-hour precipitation RMSE in mm per 6 hours."
    ),
    "wind_speed_rmse_m_per_s": (
        "Latitude-weighted wind-speed RMSE in metres per second."
    ),
    "peak_vram_bytes": "Measured peak accelerator memory in bytes.",
    "gpu_hours": "Measured accelerator runtime in GPU-hours.",
}


class MlArtifactCollector:
    """Prometheus collector that validates the artifact bundle on every scrape."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._validation_failures = 0
        self._last_invalid_key: tuple[object, ...] | None = None

    def collect(self) -> Iterator[Metric]:
        """Yield readiness and scientific metrics only for a valid snapshot."""
        try:
            snapshot = load_ml_artifact_snapshot(self._root)
        except MlArtifactContractError:
            self._record_validation_failure()
            yield _readiness_metric(0.0)
            yield self._failure_metric()
            return

        with self._lock:
            self._last_invalid_key = None
        yield _readiness_metric(1.0)
        yield self._failure_metric()
        yield from _snapshot_metrics(snapshot)

    def _record_validation_failure(self) -> None:
        invalid_key = _artifact_change_key(self._root)
        with self._lock:
            if invalid_key != self._last_invalid_key:
                self._validation_failures += 1
                self._last_invalid_key = invalid_key

    def _failure_metric(self) -> Metric:
        with self._lock:
            value = float(self._validation_failures)
        metric = CounterMetricFamily(
            "era5_codec_artifact_validation_failures_total",
            "Distinct invalid artifact bundle versions observed by the exporter.",
        )
        metric.add_metric([], value)
        return metric


def _readiness_metric(value: float) -> Metric:
    metric = GaugeMetricFamily(
        "era5_codec_artifact_ready",
        "Whether the selected ML artifact bundle satisfies the scientific contract.",
    )
    metric.add_metric([], value)
    return metric


def _snapshot_metrics(snapshot: MlArtifactSnapshot) -> Iterator[Metric]:
    label_names = ["grid", "target_cr", "split"]
    label_values = [snapshot.labels[name] for name in label_names]
    for name, value in snapshot.values.items():
        metric = GaugeMetricFamily(
            f"era5_codec_{name}",
            _METRIC_DESCRIPTIONS[name],
            labels=label_names,
        )
        metric.add_metric(label_values, value)
        yield metric

    modified = GaugeMetricFamily(
        "era5_codec_artifact_last_modified_timestamp_seconds",
        "Latest modification time of the validated metrics and report files.",
        labels=label_names,
    )
    modified.add_metric(label_values, snapshot.last_modified_timestamp)
    yield modified

    nrmse = GaugeMetricFamily(
        "era5_codec_channel_nrmse",
        "Latitude-weighted per-channel NRMSE in physical space.",
        labels=["channel"],
    )
    psnr = GaugeMetricFamily(
        "era5_codec_channel_psnr_db",
        "Latitude-weighted per-channel PSNR in decibels.",
        labels=["channel"],
    )
    for channel, value in snapshot.channel_nrmse.items():
        nrmse.add_metric([channel], value)
        psnr.add_metric([channel], snapshot.channel_psnr_db[channel])
    yield nrmse
    yield psnr


def _artifact_change_key(root: Path) -> tuple[object, ...]:
    key: list[object] = [str(root)]
    for name in ("metrics.json", "report.json"):
        try:
            stat = (root / name).stat()
        except OSError:
            key.extend((name, None, None))
        else:
            key.extend((name, stat.st_mtime_ns, stat.st_size))
    return tuple(key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the selected bundle once and exit",
    )
    return parser


def main() -> int:
    """Run the exporter server or validate its selected artifact once."""
    args = _build_parser().parse_args()
    root = Path(
        os.environ.get("ERA5_ML_METRICS_ROOT", "/workspace/artifacts/model-n32")
    )
    if args.check:
        try:
            snapshot = load_ml_artifact_snapshot(root)
        except MlArtifactContractError as exc:
            print(f"valid=0 error={exc}")
            return 1
        print(
            "valid=1 "
            f"channels={len(snapshot.channel_nrmse)} "
            f"grid={snapshot.labels['grid']} "
            f"target_cr={snapshot.labels['target_cr']} "
            f"split={snapshot.labels['split']}"
        )
        return 0

    host = os.environ.get("ERA5_ML_EXPORTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ERA5_ML_EXPORTER_PORT", "9101"))
    registry = CollectorRegistry()
    registry.register(MlArtifactCollector(root))
    start_http_server(port, addr=host, registry=registry)
    LOGGER.info("ERA5 ML metrics exporter listening on %s:%d for %s", host, port, root)
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
