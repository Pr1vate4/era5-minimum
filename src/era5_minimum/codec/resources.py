from __future__ import annotations

import json
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _peak_ram_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def _peak_vram_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def build_resource_usage_record(
    *,
    run_id: str,
    operation: str,
    grid_resolution: str,
    dataset_size: int,
    started_at: str | None,
    finished_at: str | None,
    wall_clock_seconds: float,
    gpu_hours: float | None,
    gpu_name: str | None,
    visible_gpu_count: int,
    peak_vram_bytes: int | None,
    peak_ram_bytes: int | None,
    trainable_parameter_count: int | None,
    total_parameter_count: int | None,
    optimizer_steps: int | None,
    examples_seen: int | None,
    unique_train_timestamps: int | None,
    patches_seen: int | None,
    encode_seconds: float | None,
    decode_seconds: float | None,
    bitstream_bytes: int | None,
    parameter_limit: int | None = None,
    max_vram_bytes: int | None = None,
    max_gpu_hours: float | None = None,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    compliance = True
    if parameter_limit is not None and trainable_parameter_count is not None and trainable_parameter_count > parameter_limit:
        compliance = False
    if max_vram_bytes is not None and peak_vram_bytes is not None and peak_vram_bytes > max_vram_bytes:
        compliance = False
    if max_gpu_hours is not None and gpu_hours is not None and gpu_hours > max_gpu_hours:
        compliance = False
    if max_optimizer_steps is not None and optimizer_steps is not None and optimizer_steps > max_optimizer_steps:
        compliance = False

    return {
        "run_id": run_id,
        "operation": operation,
        "grid_resolution": grid_resolution,
        "dataset_size": int(dataset_size),
        "started_at": started_at or _utc_now(),
        "finished_at": finished_at or _utc_now(),
        "wall_clock_seconds": float(wall_clock_seconds),
        "gpu_hours": gpu_hours,
        "gpu_name": gpu_name,
        "visible_gpu_count": int(visible_gpu_count),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_ram_bytes": peak_ram_bytes,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "unique_train_timestamps": unique_train_timestamps,
        "patches_seen": patches_seen,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "bitstream_bytes": bitstream_bytes,
        "resource_compliance": compliance,
    }


def write_resource_usage(path: Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def measure_runtime_resources(
    *,
    run_id: str,
    operation: str,
    grid_resolution: str,
    dataset_size: int,
    started_at: str,
    finished_at: str,
    wall_clock_seconds: float,
    trainable_parameter_count: int | None,
    total_parameter_count: int | None,
    optimizer_steps: int | None,
    examples_seen: int | None,
    unique_train_timestamps: int | None,
    patches_seen: int | None,
    encode_seconds: float | None,
    decode_seconds: float | None,
    bitstream_bytes: int | None,
    parameter_limit: int | None = None,
    max_vram_gb: float | None = None,
    max_gpu_hours: float | None = None,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    peak_vram = _peak_vram_bytes()
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_hours = wall_clock_seconds / 3600.0 if torch.cuda.is_available() else None
    max_vram_bytes = int(max_vram_gb * 1024**3) if max_vram_gb is not None else None
    return build_resource_usage_record(
        run_id=run_id,
        operation=operation,
        grid_resolution=grid_resolution,
        dataset_size=dataset_size,
        started_at=started_at,
        finished_at=finished_at,
        wall_clock_seconds=wall_clock_seconds,
        gpu_hours=gpu_hours,
        gpu_name=gpu_name,
        visible_gpu_count=visible_gpu_count,
        peak_vram_bytes=peak_vram,
        peak_ram_bytes=_peak_ram_bytes(),
        trainable_parameter_count=trainable_parameter_count,
        total_parameter_count=total_parameter_count,
        optimizer_steps=optimizer_steps,
        examples_seen=examples_seen,
        unique_train_timestamps=unique_train_timestamps,
        patches_seen=patches_seen,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        bitstream_bytes=bitstream_bytes,
        parameter_limit=parameter_limit,
        max_vram_bytes=max_vram_bytes,
        max_gpu_hours=max_gpu_hours,
        max_optimizer_steps=max_optimizer_steps,
    )
