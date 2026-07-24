"""
Формирует JSON-словари, соответствующие схемам из
docs/ARTIFACT_API_CONTRACT.md (разделы 6, 7, 9): summary, experiments,
reconstructions. Это единственное место, где PCA baseline "разговаривает"
на языке контракта артефактов backend'а — evaluate_pca_baseline.py не
собирает JSON вручную, а вызывает функции отсюда.

Также содержит канонический список допустимых имён каналов и явную
проверку запрещённых обозначений (mslp, tp6h) — раздел 4 контракта.
В частности, это гарантирует требование PCA_BASELINE.md
"tp1h не является tp6h" на уровне кода, а не только документации.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

# Раздел 4 ARTIFACT_API_CONTRACT.md — канонические названия каналов и их units.
CANONICAL_CHANNELS: dict[str, str] = {
    "u10": "m s**-1",
    "v10": "m s**-1",
    "t2m": "K",
    "msl": "Pa",
    "sst": "K",
    "tcc": "0-1",
    "tcwv": "kg m**-2",
    "tp1h": "m",
}

# Раздел 4 контракта: явно запрещённые обозначения.
FORBIDDEN_CHANNEL_NAMES: frozenset[str] = frozenset({"mslp", "tp6h"})


def validate_channel_name(channel: str, units: str | None = None) -> None:
    """Бросает ValueError, если имя канала запрещено или не каноническое,
    либо если переданные units не совпадают с требуемыми контрактом.
    """
    if channel in FORBIDDEN_CHANNEL_NAMES:
        raise ValueError(
            f"Имя канала '{channel}' запрещено контрактом артефактов "
            f"(ARTIFACT_API_CONTRACT.md, раздел 4). Например, tp1h нельзя "
            f"агрегировать/переименовывать в tp6h."
        )
    if channel not in CANONICAL_CHANNELS:
        raise ValueError(
            f"Имя канала '{channel}' не входит в канонический список "
            f"{sorted(CANONICAL_CHANNELS)} (ARTIFACT_API_CONTRACT.md, раздел 4)"
        )
    if units is not None and units != CANONICAL_CHANNELS[channel]:
        raise ValueError(
            f"Единицы измерения '{units}' не соответствуют каналу '{channel}': "
            f"ожидалось '{CANONICAL_CHANNELS[channel]}' (раздел 5 контракта)"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_summary_artifact(
    experiment_count: int, completed: int, failed: int, is_demo: bool
) -> dict[str, Any]:
    """summary.json — раздел 6 контракта."""
    return {
        "is_demo": bool(is_demo),
        "experiment_count": int(experiment_count),
        "completed": int(completed),
        "failed": int(failed),
        "generated_at": _now_iso(),
    }


def build_experiment_artifact(
    experiment_id: str,
    name: str,
    status: str,
    is_demo: bool,
    compression_type: str,
    payload_ratio: float,
    sample_count: int,
    selection_strategy: str,
    metrics: dict[str, float],
    model: str = "patch_pca",
) -> dict[str, Any]:
    """experiments.json — раздел 7 контракта. `model` фиксировано как
    "patch_pca", как того требует п.17 задачи ("model = patch_pca")."""
    return {
        "id": experiment_id,
        "name": name,
        "model": model,
        "status": status,
        "is_demo": bool(is_demo),
        "compression": {
            "type": compression_type,
            "ratio": round(float(payload_ratio), 4),
        },
        "training": {
            "sample_count": int(sample_count),
            "selection_strategy": selection_strategy,
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
    }


def build_reconstruction_artifact(
    experiment_id: str,
    channel: str,
    timestamp: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    absolute_error: np.ndarray,
    is_demo: bool,
) -> dict[str, Any]:
    """Файл реконструкции для одного канала/timestamp — раздел 9 контракта.
    Проверяет соответствие размеров массивов и координатной сетки (раздел
    10 контракта: "соответствие количества широт числу строк массива" и т.д.).
    """
    units = CANONICAL_CHANNELS.get(channel)
    validate_channel_name(channel, units)

    original = np.asarray(original)
    reconstruction = np.asarray(reconstruction)
    absolute_error = np.asarray(absolute_error)
    if not (original.shape == reconstruction.shape == absolute_error.shape):
        raise ValueError(
            "original/reconstruction/absolute_error должны иметь одинаковый размер: "
            f"{original.shape}, {reconstruction.shape}, {absolute_error.shape}"
        )
    if original.shape[0] != len(latitude):
        raise ValueError(
            f"Число строк массива ({original.shape[0]}) не совпадает с числом latitude ({len(latitude)})"
        )
    if original.shape[1] != len(longitude):
        raise ValueError(
            f"Число столбцов массива ({original.shape[1]}) не совпадает с числом longitude ({len(longitude)})"
        )

    return {
        "experiment_id": experiment_id,
        "channel": channel,
        "units": units,
        "timestamp": timestamp,
        "latitude": np.asarray(latitude).tolist(),
        "longitude": np.asarray(longitude).tolist(),
        "original": original.tolist(),
        "reconstruction": reconstruction.tolist(),
        "absolute_error": absolute_error.tolist(),
        "is_demo": bool(is_demo),
    }