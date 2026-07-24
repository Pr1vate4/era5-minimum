"""
Pydantic-схемы контракта артефактов ERA5-Minimum.

Схемы разбиты в точности так, как перечислено в задаче API-001:
    summary; experiment; training metadata; compression metadata;
    metrics; sample-efficiency data; reconstruction data;
    API error responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Канонические каналы и единицы измерения (контракт, разделы 4-5).
# ---------------------------------------------------------------------------
CANONICAL_CHANNEL_UNITS: dict[str, str] = {
    "u10": "m s**-1",
    "v10": "m s**-1",
    "t2m": "K",
    "msl": "Pa",
    "sst": "K",
    "tcc": "0–1",
    "tcwv": "kg m**-2",
    "tp1h": "m",
}
CANONICAL_CHANNELS: frozenset[str] = frozenset(CANONICAL_CHANNEL_UNITS)
FORBIDDEN_CHANNELS: frozenset[str] = frozenset({"mslp", "tp6h"})


# ---------------------------------------------------------------------------
# Summary (контракт, раздел 6).
# ---------------------------------------------------------------------------
class SummaryArtifact(BaseModel):
    is_demo: bool
    experiment_count: int
    completed: int
    failed: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Experiment (контракт, раздел 7): compression / training / metrics metadata.
# ---------------------------------------------------------------------------
class CompressionMetadata(BaseModel):
    type: str
    ratio: int


class TrainingMetadata(BaseModel):
    sample_count: int
    selection_strategy: str


class MetricsMetadata(BaseModel):
    normalized_rmse: float
    mae: float


class ExperimentArtifact(BaseModel):
    id: str
    name: str
    model: str
    status: str
    is_demo: bool
    compression: CompressionMetadata
    training: TrainingMetadata
    metrics: MetricsMetadata


# ---------------------------------------------------------------------------
# Sample efficiency (контракт, раздел 8).
# ---------------------------------------------------------------------------
class SampleEfficiencyPoint(BaseModel):
    sample_count: int
    value: float


class SampleEfficiencyArtifact(BaseModel):
    is_demo: bool
    metric: str
    points: List[SampleEfficiencyPoint]

    @model_validator(mode="after")
    def _points_not_empty(self) -> "SampleEfficiencyArtifact":
        if not self.points:
            raise ValueError("points must not be empty")
        return self


# ---------------------------------------------------------------------------
# Reconstruction (контракт, раздел 9). Основная часть проверок раздела 10.
# ---------------------------------------------------------------------------
def _matrix_shape(matrix: List[List[float]]) -> tuple[int, int]:
    rows = len(matrix)
    if rows == 0:
        return (0, 0)
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != cols:
            raise ValueError("matrix rows have inconsistent lengths")
    return (rows, cols)


class ReconstructionArtifact(BaseModel):
    experiment_id: str
    channel: str
    units: str
    timestamp: datetime
    latitude: List[float]
    longitude: List[float]
    original: List[List[float]]
    reconstruction: List[List[float]]
    absolute_error: List[List[float]] = Field(default_factory=list)
    is_demo: bool

    @model_validator(mode="after")
    def _channel_is_canonical(self) -> "ReconstructionArtifact":
        if self.channel in FORBIDDEN_CHANNELS:
            raise ValueError(
                f"channel '{self.channel}' is forbidden by the contract"
            )
        if self.channel not in CANONICAL_CHANNEL_UNITS:
            raise ValueError(
                f"channel '{self.channel}' is not a canonical channel"
            )
        return self

    @model_validator(mode="after")
    def _units_match_channel(self) -> "ReconstructionArtifact":
        expected = CANONICAL_CHANNEL_UNITS.get(self.channel)
        if expected is not None and self.units != expected:
            raise ValueError(
                f"units '{self.units}' do not match channel "
                f"'{self.channel}' (expected '{expected}')"
            )
        return self

    @model_validator(mode="after")
    def _matrix_shapes_consistent(self) -> "ReconstructionArtifact":
        original_shape = _matrix_shape(self.original)

        reconstruction_shape = _matrix_shape(self.reconstruction)
        if reconstruction_shape != original_shape:
            raise ValueError(
                "reconstruction shape does not match original "
                f"({reconstruction_shape} != {original_shape})"
            )

        if self.absolute_error:
            error_shape = _matrix_shape(self.absolute_error)
            if error_shape != original_shape:
                raise ValueError(
                    "absolute_error shape does not match original "
                    f"({error_shape} != {original_shape})"
                )

        rows, cols = original_shape
        if len(self.latitude) != rows:
            raise ValueError(
                "latitude length does not match matrix row count "
                f"({len(self.latitude)} != {rows})"
            )
        if len(self.longitude) != cols:
            raise ValueError(
                "longitude length does not match matrix column count "
                f"({len(self.longitude)} != {cols})"
            )
        return self

    @model_validator(mode="after")
    def _is_demo_required(self) -> "ReconstructionArtifact":
        if not self.is_demo:
            raise ValueError("is_demo must be true for demo artifacts")
        return self


# ---------------------------------------------------------------------------
# API error response schema.
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Единый формат ответа об ошибке для всех эндпоинтов API."""

    detail: str
    error_type: Optional[str] = None
