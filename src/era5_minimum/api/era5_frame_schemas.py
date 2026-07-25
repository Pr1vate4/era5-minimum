"""Pydantic response models for timestamp-driven ERA5 frame preparation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Era5ErrorResponse(BaseModel):
    error: str
    message: str
    requested: str | None = None
    nearest_before: str | None = None
    nearest_after: str | None = None


class Era5TimestampList(BaseModel):
    timestamps: list[datetime]
    count: int
    total: int
    offset: int
    limit: int


class Era5ChannelMetadata(BaseModel):
    index: int
    name: str
    source_variable: str
    level: int | None
    unit: str


class Era5LatitudeMetadata(BaseModel):
    size: int
    order: Literal["south_to_north"]
    minimum: float
    maximum: float


class Era5LongitudeMetadata(BaseModel):
    size: int
    order: Literal["west_to_east"]
    range: tuple[float, float]
    periodic: bool


class Era5FrameValidationMetadata(BaseModel):
    valid: bool
    missing_values: int
    sst_masked_values: int
    sst_ocean_missing_values: int
    invalid_values: int


class Era5FrameMetadata(BaseModel):
    timestamp: datetime
    shape: tuple[int, int, int, int]
    dtype: Literal["float32"]
    dataset_id: str
    dataset_source: str
    dataset_split: str
    schema_version: str
    channels: list[Era5ChannelMetadata]
    latitude: Era5LatitudeMetadata
    longitude: Era5LongitudeMetadata
    validation: Era5FrameValidationMetadata


class CodecJobMetrics(BaseModel):
    serialized_compression_ratio: float
    tensor_compression_ratio: float
    bitstream_bytes: int
    exact_roundtrip: bool
    encode_seconds: float
    decode_seconds: float


class CodecJobDownloads(BaseModel):
    bitstream: str
    reconstruction: str


class CodecJobPreviews(BaseModel):
    original: str
    reconstruction: str


class CodecJobSource(BaseModel):
    type: Literal["era5"]
    timestamp: datetime
    dataset_id: str


class CodecJobResponse(BaseModel):
    id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    error: str | None
    metrics: CodecJobMetrics
    downloads: CodecJobDownloads
    previews: CodecJobPreviews
    source: CodecJobSource | None = None
