"""Adapters and provenance helpers for the external CRA5 runtime."""

from .channel_mapping import (
    CRA5_CHANNEL_MAPPING,
    CRA5_PRESSURE_LEVELS,
    CRA5_PRESSURE_VARIABLES,
    CRA5_SURFACE_VARIABLES,
    Cra5ChannelMapping,
    validate_cra5_channel_mapping,
)
from .provenance import (
    CRA5_159_SHA256,
    CRA5_159_SIZE,
    CRA5_CHECKPOINT_FILENAME,
    CRA5_CHECKPOINT_URL,
    CRA5_UPSTREAM_COMMIT,
    CheckpointRecord,
    CheckpointValidationError,
    checkpoint_cache_path,
    fetch_checkpoint,
    validate_checkpoint_file,
)

__all__ = [
    "CRA5_159_SHA256",
    "CRA5_159_SIZE",
    "CRA5_CHANNEL_MAPPING",
    "CRA5_CHECKPOINT_FILENAME",
    "CRA5_CHECKPOINT_URL",
    "CRA5_PRESSURE_LEVELS",
    "CRA5_PRESSURE_VARIABLES",
    "CRA5_SURFACE_VARIABLES",
    "CRA5_UPSTREAM_COMMIT",
    "CheckpointRecord",
    "CheckpointValidationError",
    "Cra5ChannelMapping",
    "checkpoint_cache_path",
    "fetch_checkpoint",
    "validate_checkpoint_file",
    "validate_cra5_channel_mapping",
]
