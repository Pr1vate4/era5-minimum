"""Adapters and provenance helpers for the external CRA5 runtime."""

from .adapter import (
    AdaptedCheckpointMetadata,
    adapt_cra5_checkpoint,
)
from .bridge import (
    BRIDGE_PROTOCOL_VERSION,
    Cra5BridgeError,
    Cra5BridgeRequest,
    Cra5BridgeResponse,
    run_cra5_bridge,
)
from .channel_mapping import (
    CRA5_CHANNEL_MAPPING,
    CRA5_PRESSURE_LEVELS,
    CRA5_PRESSURE_VARIABLES,
    CRA5_SURFACE_VARIABLES,
    Cra5ChannelMapping,
    validate_cra5_channel_mapping,
)
from .codec_workflow import (
    Cra5CodecMetadata,
    decode_cra5,
    encode_and_serialize_cra5,
    encode_cra5,
    serialize_cra5_bitstream,
)
from .model import (
    Cra5Vaeformer28,
    build_cra5_model,
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
    "AdaptedCheckpointMetadata",
    "BRIDGE_PROTOCOL_VERSION",
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
    "Cra5BridgeError",
    "Cra5BridgeRequest",
    "Cra5BridgeResponse",
    "Cra5CodecMetadata",
    "Cra5Vaeformer28",
    "Cra5ChannelMapping",
    "adapt_cra5_checkpoint",
    "build_cra5_model",
    "checkpoint_cache_path",
    "decode_cra5",
    "encode_and_serialize_cra5",
    "encode_cra5",
    "fetch_checkpoint",
    "serialize_cra5_bitstream",
    "validate_checkpoint_file",
    "validate_cra5_channel_mapping",
    "run_cra5_bridge",
]
