"""Codec primitives for ERA5-Minimum."""

from .bitstream import BitstreamError, CanonicalHuffmanCoder
from .harness import CodecHarness
from .normalization import NormalizationSpec
from .resources import build_resource_usage_record, measure_runtime_resources, write_resource_usage
from .tiling import compute_tile_seam_error, run_tiled_inference
from .quantization import ScalarQuantizer
from .workflow import SMOKE_CHANNELS, build_smoke_tensor, run_codec_smoke
from .types import CodecConfig, CodecResult

__all__ = [
    "BitstreamError",
    "CanonicalHuffmanCoder",
    "CodecConfig",
    "CodecHarness",
    "CodecResult",
    "SMOKE_CHANNELS",
    "build_resource_usage_record",
    "build_smoke_tensor",
    "compute_tile_seam_error",
    "NormalizationSpec",
    "measure_runtime_resources",
    "run_tiled_inference",
    "run_codec_smoke",
    "ScalarQuantizer",
    "write_resource_usage",
]
