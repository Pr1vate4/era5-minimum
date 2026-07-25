"""Codec primitives for ERA5-Minimum."""

from .bitstream import BitstreamError, CanonicalHuffmanCoder
from .harness import CodecHarness
from .normalization import NormalizationSpec, denormalize_reconstruction, normalize_physical_tensor
from .rate_distortion import (
    DistortionComponents,
    FactorizedLogisticEntropyModel,
    grouped_latitude_distortion,
    quantize_with_uniform_noise,
)
from .resources import build_resource_usage_record, measure_runtime_resources, write_resource_usage
from .quantization import ScalarQuantizer
from .tiling import compute_tile_seam_error, decode_latent_tiled, run_tiled_inference
from .types import CodecConfig, CodecResult
from .workflow import SMOKE_CHANNELS, build_smoke_tensor, run_codec_smoke

__all__ = [
    "BitstreamError",
    "CanonicalHuffmanCoder",
    "CodecConfig",
    "CodecHarness",
    "CodecResult",
    "DistortionComponents",
    "FactorizedLogisticEntropyModel",
    "SMOKE_CHANNELS",
    "build_resource_usage_record",
    "build_smoke_tensor",
    "compute_tile_seam_error",
    "denormalize_reconstruction",
    "decode_latent_tiled",
    "grouped_latitude_distortion",
    "NormalizationSpec",
    "measure_runtime_resources",
    "normalize_physical_tensor",
    "quantize_with_uniform_noise",
    "run_tiled_inference",
    "run_codec_smoke",
    "ScalarQuantizer",
    "write_resource_usage",
]
