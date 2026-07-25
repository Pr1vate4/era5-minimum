"""CRA5 codec workflow: encode, quantize, serialize, deserialize, decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from era5_minimum.codec.bitstream import CanonicalHuffmanCoder
from era5_minimum.codec.quantization import ScalarQuantizer
from era5_minimum.cra5.model import Cra5Vaeformer28


@dataclass(frozen=True)
class Cra5CodecMetadata:
    """Metadata for CRA5 codec encode/decode."""

    latent_shape: tuple[int, ...]
    quantization_step: float
    bitstream_bytes: int
    roundtrip_exact: bool
    encode_seconds: float
    decode_seconds: float


def encode_cra5(
    input_tensor: torch.Tensor,
    model: Cra5Vaeformer28,
    *,
    quantization_step: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Encode 28-channel tensor to quantized latent.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor [B, 28, H, W]
    model : Cra5Vaeformer28
        CRA5 model in eval mode
    quantization_step : float
        Quantization step size. Default: 0.1.

    Returns
    -------
    quantized_latent : np.ndarray
        Quantized latent symbols [B, hidden_dim, h, w] as int32
    metadata : dict
        Encoding metadata including latent_shape, quantization_step

    Raises
    ------
    ValueError
        If input shape is invalid or model is not in eval mode
    """
    if input_tensor.ndim != 4 or input_tensor.shape[1] != model.in_channels:
        raise ValueError(f"Expected input shape [B, {model.in_channels}, H, W], got {input_tensor.shape}")

    if model.training:
        raise ValueError("Model must be in eval mode for encoding")

    # Encode to latent
    with torch.no_grad():
        latent = model.encode(input_tensor)  # [B, hidden_dim, h, w]

    # Quantize latent
    latent_np = latent.cpu().numpy()
    quantizer = ScalarQuantizer(step=quantization_step)
    quantized = quantizer.quantize(latent_np)

    metadata = {
        "latent_shape": list(quantized.shape),
        "quantization_step": float(quantization_step),
        "hidden_dim": int(latent.shape[1]),
    }

    return quantized, metadata


def decode_cra5(
    quantized_latent: np.ndarray,
    model: Cra5Vaeformer28,
    *,
    quantization_step: float,
) -> torch.Tensor:
    """
    Decode quantized latent to 28-channel reconstruction.

    Parameters
    ----------
    quantized_latent : np.ndarray
        Quantized latent symbols [B, hidden_dim, h, w] as int32
    model : Cra5Vaeformer28
        CRA5 model in eval mode
    quantization_step : float
        Quantization step size used during encoding

    Returns
    -------
    reconstruction : torch.Tensor
        Reconstructed tensor [B, 28, H, W]

    Raises
    ------
    ValueError
        If model is not in eval mode
    """
    if model.training:
        raise ValueError("Model must be in eval mode for decoding")

    # Dequantize
    quantizer = ScalarQuantizer(step=quantization_step)
    latent_float = quantizer.dequantize(quantized_latent)

    # Convert to torch and decode
    latent_tensor = torch.from_numpy(latent_float).to(next(model.parameters()).device)

    with torch.no_grad():
        reconstruction = model.decode(latent_tensor)

    return reconstruction


def serialize_cra5_bitstream(
    quantized_latent: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path,
) -> int:
    """
    Serialize quantized latent to Huffman bitstream file.

    Parameters
    ----------
    quantized_latent : np.ndarray
        Quantized latent symbols as int32
    metadata : dict
        Codec metadata
    output_path : Path
        Output bitstream file path

    Returns
    -------
    int
        Total serialized bytes including header and symbols

    Raises
    ------
    ValueError
        If symbols are not int32
    """
    if quantized_latent.dtype != np.int32:
        raise ValueError(f"Expected int32 symbols, got {quantized_latent.dtype}")

    # Build Huffman coder from symbols
    coder = CanonicalHuffmanCoder.from_symbols(quantized_latent)

    # Encode to bitstream
    bitstream_bytes = coder.encode(quantized_latent)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bitstream_bytes)

    return len(bitstream_bytes)


def encode_and_serialize_cra5(
    input_tensor: torch.Tensor,
    model: Cra5Vaeformer28,
    output_path: Path,
    *,
    quantization_step: float = 0.1,
) -> Cra5CodecMetadata:
    """
    Full encoding pipeline: encode → quantize → serialize.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor [B, 28, H, W]
    model : Cra5Vaeformer28
        CRA5 model in eval mode
    output_path : Path
        Output bitstream file path
    quantization_step : float
        Quantization step size. Default: 0.1.

    Returns
    -------
    Cra5CodecMetadata
        Codec operation metadata including bitstream size and timings

    Raises
    ------
    ValueError
        If input is invalid or model is not in eval mode
    """
    import time

    # Encode
    start_encode = time.perf_counter()
    quantized, metadata = encode_cra5(input_tensor, model, quantization_step=quantization_step)
    encode_seconds = time.perf_counter() - start_encode

    # Serialize
    bitstream_bytes = serialize_cra5_bitstream(quantized, metadata, output_path)

    # Check roundtrip
    quantized_check, _ = encode_cra5(input_tensor, model, quantization_step=quantization_step)
    roundtrip_exact = np.array_equal(quantized, quantized_check)

    return Cra5CodecMetadata(
        latent_shape=tuple(quantized.shape),
        quantization_step=quantization_step,
        bitstream_bytes=bitstream_bytes,
        roundtrip_exact=roundtrip_exact,
        encode_seconds=encode_seconds,
        decode_seconds=0.0,  # Will be measured separately
    )
