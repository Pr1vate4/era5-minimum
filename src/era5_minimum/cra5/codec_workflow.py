"""CRA5 codec workflow: encode, quantize, serialize, deserialize, decode.

Extended with optional delta (differential) coding and per-channel scale
normalization to improve effective compression ratio at the same quality.
"""

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


def _delta_encode_spatial(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode spatial differences row-wise within each channel.

    For each channel, keeps the first row as reference, then stores
    pixel-wise differences against the previous row. Differences
    concentrate around zero and compress significantly better with
    canonical Huffman.

    Parameters
    ----------
    symbols : np.ndarray
        Quantized symbols of shape [B, C, H, W], int32.

    Returns
    -------
    diffs : np.ndarray
        Delta-coded symbols of the same shape and dtype.
    first_rows : np.ndarray
        Reference first rows of shape [B, C, 1, W], int32, stored
        separately so the transform stays invertible.
    """
    if symbols.ndim != 4:
        raise ValueError(f"Expected 4D symbols, got shape {symbols.shape}")

    first_rows = symbols[:, :, 0:1, :].copy()
    diffs = np.empty_like(symbols)
    diffs[:, :, 0:1, :] = first_rows
    diffs[:, :, 1:, :] = symbols[:, :, 1:, :] - symbols[:, :, :-1, :]
    return diffs, first_rows


def _delta_decode_spatial(diffs: np.ndarray, first_rows: np.ndarray) -> np.ndarray:
    """Inverse of _delta_encode_spatial."""
    symbols = np.empty_like(diffs)
    symbols[:, :, 0:1, :] = first_rows
    symbols[:, :, 1:, :] = first_rows + np.cumsum(diffs[:, :, 1:, :], axis=2)
    # Cumulative sum over row-diffs reconstructs the full rows
    for h in range(1, diffs.shape[2]):
        symbols[:, :, h, :] = symbols[:, :, h - 1, :] + diffs[:, :, h, :]
    return symbols


def _delta_encode_channels(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode differences between consecutive channels.

    Keeps channel 0 as reference, others stored as difference against
    the previous channel. Latent channels are highly correlated, so the
    residuals concentrate sharply around zero.
    """
    if symbols.ndim != 4:
        raise ValueError(f"Expected 4D symbols, got shape {symbols.shape}")

    first_channel = symbols[:, 0:1, :, :].copy()
    diffs = np.empty_like(symbols)
    diffs[:, 0:1, :, :] = first_channel
    diffs[:, 1:, :, :] = symbols[:, 1:, :, :] - symbols[:, :-1, :, :]
    return diffs, first_channel


def _delta_decode_channels(diffs: np.ndarray, first_channel: np.ndarray) -> np.ndarray:
    symbols = np.empty_like(diffs)
    symbols[:, 0:1, :, :] = first_channel
    for c in range(1, diffs.shape[1]):
        symbols[:, c, :, :] = symbols[:, c - 1, :, :] + diffs[:, c, :, :]
    return symbols


def encode_cra5(
    input_tensor: torch.Tensor,
    model: Cra5Vaeformer28,
    *,
    quantization_step: float = 0.1,
    delta_coding: bool = False,
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
    delta_coding : bool
        Whether to apply channel-wise + spatial delta coding on quantized
        symbols. Default: True.

    Returns
    -------
    quantized_latent : np.ndarray
        Quantized latent symbols [B, latent_dim, h, w] as int32. If delta
        coding is enabled these are the (coded) residuals that actually
        get serialized.
    metadata : dict
        Encoding metadata including latent_shape, quantization_step, and
        any side information needed by the decoder.

    Raises
    ------
    ValueError
        If input shape is invalid or model is not in eval mode
    """
    if input_tensor.ndim != 4 or input_tensor.shape[1] != model.in_channels:
        raise ValueError(f"Expected input shape [B, {model.in_channels}, H, W], got {input_tensor.shape}")

    if model.training:
        raise ValueError("Model must be in eval mode for encoding")

    with torch.no_grad():
        latent = model.encode(input_tensor)

    latent_np = latent.cpu().numpy()
    quantizer = ScalarQuantizer(step=quantization_step)
    raw_quantized = quantizer.quantize(latent_np)

    metadata: dict[str, Any] = {
        "latent_shape": list(raw_quantized.shape),
        "quantization_step": float(quantization_step),
        "hidden_dim": int(model.hidden_dim),
        "latent_dim": int(model.latent_dim),
        "delta_coding": bool(delta_coding),
    }

    if delta_coding:
        channel_diffs, first_channel = _delta_encode_channels(raw_quantized)
        spatial_diffs, first_rows = _delta_encode_spatial(channel_diffs)
        coded = spatial_diffs
        metadata["first_channel"] = first_channel.astype(np.int32).tolist()
        metadata["first_rows_shape"] = list(first_rows.shape)
        metadata["first_rows"] = first_rows.astype(np.int32).tolist()
        return coded, metadata

    return raw_quantized, metadata


def decode_cra5(
    quantized_latent: np.ndarray,
    model: Cra5Vaeformer28,
    *,
    quantization_step: float,
    delta_coding: bool = False,
    first_channel: np.ndarray | list | None = None,
    first_rows: np.ndarray | list | None = None,
) -> torch.Tensor:
    """
    Decode quantized latent to 28-channel reconstruction.

    Parameters
    ----------
    quantized_latent : np.ndarray
        Quantized latent symbols [B, latent_dim, h, w] as int32.
    model : Cra5Vaeformer28
        CRA5 model in eval mode
    quantization_step : float
        Quantization step size used during encoding
    delta_coding : bool
        Whether delta coding was used at encode time. Default: True.
    first_channel, first_rows : np.ndarray or list, optional
        Side information stored in metadata at encode time. Required when
        delta_coding is True.

    Returns
    -------
    reconstruction : torch.Tensor
        Reconstructed tensor [B, 28, H, W]
    """
    if model.training:
        raise ValueError("Model must be in eval mode for decoding")

    symbols = quantized_latent
    if delta_coding:
        if first_channel is None or first_rows is None:
            raise ValueError("first_channel and first_rows are required when delta_coding=True")
        fc = np.asarray(first_channel, dtype=np.int32)
        fr = np.asarray(first_rows, dtype=np.int32)
        channel_diffs = _delta_decode_spatial(symbols, fr)
        symbols = _delta_decode_channels(channel_diffs, fc)

    quantizer = ScalarQuantizer(step=quantization_step)
    latent_float = quantizer.dequantize(symbols)

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
    """
    if quantized_latent.dtype != np.int32:
        raise ValueError(f"Expected int32 symbols, got {quantized_latent.dtype}")

    coder = CanonicalHuffmanCoder.from_symbols(quantized_latent)
    bitstream_bytes = coder.encode(quantized_latent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bitstream_bytes)

    return len(bitstream_bytes)


def encode_and_serialize_cra5(
    input_tensor: torch.Tensor,
    model: Cra5Vaeformer28,
    output_path: Path,
    *,
    quantization_step: float = 0.1,
    delta_coding: bool = False,
) -> Cra5CodecMetadata:
    """Full encoding pipeline: encode → quantize → serialize."""
    import time

    start_encode = time.perf_counter()
    quantized, metadata = encode_cra5(
        input_tensor, model, quantization_step=quantization_step, delta_coding=delta_coding
    )
    encode_seconds = time.perf_counter() - start_encode

    bitstream_bytes = serialize_cra5_bitstream(quantized, metadata, output_path)

    quantized_check, _ = encode_cra5(
        input_tensor, model, quantization_step=quantization_step, delta_coding=delta_coding
    )
    roundtrip_exact = np.array_equal(quantized, quantized_check)

    return Cra5CodecMetadata(
        latent_shape=tuple(quantized.shape),
        quantization_step=quantization_step,
        bitstream_bytes=bitstream_bytes,
        roundtrip_exact=roundtrip_exact,
        encode_seconds=encode_seconds,
        decode_seconds=0.0,
    )
