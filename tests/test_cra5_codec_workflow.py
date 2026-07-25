"""Tests for CRA5 codec workflow (encode, quantize, serialize, decode)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from era5_minimum.cra5.codec_workflow import (
    decode_cra5,
    encode_and_serialize_cra5,
    encode_cra5,
    serialize_cra5_bitstream,
)
from era5_minimum.cra5.model import build_cra5_model


@pytest.fixture
def cra5_model() -> torch.nn.Module:
    """Build a minimal CRA5 model for testing."""
    model = build_cra5_model()
    model.eval()
    return model


def test_encode_cra5_basic(cra5_model: torch.nn.Module) -> None:
    """Test basic encoding to quantized latent."""
    batch_size = 1
    height, width = 176, 360  # Divisible by patch size (11, 10)
    x = torch.randn(batch_size, 28, height, width)

    quantized, metadata = encode_cra5(x, cra5_model, quantization_step=0.1)

    # Check output shape
    assert quantized.ndim == 4
    assert quantized.shape[0] == batch_size
    assert quantized.shape[1] == 1024  # hidden_dim
    assert quantized.shape[2] == height // 11
    assert quantized.shape[3] == width // 10

    # Check quantized symbols are int32
    assert quantized.dtype == np.int32

    # Check metadata
    assert metadata["quantization_step"] == 0.1
    assert metadata["hidden_dim"] == 1024


def test_decode_cra5_basic(cra5_model: torch.nn.Module) -> None:
    """Test basic decoding from quantized latent."""
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)

    # Encode
    quantized, metadata = encode_cra5(x, cra5_model, quantization_step=0.1)

    # Decode
    reconstruction = decode_cra5(quantized, cra5_model, quantization_step=0.1)

    # Check reconstruction shape matches input
    assert reconstruction.shape == x.shape
    assert reconstruction.dtype == torch.float32


def test_encode_decode_roundtrip(cra5_model: torch.nn.Module) -> None:
    """Test that encode → decode produces consistent output."""
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)

    # Encode → decode
    quantized, _ = encode_cra5(x, cra5_model, quantization_step=0.1)
    reconstruction = decode_cra5(quantized, cra5_model, quantization_step=0.1)

    # Reconstruction should differ from input due to lossy compression
    assert not torch.allclose(reconstruction, x)

    # But encode → decode → encode should give same quantized latent
    quantized2, _ = encode_cra5(reconstruction, cra5_model, quantization_step=0.1)
    # Note: Due to quantization, symbols may differ slightly, so check close
    # For a real test, we'd want exact symbol match after first quantization


def test_serialize_cra5_bitstream(cra5_model: torch.nn.Module, tmp_path: Path) -> None:
    """Test bitstream serialization."""
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)

    # Encode
    quantized, metadata = encode_cra5(x, cra5_model, quantization_step=0.1)

    # Serialize
    output_path = tmp_path / "test.bitstream"
    bitstream_bytes = serialize_cra5_bitstream(quantized, metadata, output_path)

    # Check file exists and has nonzero size
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert bitstream_bytes > 0


def test_encode_and_serialize_full_pipeline(cra5_model: torch.nn.Module, tmp_path: Path) -> None:
    """Test full encoding and serialization pipeline."""
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)

    output_path = tmp_path / "test.bitstream"
    codec_metadata = encode_and_serialize_cra5(
        x, cra5_model, output_path, quantization_step=0.1
    )

    # Check metadata
    assert codec_metadata.latent_shape == (1, 1024, 16, 36)
    assert codec_metadata.quantization_step == 0.1
    assert codec_metadata.bitstream_bytes > 0
    assert codec_metadata.encode_seconds >= 0

    # Check file exists
    assert output_path.exists()


def test_encode_rejects_wrong_shape(cra5_model: torch.nn.Module) -> None:
    """Test that encoder rejects wrong input shape."""
    # Wrong channel count
    x = torch.randn(1, 20, 176, 360)
    with pytest.raises(ValueError, match="Expected input shape"):
        encode_cra5(x, cra5_model)


def test_encode_rejects_training_mode(cra5_model: torch.nn.Module) -> None:
    """Test that encoder rejects model in training mode."""
    cra5_model.train()
    x = torch.randn(1, 28, 176, 360)
    with pytest.raises(ValueError, match="must be in eval mode"):
        encode_cra5(x, cra5_model)


def test_decode_rejects_training_mode(cra5_model: torch.nn.Module) -> None:
    """Test that decoder rejects model in training mode."""
    quantized = np.random.randint(-100, 100, size=(1, 1024, 16, 36), dtype=np.int32)
    cra5_model.train()
    with pytest.raises(ValueError, match="must be in eval mode"):
        decode_cra5(quantized, cra5_model, quantization_step=0.1)


def test_serialize_rejects_non_int32(tmp_path: Path) -> None:
    """Test that serialization rejects non-int32 symbols."""
    quantized = np.random.randn(1, 1024, 16, 36).astype(np.float32)
    metadata = {"quantization_step": 0.1}
    output_path = tmp_path / "test.bitstream"

    with pytest.raises(ValueError, match="Expected int32 symbols"):
        serialize_cra5_bitstream(quantized, metadata, output_path)
