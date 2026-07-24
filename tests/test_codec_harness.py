from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from era5_minimum.codec import CodecConfig, NormalizationSpec
from era5_minimum.codec.harness import CodecHarness


def test_codec_harness_writes_metadata_and_separates_ratios(tmp_path: Path) -> None:
    tensor = np.array([[[[0.0, 0.25], [0.5, 0.75]]]], dtype=np.float32)
    config = CodecConfig(
        version="ml-001",
        channel_order=("t2m",),
        grid="0.5deg",
        quantization_step=0.25,
        seed=7,
        git_commit="deadbeef",
    )
    normalization = NormalizationSpec(
        channel_order=("t2m",),
        mean=np.array([0.0], dtype=np.float32),
        std=np.array([1.0], dtype=np.float32),
        source_manifest_sha256="abc123",
        train_only=True,
    )

    result = CodecHarness(config=config, normalization=normalization).encode_decode(tensor, tmp_path)

    assert result.roundtrip_ok is True
    assert result.tensor_ratio == 1.0
    assert result.serialized_ratio > 0.0
    assert result.bitstream_path.exists()
    assert result.metadata_path.exists()

    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["compression"]["tensor_ratio"] == 1.0
    assert payload["compression"]["serialized_ratio"] == result.serialized_ratio
    assert payload["normalization"]["source_manifest_sha256"] == "abc123"
    assert payload["roundtrip"]["exact_symbol_match"] is True


def test_codec_harness_rejects_channel_mismatch(tmp_path: Path) -> None:
    tensor = np.zeros((1, 2, 2, 2), dtype=np.float32)
    config = CodecConfig(
        version="ml-001",
        channel_order=("t2m",),
        grid="0.5deg",
        quantization_step=0.25,
        seed=7,
        git_commit="deadbeef",
    )
    normalization = NormalizationSpec(
        channel_order=("t2m",),
        mean=np.array([0.0], dtype=np.float32),
        std=np.array([1.0], dtype=np.float32),
        source_manifest_sha256="abc123",
        train_only=True,
    )

    with pytest.raises(ValueError, match="channel count"):
        CodecHarness(config=config, normalization=normalization).encode_decode(tensor, tmp_path)


def test_codec_harness_reports_external_latent_ratios(tmp_path: Path) -> None:
    input_tensor = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    latent = np.array([[0.0, 0.25, 0.5, 0.75]], dtype=np.float32)
    config = CodecConfig(
        version="ml-001",
        channel_order=("t2m",),
        grid="0.5deg",
        quantization_step=0.25,
        seed=7,
        git_commit="deadbeef",
    )
    normalization = NormalizationSpec(
        channel_order=("t2m",),
        mean=np.array([0.0], dtype=np.float32),
        std=np.array([1.0], dtype=np.float32),
        source_manifest_sha256="abc123",
        train_only=True,
    )

    result = CodecHarness(config=config, normalization=normalization).encode_latent(
        input_tensor=input_tensor,
        latent=latent,
        output_dir=tmp_path,
    )

    assert result.roundtrip_ok is True
    assert result.tensor_ratio == 4.0
    assert result.decoded_latent is not None
    np.testing.assert_array_equal(result.decoded_latent, latent)

    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["latent"]["shape"] == [1, 4]
    assert payload["compression"]["tensor_ratio"] == 4.0
    assert payload["compression"]["actual_compression_ratio"] == pytest.approx(result.serialized_ratio)
    assert payload["quantization"]["symbol_dtype"] == "int32"
