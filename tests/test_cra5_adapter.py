"""Tests for CRA5-159 to ERA5-28 checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from era5_minimum.cra5.adapter import (
    AdaptedCheckpointMetadata,
    adapt_cra5_checkpoint,
)
from era5_minimum.cra5.channel_mapping import CRA5_CHANNEL_MAPPING


@pytest.fixture
def synthetic_cra5_checkpoint(tmp_path: Path) -> Path:
    """Create a synthetic CRA5-159 checkpoint with correct structure."""
    state_dict = {
        # Input embedding: [hidden_dim, 159, kernel_h, kernel_w]
        "backbone.encoder.patch_embed.proj.weight": torch.randn(1024, 159, 11, 10),
        "backbone.encoder.patch_embed.proj.bias": torch.randn(1024),
        # Output projection: [hidden_dim, 159, kernel_h, kernel_w]
        "backbone.decoder.final.weight": torch.randn(1024, 159, 11, 10),
        # Some encoder block (should remain unchanged)
        "backbone.encoder.blocks.0.norm1.weight": torch.randn(1024),
        "backbone.encoder.blocks.0.norm1.bias": torch.randn(1024),
        # Some decoder block (should remain unchanged)
        "backbone.decoder.blocks.0.norm1.weight": torch.randn(1024),
        "backbone.decoder.blocks.0.norm1.bias": torch.randn(1024),
        # Loss logvar (should be adapted)
        "backbone.loss.logvar": torch.randn(1, 159, 1, 1),
    }
    checkpoint_path = tmp_path / "synthetic_cra5_159.pth"
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def test_adapt_checkpoint_shape_transformation(synthetic_cra5_checkpoint: Path) -> None:
    """Test that adapted checkpoint has 28 input/output channels."""
    adapted, metadata = adapt_cra5_checkpoint(synthetic_cra5_checkpoint)

    # Input embedding should be sliced to 28 channels
    assert "backbone.encoder.patch_embed.proj.weight" in adapted
    input_weight = adapted["backbone.encoder.patch_embed.proj.weight"]
    assert input_weight.shape == (1024, 28, 11, 10)

    # Output projection should be sliced to 28 channels
    assert "backbone.decoder.final.weight" in adapted
    output_weight = adapted["backbone.decoder.final.weight"]
    assert output_weight.shape == (1024, 28, 11, 10)

    # Loss logvar should be adapted to 28 channels
    assert "backbone.loss.logvar" in adapted
    logvar = adapted["backbone.loss.logvar"]
    assert logvar.shape == (1, 28, 1, 1)

    # Other layers should remain unchanged
    assert adapted["backbone.encoder.blocks.0.norm1.weight"].shape == (1024,)
    assert adapted["backbone.decoder.blocks.0.norm1.weight"].shape == (1024,)


def test_adapt_checkpoint_channel_mapping(synthetic_cra5_checkpoint: Path) -> None:
    """Test that channels are copied according to explicit mapping."""
    original = torch.load(synthetic_cra5_checkpoint, map_location="cpu", weights_only=True)
    adapted, _ = adapt_cra5_checkpoint(synthetic_cra5_checkpoint)

    input_original = original["backbone.encoder.patch_embed.proj.weight"]
    input_adapted = adapted["backbone.encoder.patch_embed.proj.weight"]

    # Check that copied channels match their source indices
    copied_count = 0
    learned_count = 0
    for i, mapping in enumerate(CRA5_CHANNEL_MAPPING):
        if mapping.initialization == "copy":
            assert mapping.cra5_index is not None
            # Input channel i should equal original channel cra5_index
            torch.testing.assert_close(
                input_adapted[:, i, :, :],
                input_original[:, mapping.cra5_index, :, :],
                msg=f"Channel {i} ({mapping.canonical_name}) mismatch",
            )
            copied_count += 1
        elif mapping.initialization == "learned_boundary":
            # Learned boundary channels should be zero-initialized
            assert torch.allclose(input_adapted[:, i, :, :], torch.zeros_like(input_adapted[:, i, :, :]))
            learned_count += 1

    assert copied_count == 26
    assert learned_count == 2


def test_adapt_checkpoint_output_projection(synthetic_cra5_checkpoint: Path) -> None:
    """Test that output projection is correctly sliced."""
    original = torch.load(synthetic_cra5_checkpoint, map_location="cpu", weights_only=True)
    adapted, _ = adapt_cra5_checkpoint(synthetic_cra5_checkpoint)

    output_original = original["backbone.decoder.final.weight"]
    output_adapted = adapted["backbone.decoder.final.weight"]

    # Output projection: each output channel corresponds to one canonical channel
    for i, mapping in enumerate(CRA5_CHANNEL_MAPPING):
        if mapping.initialization == "copy":
            assert mapping.cra5_index is not None
            # Output channel i should be copied from original channel cra5_index
            torch.testing.assert_close(
                output_adapted[:, i, :, :],
                output_original[:, mapping.cra5_index, :, :],
                msg=f"Output channel {i} ({mapping.canonical_name}) mismatch",
            )
        elif mapping.initialization == "learned_boundary":
            # Learned boundary output channels zero-initialized
            assert torch.allclose(output_adapted[:, i, :, :], torch.zeros_like(output_adapted[:, i, :, :]))


def test_adapt_checkpoint_metadata(synthetic_cra5_checkpoint: Path) -> None:
    """Test that metadata records the adaptation."""
    _, metadata = adapt_cra5_checkpoint(synthetic_cra5_checkpoint)

    assert metadata.source_checkpoint == str(synthetic_cra5_checkpoint)
    assert metadata.source_channels == 159
    assert metadata.target_channels == 28
    assert metadata.copied_channels == 26
    assert metadata.learned_boundary_channels == 2
    assert len(metadata.learned_boundary_names) == 2
    assert "sst" in metadata.learned_boundary_names
    assert "tcwv" in metadata.learned_boundary_names


def test_adapt_checkpoint_preserves_other_layers(synthetic_cra5_checkpoint: Path) -> None:
    """Test that encoder/decoder blocks remain unchanged."""
    original = torch.load(synthetic_cra5_checkpoint, map_location="cpu", weights_only=True)
    adapted, _ = adapt_cra5_checkpoint(synthetic_cra5_checkpoint)

    # Encoder block should be identical
    torch.testing.assert_close(
        adapted["backbone.encoder.blocks.0.norm1.weight"],
        original["backbone.encoder.blocks.0.norm1.weight"],
    )
    torch.testing.assert_close(
        adapted["backbone.encoder.blocks.0.norm1.bias"],
        original["backbone.encoder.blocks.0.norm1.bias"],
    )

    # Decoder block should be identical
    torch.testing.assert_close(
        adapted["backbone.decoder.blocks.0.norm1.weight"],
        original["backbone.decoder.blocks.0.norm1.weight"],
    )


def test_adapt_checkpoint_rejects_wrong_structure(tmp_path: Path) -> None:
    """Test that adapter rejects checkpoints with unexpected structure."""
    # Missing input embedding
    bad_checkpoint = tmp_path / "bad.pth"
    torch.save({"some.other.key": torch.randn(10)}, bad_checkpoint)

    with pytest.raises(ValueError, match="Missing required key"):
        adapt_cra5_checkpoint(bad_checkpoint)


def test_adapt_checkpoint_rejects_wrong_channel_count(tmp_path: Path) -> None:
    """Test that adapter rejects checkpoints with wrong channel count."""
    bad_checkpoint = tmp_path / "bad_channels.pth"
    state_dict = {
        "backbone.encoder.patch_embed.proj.weight": torch.randn(1024, 100, 11, 10),  # Wrong: 100 instead of 159
        "backbone.decoder.final.weight": torch.randn(1024, 100, 11, 10),
        "backbone.loss.logvar": torch.randn(1, 100, 1, 1),
    }
    torch.save(state_dict, bad_checkpoint)

    with pytest.raises(ValueError, match="Expected 159 channels"):
        adapt_cra5_checkpoint(bad_checkpoint)


def test_adapt_checkpoint_device_placement(synthetic_cra5_checkpoint: Path) -> None:
    """Test that adapted checkpoint respects device parameter."""
    adapted_cpu, _ = adapt_cra5_checkpoint(synthetic_cra5_checkpoint, device="cpu")
    assert adapted_cpu["backbone.encoder.patch_embed.proj.weight"].device.type == "cpu"

    # If CUDA available, test GPU placement
    if torch.cuda.is_available():
        adapted_gpu, _ = adapt_cra5_checkpoint(synthetic_cra5_checkpoint, device="cuda")
        assert adapted_gpu["backbone.encoder.patch_embed.proj.weight"].device.type == "cuda"


def test_adapt_real_cra5_checkpoint() -> None:
    """Test adaptation of the real CRA5-159v checkpoint (if available)."""
    checkpoint_path = Path.home() / ".cache/era5-minimum/cra5/cra5_159v_150k.pth"
    if not checkpoint_path.exists():
        pytest.skip("Real CRA5 checkpoint not available")

    adapted, metadata = adapt_cra5_checkpoint(checkpoint_path)

    # Validate shapes
    assert adapted["backbone.encoder.patch_embed.proj.weight"].shape == (1024, 28, 11, 10)
    assert adapted["backbone.decoder.final.weight"].shape == (1024, 28, 11, 10)
    assert adapted["backbone.loss.logvar"].shape == (1, 28, 1, 1)

    # Validate metadata
    assert metadata.source_channels == 159
    assert metadata.target_channels == 28
    assert metadata.copied_channels == 26
    assert metadata.learned_boundary_channels == 2
