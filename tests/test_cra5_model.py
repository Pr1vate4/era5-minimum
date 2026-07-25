"""Tests for CRA5 VAEformer-28 model definition."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from era5_minimum.cra5.adapter import adapt_cra5_checkpoint
from era5_minimum.cra5.model import Cra5Vaeformer28, build_cra5_model


@pytest.fixture
def adapted_checkpoint(tmp_path: Path) -> dict[str, torch.Tensor]:
    """Create a minimal adapted checkpoint for testing."""
    # This is a simplified version matching the adapter output
    return {
        "backbone.encoder.patch_embed.proj.weight": torch.randn(1024, 28, 11, 10),
        "backbone.encoder.patch_embed.proj.bias": torch.randn(1024),
        # pos_embed will match 16x36 patch grid = 576 patches
        "backbone.encoder.pos_embed": torch.randn(1, 576, 1024),
        "backbone.encoder.blocks.0.norm1.weight": torch.randn(1024),
        "backbone.encoder.blocks.0.norm1.bias": torch.randn(1024),
        "backbone.encoder.blocks.0.attn.qkv.weight": torch.randn(3072, 1024),
        "backbone.encoder.blocks.0.attn.qkv.bias": torch.randn(3072),
        "backbone.encoder.blocks.0.attn.proj.weight": torch.randn(1024, 1024),
        "backbone.encoder.blocks.0.attn.proj.bias": torch.randn(1024),
        "backbone.encoder.blocks.0.norm2.weight": torch.randn(1024),
        "backbone.encoder.blocks.0.norm2.bias": torch.randn(1024),
        "backbone.encoder.blocks.0.mlp.fc1.weight": torch.randn(4096, 1024),
        "backbone.encoder.blocks.0.mlp.fc1.bias": torch.randn(4096),
        "backbone.encoder.blocks.0.mlp.fc2.weight": torch.randn(1024, 4096),
        "backbone.encoder.blocks.0.mlp.fc2.bias": torch.randn(1024),
        "backbone.decoder.blocks.0.norm1.weight": torch.randn(1024),
        "backbone.decoder.blocks.0.norm1.bias": torch.randn(1024),
        "backbone.decoder.blocks.0.attn.qkv.weight": torch.randn(3072, 1024),
        "backbone.decoder.blocks.0.attn.qkv.bias": torch.randn(3072),
        "backbone.decoder.blocks.0.attn.proj.weight": torch.randn(1024, 1024),
        "backbone.decoder.blocks.0.attn.proj.bias": torch.randn(1024),
        "backbone.decoder.blocks.0.norm2.weight": torch.randn(1024),
        "backbone.decoder.blocks.0.norm2.bias": torch.randn(1024),
        "backbone.decoder.blocks.0.mlp.fc1.weight": torch.randn(4096, 1024),
        "backbone.decoder.blocks.0.mlp.fc1.bias": torch.randn(4096),
        "backbone.decoder.blocks.0.mlp.fc2.weight": torch.randn(1024, 4096),
        "backbone.decoder.blocks.0.mlp.fc2.bias": torch.randn(1024),
        "backbone.decoder.final.weight": torch.randn(1024, 28, 11, 10),
        "backbone.loss.logvar": torch.randn(1, 28, 1, 1),
    }


def test_build_cra5_model_default() -> None:
    """Test that CRA5 model can be built with default parameters."""
    model = build_cra5_model()
    assert isinstance(model, Cra5Vaeformer28)
    assert model.in_channels == 28
    assert model.out_channels == 28


def test_cra5_model_load_adapted_weights(adapted_checkpoint: dict[str, torch.Tensor]) -> None:
    """Test that model can load adapted checkpoint weights."""
    model = build_cra5_model()
    
    # Load weights (strict=False because we have minimal checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(adapted_checkpoint, strict=False)
    
    # Input and output projections should be loaded
    assert "backbone.encoder.patch_embed.proj.weight" not in missing_keys
    assert "backbone.decoder.final.weight" not in missing_keys


def test_cra5_model_forward_pass() -> None:
    """Test that model can perform forward pass with 28-channel input."""
    model = build_cra5_model()
    model.eval()
    
    # Synthetic 28-channel input: [batch, channels, height, width]
    # CRA5 expects input divisible by patch size (11x10)
    batch_size = 2
    height, width = 176, 360  # 16x36 patches with 11x10 patch size
    x = torch.randn(batch_size, 28, height, width)
    
    with torch.no_grad():
        # Forward pass should return reconstruction
        output = model(x)
    
    assert output.shape == (batch_size, 28, height, width)
    assert output.dtype == torch.float32


def test_cra5_model_encoder_output_shape() -> None:
    """Test that encoder produces expected latent shape."""
    model = build_cra5_model()
    model.eval()
    
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)
    
    with torch.no_grad():
        latent = model.encode(x)
    
    # Latent should have reduced spatial dimensions
    assert latent.ndim == 4
    assert latent.shape[0] == batch_size
    # Hidden dimension should be 1024 (CRA5 default)
    assert latent.shape[1] == 1024
    # Spatial dimensions should be reduced by patch size
    assert latent.shape[2] == height // 11
    assert latent.shape[3] == width // 10


def test_cra5_model_decoder_reconstruction() -> None:
    """Test that decoder reconstructs from latent."""
    model = build_cra5_model()
    model.eval()
    
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)
    
    with torch.no_grad():
        latent = model.encode(x)
        reconstruction = model.decode(latent)
    
    assert reconstruction.shape == x.shape


def test_cra5_model_frozen_vs_trainable() -> None:
    """Test that model correctly marks frozen vs trainable parameters."""
    model = build_cra5_model(freeze_backbone=True)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # With freeze_backbone=True, most parameters should be frozen
    assert frozen_params > trainable_params
    
    # Input/output projections should be trainable
    assert model.backbone.encoder.patch_embed.proj.weight.requires_grad
    assert model.backbone.decoder.final.weight.requires_grad


def test_cra5_model_all_trainable() -> None:
    """Test that model can be made fully trainable."""
    model = build_cra5_model(freeze_backbone=False)
    
    # All parameters should be trainable
    for param in model.parameters():
        assert param.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cra5_model_gpu_placement() -> None:
    """Test that model works on GPU."""
    model = build_cra5_model().cuda()
    model.eval()
    
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width).cuda()
    
    with torch.no_grad():
        output = model(x)
    
    assert output.device.type == "cuda"
    assert output.shape == x.shape


def test_cra5_model_with_real_adapted_checkpoint() -> None:
    """Test model loading with real adapted CRA5 checkpoint (if available)."""
    checkpoint_path = Path.home() / ".cache/era5-minimum/cra5/cra5_159v_150k.pth"
    if not checkpoint_path.exists():
        pytest.skip("Real CRA5 checkpoint not available")
    
    # Adapt checkpoint
    adapted, metadata = adapt_cra5_checkpoint(checkpoint_path)
    
    # Build model and load adapted weights
    model = build_cra5_model()
    missing_keys, unexpected_keys = model.load_state_dict(adapted, strict=False)
    
    # Critical keys should be loaded
    assert "backbone.encoder.patch_embed.proj.weight" not in missing_keys
    assert "backbone.decoder.final.weight" not in missing_keys
    
    # Test forward pass
    model.eval()
    batch_size = 1
    height, width = 176, 360
    x = torch.randn(batch_size, 28, height, width)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == x.shape
