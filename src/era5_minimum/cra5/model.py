"""Minimal VAEformer-28 model definition for CRA5 adapter."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """2D patch embedding layer."""

    def __init__(
        self,
        in_channels: int = 28,
        hidden_dim: int = 1024,
        patch_size: tuple[int, int] = (11, 10),
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to patches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Patch embeddings [B, hidden_dim, H//patch_h, W//patch_w]
        """
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Simplified transformer block."""

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, N, hidden_dim]

        Returns
        -------
        torch.Tensor
            Output tensor [B, N, hidden_dim]
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """VAEformer encoder."""

    def __init__(
        self,
        in_channels: int = 28,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
        max_patches: int = 2304,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, hidden_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, hidden_dim))
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Latent tensor [B, hidden_dim, H//patch_h, W//patch_w]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_dim, h, w]
        
        B, C, h, w = x.shape
        
        # Flatten spatial dims for transformer
        x_flat = x.flatten(2).transpose(1, 2)  # [B, h*w, hidden_dim]
        
        # Add positional embedding if available and matches shape
        if self.pos_embed is not None and self.pos_embed.shape[1] == h * w:
            x_flat = x_flat + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x_flat = block(x_flat)
        
        # Reshape back to spatial
        x = x_flat.transpose(1, 2).reshape(B, C, h, w)
        return x


class Decoder(nn.Module):
    """VAEformer decoder."""

    def __init__(
        self,
        out_channels: int = 28,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        self.final = nn.ConvTranspose2d(
            hidden_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor [B, hidden_dim, h, w]

        Returns
        -------
        torch.Tensor
            Reconstructed tensor [B, out_channels, H, W]
        """
        B, C, h, w = x.shape
        
        # Flatten spatial dims for transformer
        x_flat = x.flatten(2).transpose(1, 2)  # [B, h*w, hidden_dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x_flat = block(x_flat)
        
        # Reshape back to spatial
        x = x_flat.transpose(1, 2).reshape(B, C, h, w)
        
        # Unpatch to original resolution
        x = self.final(x)
        return x


class VAEformerBackbone(nn.Module):
    """Combined encoder-decoder backbone."""

    def __init__(
        self,
        in_channels: int = 28,
        out_channels: int = 28,
        hidden_dim: int = 1024,
        num_encoder_blocks: int = 4,
        num_decoder_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, num_encoder_blocks, num_heads, patch_size)
        self.decoder = Decoder(out_channels, hidden_dim, num_decoder_blocks, num_heads, patch_size)


class Cra5Vaeformer28(nn.Module):
    """
    CRA5 VAEformer adapted to 28 channels.

    This is a simplified implementation compatible with adapted CRA5-159 checkpoints.
    """

    def __init__(
        self,
        in_channels: int = 28,
        out_channels: int = 28,
        hidden_dim: int = 1024,
        num_encoder_blocks: int = 4,
        num_decoder_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.backbone = VAEformerBackbone(
            in_channels,
            out_channels,
            hidden_dim,
            num_encoder_blocks,
            num_decoder_blocks,
            num_heads,
            patch_size,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, in_channels, H, W]

        Returns
        -------
        torch.Tensor
            Latent tensor [B, hidden_dim, h, w]
        """
        return self.backbone.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Parameters
        ----------
        latent : torch.Tensor
            Latent tensor [B, hidden_dim, h, w]

        Returns
        -------
        torch.Tensor
            Reconstructed tensor [B, out_channels, H, W]
        """
        return self.backbone.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode and decode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, in_channels, H, W]

        Returns
        -------
        torch.Tensor
            Reconstructed tensor [B, out_channels, H, W]
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def freeze_backbone(self) -> None:
        """Freeze encoder and decoder parameters."""
        for param in self.backbone.encoder.parameters():
            # Freeze encoder except patch_embed
            param.requires_grad = False
        for param in self.backbone.decoder.parameters():
            # Freeze decoder except final
            param.requires_grad = False
        
        # Make input/output projections trainable
        self.backbone.encoder.patch_embed.proj.weight.requires_grad = True
        self.backbone.encoder.patch_embed.proj.bias.requires_grad = True
        self.backbone.decoder.final.weight.requires_grad = True


def build_cra5_model(
    in_channels: int = 28,
    out_channels: int = 28,
    hidden_dim: int = 1024,
    num_encoder_blocks: int = 4,
    num_decoder_blocks: int = 4,
    num_heads: int = 8,
    patch_size: tuple[int, int] = (11, 10),
    freeze_backbone: bool = False,
) -> Cra5Vaeformer28:
    """
    Build CRA5 VAEformer-28 model.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default: 28.
    out_channels : int
        Number of output channels. Default: 28.
    hidden_dim : int
        Hidden dimension size. Default: 1024.
    num_encoder_blocks : int
        Number of encoder transformer blocks. Default: 4.
    num_decoder_blocks : int
        Number of decoder transformer blocks. Default: 4.
    num_heads : int
        Number of attention heads. Default: 8.
    patch_size : tuple[int, int]
        Patch size (height, width). Default: (11, 10).
    freeze_backbone : bool
        Whether to freeze encoder/decoder and only train projections. Default: False.

    Returns
    -------
    Cra5Vaeformer28
        Initialized model.
    """
    model = Cra5Vaeformer28(
        in_channels,
        out_channels,
        hidden_dim,
        num_encoder_blocks,
        num_decoder_blocks,
        num_heads,
        patch_size,
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model
