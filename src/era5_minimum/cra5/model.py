"""Minimal VAEformer-28 model definition for CRA5 adapter with compression controls."""

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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LatentBottleneck(nn.Module):
    """Channel-wise compression bottleneck applied after encoder / before decoder.

    Reduces hidden_dim -> latent_dim channels and back.
    This is the primary lever for achieving 32x-64x compression ratios.
    """

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        if latent_dim == hidden_dim:
            self.compress = nn.Identity()
            self.expand = nn.Identity()
        else:
            self.compress = nn.Sequential(
                nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, stride=1, bias=True),
            )
            self.expand = nn.Sequential(
                nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, stride=1, bias=True),
            )

    def down(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(x)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(x)


class Encoder(nn.Module):
    """VAEformer encoder."""

    def __init__(
        self,
        in_channels: int = 28,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
        max_patches: int = 10368,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, hidden_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, hidden_dim))
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        self.patch_size = patch_size
        self._cached_pos_embed: torch.Tensor | None = None
        self._cached_pos_grid: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        B, C, h, w = x.shape
        num_patches = h * w

        x_flat = x.flatten(2).transpose(1, 2)

        pos = self.pos_embed
        if pos is not None and pos.shape[1] != num_patches:
            # Interpolate learned positional embeddings to the current patch grid.
            # Source pos_embed is stored as [1, N_src, C]; infer a rectangular grid
            # using the original CRA5-159 aspect ratio (1:2) as a best-effort prior.
            if self._cached_pos_grid != (h, w):
                n_src = pos.shape[1]
                src_h = int(round((n_src / 2.0) ** 0.5))
                src_w = n_src // src_h
                while src_h * src_w != n_src and src_h > 1:
                    src_h -= 1
                    src_w = n_src // src_h
                if src_h * src_w != n_src:
                    src_h, src_w = 72, 144
                pos_shaped = pos.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2).contiguous()
                pos_resized = torch.nn.functional.interpolate(
                    pos_shaped, size=(h, w), mode="bicubic", align_corners=False
                )
                pos_resized = pos_resized.permute(0, 2, 3, 1).reshape(1, h * w, C).contiguous()
                self._cached_pos_embed = pos_resized
                self._cached_pos_grid = (h, w)
            x_flat = x_flat + self._cached_pos_embed.to(x_flat.device)
        elif pos is not None:
            x_flat = x_flat + pos

        for block in self.blocks:
            x_flat = block(x_flat)

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
        B, C, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        for block in self.blocks:
            x_flat = block(x_flat)

        x = x_flat.transpose(1, 2).reshape(B, C, h, w)

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
    CRA5 VAEformer adapted to 28 channels with compression bottleneck.

    Supports two compression strategies:
    1. Reduce hidden_dim directly (random init only, no checkpoint loading)
    2. Keep hidden_dim=1024 for checkpoint compatibility and add a 1x1 conv
       bottleneck that reduces channels to latent_dim (e.g. 256, 192, 128).
    """

    def __init__(
        self,
        in_channels: int = 28,
        out_channels: int = 28,
        hidden_dim: int = 1024,
        latent_dim: int | None = None,
        num_encoder_blocks: int = 4,
        num_decoder_blocks: int = 4,
        num_heads: int = 8,
        patch_size: tuple[int, int] = (11, 10),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim
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

        self.bottleneck = LatentBottleneck(hidden_dim, self.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.encoder(x)
        latent = self.bottleneck.down(hidden)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.bottleneck.up(latent)
        reconstruction = self.backbone.decoder(hidden)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def freeze_backbone(self) -> None:
        """Freeze encoder and decoder parameters, keep bottleneck + projections trainable."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.decoder.parameters():
            param.requires_grad = False

        self.backbone.encoder.patch_embed.proj.weight.requires_grad = True
        self.backbone.encoder.patch_embed.proj.bias.requires_grad = True
        self.backbone.decoder.final.weight.requires_grad = True

        if self.latent_dim != self.hidden_dim:
            for param in self.bottleneck.parameters():
                param.requires_grad = True


def build_cra5_model(
    in_channels: int = 28,
    out_channels: int = 28,
    hidden_dim: int = 1024,
    latent_dim: int | None = None,
    num_encoder_blocks: int = 4,
    num_decoder_blocks: int = 4,
    num_heads: int = 8,
    patch_size: tuple[int, int] = (11, 10),
    freeze_backbone: bool = False,
) -> Cra5Vaeformer28:
    """
    Build CRA5 VAEformer-28 model with configurable compression.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default: 28.
    out_channels : int
        Number of output channels. Default: 28.
    hidden_dim : int
        Transformer hidden dimension. Default: 1024.
        Reduce this (e.g. to 512 or 256) for smaller model at the cost of
        not being able to load the CRA5-159 pretrained checkpoint.
    latent_dim : int or None
        If set and different from hidden_dim, a 1x1 conv bottleneck is
        inserted after encoder / before decoder to project to this many channels.
        This is the recommended way to reach 32x-64x compression while still
        being able to initialize the transformer backbone from a CRA5 checkpoint.
        Examples: 256 (4x smaller latent), 192 (~5.3x), 128 (8x), 96 (~10.7x).
    num_encoder_blocks : int
        Number of encoder transformer blocks. Default: 4.
    num_decoder_blocks : int
        Number of decoder transformer blocks. Default: 4.
    num_heads : int
        Number of attention heads. Default: 8.
    patch_size : tuple[int, int]
        Patch size (height, width). Default: (11, 10).
    freeze_backbone : bool
        Whether to freeze encoder/decoder and only train projections + bottleneck. Default: False.

    Returns
    -------
    Cra5Vaeformer28
        Initialized model.
    """
    model = Cra5Vaeformer28(
        in_channels,
        out_channels,
        hidden_dim,
        latent_dim,
        num_encoder_blocks,
        num_decoder_blocks,
        num_heads,
        patch_size,
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model
