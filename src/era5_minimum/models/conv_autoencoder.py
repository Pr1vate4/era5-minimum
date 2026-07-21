from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Small deterministic convolutional baseline for 32x/64x tensor compression."""

    def __init__(self, in_channels: int, latent_channels: int = 16) -> None:
        super().__init__()
        if in_channels < 1 or latent_channels < 1:
            raise ValueError("channel counts must be positive")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def tensor_compression_ratio(self, sample: torch.Tensor) -> float:
        """Return input-value count divided by latent-value count."""
        latent = self.encode(sample)
        return float(sample.numel() / latent.numel())
