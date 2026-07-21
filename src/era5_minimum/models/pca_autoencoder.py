from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAFitResult:
    runtime_seconds: float
    retained_variance_fraction: float


class PCAAutoencoder:
    """Fast linear autoencoder used as a CPU-safe end-to-end baseline."""

    def __init__(self, latent_dim: int) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.input_shape_: tuple[int, ...] | None = None

    def fit(self, data: np.ndarray) -> PCAFitResult:
        if data.ndim != 4:
            raise ValueError("data must have shape [time, channel, latitude, longitude]")
        flat = data.reshape(data.shape[0], -1).astype(np.float64)
        if self.latent_dim >= min(flat.shape):
            raise ValueError(
                f"latent_dim={self.latent_dim} must be smaller than min(samples, features)={min(flat.shape)}"
            )
        started = time.perf_counter()
        self.mean_ = flat.mean(axis=0, keepdims=True)
        centered = flat - self.mean_
        # Compute PCA through the sample-space Gram matrix. This is much faster
        # than a full SVD when features greatly outnumber training samples.
        gram = centered @ centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        selected_values = eigenvalues[: self.latent_dim]
        selected_vectors = eigenvectors[:, : self.latent_dim]
        scale = np.sqrt(np.maximum(selected_values, 1e-12))[None, :]
        components = (centered.T @ selected_vectors) / scale
        self.components_ = components.T.astype(np.float32)
        self.input_shape_ = tuple(data.shape[1:])
        total_variance = float(np.sum(eigenvalues))
        kept_variance = float(np.sum(selected_values))
        return PCAFitResult(
            runtime_seconds=time.perf_counter() - started,
            retained_variance_fraction=kept_variance / max(total_variance, 1e-12),
        )

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.components_ is None or self.input_shape_ is None:
            raise RuntimeError("PCAAutoencoder must be fitted before use")

    def encode(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        flat = data.reshape(data.shape[0], -1)
        return ((flat - self.mean_) @ self.components_.T).astype(np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._check_fitted()
        flat = latent @ self.components_ + self.mean_
        return flat.reshape((latent.shape[0], *self.input_shape_)).astype(np.float32)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def tensor_compression_ratio(self) -> float:
        self._check_fitted()
        input_values = int(np.prod(self.input_shape_))
        return float(input_values / self.latent_dim)
