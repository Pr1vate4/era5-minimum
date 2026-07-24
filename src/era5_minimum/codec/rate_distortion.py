from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistortionComponents:
    """Grouped differentiable distortion values."""

    total: torch.Tensor
    surface: torch.Tensor
    pressure: torch.Tensor


def _validate_distortion_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
    latitudes: torch.Tensor,
) -> None:
    if prediction.ndim != 4 or prediction.shape[1] != 28:
        raise ValueError("prediction must have shape [B, 28, H, W]")
    if target.shape != prediction.shape:
        raise ValueError("target must have the same [B, 28, H, W] shape as prediction")
    if validity_mask.shape != prediction.shape:
        raise ValueError(
            "validity_mask must have the same [B, 28, H, W] shape as prediction"
        )
    if latitudes.ndim != 1 or latitudes.numel() != prediction.shape[2]:
        raise ValueError("latitudes must be one-dimensional with one value per grid row")


def _masked_latitude_mean(
    elementwise_loss: torch.Tensor,
    validity_mask: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    elementwise_loss = elementwise_loss.float()
    valid = validity_mask.to(dtype=torch.bool)
    weighted_validity = valid.to(dtype=torch.float32) * latitude_weights.float()
    numerator = (elementwise_loss * weighted_validity).sum()
    denominator = weighted_validity.sum()
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    return numerator / safe_denominator


def grouped_latitude_distortion(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
    *,
    latitudes: torch.Tensor,
    loss_type: str,
    surface_weight: float,
    pressure_weight: float,
) -> DistortionComponents:
    """Compute separate masked distortion for surface and pressure channels."""

    _validate_distortion_inputs(prediction, target, validity_mask, latitudes)
    if loss_type not in {"mse", "l1", "smooth_l1"}:
        raise ValueError("loss_type must be one of: mse, l1, smooth_l1")
    if surface_weight < 0 or pressure_weight < 0:
        raise ValueError("surface_weight and pressure_weight must be non-negative")
    if not math.isfinite(surface_weight) or not math.isfinite(pressure_weight):
        raise ValueError("surface_weight and pressure_weight must be finite")
    if surface_weight + pressure_weight <= 0:
        raise ValueError("surface_weight and pressure_weight sum must be positive")

    valid = validity_mask.to(dtype=torch.bool)
    if not valid[:, :8].any():
        raise ValueError("surface group has no valid values")
    if not valid[:, 8:].any():
        raise ValueError("pressure group has no valid values")

    prediction_float = prediction.float()
    target_float = target.float()
    safe_prediction = torch.where(
        valid,
        prediction_float,
        torch.zeros_like(prediction_float),
    )
    safe_target = torch.where(
        valid,
        target_float,
        torch.zeros_like(target_float),
    )
    if loss_type == "mse":
        elementwise_loss = (safe_prediction - safe_target).square()
    elif loss_type == "l1":
        elementwise_loss = (safe_prediction - safe_target).abs()
    else:
        elementwise_loss = F.smooth_l1_loss(
            safe_prediction,
            safe_target,
            reduction="none",
        )

    latitude_weights = torch.cos(
        torch.deg2rad(latitudes.to(device=prediction.device, dtype=torch.float32))
    ).clamp_min(0.0)
    latitude_weights = latitude_weights.view(1, 1, prediction.shape[2], 1)
    surface = _masked_latitude_mean(
        elementwise_loss[:, :8],
        valid[:, :8],
        latitude_weights,
    )
    pressure = _masked_latitude_mean(
        elementwise_loss[:, 8:],
        valid[:, 8:],
        latitude_weights,
    )
    total = surface_weight * surface + pressure_weight * pressure
    return DistortionComponents(total=total, surface=surface, pressure=pressure)


def quantize_with_uniform_noise(
    latent: torch.Tensor,
    quantization_step: float,
) -> torch.Tensor:
    """Apply a differentiable additive uniform-noise quantization proxy."""

    if not math.isfinite(quantization_step) or quantization_step <= 0:
        raise ValueError("quantization_step must be positive and finite")
    half_step = quantization_step / 2.0
    noise = torch.empty_like(latent).uniform_(-half_step, half_step)
    return latent + noise


class FactorizedLogisticEntropyModel(nn.Module):
    """Estimate independent logistic symbol rates per latent channel."""

    # Keeps value gradients below float16 overflow for mixed-precision training.
    MIN_SCALE = 1e-3

    def __init__(self, channels: int, min_probability: float = 1e-9) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not math.isfinite(min_probability) or not 0 < min_probability <= 1:
            raise ValueError("min_probability must be finite and in (0, 1]")
        self.channels = channels
        self.min_probability = min_probability
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def estimated_bits(
        self,
        values: torch.Tensor,
        quantization_step: float,
    ) -> torch.Tensor:
        """Return elementwise negative log2 logistic interval masses."""

        if not math.isfinite(quantization_step) or quantization_step <= 0:
            raise ValueError("quantization_step must be positive and finite")
        if values.ndim < 2 or values.shape[1] != self.channels:
            raise ValueError(
                f"values must have {self.channels} channels in dimension 1"
            )

        scale_shape = (1, self.channels, *((1,) * (values.ndim - 2)))
        scale = (
            F.softplus(self.log_scale.float())
            .clamp_min(self.MIN_SCALE)
            .clamp_max(1e6)
            .view(scale_shape)
        )
        values_float = values.float()
        half_step = quantization_step / 2.0
        upper = (values_float + half_step) / scale
        lower = (values_float - half_step) / scale
        delta = (quantization_step / scale).clamp_min(torch.finfo(torch.float32).tiny)
        log_probability = (
            F.logsigmoid(upper)
            + F.logsigmoid(-lower)
            + torch.log(-torch.expm1(-delta))
        )
        log_probability = log_probability.clamp(
            min=math.log(self.min_probability),
            max=0.0,
        )
        return -log_probability / math.log(2.0)
