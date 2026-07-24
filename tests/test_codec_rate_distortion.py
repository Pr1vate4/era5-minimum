from __future__ import annotations

import pytest
import torch

from era5_minimum.codec import (
    DistortionComponents,
    FactorizedLogisticEntropyModel,
    grouped_latitude_distortion,
    quantize_with_uniform_noise,
)


def _distortion(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
    *,
    latitudes: torch.Tensor,
    loss_type: str = "mse",
    surface_weight: float = 0.5,
    pressure_weight: float = 0.5,
) -> DistortionComponents:
    return grouped_latitude_distortion(
        prediction,
        target,
        validity_mask,
        latitudes=latitudes,
        loss_type=loss_type,
        surface_weight=surface_weight,
        pressure_weight=pressure_weight,
    )


def test_grouped_distortion_weights_surface_and_pressure_equally() -> None:
    target = torch.zeros(1, 28, 2, 2)
    prediction = target.clone()
    prediction[:, :8] = 1.0
    prediction[:, 8:] = 2.0

    loss = _distortion(
        prediction,
        target,
        torch.ones_like(target),
        latitudes=torch.tensor([0.0, 0.0]),
    )

    assert isinstance(loss, DistortionComponents)
    assert loss.surface.item() == pytest.approx(1.0)
    assert loss.pressure.item() == pytest.approx(4.0)
    assert loss.total.item() == pytest.approx(2.5)


def test_grouped_distortion_applies_cosine_latitude_weights_in_degrees() -> None:
    target = torch.zeros(1, 28, 2, 1)
    prediction = target.clone()
    prediction[:, :8, 0] = 1.0
    prediction[:, :8, 1] = 3.0

    loss = _distortion(
        prediction,
        target,
        torch.ones_like(target),
        latitudes=torch.tensor([0.0, 60.0]),
        surface_weight=1.0,
        pressure_weight=0.0,
    )

    assert loss.surface.item() == pytest.approx((1.0 + 0.5 * 9.0) / 1.5)
    assert loss.total.item() == pytest.approx(loss.surface.item())


def test_grouped_distortion_excludes_invalid_values_from_both_sides() -> None:
    target = torch.zeros(1, 28, 1, 2)
    prediction = target.clone()
    prediction[:, :8, 0, 0] = 1.0
    prediction[:, :8, 0, 1] = 100.0
    mask = torch.ones_like(target)
    mask[:, :8, 0, 1] = 0.0

    loss = _distortion(
        prediction,
        target,
        mask,
        latitudes=torch.tensor([0.0]),
        surface_weight=1.0,
        pressure_weight=0.0,
    )

    assert loss.surface.item() == pytest.approx(1.0)
    assert loss.total.item() == pytest.approx(1.0)


def test_grouped_distortion_does_not_clip_small_valid_weight_denominator() -> None:
    target = torch.zeros(1, 28, 1, 1)
    prediction = target.clone()
    prediction[:, 0] = 2.0
    mask = torch.zeros_like(target)
    mask[:, 0] = 1.0

    loss = _distortion(
        prediction,
        target,
        mask,
        latitudes=torch.tensor([89.0]),
        surface_weight=1.0,
        pressure_weight=0.0,
    )

    assert loss.surface.item() == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("loss_type", "expected"),
    [
        ("mse", 4.0),
        ("l1", 2.0),
        ("smooth_l1", 1.5),
    ],
)
def test_grouped_distortion_supports_configured_loss_types(
    loss_type: str,
    expected: float,
) -> None:
    target = torch.zeros(1, 28, 1, 1)
    prediction = target.clone()
    prediction[:, :8] = 2.0

    loss = _distortion(
        prediction,
        target,
        torch.ones_like(target),
        latitudes=torch.tensor([0.0]),
        loss_type=loss_type,
        surface_weight=1.0,
        pressure_weight=0.0,
    )

    assert loss.surface.item() == pytest.approx(expected)
    assert loss.total.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_type": "mae"}, "loss_type"),
        ({"surface_weight": -0.1}, "non-negative"),
        ({"pressure_weight": -0.1}, "non-negative"),
        ({"surface_weight": 0.0, "pressure_weight": 0.0}, "sum"),
    ],
)
def test_grouped_distortion_validates_configuration(
    overrides: dict[str, str | float],
    message: str,
) -> None:
    values = torch.zeros(1, 28, 1, 1)
    kwargs: dict[str, object] = {
        "latitudes": torch.tensor([0.0]),
        "loss_type": "mse",
        "surface_weight": 0.5,
        "pressure_weight": 0.5,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        grouped_latitude_distortion(values, values, torch.ones_like(values), **kwargs)


def test_grouped_distortion_validates_tensor_shapes() -> None:
    values = torch.zeros(1, 28, 2, 1)

    with pytest.raises(ValueError, match=r"\[B, 28, H, W\]"):
        grouped_latitude_distortion(
            values[:, :27],
            values[:, :27],
            torch.ones_like(values[:, :27]),
            latitudes=torch.tensor([0.0, 10.0]),
            loss_type="mse",
            surface_weight=0.5,
            pressure_weight=0.5,
        )

    with pytest.raises(ValueError, match="latitudes"):
        _distortion(
            values,
            values,
            torch.ones_like(values),
            latitudes=torch.tensor([0.0]),
        )


def test_factorized_logistic_rate_returns_finite_bits_and_gradients() -> None:
    model = FactorizedLogisticEntropyModel(channels=3, min_probability=1e-9)
    latent = torch.randn(2, 3, 4, 5, requires_grad=True)

    bits = model.estimated_bits(latent, quantization_step=0.25)
    bits.mean().backward()

    assert bits.shape == latent.shape
    assert torch.isfinite(bits).all()
    assert (bits >= 0).all()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert model.log_scale.grad is not None
    assert torch.isfinite(model.log_scale.grad).all()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FactorizedLogisticEntropyModel(channels=0), "channels"),
        (
            lambda: FactorizedLogisticEntropyModel(channels=1, min_probability=0.0),
            "min_probability",
        ),
        (
            lambda: FactorizedLogisticEntropyModel(channels=1, min_probability=1.1),
            "min_probability",
        ),
    ],
)
def test_entropy_model_validates_configuration(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_entropy_model_validates_step_and_channel_count() -> None:
    model = FactorizedLogisticEntropyModel(channels=2)

    with pytest.raises(ValueError, match="quantization_step"):
        model.estimated_bits(torch.zeros(1, 2, 1, 1), quantization_step=0.0)

    with pytest.raises(ValueError, match="channels"):
        model.estimated_bits(torch.zeros(1, 3, 1, 1), quantization_step=1.0)


def test_uniform_noise_is_seeded_bounded_and_preserves_gradients() -> None:
    latent = torch.zeros(2, 3, 4, requires_grad=True)

    torch.manual_seed(1234)
    first = quantize_with_uniform_noise(latent, quantization_step=0.5)
    torch.manual_seed(1234)
    second = quantize_with_uniform_noise(latent, quantization_step=0.5)

    assert torch.equal(first, second)
    assert (first >= -0.25).all()
    assert (first <= 0.25).all()
    first.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))


def test_uniform_noise_requires_positive_step() -> None:
    with pytest.raises(ValueError, match="quantization_step"):
        quantize_with_uniform_noise(torch.zeros(1), quantization_step=0.0)
