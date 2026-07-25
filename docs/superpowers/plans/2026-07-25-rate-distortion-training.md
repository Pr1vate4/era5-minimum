# ML-001 Rate-Distortion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the smoke codec's plain masked MSE training with a configurable latitude-weighted rate-distortion objective and a trainable factorized entropy-rate model.

**Architecture:** Add focused PyTorch primitives in `codec/rate_distortion.py`: additive uniform quantization noise, a per-latent-channel factorized logistic model, and grouped surface/pressure distortion. The workflow will explicitly encode, apply the training quantization proxy, decode, optimize model plus entropy parameters, and record every loss component without changing actual bitstream ratio semantics.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, YAML.

---

### Task 1: Rate-distortion primitives

**Files:**
- Create: `src/era5_minimum/codec/rate_distortion.py`
- Create: `tests/test_codec_rate_distortion.py`
- Modify: `src/era5_minimum/codec/__init__.py`

- [ ] **Step 1: Write failing grouped-distortion tests**

```python
def test_grouped_distortion_weights_surface_and_pressure_equally() -> None:
    target = torch.zeros(1, 28, 2, 2)
    prediction = target.clone()
    prediction[:, :8] = 1.0
    prediction[:, 8:] = 2.0
    mask = torch.ones_like(target)
    loss = grouped_latitude_distortion(
        prediction,
        target,
        mask,
        latitudes=torch.tensor([0.0, 0.0]),
        loss_type="mse",
        surface_weight=0.5,
        pressure_weight=0.5,
    )
    assert loss.surface.item() == pytest.approx(1.0)
    assert loss.pressure.item() == pytest.approx(4.0)
    assert loss.total.item() == pytest.approx(2.5)
```

Run: `pytest tests/test_codec_rate_distortion.py -q`
Expected: FAIL because `era5_minimum.codec.rate_distortion` does not exist.

- [ ] **Step 2: Implement grouped latitude-weighted distortion**

Implement `DistortionComponents`, masked weighted reduction, and `grouped_latitude_distortion` for `mse`, `l1`, and `smooth_l1`. Latitude weights are `cos(latitude)`, invalid values are excluded from numerator and denominator, channels `0:8` are surface, and channels `8:28` are pressure.

- [ ] **Step 3: Add failing entropy-rate tests**

```python
def test_factorized_logistic_rate_returns_finite_bits_and_gradients() -> None:
    model = FactorizedLogisticEntropyModel(channels=3)
    latent = torch.randn(2, 3, 4, 5, requires_grad=True)
    bits = model.estimated_bits(latent, quantization_step=0.25)
    bits.mean().backward()
    assert torch.isfinite(bits).all()
    assert latent.grad is not None
    assert model.log_scale.grad is not None
```

Run: `pytest tests/test_codec_rate_distortion.py -q`
Expected: FAIL because the entropy model is absent.

- [ ] **Step 4: Implement factorized logistic model and training quantization**

Use one trainable `log_scale` per latent channel. Estimate symbol probability as logistic CDF mass over `[value-step/2, value+step/2]`, clamp by configured minimum probability, and return `-log2(probability)`. `quantize_with_uniform_noise` adds reproducible `U(-step/2, step/2)` noise during training.

- [ ] **Step 5: Run primitive tests**

Run: `pytest tests/test_codec_rate_distortion.py -q`
Expected: all tests PASS.

### Task 2: Training workflow integration

**Files:**
- Modify: `src/era5_minimum/codec/workflow.py`
- Modify: `tests/test_codec_smoke.py`

- [ ] **Step 1: Write failing artifact/history test**

Extend the smoke config with `entropy` and `loss`, then assert every history row contains `loss`, `distortion`, `surface_distortion`, `pressure_distortion`, and `estimated_rate_bits_per_input_value`. Assert checkpoint metadata and `run_summary.json` preserve the resolved loss/entropy configuration.

Run: `pytest tests/test_codec_smoke.py::test_codec_smoke_records_rate_distortion_training -q`
Expected: FAIL because the fields are absent.

- [ ] **Step 2: Integrate explicit encode/quantize/decode training**

Create `FactorizedLogisticEntropyModel(latent_channels)`, include its parameters in the optimizer and parameter-limit count, and replace `model(batch_inputs)` with:

```python
latent = model.encode(batch_inputs)
noisy_latent = quantize_with_uniform_noise(latent, quantization_step)
outputs = model.decode(noisy_latent, output_size=batch_inputs.shape[-2:])
distortion = grouped_latitude_distortion(...)
rate_bpv = entropy_model.estimated_bits(noisy_latent, quantization_step).sum() / batch_inputs.numel()
loss = distortion.total + rate_lambda * rate_bpv
```

Pass fixed grid latitudes into training. Save entropy state/config in the checkpoint, write all components to JSONL, and expose final components in summary metadata.

- [ ] **Step 3: Run smoke and CLI regression tests**

Run: `pytest tests/test_codec_smoke.py tests/test_codec_cli_roundtrip.py tests/test_train_codec_script.py -q`
Expected: all tests PASS and exact symbol roundtrip remains true.

### Task 3: Configs, decision record, and verification

**Files:**
- Modify: `configs/codec/smoke.yaml`
- Modify: `configs/codec/codec_0p5_32x.yaml`
- Modify: `configs/codec/codec_0p5_64x.yaml`
- Modify: `configs/codec/codec_0p25_32x.yaml`
- Modify: `configs/codec/codec_0p25_64x.yaml`
- Modify: `docs/DECISIONS.md`

- [ ] **Step 1: Add explicit entropy and loss configuration**

Each config must define `entropy.model_type: factorized_logistic`, `entropy.min_probability`, and a `loss` block containing `type`, `latitude_weighting`, `surface_weight`, `pressure_weight`, and `rate_lambda`. Keep actual Huffman bitstream metrics separate from the learned estimated-rate proxy.

- [ ] **Step 2: Document the rate-model decision**

Record that the factorized logistic model drives training only, while canonical Huffman remains the actual serialized coder. Document that estimated rate and actual serialized rate are separate and may diverge.

- [ ] **Step 3: Run full verification**

Run: `make verify`
Expected: pytest passes, `demo/mock` artifact validation succeeds, and the MVP experiment command succeeds.

- [ ] **Step 4: Run codec smoke and inspect outputs**

Run: `python scripts/train_codec.py --config configs/codec/smoke.yaml`
Expected: `training_history.jsonl` contains finite rate-distortion components, checkpoint contains entropy state, and `run_summary.json` keeps honest actual and estimated rate fields.

- [ ] **Step 5: Commit**

```bash
git add src/era5_minimum/codec tests/test_codec_rate_distortion.py tests/test_codec_smoke.py configs/codec docs/DECISIONS.md docs/superpowers/plans/2026-07-25-rate-distortion-training.md
git commit -m "feat(codec): train with rate-distortion objective"
```
