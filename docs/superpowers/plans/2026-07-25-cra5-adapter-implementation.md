# CRA5 Adapter Implementation Plan

**Goal:** Implement the CRA5-159 to ERA5-28 adapter that copies 26 checkpoint channels, learns sst/tcwv boundaries, and produces 32x/64x compression with real bitstreams.

**Status:** in-progress

---

## Task 1: CRA5 Checkpoint Adapter (Weight Slicing)

**Objective:** Load CRA5-159v checkpoint, slice 26 channels by explicit mapping, zero-initialize sst/tcwv boundaries.

**Files:**
- Create: `src/era5_minimum/cra5/adapter.py`
- Create: `tests/test_cra5_adapter.py`

**Steps:**

1. **Write failing tests for state dict adaptation**
   - Test: Load synthetic CRA5-159 state dict, extract 26 channel indices from input/output layers
   - Test: Zero-initialize 2 learned-boundary channels (sst, tcwv)
   - Test: Validate adapted state dict has correct shapes (28 input/output channels)
   - Test: Reject missing keys, shape mismatches, invalid checkpoint

2. **Implement checkpoint adapter**
   ```python
   def adapt_cra5_checkpoint(
       checkpoint_path: Path,
       *,
       mapping: tuple[Cra5ChannelMapping, ...] = CRA5_CHANNEL_MAPPING,
       device: str = "cpu",
   ) -> dict[str, torch.Tensor]:
       """Load CRA5-159 checkpoint and adapt to ERA5-28."""
   ```

3. **Test with real checkpoint structure**
   - Extract actual layer names from downloaded CRA5-159v checkpoint
   - Validate input_proj and output_proj layer shapes
   - Verify encoder/decoder/hyperprior layers remain unchanged

4. **Commit:** `feat(cra5): add checkpoint weight adapter`

---

## Task 2: VAEformer-28 Model Definition

**Objective:** Define 28-channel VAEformer architecture that accepts adapted weights.

**Files:**
- Create: `src/era5_minimum/cra5/model.py`
- Create: `tests/test_cra5_model.py`

**Steps:**

1. **Write failing model construction tests**
   - Test: Build VAEformer-28 with 28 input/output channels
   - Test: Load adapted state dict without shape errors
   - Test: Forward pass with synthetic 28-channel input
   - Test: Encoder output has expected latent dimensions
   - Test: Decoder reconstructs 28 channels

2. **Implement minimal VAEformer-28**
   - Reuse CRA5 encoder/decoder architecture
   - 28-channel input embedding and output projection
   - Keep hyperprior structure
   - No training code yet, just forward pass

3. **Test frozen vs trainable parameters**
   - Mark encoder/decoder/hyperprior as frozen
   - Mark input_proj/output_proj as trainable
   - Validate parameter counts

4. **Commit:** `feat(cra5): add VAEformer-28 model definition`

---

## Task 3: CRA5 Codec Workflow (Encode/Decode/Bitstream)

**Objective:** Integrate adapter + model into codec workflow with real entropy coding.

**Files:**
- Create: `src/era5_minimum/cra5/codec_workflow.py`
- Modify: `src/era5_minimum/codec/workflow.py` (add CRA5 codec type)
- Create: `tests/test_cra5_codec_workflow.py`

**Steps:**

1. **Write failing codec tests**
   - Test: Encode 28-channel input tensor → latent → quantized symbols
   - Test: Decode quantized symbols → reconstruction
   - Test: Exact symbol roundtrip (encode → decode → same quantized latent)
   - Test: Bitstream serialization (Huffman-encoded)
   - Test: Serialized bytes + metadata accounting

2. **Implement encode/decode functions**
   ```python
   def encode_cra5(
       input_tensor: torch.Tensor,
       model: VAEformer28,
       *,
       bits_per_channel: int = 8,
   ) -> tuple[np.ndarray, dict]:
       """Encode 28-channel tensor to quantized latent + metadata."""

   def decode_cra5(
       quantized_latent: np.ndarray,
       model: VAEformer28,
       metadata: dict,
   ) -> torch.Tensor:
       """Decode quantized latent to 28-channel reconstruction."""
   ```

3. **Integrate with existing Huffman codec**
   - Reuse `canonical_huffman.py` for bitstream serialization
   - Record exact serialized bytes
   - Validate roundtrip: encode → serialize → deserialize → decode

4. **Commit:** `feat(cra5): add codec encode/decode workflow`

---

## Task 4: Training Configs (16/32/64/128 samples)

**Objective:** Create reproducible experiment configs for each sample size.

**Files:**
- Create: `configs/cra5/cra5_adapter_n16.yaml`
- Create: `configs/cra5/cra5_adapter_n32.yaml`
- Create: `configs/cra5/cra5_adapter_n64.yaml`
- Create: `configs/cra5/cra5_adapter_n128.yaml`
- Create: `scripts/train_cra5_adapter.py`

**Steps:**

1. **Define config schema**
   ```yaml
   experiment:
     name: cra5-adapter-n16
     seed: 42
   
   data:
     manifest: /path/to/samples.json
     subset_size: 16
     validation_split: 2020
     test_split: 2021
   
   model:
     checkpoint: ~/.cache/era5-minimum/cra5/cra5_159v_150k.pth
     frozen_blocks: [encoder, decoder, hyperprior]
     trainable_blocks: [input_proj, output_proj]
   
   training:
     epochs: 50
     batch_size: 4
     lr: 1e-4
     target_compression: 32x
   
   codec:
     bits_per_channel: 8
     entropy_coding: huffman
   ```

2. **Implement training script**
   - Load manifest, subset timestamps
   - Adapt checkpoint, build model
   - Train input_proj/output_proj only
   - Compute validation metrics every epoch
   - Save best checkpoint + metrics

3. **Add smoke config**
   - `configs/cra5/smoke.yaml` with n=16, 5 epochs, synthetic data

4. **Commit:** `feat(cra5): add training configs and script`

---

## Task 5: Real Runtime Smoke Test

**Objective:** Encode/decode one validation timestamp with actual CRA5 runtime on GPU.

**Files:**
- Create: `scripts/smoke_cra5_runtime.py`
- Create: `tests/test_cra5_runtime_smoke.py`

**Steps:**

1. **Write runtime smoke test**
   - Load one validation timestamp from demo data
   - Normalize, encode with adapted model
   - Serialize bitstream
   - Deserialize, decode, inverse normalize
   - Compute physical RMSE
   - Record: runtime verified, GPU available, bitstream bytes

2. **Implement smoke script**
   ```python
   # scripts/smoke_cra5_runtime.py
   # 1. Check GPU availability
   # 2. Load CRA5-159v checkpoint
   # 3. Adapt to ERA5-28
   # 4. Load one validation timestamp
   # 5. Encode → bitstream → decode
   # 6. Compute metrics
   # 7. Write smoke_cra5_runtime.json
   ```

3. **Test without GPU**
   - Smoke test should work on CPU (slower)
   - Mark GPU-required tests as `@pytest.mark.gpu`

4. **Commit:** `feat(cra5): add runtime smoke test`

---

## Task 6: Experiment Ladder (PCA → ConvAE → CRA5)

**Objective:** Run all three codec paths and compare results.

**Files:**
- Create: `scripts/run_experiment_ladder.py`
- Create: `configs/experiment_ladder.yaml`

**Steps:**

1. **Define ladder config**
   ```yaml
   experiments:
     - name: pca-32x
       type: pca
       config: configs/patch_pca_32x.yaml
     
     - name: convae-smoke
       type: convae
       config: configs/codec/smoke.yaml
     
     - name: cra5-n16
       type: cra5
       config: configs/cra5/cra5_adapter_n16.yaml
     
     - name: cra5-n32
       type: cra5
       config: configs/cra5/cra5_adapter_n32.yaml
   ```

2. **Implement ladder runner**
   - Run each experiment sequentially
   - Collect all local_evaluation.json artifacts
   - Generate comparison table
   - Report: train_size, tensor_CR, serialized_CR, physical_RMSE, runtime

3. **Add stopping rule**
   - Stop ladder when next larger n doesn't improve validation RMSE by >5%
   - Report minimum sample count

4. **Commit:** `feat(experiments): add experiment ladder runner`

---

## Task 7: Documentation and Integration

**Objective:** Document workflow, update DECISIONS.md, verify full pipeline.

**Files:**
- Modify: `docs/DECISIONS.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `Makefile`

**Steps:**

1. **Add Make targets**
   ```makefile
   cra5-smoke: ## Run CRA5 runtime smoke test
   cra5-train-n16: ## Train CRA5 adapter with n=16
   experiment-ladder: ## Run full experiment ladder
   ```

2. **Document decisions**
   - DEC-033: CRA5 adapter copies 26 channels, learns sst/tcwv boundaries
   - DEC-034: Input/output projection trainable, encoder/decoder frozen
   - DEC-035: Minimum sample count determined by validation plateau

3. **Final verification**
   ```bash
   make verify
   make cra5-smoke
   make experiment-ladder
   ```

4. **Commit:** `docs(cra5): document adapter workflow and decisions`

---

## Success Criteria

- [ ] CRA5-159v checkpoint successfully adapted to ERA5-28
- [ ] 26 channels copied by explicit index, 2 learned from zero
- [ ] VAEformer-28 model loads adapted weights without errors
- [ ] Encode → bitstream → decode produces exact symbol roundtrip
- [ ] Real serialized compression ratio computed from bitstream bytes
- [ ] Training configs for n=16/32/64/128 with deterministic seeds
- [ ] Runtime smoke test passes on CPU and GPU
- [ ] Experiment ladder compares PCA, ConvAE, CRA5 on same metrics
- [ ] Minimum sample count identified by validation plateau
- [ ] All tests passing: `make verify` green
- [ ] Full pipeline reproducible: one command from config to metrics

