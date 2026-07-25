#!/usr/bin/env python3
"""Quick demo: CRA5 adapter compresses real ERA5 data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.era5_minimum.cra5 import (
    adapt_cra5_checkpoint,
    build_cra5_model,
    decode_cra5,
    encode_and_serialize_cra5,
    encode_cra5,
)
from src.era5_minimum.data.channel_spec import CHANNEL_NAMES


def main() -> None:
    """Run quick compression demo."""
    print("=" * 60)
    print("CRA5 Adapter Compression Demo")
    print("=" * 60)

    # 1. Load adapted checkpoint
    print("\n[1/5] Loading CRA5-159v checkpoint...")
    checkpoint_path = Path.home() / ".cache/era5-minimum/cra5/cra5_159v_150k.pth"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Run: python scripts/fetch_cra5_checkpoint.py --download")
        return

    start = time.perf_counter()
    adapted_state, metadata = adapt_cra5_checkpoint(checkpoint_path, device="cpu")
    print(f"✅ Adapted checkpoint: {metadata.copied_channels} copied, {metadata.learned_boundary_channels} learned")
    print(f"   Time: {time.perf_counter() - start:.2f}s")

    # 2. Build model and load weights
    print("\n[2/5] Building VAEformer-28 model...")
    start = time.perf_counter()
    model = build_cra5_model()
    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    model.eval()
    print(f"✅ Model ready (missing: {len(missing)}, unexpected: {len(unexpected)})")
    print(f"   Time: {time.perf_counter() - start:.2f}s")

    # 3. Load demo data
    print("\n[3/5] Loading demo ERA5 timestamp...")
    demo_dir = Path("data/era5_28ch_demo")
    if not demo_dir.exists():
        print(f"❌ Demo data not found: {demo_dir}")
        return

    ds = xr.open_zarr(demo_dir / "data.zarr")
    # Take first timestamp
    data_sample = ds.isel(time=0)

    # Stack channels in canonical order
    arrays = []
    for ch in CHANNEL_NAMES:
        arrays.append(data_sample[ch].values)
    data_np = np.stack(arrays, axis=0)  # [28, H, W]

    # Add batch dimension
    data_tensor = torch.from_numpy(data_np).unsqueeze(0).float()  # [1, 28, H, W]
    
    print(f"✅ Loaded data: shape={data_tensor.shape}, dtype={data_tensor.dtype}")
    print(f"   Shape: {list(data_tensor.shape)}")
    print(f"   Value range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")

    # 4. Encode and compress
    print("\n[4/5] Encoding and compressing...")
    output_path = Path("/tmp/cra5_demo.bitstream")
    start = time.perf_counter()
    
    # First encode to get latent
    quantized, enc_metadata = encode_cra5(data_tensor, model, quantization_step=0.1)
    encode_time = time.perf_counter() - start
    
    # Serialize to bitstream
    from src.era5_minimum.cra5.codec_workflow import serialize_cra5_bitstream
    bitstream_bytes = serialize_cra5_bitstream(quantized, enc_metadata, output_path)
    
    print(f"✅ Encoded and compressed")
    print(f"   Encode time: {encode_time:.3f}s")
    print(f"   Latent shape: {quantized.shape}")
    print(f"   Bitstream size: {bitstream_bytes:,} bytes")

    # Calculate compression ratio
    original_bytes = data_tensor.numel() * 4  # float32
    tensor_cr = data_tensor.numel() / quantized.size
    serialized_cr = original_bytes / bitstream_bytes

    print(f"   Original size: {original_bytes:,} bytes")
    print(f"   Tensor compression: {tensor_cr:.1f}x")
    print(f"   Serialized compression: {serialized_cr:.1f}x")

    # 5. Decode
    print("\n[5/5] Decoding...")
    start = time.perf_counter()
    reconstruction = decode_cra5(quantized, model, quantization_step=0.1)
    decode_time = time.perf_counter() - start
    print(f"✅ Decoded: {reconstruction.shape}")
    print(f"   Decode time: {decode_time:.3f}s")

    # Calculate reconstruction error
    mse = torch.mean((reconstruction - data_tensor) ** 2).item()
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(data_tensor.max().item() / rmse) if rmse > 0 else float('inf')

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Original shape:       {list(data_tensor.shape)}")
    print(f"Latent shape:         {list(quantized.shape)}")
    print(f"Original bytes:       {original_bytes:,}")
    print(f"Bitstream bytes:      {bitstream_bytes:,}")
    print(f"Tensor CR:            {tensor_cr:.1f}x")
    print(f"Serialized CR:        {serialized_cr:.1f}x")
    print(f"RMSE:                 {rmse:.6f}")
    print(f"PSNR:                 {psnr:.2f} dB")
    print(f"Encode time:          {encode_time:.3f}s")
    print(f"Decode time:          {decode_time:.3f}s")
    print("=" * 60)

    print(f"\n✅ Demo complete! Bitstream saved to: {output_path}")


if __name__ == "__main__":
    main()
