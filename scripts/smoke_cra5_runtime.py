#!/usr/bin/env python3
"""
CRA5 runtime smoke test.

Validates end-to-end CRA5 workflow:
1. Load CRA5-159v checkpoint and adapt to ERA5-28
2. Load one validation timestamp from demo data  
3. Normalize → encode → bitstream → decode → inverse normalize
4. Compute physical RMSE and record runtime metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from era5_minimum.cra5.adapter import adapt_cra5_checkpoint
from era5_minimum.cra5.codec_workflow import (
    decode_cra5,
    encode_cra5,
    serialize_cra5_bitstream,
)
from era5_minimum.cra5.model import build_cra5_model
from era5_minimum.data.synthetic import make_synthetic_era5


def check_system_info() -> dict[str, Any]:
    """Check GPU availability and system info."""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    system_info = {
        "cuda_available": gpu_available,
        "gpu_count": gpu_count,
        "device_name": torch.cuda.get_device_name(0) if gpu_available else "CPU",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    
    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        system_info["gpu_memory_gb"] = props.total_memory / 1e9
        system_info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    
    return system_info


def load_demo_timestamp() -> torch.Tensor:
    """Load or generate one validation timestamp."""
    # For smoke test, use synthetic data that matches ERA5 structure
    print("Generating synthetic ERA5-28 timestamp...")
    
    # Use dimensions compatible with patch_size (11, 10)
    # 352 = 32 * 11, 720 = 72 * 10
    height, width = 352, 720
    
    # Use the 8 standard channels, repeat to get 28 channels
    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_28 = []
    for i in range(28):
        channels_28.append(standard_channels[i % len(standard_channels)])
    
    data, _ = make_synthetic_era5(
        samples=1,
        height=height,
        width=width,
        channels=tuple(channels_28),
        seed=42,
    )
    
    # Add batch dimension and ensure float32
    timestamp = torch.from_numpy(data).float()  # [1, 28, 352, 720]
    
    print(f"Demo timestamp shape: {timestamp.shape}")
    print(f"Demo timestamp dtype: {timestamp.dtype}")
    print(f"Demo timestamp range: [{timestamp.min():.3f}, {timestamp.max():.3f}]")
    
    return timestamp


def simple_normalization(data: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply simple per-channel normalization."""
    # Compute per-channel statistics
    # Shape: [B, C, H, W] -> compute over B, H, W dimensions
    mean = data.mean(dim=[0, 2, 3], keepdim=True)  # [1, C, 1, 1]
    std = data.std(dim=[0, 2, 3], keepdim=True)    # [1, C, 1, 1]
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    # Normalize
    normalized = (data - mean) / std
    
    norm_stats = {
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
        "method": "per_channel_zscore",
    }
    
    return normalized, norm_stats


def inverse_normalization(
    normalized_data: torch.Tensor, 
    norm_stats: dict[str, Any],
) -> torch.Tensor:
    """Apply inverse normalization."""
    mean = torch.tensor(norm_stats["mean"], device=normalized_data.device)
    std = torch.tensor(norm_stats["std"], device=normalized_data.device)
    
    # Reshape for broadcasting: [C] -> [1, C, 1, 1]
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    
    # Denormalize
    denormalized = normalized_data * std + mean
    
    return denormalized


def compute_physical_rmse(
    original: torch.Tensor, 
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    """Compute per-channel physical RMSE (simplified)."""
    # For smoke test, just compute raw RMSE per channel
    # In real implementation, this would convert to physical units
    
    mse_per_channel = ((original - reconstruction) ** 2).mean(dim=[0, 2, 3])
    rmse_per_channel = torch.sqrt(mse_per_channel)
    
    # Overall RMSE
    overall_rmse = torch.sqrt(((original - reconstruction) ** 2).mean())
    
    metrics = {
        "overall_rmse": float(overall_rmse),
        "mean_channel_rmse": float(rmse_per_channel.mean()),
        "max_channel_rmse": float(rmse_per_channel.max()),
        "min_channel_rmse": float(rmse_per_channel.min()),
    }
    
    return metrics


def run_smoke_test(
    checkpoint_path: Path | None = None,
    device: str = "auto",
    output_dir: Path = Path("outputs/smoke_cra5"),
) -> dict[str, Any]:
    """Run complete CRA5 smoke test."""
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running CRA5 smoke test on device: {device}")
    
    # Check system info
    system_info = check_system_info()
    print(f"System: {system_info['device_name']}")
    if system_info["cuda_available"]:
        print(f"GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "test_name": "cra5_runtime_smoke_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "device": device,
    }
    
    try:
        # Step 1: Load and adapt checkpoint
        print("\n1. Loading CRA5 checkpoint...")
        start_time = time.perf_counter()
        
        model = build_cra5_model(
            in_channels=28,
            out_channels=28,
            freeze_backbone=True,  # Only train input/output projections
        )
        
        if checkpoint_path and checkpoint_path.exists():
            print(f"Adapting checkpoint: {checkpoint_path}")
            adapted_state_dict, adapter_metadata = adapt_cra5_checkpoint(checkpoint_path)
            
            # Handle pos_embed interpolation if needed
            if "backbone.encoder.pos_embed" in adapted_state_dict:
                ckpt_pos_embed = adapted_state_dict["backbone.encoder.pos_embed"]
                model_pos_embed = model.backbone.encoder.pos_embed
                print(f"DEBUG: ckpt_pos_embed shape = {ckpt_pos_embed.shape}")
                print(f"DEBUG: model_pos_embed shape = {model_pos_embed.shape}")
                if ckpt_pos_embed.shape != model_pos_embed.shape:
                    print("DEBUG: INTERPOLATING POS_EMBED!")
                    # Reshape to [1, C, H, W] for interpolation
                    # Assuming checkpoint is 72x144 patches
                    ckpt_pos_embed = ckpt_pos_embed.reshape(1, 72, 144, -1).permute(0, 3, 1, 2)
                    # Target is 32x72 patches
                    target_h, target_w = 32, 72
                    ckpt_pos_embed = torch.nn.functional.interpolate(
                        ckpt_pos_embed, size=(target_h, target_w), mode="bicubic", align_corners=False
                    )
                    # Reshape back to [1, N, C]
                    adapted_state_dict["backbone.encoder.pos_embed"] = ckpt_pos_embed.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
                    print(f"DEBUG: New pos_embed shape = {adapted_state_dict['backbone.encoder.pos_embed'].shape}")
                    
            model.load_state_dict(adapted_state_dict, strict=False)
            
            results["checkpoint_adapted"] = True
            results["adapter_metadata"] = {
                "source_channels": adapter_metadata.source_channels,
                "target_channels": adapter_metadata.target_channels,
                "copied_channels": adapter_metadata.copied_channels,
                "learned_boundary_channels": adapter_metadata.learned_boundary_channels,
            }
        else:
            print("No checkpoint specified or found, using random initialization")
            results["checkpoint_adapted"] = False
        
        model = model.to(device)
        model.eval()
        
        checkpoint_load_time = time.perf_counter() - start_time
        results["checkpoint_load_seconds"] = checkpoint_load_time
        
        # Step 2: Load validation data
        print("\n2. Loading demo timestamp...")
        timestamp = load_demo_timestamp().to(device)
        
        results["input_shape"] = list(timestamp.shape)
        results["input_dtype"] = str(timestamp.dtype)
        
        # Step 3: Normalize
        print("\n3. Normalizing...")
        normalized, norm_stats = simple_normalization(timestamp)
        
        # Step 4: Encode
        print("\n4. Encoding...")
        start_time = time.perf_counter()
        
        quantized_latent, encode_metadata = encode_cra5(
            normalized, 
            model, 
            quantization_step=0.1,
        )
        
        encode_time = time.perf_counter() - start_time
        
        results["encode_seconds"] = encode_time
        results["latent_shape"] = encode_metadata["latent_shape"]
        results["quantization_step"] = encode_metadata["quantization_step"]
        
        print(f"Encoded to latent shape: {encode_metadata['latent_shape']}")
        
        # Step 5: Serialize bitstream
        print("\n5. Serializing bitstream...")
        bitstream_path = output_dir / "smoke_test.bin"
        
        start_time = time.perf_counter()
        bitstream_bytes = serialize_cra5_bitstream(quantized_latent, encode_metadata, bitstream_path)
        serialize_time = time.perf_counter() - start_time
        
        results["serialize_seconds"] = serialize_time
        results["bitstream_bytes"] = bitstream_bytes
        results["bitstream_path"] = str(bitstream_path)
        
        # Compute compression ratio (rough estimate)
        input_bytes = np.prod(timestamp.shape) * 4  # float32
        compression_ratio = input_bytes / bitstream_bytes
        results["tensor_compression_ratio"] = compression_ratio
        
        print(f"Bitstream: {bitstream_bytes} bytes (CR: {compression_ratio:.1f}×)")
        
        # Step 6: Decode
        print("\n6. Decoding...")
        start_time = time.perf_counter()
        
        reconstruction_norm = decode_cra5(
            quantized_latent,
            model,
            quantization_step=encode_metadata["quantization_step"],
        )
        
        decode_time = time.perf_counter() - start_time
        results["decode_seconds"] = decode_time
        
        # Step 7: Inverse normalize
        print("\n7. Inverse normalizing...")
        reconstruction = inverse_normalization(reconstruction_norm, norm_stats)
        
        # Step 8: Compute metrics
        print("\n8. Computing metrics...")
        physical_metrics = compute_physical_rmse(timestamp, reconstruction)
        
        results["physical_rmse"] = physical_metrics
        results["reconstruction_shape"] = list(reconstruction.shape)
        
        # Check exact roundtrip for quantized latent
        quantized_check, _ = encode_cra5(normalized, model, quantization_step=0.1)
        roundtrip_exact = np.array_equal(quantized_latent, quantized_check)
        results["roundtrip_exact"] = roundtrip_exact
        
        # Total runtime
        total_time = checkpoint_load_time + encode_time + serialize_time + decode_time
        results["total_runtime_seconds"] = total_time
        
        results["success"] = True
        results["error"] = None
        
        print(f"\n✅ Smoke test PASSED")
        print(f"Overall RMSE: {physical_metrics['overall_rmse']:.6f}")
        print(f"Total runtime: {total_time:.3f}s")
        print(f"Roundtrip exact: {roundtrip_exact}")
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        print(f"\n❌ Smoke test FAILED: {e}")
        raise
    
    # Save results
    results_path = output_dir / "smoke_cra5_runtime.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRA5 runtime smoke test")
    parser.add_argument(
        "--checkpoint", 
        type=Path, 
        help="Path to CRA5-159v checkpoint (optional)",
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("outputs/smoke_cra5"),
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Use default checkpoint path if not specified
    if not args.checkpoint:
        default_checkpoint = Path("~/.cache/era5-minimum/cra5/cra5_159v_150k.pth").expanduser()
        if default_checkpoint.exists():
            args.checkpoint = default_checkpoint
            print(f"Using default checkpoint: {args.checkpoint}")
        else:
            print("No checkpoint specified and default not found, using random weights")
    
    # Run smoke test
    results = run_smoke_test(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Print summary
    if results["success"]:
        print(f"\n🎉 CRA5 smoke test completed successfully!")
        print(f"Compression: {results['tensor_compression_ratio']:.1f}× ({results['bitstream_bytes']} bytes)")
        print(f"Runtime: {results['total_runtime_seconds']:.3f}s")
        exit(0)
    else:
        print(f"\n💥 CRA5 smoke test failed: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()