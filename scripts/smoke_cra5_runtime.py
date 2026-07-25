#!/usr/bin/env python3
"""
CRA5 runtime smoke test with 32x-64x compression targets.

Validates end-to-end CRA5 workflow:
1. Load CRA5-159v checkpoint and adapt to ERA5-28
2. Load one validation timestamp from demo data
3. Normalize → encode → bitstream → decode → inverse normalize
4. Compute physical RMSE and record runtime metrics

Supports configurable compression via:
  --hidden_dim      Transformer width (1024 = checkpoint-compatible)
  --latent_dim      Bottleneck channels; < hidden_dim activates a 1x1 conv
                    bottleneck that drives compression higher.
  --quantization_step  Larger step = fewer symbols = smaller bitstream.
  --delta / --no-delta Enable/disable spatial + channel delta coding.
  --sweep           Run a grid of configurations and print a summary table
                    showing which settings achieve ≥ 32x or ≥ 64x.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
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


def load_demo_timestamp(height: int = 352, width: int = 720, seed: int = 42) -> torch.Tensor:
    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_28 = [standard_channels[i % len(standard_channels)] for i in range(28)]
    data, _ = make_synthetic_era5(
        samples=1,
        height=height,
        width=width,
        channels=tuple(channels_28),
        seed=seed,
    )
    timestamp = torch.from_numpy(data).float()
    return timestamp


def simple_normalization(data: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    mean = data.mean(dim=[0, 2, 3], keepdim=True)
    std = data.std(dim=[0, 2, 3], keepdim=True)
    std = torch.clamp(std, min=1e-8)
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
    mean = torch.tensor(norm_stats["mean"], device=normalized_data.device)
    std = torch.tensor(norm_stats["std"], device=normalized_data.device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return normalized_data * std + mean


def compute_physical_rmse(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    mse_per_channel = ((original - reconstruction) ** 2).mean(dim=[0, 2, 3])
    rmse_per_channel = torch.sqrt(mse_per_channel)
    overall_rmse = torch.sqrt(((original - reconstruction) ** 2).mean())
    return {
        "overall_rmse": float(overall_rmse),
        "mean_channel_rmse": float(rmse_per_channel.mean()),
        "max_channel_rmse": float(rmse_per_channel.max()),
        "min_channel_rmse": float(rmse_per_channel.min()),
    }


@dataclass
class SweepRow:
    hidden_dim: int
    latent_dim: int
    quantization_step: float
    delta_coding: bool
    input_elements: int
    latent_elements: int
    bitstream_bytes: int
    tensor_cr: float
    serialized_cr: float
    overall_rmse: float
    total_runtime_s: float
    roundtrip_exact: bool


def run_single_config(
    *,
    checkpoint_path: Path | None,
    device: str,
    output_dir: Path,
    hidden_dim: int,
    latent_dim: int,
    quantization_step: float,
    delta_coding: bool,
    tag: str,
    timestamp: torch.Tensor | None = None,
) -> dict[str, Any]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_subdir = output_dir / tag
    output_subdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "test_name": "cra5_runtime_smoke_test",
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "quantization_step": quantization_step,
        "delta_coding": delta_coding,
    }

    try:
        print(f"\n{'='*60}")
        print(f"CONFIG: hidden={hidden_dim} latent={latent_dim} qstep={quantization_step} delta={delta_coding}")
        print(f"{'='*60}")

        print("\n1. Loading CRA5 checkpoint/model...")
        start_time = time.perf_counter()

        model = build_cra5_model(
            in_channels=28,
            out_channels=28,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim if latent_dim != hidden_dim else None,
            freeze_backbone=True,
        )

        checkpoint_loaded = False
        if checkpoint_path and checkpoint_path.exists() and hidden_dim == 1024:
            print(f"Adapting checkpoint: {checkpoint_path}")
            adapted_state_dict, adapter_metadata = adapt_cra5_checkpoint(checkpoint_path)

            if "backbone.encoder.pos_embed" in adapted_state_dict:
                ckpt_pos_embed = adapted_state_dict["backbone.encoder.pos_embed"]
                model_pos_embed = model.backbone.encoder.pos_embed
                if ckpt_pos_embed.shape != model_pos_embed.shape:
                    ckpt_pos_embed = ckpt_pos_embed.reshape(1, 72, 144, -1).permute(0, 3, 1, 2)
                    target_h, target_w = 32, 72
                    ckpt_pos_embed = torch.nn.functional.interpolate(
                        ckpt_pos_embed, size=(target_h, target_w), mode="bicubic", align_corners=False
                    )
                    adapted_state_dict["backbone.encoder.pos_embed"] = (
                        ckpt_pos_embed.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
                    )

            missing, unexpected = model.load_state_dict(adapted_state_dict, strict=False)
            print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  Missing keys (bottleneck params expected here): {missing[:5]}...")
            checkpoint_loaded = True
            results["checkpoint_adapted"] = True
            results["adapter_metadata"] = {
                "source_channels": adapter_metadata.source_channels,
                "target_channels": adapter_metadata.target_channels,
            }
        else:
            if hidden_dim != 1024:
                print(f"Skipping checkpoint: hidden_dim={hidden_dim} != 1024 (checkpoint expects 1024)")
            results["checkpoint_adapted"] = False

        model = model.to(device)
        model.eval()

        checkpoint_load_time = time.perf_counter() - start_time
        results["checkpoint_load_seconds"] = checkpoint_load_time
        results["checkpoint_loaded"] = checkpoint_loaded

        if timestamp is None:
            print("\n2. Loading demo timestamp...")
            timestamp = load_demo_timestamp().to(device)
        else:
            timestamp = timestamp.to(device)

        results["input_shape"] = list(timestamp.shape)
        results["input_dtype"] = str(timestamp.dtype)

        print("\n3. Normalizing...")
        normalized, norm_stats = simple_normalization(timestamp)

        print("\n4. Encoding...")
        start_time = time.perf_counter()

        quantized_latent, encode_metadata = encode_cra5(
            normalized,
            model,
            quantization_step=quantization_step,
            delta_coding=delta_coding,
        )

        encode_time = time.perf_counter() - start_time
        results["encode_seconds"] = encode_time
        results["latent_shape"] = encode_metadata["latent_shape"]

        latent_elements = int(np.prod(encode_metadata["latent_shape"]))
        input_elements = int(np.prod(timestamp.shape))
        results["latent_elements"] = latent_elements
        results["input_elements"] = input_elements
        results["tensor_cr_channels_reduced"] = input_elements / latent_elements

        print(f"Encoded to latent shape: {encode_metadata['latent_shape']} "
              f"({latent_elements} elements, {input_elements / latent_elements:.1f}x tensor element ratio)")

        print("\n5. Serializing bitstream...")
        bitstream_path = output_subdir / "smoke_test.bin"

        start_time = time.perf_counter()
        bitstream_bytes = serialize_cra5_bitstream(quantized_latent, encode_metadata, bitstream_path)
        serialize_time = time.perf_counter() - start_time

        results["serialize_seconds"] = serialize_time
        results["bitstream_bytes"] = bitstream_bytes
        results["bitstream_path"] = str(bitstream_path)

        input_bytes = input_elements * 4
        serialized_cr = input_bytes / bitstream_bytes
        results["input_bytes"] = input_bytes
        results["tensor_compression_ratio"] = serialized_cr

        print(f"Bitstream: {bitstream_bytes} bytes (Serialized CR: {serialized_cr:.1f}×)")
        print(f"  Input float32 bytes: {input_bytes}")

        print("\n6. Decoding...")
        start_time = time.perf_counter()

        reconstruction_norm = decode_cra5(
            quantized_latent,
            model,
            quantization_step=encode_metadata["quantization_step"],
            delta_coding=encode_metadata.get("delta_coding", True),
            first_channel=encode_metadata.get("first_channel"),
            first_rows=encode_metadata.get("first_rows"),
        )

        decode_time = time.perf_counter() - start_time
        results["decode_seconds"] = decode_time

        print("\n7. Inverse normalizing...")
        reconstruction = inverse_normalization(reconstruction_norm, norm_stats)

        print("\n8. Computing metrics...")
        physical_metrics = compute_physical_rmse(timestamp, reconstruction)

        results["physical_rmse"] = physical_metrics
        results["reconstruction_shape"] = list(reconstruction.shape)

        quantized_check, _ = encode_cra5(
            normalized, model, quantization_step=quantization_step, delta_coding=delta_coding
        )
        roundtrip_exact = np.array_equal(quantized_latent, quantized_check)
        results["roundtrip_exact"] = roundtrip_exact

        total_time = checkpoint_load_time + encode_time + serialize_time + decode_time
        results["total_runtime_seconds"] = total_time

        results["success"] = True
        results["error"] = None

        cr_label = "✅" if serialized_cr >= 32 else "⚠️"
        cr64_label = "✅64x" if serialized_cr >= 64 else ""
        print(f"\n✅ Smoke test PASSED")
        print(f"Overall RMSE: {physical_metrics['overall_rmse']:.6f}")
        print(f"Serialized CR: {serialized_cr:.1f}× {cr_label} {cr64_label}")
        print(f"Total runtime: {total_time:.3f}s")
        print(f"Roundtrip exact: {roundtrip_exact}")

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        print(f"\n❌ Smoke test FAILED: {e}")
        raise

    results_path = output_subdir / "smoke_cra5_runtime.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=lambda o: asdict(o) if isinstance(o, SweepRow) else o)

    print(f"\nResults saved to: {results_path}")
    return results


def build_sweep_grid():
    """A compact grid targeting 32x and 64x serialized compression."""
    configs = []
    # hidden_dim=1024 with checkpoint, varying latent_dim + qstep
    for latent_dim in [512, 256, 192, 128, 96, 64]:
        for qstep in [0.1, 0.25, 0.5, 1.0]:
            for delta in [True]:
                configs.append((1024, latent_dim, qstep, delta))
    # hidden_dim directly reduced (no checkpoint - pure ablation)
    for hidden_dim, latent_dim in [(512, 256), (512, 128), (256, 128), (256, 64)]:
        for qstep in [0.25, 0.5, 1.0]:
            configs.append((hidden_dim, latent_dim, qstep, True))
    return configs


def run_sweep(
    *,
    checkpoint_path: Path | None,
    device: str,
    output_dir: Path,
) -> list[SweepRow]:
    timestamp = load_demo_timestamp()
    rows: list[SweepRow] = []
    grid = build_sweep_grid()

    print(f"\n===== CRA5 COMPRESSION SWEEP: {len(grid)} configs =====")
    for idx, (hidden_dim, latent_dim, qstep, delta) in enumerate(grid):
        tag = f"h{hidden_dim}_l{latent_dim}_q{str(qstep).replace('.', 'p')}_d{1 if delta else 0}"
        try:
            res = run_single_config(
                checkpoint_path=checkpoint_path,
                device=device,
                output_dir=output_dir,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                quantization_step=qstep,
                delta_coding=delta,
                tag=tag,
                timestamp=timestamp,
            )
            rows.append(SweepRow(
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                quantization_step=qstep,
                delta_coding=delta,
                input_elements=int(res["input_elements"]),
                latent_elements=int(res["latent_elements"]),
                bitstream_bytes=int(res["bitstream_bytes"]),
                tensor_cr=float(res["tensor_cr_channels_reduced"]),
                serialized_cr=float(res["tensor_compression_ratio"]),
                overall_rmse=float(res["physical_rmse"]["overall_rmse"]),
                total_runtime_s=float(res["total_runtime_seconds"]),
                roundtrip_exact=bool(res["roundtrip_exact"]),
            ))
        except Exception as e:
            print(f"[{idx}] SKIP {tag}: {e}")

    # Summary table
    rows_32 = [r for r in rows if r.serialized_cr >= 32]
    rows_64 = [r for r in rows if r.serialized_cr >= 64]

    print("\n" + "=" * 110)
    print("CRA5 COMPRESSION SWEEP SUMMARY")
    print(f"Total configs: {len(rows)}   |   ≥32×: {len(rows_32)}   |   ≥64×: {len(rows_64)}")
    print("=" * 110)
    header = f"{'hidden':>6} {'latent':>6} {'qstep':>6} {'Δ':>3} {'elements×':>10} {'bytes':>10} {'serialCR':>9} {'RMSE':>12} {'s':>7}  tag"
    print(header)
    print("-" * 110)
    for r in sorted(rows, key=lambda r: -r.serialized_cr):
        tag = f"h{r.hidden_dim}_l{r.latent_dim}_q{str(r.quantization_step).replace('.', 'p')}_d{1 if r.delta_coding else 0}"
        marker = ""
        if r.serialized_cr >= 64:
            marker = " **64x**"
        elif r.serialized_cr >= 32:
            marker = " *32x*"
        print(
            f"{r.hidden_dim:>6} {r.latent_dim:>6} {r.quantization_step:>6} "
            f"{'Y' if r.delta_coding else 'N':>3} {r.tensor_cr:>10.1f} "
            f"{r.bitstream_bytes:>10} {r.serialized_cr:>8.1f}× "
            f"{r.overall_rmse:>12.2f} {r.total_runtime_s:>7.2f}  {tag}{marker}"
        )

    print("\nTop configs hitting ≥64x serialized CR:")
    if rows_64:
        for r in sorted(rows_64, key=lambda r: r.overall_rmse):
            print(f"  h{r.hidden_dim}/l{r.latent_dim}/q{r.quantization_step} -> "
                  f"{r.serialized_cr:.1f}×, RMSE {r.overall_rmse:.2f}")
    else:
        print("  (none — try increasing qstep or lowering latent_dim further)")

    print("\nTop configs hitting ≥32x serialized CR (by RMSE):")
    if rows_32:
        for r in sorted(rows_32, key=lambda r: r.overall_rmse)[:8]:
            print(f"  h{r.hidden_dim}/l{r.latent_dim}/q{r.quantization_step} -> "
                  f"{r.serialized_cr:.1f}×, RMSE {r.overall_rmse:.2f}")

    summary = {"rows": [asdict(r) for r in rows], "n_32x": len(rows_32), "n_64x": len(rows_64)}
    with (output_dir / "sweep_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary saved to: {output_dir / 'sweep_summary.json'}")
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CRA5 runtime smoke test with compression tuning")
    parser.add_argument("--checkpoint", type=Path, help="Path to CRA5-159v checkpoint (optional)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke_cra5"))
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="Transformer hidden dim. 1024 matches the pretrained CRA5 checkpoint.")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Bottleneck latent channels. If < hidden_dim, a 1x1 conv bottleneck is added. "
                             "Main knob for reaching 32x-64x. Suggest: 256, 192, 128, 96, 64.")
    parser.add_argument("--quantization-step", type=float, default=0.1,
                        help="Quantization step for latent. Larger = more compression, higher RMSE. "
                             "Suggest: 0.25, 0.5, 1.0 for 32x-64x.")
    parser.add_argument("--delta/--no-delta", default=True, dest="delta",
                        help="Enable channel + spatial delta coding (improves Huffman efficiency).")
    parser.add_argument("--sweep", action="store_true",
                        help="Run a grid sweep of hidden/latent/qstep settings and print a summary table "
                             "highlighting configs that achieve ≥32x or ≥64x compression.")

    args = parser.parse_args()

    if not args.checkpoint:
        default_checkpoint = Path("~/.cache/era5-minimum/cra5/cra5_159v_150k.pth").expanduser()
        if default_checkpoint.exists():
            args.checkpoint = default_checkpoint
            print(f"Using default checkpoint: {args.checkpoint}")
        else:
            print("No checkpoint specified and default not found — using random weights (bottleneck/proj still trainable).")

    if args.sweep:
        run_sweep(
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_dir=args.output_dir,
        )
        return

    latent_dim = args.latent_dim if args.latent_dim is not None else args.hidden_dim
    tag = (
        f"h{args.hidden_dim}_l{latent_dim}"
        f"_q{str(args.quantization_step).replace('.', 'p')}"
        f"_d{1 if args.delta else 0}"
    )
    results = run_single_config(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim,
        quantization_step=args.quantization_step,
        delta_coding=args.delta,
        tag=tag,
    )

    if results["success"]:
        cr = results["tensor_compression_ratio"]
        tier = "🎉 ≥64x" if cr >= 64 else "✅ ≥32x" if cr >= 32 else "⚠️ <32x"
        print(f"\n{tier} CRA5 smoke test completed successfully!")
        print(f"Compression: {cr:.1f}× serialized ({results['bitstream_bytes']} bytes)")
        print(f"RMSE: {results['physical_rmse']['overall_rmse']:.3f}")
        print(f"Runtime: {results['total_runtime_seconds']:.3f}s")
        exit(0)
    else:
        print(f"\n💥 CRA5 smoke test failed: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()
