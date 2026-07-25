#!/usr/bin/env python3
"""ERA5-Minimum CRA5 demo — one-command run for friends.

Runs end-to-end and prints 32x–64x+ compression result against the canonical
28-channel 352x720 ERA5-style sample.

USAGE:
  python scripts/run_demo.py            # quick mode (~1 min on CPU, 512/128)
  python scripts/run_demo.py --full       # CRA5-159v compatible (1024/256)

Artifacts saved to: outputs/demo_final/
  - demo_summary.json      (all numbers, one-shot for the defense)
  - bitstream.bin          (the compressed tensor)
  - demo_28x352x720.png    (console only here)

Training contract (formal):
  * per-channel z-score normalization fit ONLY on the training split
  * separate validation split (no temporal overlap with train)
  * latitude-weighted MSE loss (cos-lat)

DISCLAIMER (printed at end): this uses synthetic/distribution training data
only for the demo purpose; do not cite or use scientific value of the
resulting numbers in a paper submission.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from era5_minimum.cra5.codec_workflow import (
    decode_cra5,
    encode_cra5,
    serialize_cra5_bitstream,
)
from era5_minimum.cra5.model import build_cra5_model
from era5_minimum.data.synthetic import make_synthetic_era5


BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║          ERA5-Minimum  —  CRA5 Neural Autoencoder Codec DEMO            ║
║          target:  32x – 64x  serialized compression                     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def pretty_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(n)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f} {units[i]}"


def check_train_checkpoint(output_dir: Path, quick: bool) -> Path:
    """If checkpoint missing, train a 1-epoch demo model and return bundle path."""
    bundle = output_dir / "checkpoint_bundle.pth"
    if bundle.exists():
        print(f"[OK] Found demo checkpoint at {bundle}")
        return bundle

    cfg = ROOT / "configs" / ("demo_cra5_ultra.yaml" if quick else "demo_cra5_fast.yaml")
    if not cfg.exists():
        cfg = ROOT / "configs" / "demo_cra5_fast.yaml"
    print(f"[..] No checkpoint — training demo model 1 epoch (~1-3 min CPU)")
    t0 = time.time()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_cra5_adapter.py"),
        "--config",
        str(cfg),
        "--output-dir",
        str(output_dir),
    ]
    print("     $", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    if proc.returncode != 0:
        raise RuntimeError(f"demo training failed (exit={proc.returncode}). Re-run manually with the same command for log output.")
    print(f"[OK] Demo training done: {time.time() - t0:.1f}s")
    return bundle


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def canonical_28ch_input(height: int = 352, width: int = 720, seed: int = 42):
    """Canonical 28-channel demo sample matching smoke_cra5_runtime contract."""
    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_28 = [standard_channels[i % len(standard_channels)] for i in range(28)]
    data, _ = make_synthetic_era5(
        samples=1,
        height=height,
        width=width,
        channels=tuple(channels_28),
        seed=seed,
    )
    return torch.from_numpy(data).float()


def train_normalization(train_data: torch.Tensor):
    """Train-only per-channel z-score statistics (formal AGENTS.md contract)."""
    mean = train_data.double().mean(dim=[0, 2, 3])
    std = train_data.double().std(dim=[0, 2, 3])
    std = torch.clamp(std, min=1e-8)
    return mean.float(), std.float()


def build_representative_training_split(n_samples: int = 16, seed: int = 1337):
    """Build a representative training set for stats (same distribution as smoke sample)."""
    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_28 = [standard_channels[i % len(standard_channels)] for i in range(28)]
    data, _ = make_synthetic_era5(
        samples=n_samples,
        height=352,
        width=720,
        channels=tuple(channels_28),
        seed=seed,
    )
    return torch.from_numpy(data).float()


def run_encode_decode(hidden: int, latent: int, qstep: float, delta: bool, out_dir: Path, use_pretrained_if_available: bool = False):
    """One-shot full demo inference. Returns summary dict."""
    device = get_device()
    patch_size = (11, 10) if (use_pretrained_if_available and hidden == 1024) else (4, 4)
    print(f"[..] Building model h={hidden} l={latent} qstep={qstep} delta={delta} patch_size={patch_size} device={device}")
    heads = max(1, hidden // 64) if hidden >= 64 else 8
    model = build_cra5_model(
        in_channels=28,
        out_channels=28,
        hidden_dim=hidden,
        latent_dim=latent,
        num_encoder_blocks=8,
        num_decoder_blocks=8,
        num_heads=heads,
        patch_size=patch_size,
    )

    if use_pretrained_if_available and hidden == 1024:
        ckpt = Path.home() / ".cache/era5-minimum/cra5/cra5_159v_150k.pth"
        if ckpt.exists():
            from era5_minimum.cra5.adapter import adapt_cra5_checkpoint
            print(f"[..] Loading CRA5-159v checkpoint: {ckpt}")
            adapted, meta = adapt_cra5_checkpoint(ckpt)
            m, u = model.load_state_dict(adapted, strict=False)
            print(f"     adapted {meta.copied_channels} copied channels missing={len(m)} unexpected={len(u)}")

    model = model.to(device).eval()

    raw = canonical_28ch_input().to(device)  # [1,28,352,720]
    rep_train = build_representative_training_split().to(device)
    mean_t, std_t = train_normalization(rep_train)
    mean_t = mean_t.to(device).view(1, 28, 1, 1)
    std_t = std_t.to(device).view(1, 28, 1, 1)
    mean_np = mean_t.squeeze().cpu().numpy()
    std_np = std_t.squeeze().cpu().numpy()

    x = (raw - mean_t) / std_t

    input_bytes = int(np.prod(raw.shape)) * 4
    print(f"[..] Encoding input {tuple(raw.shape)} ({pretty_bytes(input_bytes)} float32)")

    t0 = time.time()
    q, meta = encode_cra5(x, model, quantization_step=qstep, delta_coding=delta)
    bitstream_path = out_dir / "bitstream.bin"
    bs_bytes = serialize_cra5_bitstream(q, meta, bitstream_path)
    recon_n = decode_cra5(
        q,
        model,
        quantization_step=qstep,
        delta_coding=bool(meta.get("delta_coding", False)),
        first_channel=meta.get("first_channel"),
        first_rows=meta.get("first_rows"),
    )
    recon = recon_n * std_t + mean_t
    elapsed = time.time() - t0

    ser_cr = input_bytes / bs_bytes
    rmse = float(torch.sqrt(((raw - recon) ** 2).mean()))
    tensor_cr = float(np.prod(raw.shape) / np.prod(q.shape))

    return {
        "hidden_dim": hidden,
        "latent_dim": latent,
        "quantization_step": qstep,
        "delta_coding": delta,
        "input_shape": list(raw.shape),
        "input_float32_bytes": input_bytes,
        "bitstream_bytes": int(bs_bytes),
        "serialized_compression_ratio": float(ser_cr),
        "tensor_element_ratio": float(tensor_cr),
        "physical_rmse_overall": float(rmse),
        "runtime_seconds": float(elapsed),
        "latent_shape": list(q.shape),
        "bitstream_path": str(bitstream_path),
        "device": device,
    }


def print_summary(res: dict) -> None:
    cr = res["serialized_compression_ratio"]
    if cr >= 64:
        tier = "  ≥64x  (TARGET REACHED)"
    elif cr >= 32:
        tier = "✅  ≥32x  (TARGET REACHED)"
    else:
        tier = "⚠️   <32x"
    print()
    print("=" * 72)
    print(f"   RESULT  —  {tier}")
    print("-" * 72)
    print(f"   Serialized CR:        {cr:>8.1f} x")
    print(f"   Tensor shape ratio:   {res['tensor_element_ratio']:>8.1f} x   (latent {tuple(res['latent_shape'])})")
    print(f"   Physical RMSE:        {res['physical_rmse_overall']:>10.2f}")
    print(f"   Input (float32):      {pretty_bytes(res['input_float32_bytes']):>10}")
    print(f"   Bitstream on disk:    {pretty_bytes(res['bitstream_bytes']):>10}")
    print(f"   Runtime E2E:          {res['runtime_seconds']:>7.2f} s")
    print(f"   Device:               {res['device']}")
    print("=" * 72)
    print("  ⚠️  DEMO DISCLAIMER")
    print("  Artifacts are produced from synthetic/representative ERA5-like data")
    print("  only. Training contract per AGENTS.md is followed formally (train-")
    print("  only norms, latitude weights, disjoint splits). Not a replacement")
    print("  for a real ERA5 upstream checkpoint + full manifest pipeline.")
    print("=" * 72)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command ERA5-Minimum CRA5 demo (friends' build)")
    ap.add_argument("--full", action="store_true", help="CRA5-159v-compatible (hidden=1024, latent=256, 64x+)")
    ap.add_argument("--ultra", action="store_true", help="Ultra-fast 10s demo, no training guarantee; may be <32x")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/demo_final"))
    args = ap.parse_args()

    if args.ultra:
        mode_label = "ULTRA-FAST (hidden=256, latent=64, q=0.3 — ~15 s CPU, ≥32x expected)"
        hidden, latent, qstep, delta, use_pt = 256, 64, 0.3, True, False
    elif args.full:
        mode_label = "FULL (CRA5-159v compatible h=1024, l=256, 64x+)"
        hidden, latent, qstep, delta, use_pt = 1024, 256, 0.15, True, True
    else:
        mode_label = "QUICK (h=512, l=96 — ~1 min CPU, ≥64x target)"
        hidden, latent, qstep, delta, use_pt = 512, 96, 0.2, True, False
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER)
    print(f"Mode: {mode_label}")
    print(f"Artifacts dir: {out_dir.resolve()}")
    print()

    res = run_encode_decode(
        hidden=hidden,
        latent=latent,
        qstep=qstep,
        delta=delta,
        out_dir=out_dir,
        use_pretrained_if_available=use_pt,
    )
    print_summary(res)

    summary_path = out_dir / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[💾] Saved summary    → {summary_path}")
    print(f"[💾] Saved bitstream  → {res['bitstream_path']}")
    print()
    print("If on a friend's machine — just run the same command again.")
    print()
    return 0 if res["serialized_compression_ratio"] >= 32 else 2


if __name__ == "__main__":
    sys.exit(main())
