#!/usr/bin/env python3
"""Train CRA5 adapter with proper train-only normalization, bottleneck and artifact export.

This script intentionally produces a SYNTHETIC-TRAINED checkpoint for
DEMONSTRATION / DEFENSE purposes only. The resulting artifacts must not be
used in production or claimed to represent real ERA5 data.

Compared to the previous implementation, this script adds:
  * Train-only per-channel mean/std (formal AGENTS.md contract).
  * 1x1 conv bottleneck via latent_dim (for 32x-64x serialized CR).
  * Saves best_model.pth + normalization.json + training_metrics.json +
    codec artifact (bitstream + inverse-normalized RMSE) to output dir.
  * Always saves checkpoints by default (save_best default override).
  * Disambiguates train vs val splits in normalization logs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from era5_minimum.cra5.adapter import adapt_cra5_checkpoint
from era5_minimum.cra5.codec_workflow import (
    decode_cra5,
    encode_cra5,
    serialize_cra5_bitstream,
)
from era5_minimum.cra5.model import build_cra5_model
from era5_minimum.data.synthetic import make_synthetic_era5


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def create_synthetic_data(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training and validation data."""
    data_config = config["data"]
    subset_size = data_config["subset_size"]
    channels = data_config["channels"]
    shape = data_config.get("synthetic_shape", [28, 352, 720])

    total_samples = max(subset_size + 8, 24)

    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_list = [standard_channels[i % len(standard_channels)] for i in range(channels)]

    data, _ = make_synthetic_era5(
        samples=total_samples,
        height=shape[1],
        width=shape[2],
        channels=tuple(channels_list),
        seed=config["experiment"]["seed"],
    )

    data_tensor = torch.from_numpy(data).float()
    train_data = data_tensor[:subset_size]
    val_size = min(8, total_samples - subset_size)
    val_data = data_tensor[subset_size : subset_size + val_size]
    return train_data, val_data


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    method: str
    fit_on: str = "train_split_only"

    def as_tensor(self, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        mean_t = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
        std_t = torch.tensor(self.std, device=device).view(1, -1, 1, 1)
        return mean_t, std_t


def fit_train_norm(train_data: torch.Tensor, eps: float = 1e-8) -> NormStats:
    """Formal per-channel z-score normalization, fit ONLY on training split."""
    dtype64 = train_data.double()
    mean = dtype64.mean(dim=[0, 2, 3])
    std = dtype64.std(dim=[0, 2, 3])
    std = torch.clamp(std, min=eps)
    return NormStats(
        mean=mean.float().tolist(),
        std=std.float().tolist(),
        method="per_channel_zscore_train_only",
    )


def apply_norm(data: torch.Tensor, stats: NormStats) -> torch.Tensor:
    mean_t, std_t = stats.as_tensor(data.device)
    return (data - mean_t) / std_t


def apply_denorm(data: torch.Tensor, stats: NormStats) -> torch.Tensor:
    mean_t, std_t = stats.as_tensor(data.device)
    return data * std_t + mean_t


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build CRA5 model with optional checkpoint adaptation + bottleneck."""
    model_config = config["model"]

    latent_dim = model_config.get("latent_dim")
    if latent_dim == model_config["hidden_dim"]:
        latent_dim = None

    model = build_cra5_model(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=latent_dim,
        num_encoder_blocks=model_config["num_encoder_blocks"],
        num_decoder_blocks=model_config["num_decoder_blocks"],
        num_heads=model_config["num_heads"],
        patch_size=tuple(model_config["patch_size"]),
        freeze_backbone=model_config.get("freeze_backbone", False),
    )

    if model_config.get("use_pretrained", True):
        default_ckpt = Path("~/.cache/era5-minimum/cra5/cra5_159v_150k.pth").expanduser()
        checkpoint_path = Path(model_config.get("checkpoint_path", str(default_ckpt))).expanduser()

        if checkpoint_path.exists() and model_config["hidden_dim"] == 1024:
            print(f"Loading and adapting CRA5-159v checkpoint: {checkpoint_path}")
            adapted_state_dict, metadata = adapt_cra5_checkpoint(checkpoint_path)

            if "backbone.encoder.pos_embed" in adapted_state_dict:
                ckpt_pos_embed = adapted_state_dict["backbone.encoder.pos_embed"]
                model_pos_embed = model.backbone.encoder.pos_embed
                if ckpt_pos_embed.shape != model_pos_embed.shape:
                    n_src = ckpt_pos_embed.shape[1]
                    src_h, src_w = 72, 144
                    if hasattr(model.backbone.encoder, "patch_size"):
                        pass
                    C = model_config["hidden_dim"]
                    ckpt_pos_embed = ckpt_pos_embed.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)
                    target_h, target_w = 32, 72
                    ckpt_pos_embed = torch.nn.functional.interpolate(
                        ckpt_pos_embed, size=(target_h, target_w), mode="bicubic", align_corners=False
                    )
                    adapted_state_dict["backbone.encoder.pos_embed"] = (
                        ckpt_pos_embed.permute(0, 2, 3, 1).reshape(1, target_h * target_w, C)
                    )

            missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
            print(f"Loaded checkpoint. missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
            if missing_keys:
                print(f"  Missing (expected for bottleneck): {missing_keys[:6]}")
            print(f"Adapted checkpoint: {metadata.copied_channels} copied, "
                  f"{metadata.learned_boundary_channels} learned boundary channels")
        else:
            why = "hidden_dim!=1024" if model_config["hidden_dim"] != 1024 else f"not found at {checkpoint_path}"
            print(f"Skipping pretrained load ({why}). Using random initialization — "
                  "THIS IS DEMO / SYNTHETIC TRAINING ONLY.")

    return model


class LatitudeWeightedMSELoss(nn.Module):
    def __init__(self, height: int, device: str) -> None:
        super().__init__()
        latitudes = np.linspace(90, -90, height)
        values = np.cos(np.deg2rad(latitudes))
        values = np.clip(values, 0.0, None)
        values = values / values.mean()
        weights = torch.from_numpy(values.astype(np.float32)).view(1, 1, -1, 1).to(device)
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared_error = (pred - target) ** 2
        return torch.mean(squared_error * self.weights)


def train_epoch(
    model: nn.Module,
    data_norm: torch.Tensor,
    optimizer: optim.Optimizer,
    batch_size: int,
    device: str,
    criterion: nn.Module,
    grad_clip: float = 1.0,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    perm = torch.randperm(len(data_norm))
    for i in range(0, len(data_norm), batch_size):
        idx = perm[i : i + batch_size]
        batch = data_norm[idx].to(device)
        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if scheduler._step_count < scheduler.total_steps:
                scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def validate(
    model: nn.Module,
    data_norm: torch.Tensor,
    device: str,
    criterion: nn.Module,
) -> float:
    model.eval()
    data_device = data_norm.to(device)
    reconstruction = model(data_device)
    return criterion(reconstruction, data_device).item()


def compute_physical_metrics(
    model: nn.Module,
    data_raw: torch.Tensor,
    norm: NormStats,
    codec_cfg: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Full end-to-end codec evaluation: norm → encode → serialize → decode → denorm → physical RMSE."""
    from sklearn.metrics import mean_squared_error  # noqa: F401  (kept for potential extension)

    model.eval()
    sample_raw = data_raw[:1].to(device)
    sample_n = apply_norm(sample_raw, norm)

    qstep = float(codec_cfg["quantization_step"])
    delta = bool(codec_cfg.get("delta_coding", True))

    q, meta = encode_cra5(sample_n, model, quantization_step=qstep, delta_coding=delta)

    bitstream_path = output_dir / "demo_encoded.bin"
    bs_bytes = serialize_cra5_bitstream(q, meta, bitstream_path)

    recon_n = decode_cra5(
        q, model,
        quantization_step=qstep,
        delta_coding=meta.get("delta_coding", False),
        first_channel=meta.get("first_channel"),
        first_rows=meta.get("first_rows"),
    )
    recon_raw = apply_denorm(recon_n, norm)

    se = ((sample_raw - recon_raw) ** 2).mean(dim=[0, 2, 3])
    rmse_per_ch = torch.sqrt(se)
    overall_rmse = float(torch.sqrt(((sample_raw - recon_raw) ** 2).mean()))
    input_bytes = int(np.prod(sample_raw.shape)) * 4
    serialized_cr = input_bytes / bs_bytes

    result = {
        "sample_shape": list(sample_raw.shape),
        "input_float32_bytes": input_bytes,
        "bitstream_bytes": int(bs_bytes),
        "bitstream_path": str(bitstream_path.name),
        "serialized_compression_ratio": float(serialized_cr),
        "tensor_element_ratio": float(np.prod(sample_raw.shape) / np.prod(q.shape)),
        "quantization_step": qstep,
        "delta_coding": delta,
        "latent_shape": list(q.shape),
        "physical_rmse_overall": overall_rmse,
        "physical_rmse_per_channel_mean": float(rmse_per_ch.mean()),
        "physical_rmse_per_channel_max": float(rmse_per_ch.max()),
        "physical_rmse_per_channel_min": float(rmse_per_ch.min()),
        "roundtrip_exact_symbols": bool(np.array_equal(q, encode_cra5(
            sample_n, model, quantization_step=qstep, delta_coding=delta
        )[0])),
    }
    return result


def save_metrics(
    output_dir: Path,
    config: dict[str, Any],
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
    training_time: float,
    norm: NormStats,
    physical: dict[str, Any],
) -> None:
    metrics = {
        "experiment_name": config["experiment"]["name"],
        "subset_size": config["data"]["subset_size"],
        "epochs": len(train_losses),
        "best_val_loss_normalized": float(best_val_loss),
        "final_train_loss_normalized": float(train_losses[-1]) if train_losses else None,
        "training_time_seconds": float(training_time),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "normalization_train_only": asdict(norm),
        "codec_end_to_end": physical,
        "disclaimer": (
            "THIS IS A DEMONSTRATION RUN ON SYNTHETIC DATA. Normalization, loss, and splits "
            "follow AGENTS.md rules formally, but the underlying sample distribution has no "
            "physical ERA5 semantics. Do not use for scientific conclusions."
        ),
        "config": config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "normalization_train_only.json").open("w") as f:
        json.dump(asdict(norm), f, indent=2)
    print(f"Artifacts saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRA5 adapter (demo/synthetic — formal training contract)")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    config = load_config(args.config)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("CRA5 adapter DEMO training (SYNTHETIC DATA ONLY — see disclaimer in output JSON)")
    print(f"Device: {device}   Output: {output_dir}")
    print("=" * 80)

    train_raw, val_raw = create_synthetic_data(config)
    print(f"Train raw: {tuple(train_raw.shape)}   Val raw: {tuple(val_raw.shape)}")

    norm = fit_train_norm(train_raw)
    print(f"Fit train-only norm: {len(norm.mean)} channels, std range [{min(norm.std):.4f}, {max(norm.std):.4f}]")

    train_n = apply_norm(train_raw, norm)
    val_n = apply_norm(val_raw, norm)
    print(f"Train norm range [{train_n.min():.3f}, {train_n.max():.3f}]  "
          f"Val norm range [{val_n.min():.3f}, {val_n.max():.3f}]")

    print("Building CRA5 model...")
    model = build_model(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}   Trainable: {trainable_params:,}")
    print(f"  hidden_dim={config['model']['hidden_dim']}  "
          f"latent_dim={config['model'].get('latent_dim', config['model']['hidden_dim'])}")

    opt_config = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(opt_config["lr"]),
        betas=tuple(opt_config.get("betas", (0.9, 0.999))),
        weight_decay=float(opt_config.get("weight_decay", 0.05)),
    )

    height = train_raw.shape[2]
    criterion = LatitudeWeightedMSELoss(height=height, device=device)

    train_config = config["training"]
    epochs = int(train_config["epochs"])
    batch_size = int(train_config["batch_size"])
    val_frequency = int(train_config["val_frequency"])
    grad_clip = float(train_config.get("grad_clip", 1.0))
    save_best = bool(train_config.get("save_best", True))  # default True for demo

    scheduler = None
    if "scheduler" in config:
        sched_config = config["scheduler"]
        stype = sched_config.get("type", "constant")
        if stype == "one_cycle":
            steps_per_epoch = max(1, (len(train_n) + batch_size - 1) // batch_size)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(opt_config["lr"]),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=sched_config.get("pct_start", 0.3),
                div_factor=sched_config.get("div_factor", 25.0),
                final_div_factor=sched_config.get("final_div_factor", 1e4),
            )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_n, optimizer, batch_size, device, criterion, grad_clip, scheduler)
        train_losses.append(train_loss)

        if epoch % val_frequency == 0:
            val_loss = validate(model, val_n, device, criterion)
            val_losses.append(val_loss)
            if val_loss < best_val_loss - float(train_config.get("min_delta", 0.0)):
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    torch.save(model.state_dict(), output_dir / "best_model.pth")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "normalization": asdict(norm),
                        "config": config,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    }, output_dir / "checkpoint_bundle.pth")
            else:
                patience_counter += 1

            print(f"Epoch {epoch:3d}: train={train_loss:.6f}  val={val_loss:.6f}  best_val={best_val_loss:.6f}")
            if patience_counter >= int(train_config.get("patience", 1_000_000)):
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"Epoch {epoch:3d}: train={train_loss:.6f}")

        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.1f}s. Best val loss: {best_val_loss:.6f}")

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        print(f"Loading best_model.pth for final codec evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    physical = compute_physical_metrics(
        model, val_raw, norm, config.get("codec", {}), output_dir, device
    )

    cr = physical["serialized_compression_ratio"]
    tier = ("🎉 ≥64x" if cr >= 64 else "✅ ≥32x" if cr >= 32 else "⚠️ <32x")
    print(f"\nCodec end-to-end: {tier}  CR={cr:.1f}x  "
          f"RMSE={physical['physical_rmse_overall']:.2f}  "
          f"bytes={physical['bitstream_bytes']}")

    save_metrics(output_dir, config, train_losses, val_losses, best_val_loss, training_time, norm, physical)

    print("\n===== DEMO TRAINING COMPLETE =====")
    print("Artifacts (for project defense / demo):")
    print(f"  {output_dir / 'best_model.pth'}")
    print(f"  {output_dir / 'checkpoint_bundle.pth'}  (model + norm + config)")
    print(f"  {output_dir / 'training_metrics.json'}")
    print(f"  {output_dir / 'normalization_train_only.json'}")
    print(f"  {output_dir / 'demo_encoded.bin'}  ({physical['bitstream_bytes']} bytes)")


if __name__ == "__main__":
    main()
