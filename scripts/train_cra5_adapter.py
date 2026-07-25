#!/usr/bin/env python3
"""Train CRA5 adapter with different sample sizes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from era5_minimum.cra5.adapter import adapt_cra5_checkpoint
from era5_minimum.cra5.codec_workflow import encode_and_serialize_cra5
from era5_minimum.cra5.model import build_cra5_model
from era5_minimum.data.synthetic import make_synthetic_era5


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def create_synthetic_data(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training and validation data."""
    data_config = config["data"]
    subset_size = data_config["subset_size"]
    channels = data_config["channels"]
    shape = data_config.get("synthetic_shape", [28, 361, 720])
    
    # Generate more samples for train/val split
    total_samples = max(subset_size + 10, 32)
    
    # Use the 8 standard channels, repeat to get required number of channels
    standard_channels = ("t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc")
    channels_list = []
    for i in range(channels):
        channels_list.append(standard_channels[i % len(standard_channels)])
    
    # Generate synthetic data
    data, _ = make_synthetic_era5(
        samples=total_samples,
        height=shape[1],
        width=shape[2],
        channels=tuple(channels_list),
        seed=config["experiment"]["seed"],
    )
    
    # Convert to tensor and split
    data_tensor = torch.from_numpy(data).float()
    train_data = data_tensor[:subset_size]
    val_data = data_tensor[subset_size:subset_size + min(8, total_samples - subset_size)]
    
    return train_data, val_data


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build CRA5 model with optional checkpoint adaptation."""
    model_config = config["model"]
    
    # Build base model
    model = build_cra5_model(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        hidden_dim=model_config["hidden_dim"],
        num_encoder_blocks=model_config["num_encoder_blocks"],
        num_decoder_blocks=model_config["num_decoder_blocks"],
        num_heads=model_config["num_heads"],
        patch_size=model_config["patch_size"],
        freeze_backbone=model_config.get("freeze_backbone", False),
    )
    
    # Load adapted checkpoint if specified
    if model_config.get("use_pretrained", True) and "checkpoint_path" in model_config:
        checkpoint_path = Path(model_config["checkpoint_path"]).expanduser()
        
        if checkpoint_path.exists():
            print(f"Loading and adapting checkpoint: {checkpoint_path}")
            adapted_state_dict, metadata = adapt_cra5_checkpoint(checkpoint_path)
            
            # Load adapted weights
            missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print(f"Adapted checkpoint: {metadata.copied_channels} copied, "
                  f"{metadata.learned_boundary_channels} learned boundary channels")
        else:
            print(f"Checkpoint not found: {checkpoint_path}, using random initialization")
    
    return model


def train_epoch(
    model: nn.Module,
    data: torch.Tensor,
    optimizer: optim.Optimizer,
    batch_size: int,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    # Simple reconstruction loss
    criterion = nn.MSELoss()
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(model: nn.Module, data: torch.Tensor, device: str) -> float:
    """Validate model."""
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        data_device = data.to(device)
        reconstruction = model(data_device)
        val_loss = criterion(reconstruction, data_device).item()
    
    return val_loss


def save_metrics(
    output_dir: Path,
    config: dict[str, Any],
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
    training_time: float,
) -> None:
    """Save training metrics to JSON."""
    metrics = {
        "experiment_name": config["experiment"]["name"],
        "subset_size": config["data"]["subset_size"],
        "epochs": len(train_losses),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "training_time_seconds": training_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.json"
    
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train CRA5 adapter")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML file")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Override output dir if specified
    if args.output_dir:
        config["experiment"]["output_dir"] = str(args.output_dir)
    
    output_dir = Path(config["experiment"]["output_dir"])
    
    # Set random seed
    torch.manual_seed(config["experiment"]["seed"])
    
    # Create data
    print("Creating synthetic training data...")
    train_data, val_data = create_synthetic_data(config)
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    
    # Build model
    print("Building CRA5 model...")
    model = build_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    opt_config = config["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(opt_config["lr"]),
        betas=opt_config["betas"],
        weight_decay=float(opt_config["weight_decay"]),
    )
    
    # Setup scheduler if specified
    scheduler = None
    if "scheduler" in config and config["scheduler"]["type"] != "constant":
        sched_config = config["scheduler"]
        if sched_config["type"] == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config["T_max"],
                eta_min=float(sched_config["eta_min"]),
            )
    
    # Training loop
    train_config = config["training"]
    epochs = train_config["epochs"]
    batch_size = train_config["batch_size"]
    val_frequency = train_config["val_frequency"]
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_data, optimizer, batch_size, device)
        train_losses.append(train_loss)
        
        # Validate
        if epoch % val_frequency == 0:
            val_loss = validate(model, val_data, device)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - train_config.get("min_delta", 0.0):
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if train_config.get("save_best", False):
                    output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"best_val={best_val_loss:.6f}")
            
            # Check early stopping
            if patience_counter >= train_config.get("patience", float("inf")):
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Save final metrics
    save_metrics(output_dir, config, train_losses, val_losses, best_val_loss, training_time)
    
    # Test codec if requested
    if config.get("codec", {}).get("test_codec", False):
        print("Testing codec workflow...")
        model.eval()
        test_sample = val_data[:1].to(device)  # Single sample
        
        output_path = output_dir / "test_bitstream.bin"
        try:
            metadata = encode_and_serialize_cra5(
                test_sample,
                model,
                output_path,
                quantization_step=config["codec"]["quantization_step"],
            )
            print(f"Codec test successful: {metadata.bitstream_bytes} bytes, "
                  f"roundtrip_exact={metadata.roundtrip_exact}")
        except Exception as e:
            print(f"Codec test failed: {e}")


if __name__ == "__main__":
    main()