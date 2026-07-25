"""CRA5-159 to ERA5-28 checkpoint adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from era5_minimum.cra5.channel_mapping import CRA5_CHANNEL_MAPPING, Cra5ChannelMapping


@dataclass(frozen=True)
class AdaptedCheckpointMetadata:
    """Metadata about checkpoint adaptation."""

    source_checkpoint: str
    source_channels: int
    target_channels: int
    copied_channels: int
    learned_boundary_channels: int
    learned_boundary_names: tuple[str, ...]


def adapt_cra5_checkpoint(
    checkpoint_path: Path,
    *,
    mapping: tuple[Cra5ChannelMapping, ...] = CRA5_CHANNEL_MAPPING,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], AdaptedCheckpointMetadata]:
    """
    Load CRA5-159 checkpoint and adapt to ERA5-28.

    Parameters
    ----------
    checkpoint_path : Path
        Path to CRA5-159v checkpoint file.
    mapping : tuple[Cra5ChannelMapping, ...]
        Channel mapping specification. Defaults to canonical mapping.
    device : str
        Target device for adapted tensors. Defaults to 'cpu'.

    Returns
    -------
    adapted_state_dict : dict[str, torch.Tensor]
        State dict with 28-channel input/output projections.
    metadata : AdaptedCheckpointMetadata
        Record of adaptation operation.

    Raises
    ------
    ValueError
        If checkpoint structure is invalid or channel count mismatches.
    FileNotFoundError
        If checkpoint file does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Validate required keys
    required_keys = [
        "backbone.encoder.patch_embed.proj.weight",
        "backbone.decoder.final.weight",
        "backbone.loss.logvar",
    ]
    for key in required_keys:
        if key not in state_dict:
            raise ValueError(f"Missing required key: {key}")

    # Validate source channel count
    input_weight = state_dict["backbone.encoder.patch_embed.proj.weight"]
    if input_weight.shape[1] != 159:
        raise ValueError(f"Expected 159 channels in input projection, got {input_weight.shape[1]}")

    output_weight = state_dict["backbone.decoder.final.weight"]
    if output_weight.shape[1] != 159:
        raise ValueError(f"Expected 159 channels in output projection, got {output_weight.shape[1]}")

    logvar = state_dict["backbone.loss.logvar"]
    if logvar.shape[1] != 159:
        raise ValueError(f"Expected 159 channels in logvar, got {logvar.shape[1]}")

    # Build adapted state dict
    adapted = {}
    target_channels = len(mapping)
    copied_count = 0
    learned_boundary_names = []

    # Adapt input projection: [hidden_dim, 159, kernel_h, kernel_w] -> [hidden_dim, 28, kernel_h, kernel_w]
    adapted_input = torch.zeros(
        input_weight.shape[0],
        target_channels,
        input_weight.shape[2],
        input_weight.shape[3],
        dtype=input_weight.dtype,
        device=device,
    )
    for i, channel_mapping in enumerate(mapping):
        if channel_mapping.initialization == "copy":
            assert channel_mapping.cra5_index is not None
            adapted_input[:, i, :, :] = input_weight[:, channel_mapping.cra5_index, :, :]
            copied_count += 1
        elif channel_mapping.initialization == "learned_boundary":
            # Zero-initialized (already zeros from torch.zeros)
            learned_boundary_names.append(channel_mapping.canonical_name)

    adapted["backbone.encoder.patch_embed.proj.weight"] = adapted_input

    # Copy input projection bias unchanged
    adapted["backbone.encoder.patch_embed.proj.bias"] = state_dict["backbone.encoder.patch_embed.proj.bias"].to(device)

    # Adapt output projection: [hidden_dim, 159, kernel_h, kernel_w] -> [hidden_dim, 28, kernel_h, kernel_w]
    adapted_output = torch.zeros(
        output_weight.shape[0],
        target_channels,
        output_weight.shape[2],
        output_weight.shape[3],
        dtype=output_weight.dtype,
        device=device,
    )
    for i, channel_mapping in enumerate(mapping):
        if channel_mapping.initialization == "copy":
            assert channel_mapping.cra5_index is not None
            adapted_output[:, i, :, :] = output_weight[:, channel_mapping.cra5_index, :, :]
        # learned_boundary remains zero

    adapted["backbone.decoder.final.weight"] = adapted_output

    # Adapt logvar: [1, 159, 1, 1] -> [1, 28, 1, 1]
    adapted_logvar = torch.zeros(
        1,
        target_channels,
        1,
        1,
        dtype=logvar.dtype,
        device=device,
    )
    for i, channel_mapping in enumerate(mapping):
        if channel_mapping.initialization == "copy":
            assert channel_mapping.cra5_index is not None
            adapted_logvar[:, i, :, :] = logvar[:, channel_mapping.cra5_index, :, :]
        # learned_boundary remains zero

    adapted["backbone.loss.logvar"] = adapted_logvar

    # Copy all other keys unchanged
    for key, value in state_dict.items():
        if key not in adapted:
            adapted[key] = value.to(device)

    # Build metadata
    metadata = AdaptedCheckpointMetadata(
        source_checkpoint=str(checkpoint_path),
        source_channels=159,
        target_channels=target_channels,
        copied_channels=copied_count,
        learned_boundary_channels=len(learned_boundary_names),
        learned_boundary_names=tuple(learned_boundary_names),
    )

    return adapted, metadata
