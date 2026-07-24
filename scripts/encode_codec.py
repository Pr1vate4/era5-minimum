from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec import CodecConfig, CodecHarness, NormalizationSpec, normalize_physical_tensor
from era5_minimum.models import ConvAutoencoder


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_tensor(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "tensor" in data:
            return data["tensor"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError("npz input must contain a single array or a 'tensor' entry")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a tensor into a codec bitstream")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input tensor (.npy or .npz)")
    parser.add_argument("--output", required=True, help="Output bitstream path")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON path")
    args = parser.parse_args()

    checkpoint = _load_checkpoint(Path(args.checkpoint))
    model_cfg = checkpoint["model_config"]
    codec_cfg = checkpoint["codec_config"]
    normalization = NormalizationSpec(**checkpoint["normalization"])
    model = ConvAutoencoder(
        in_channels=int(model_cfg["in_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    tensor = _load_tensor(Path(args.input)).astype(np.float32)
    preprocessing = checkpoint.get("preprocessing") or {}
    ocean_mask = preprocessing.get("ocean_mask")
    sst_index = preprocessing.get("sst_index")
    model_input, _, invalid_value_count = normalize_physical_tensor(
        tensor,
        spec=normalization,
        ocean_mask=ocean_mask,
        sst_index=sst_index,
    )
    with torch.no_grad():
        encoded = model.encode(torch.from_numpy(model_input)).cpu().numpy()
    result = CodecHarness(
        config=CodecConfig(
            version=str(codec_cfg["version"]),
            channel_order=tuple(normalization.channel_order),
            grid=str(codec_cfg["grid"]),
            quantization_step=float(codec_cfg["quantization_step"]),
            seed=int(codec_cfg.get("seed", 0)),
            git_commit=codec_cfg.get("git_commit"),
        ),
        normalization=normalization,
    ).encode_latent(input_tensor=tensor, latent=encoded, output_dir=Path(args.output).parent)

    result.metadata["preprocessing"] = {
        "input_value_space": "physical",
        "model_input_value_space": "normalized",
        "normalization_train_only": normalization.train_only,
        "normalization_source_manifest_sha256": normalization.source_manifest_sha256,
        "invalid_value_count": invalid_value_count,
        "ocean_mask_sha256": preprocessing.get("ocean_mask_sha256"),
    }
    metadata_json = json.dumps(result.metadata, indent=2, sort_keys=True)
    result.metadata_path.write_text(metadata_json, encoding="utf-8")
    Path(args.output).write_bytes(result.bitstream_path.read_bytes())
    if args.metadata is not None:
        Path(args.metadata).write_text(metadata_json, encoding="utf-8")


if __name__ == "__main__":
    main()
