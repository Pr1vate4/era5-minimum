from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec import (
    CanonicalHuffmanCoder,
    CodecConfig,
    CodecHarness,
    NormalizationSpec,
    normalize_physical_tensor,
)
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
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify codec bitstream roundtrip")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input tensor (.npy or .npz)")
    parser.add_argument("--bitstream", required=True, help="Encoded bitstream")
    parser.add_argument("--metadata", required=True, help="Codec metadata JSON")
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
    model_input, _, _ = normalize_physical_tensor(
        tensor,
        spec=normalization,
        ocean_mask=preprocessing.get("ocean_mask"),
        sst_index=preprocessing.get("sst_index"),
    )
    with torch.no_grad():
        latent = model.encode(torch.from_numpy(model_input)).cpu().numpy()
    codec = CodecHarness(
        config=CodecConfig(
            version=str(codec_cfg["version"]),
            channel_order=tuple(normalization.channel_order),
            grid=str(codec_cfg["grid"]),
            quantization_step=float(codec_cfg["quantization_step"]),
            seed=int(codec_cfg.get("seed", 0)),
            git_commit=codec_cfg.get("git_commit") or checkpoint.get("git_commit"),
        ),
        normalization=normalization,
    )
    result = codec.encode_latent(input_tensor=tensor, latent=latent, output_dir=Path(args.bitstream).parent)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    decoder = CanonicalHuffmanCoder.from_symbols(np.array([0], dtype=np.int32))
    decoded = decoder.decode(Path(args.bitstream).read_bytes())
    assert result.decoded_latent is not None
    assert np.array_equal(decoded, codec.quantizer.quantize(latent).ravel())
    assert np.array_equal(result.decoded_latent, codec.quantizer.dequantize(decoded).reshape(metadata["latent"]["shape"]))


if __name__ == "__main__":
    main()
