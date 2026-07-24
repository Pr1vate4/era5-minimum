from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec import CanonicalHuffmanCoder, CodecConfig, NormalizationSpec, ScalarQuantizer
from era5_minimum.models import ConvAutoencoder


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a codec bitstream into a tensor")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--bitstream", required=True, help="Path to encoded bitstream")
    parser.add_argument("--metadata", required=True, help="Path to codec metadata JSON")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    args = parser.parse_args()

    checkpoint = _load_checkpoint(Path(args.checkpoint))
    model_cfg = checkpoint["model_config"]
    normalization = NormalizationSpec(**checkpoint["normalization"])
    model = ConvAutoencoder(
        in_channels=int(model_cfg["in_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    payload = Path(args.bitstream).read_bytes()
    coder = CanonicalHuffmanCoder.from_symbols(np.array([0], dtype=np.int32))
    decoded_symbols = coder.decode(payload)
    latent_shape = tuple(metadata["latent"]["shape"])
    quantizer = ScalarQuantizer(step=float(metadata["quantization"]["scale"]))
    latent = quantizer.dequantize(decoded_symbols).reshape(latent_shape)
    reconstruction = model.decode(torch.from_numpy(latent.astype(np.float32))).detach().cpu().numpy()
    np.savez_compressed(Path(args.output), reconstruction=reconstruction)


if __name__ == "__main__":
    main()
