from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec import (
    CanonicalHuffmanCoder,
    NormalizationSpec,
    ScalarQuantizer,
    denormalize_reconstruction,
    decode_latent_tiled,
)
from era5_minimum.models import ConvAutoencoder


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_decode_contract(
    *,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    payload: bytes,
) -> tuple[int, ...]:
    codec_cfg = checkpoint["codec_config"]
    model_cfg = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    metadata_cfg = metadata["config"]

    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != str(metadata["bitstream"]["sha256"]):
        raise ValueError("bitstream checksum does not match metadata")
    if int(metadata["compression"]["bitstream_bytes"]) != len(payload):
        raise ValueError("bitstream byte count does not match metadata")
    if str(metadata_cfg["version"]) != str(codec_cfg["version"]):
        raise ValueError("codec version does not match checkpoint")
    if str(metadata_cfg["grid"]) != str(codec_cfg["grid"]):
        raise ValueError("grid does not match checkpoint")
    if tuple(metadata_cfg["channel_order"]) != tuple(normalization["channel_order"]):
        raise ValueError("channel order does not match checkpoint")
    if float(metadata["quantization"]["scale"]) != float(codec_cfg["quantization_step"]):
        raise ValueError("quantization scale does not match checkpoint")
    if metadata["entropy"]["coder"] != "canonical_huffman":
        raise ValueError("unsupported entropy coder")

    input_shape = tuple(int(value) for value in metadata["input"]["shape"])
    if len(input_shape) != 4 or input_shape[1] != int(model_cfg["in_channels"]):
        raise ValueError("input shape does not match checkpoint")
    latent_shape = tuple(int(value) for value in metadata["latent"]["shape"])
    if len(latent_shape) != 4 or latent_shape[1] != int(model_cfg["latent_channels"]):
        raise ValueError("latent shape does not match checkpoint")
    value_count = math.prod(latent_shape)
    if value_count != int(metadata["latent"]["value_count"]):
        raise ValueError("latent value count does not match latent shape")
    if value_count != int(metadata["compression"]["symbol_count"]):
        raise ValueError("symbol count does not match latent shape")
    return latent_shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a codec bitstream into a tensor")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--bitstream", required=True, help="Path to encoded bitstream")
    parser.add_argument("--metadata", required=True, help="Path to codec metadata JSON")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument("--inference-mode", choices=("full_frame", "tiled"), default=None)
    parser.add_argument("--tile-height", type=int, default=None, help="Output-grid tile height")
    parser.add_argument("--tile-width", type=int, default=None, help="Output-grid tile width")
    parser.add_argument("--halo", type=int, default=None, help="Output-grid halo size")
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
    latent_shape = _validate_decode_contract(
        checkpoint=checkpoint,
        metadata=metadata,
        payload=payload,
    )
    coder = CanonicalHuffmanCoder.from_symbols(np.array([0], dtype=np.int32))
    decoded_symbols = coder.decode(payload)
    if decoded_symbols.size != math.prod(latent_shape):
        raise ValueError("decoded symbol count does not match metadata")
    quantizer = ScalarQuantizer(step=float(metadata["quantization"]["scale"]))
    latent = quantizer.dequantize(decoded_symbols).reshape(latent_shape)
    output_size = tuple(int(value) for value in metadata["input"]["shape"][-2:])
    inference_cfg = checkpoint.get("inference_config") or {}
    inference_mode = args.inference_mode or str(inference_cfg.get("mode", "full_frame"))
    if inference_mode not in {"full_frame", "tiled"}:
        raise ValueError(f"unsupported inference mode: {inference_mode}")
    tile_height = args.tile_height if args.tile_height is not None else inference_cfg.get("tile_height")
    tile_width = args.tile_width if args.tile_width is not None else inference_cfg.get("tile_width")
    halo = args.halo if args.halo is not None else inference_cfg.get("halo")
    latent_tensor = torch.from_numpy(latent.astype(np.float32))
    with torch.no_grad():
        if inference_mode == "tiled":
            if tile_height is None or tile_width is None or halo is None:
                parser.error("tiled inference requires --tile-height, --tile-width, and --halo")
            reconstruction_tensor = decode_latent_tiled(
                latent_tensor,
                output_size=output_size,
                tile_height=int(tile_height),
                tile_width=int(tile_width),
                halo=int(halo),
                scale_factor=8,
                decoder=model.decode,
            )
        else:
            reconstruction_tensor = model.decode(latent_tensor, output_size=output_size)
    reconstruction_normalized = reconstruction_tensor.cpu().numpy()
    preprocessing = checkpoint.get("preprocessing") or {}
    reconstruction = denormalize_reconstruction(
        reconstruction_normalized,
        spec=normalization,
        ocean_mask=preprocessing.get("ocean_mask"),
        sst_index=preprocessing.get("sst_index"),
    )
    np.savez_compressed(
        Path(args.output),
        reconstruction=reconstruction,
        value_space=np.asarray("physical"),
        inference_mode=np.asarray(inference_mode),
        tile_height=np.asarray(-1 if tile_height is None else int(tile_height)),
        tile_width=np.asarray(-1 if tile_width is None else int(tile_width)),
        halo=np.asarray(-1 if halo is None else int(halo)),
    )


if __name__ == "__main__":
    main()
