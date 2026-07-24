from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from era5_minimum.codec.workflow import run_codec_smoke


def test_codec_encode_decode_and_verify_cli_roundtrip(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_out"
    config = {
        "seed": 7,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 8,
            "width": 8,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "model": {
            "latent_channels": 8,
            "parameter_limit": 2_000_000,
        },
        "codec": {
            "version": "ml-001",
            "grid": "smoke-8x8",
            "quantization_step": 0.25,
            "target_compression_ratio": 32,
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 1e-3,
            "max_steps": 4,
        },
        "resources": {
            "max_vram_gb": 24,
            "max_gpu_hours": 48,
        },
    }

    run_codec_smoke(config)
    checkpoint = output_dir / "checkpoints" / "model.ckpt"
    reconstruction = np.load(output_dir / "reconstruction_samples.npz")
    validation_original = reconstruction["validation_original"]
    input_path = tmp_path / "validation.npy"
    np.save(input_path, validation_original)

    bitstream_path = tmp_path / "encoded.bin"
    metadata_path = tmp_path / "encoded.json"
    decode_output = tmp_path / "decoded.npz"

    encode = subprocess.run(
        [
            sys.executable,
            "scripts/encode_codec.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output",
            str(bitstream_path),
            "--metadata",
            str(metadata_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert encode.returncode == 0, encode.stderr
    assert bitstream_path.exists()
    assert metadata_path.exists()

    decode = subprocess.run(
        [
            sys.executable,
            "scripts/decode_codec.py",
            "--checkpoint",
            str(checkpoint),
            "--bitstream",
            str(bitstream_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(decode_output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert decode.returncode == 0, decode.stderr
    assert decode_output.exists()

    verify = subprocess.run(
        [
            sys.executable,
            "scripts/verify_codec_roundtrip.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--bitstream",
            str(bitstream_path),
            "--metadata",
            str(metadata_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["roundtrip"]["exact_symbol_match"] is True


def test_codec_decode_cli_preserves_non_multiple_of_eight_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "codec_out_odd_grid"
    config = {
        "seed": 13,
        "output_dir": str(output_dir),
        "data": {
            "samples": 20,
            "height": 9,
            "width": 16,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "model": {
            "latent_channels": 8,
            "parameter_limit": 2_000_000,
        },
        "codec": {
            "version": "ml-001",
            "grid": "smoke-9x16",
            "quantization_step": 0.25,
            "target_compression_ratio": 32,
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 1e-3,
            "max_steps": 4,
        },
        "resources": {
            "max_vram_gb": 24,
            "max_gpu_hours": 48,
        },
    }

    run_codec_smoke(config)
    checkpoint = output_dir / "checkpoints" / "model.ckpt"
    reconstruction = np.load(output_dir / "reconstruction_samples.npz")
    validation_original = reconstruction["validation_original"]
    input_path = tmp_path / "validation_odd.npy"
    np.save(input_path, validation_original)

    bitstream_path = tmp_path / "encoded_odd.bin"
    metadata_path = tmp_path / "encoded_odd.json"
    decode_output = tmp_path / "decoded_odd.npz"

    encode = subprocess.run(
        [
            sys.executable,
            "scripts/encode_codec.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output",
            str(bitstream_path),
            "--metadata",
            str(metadata_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert encode.returncode == 0, encode.stderr

    decode = subprocess.run(
        [
            sys.executable,
            "scripts/decode_codec.py",
            "--checkpoint",
            str(checkpoint),
            "--bitstream",
            str(bitstream_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(decode_output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert decode.returncode == 0, decode.stderr

    decoded = np.load(decode_output)
    assert decoded["reconstruction"].shape == validation_original.shape
