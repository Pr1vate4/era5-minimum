from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_train_codec_script_smoke_creates_run_summary(tmp_path: Path) -> None:
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
    config_path = tmp_path / "codec.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/train_codec.py", "--config", str(config_path), "--smoke-test"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["exact_roundtrip"] is True
    assert run_summary["bitstream_bytes"] > 0
