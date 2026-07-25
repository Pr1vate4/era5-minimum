from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from era5_minimum.codec.workflow import run_codec_smoke


def test_train_latent_probe_script_writes_probe_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe_out"
    config = {
        "seed": 7,
        "output_dir": str(output_dir),
        "data": {
            "source": "synthetic",
            "samples": 20,
            "channels": ["t2m", "mslp"],
            "height": 8,
            "width": 8,
            "train_samples": 12,
            "validation_samples": 8,
        },
        "model": {
            "latent_dim": 4,
        },
        "probe": {
            "hidden_dim": 16,
            "batch_size": 4,
            "learning_rate": 0.05,
            "max_steps": 40,
            "parameter_limit": 2000,
        },
    }
    config_path = tmp_path / "latent_probe.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/train_latent_probe.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads((output_dir / "probe_metrics.json").read_text(encoding="utf-8"))
    assert payload["train_pair_count"] == 11
    assert payload["validation_pair_count"] == 7
    assert payload["optimizer_steps"] == 40
    resource_usage = json.loads((output_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource_usage["operation"] == "train_latent_probe"
    assert resource_usage["optimizer_steps"] == 40


def test_train_latent_probe_script_supports_codec_checkpoint_mode(tmp_path: Path) -> None:
    codec_output = tmp_path / "codec_out"
    probe_output = tmp_path / "probe_out"
    run_codec_smoke(
        {
            "seed": 7,
            "output_dir": str(codec_output),
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
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent_probe.py",
            "--codec-checkpoint",
            str(codec_output / "checkpoints" / "model.ckpt"),
            "--pair-count",
            "8",
            "--max-steps",
            "12",
            "--output-dir",
            str(probe_output),
            "--smoke-test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads((probe_output / "probe_metrics.json").read_text(encoding="utf-8"))
    assert payload["train_pair_count"] == 8
    assert payload["optimizer_steps"] == 12
    assert payload["forecast_horizon_hours"] == 6
    assert payload["encoder_frozen"] is True
    assert payload["decoder_frozen"] is True
    resource_usage = json.loads((probe_output / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource_usage["operation"] == "train_latent_probe"
    assert resource_usage["optimizer_steps"] == 12
