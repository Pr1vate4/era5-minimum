from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


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
