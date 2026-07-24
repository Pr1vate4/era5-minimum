from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from era5_minimum.experiments import run


def test_pca_run_writes_codec_artifacts_and_actual_ratio(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config = {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(output_dir),
        "data": {
            "source": "synthetic",
            "samples": 16,
            "channels": ["t2m", "mslp"],
            "height": 8,
            "width": 8,
            "validation_samples": 4,
        },
        "model": {
            "type": "pca",
            "latent_dim": 4,
            "codec": {
                "version": "ml-001",
                "grid": "synthetic-8x8",
                "quantization_step": 0.25,
            },
        },
        "training": {
            "train_sizes": [8],
        },
    }
    config_path = tmp_path / "codec_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows = run(config_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["latent_reduction_ratio"] == pytest.approx(32.0)
    assert row["actual_compression_ratio"] > 0.0
    assert row["codec_roundtrip_exact"] is True

    codec_dir = output_dir / "pca_train_8_codec"
    payload = json.loads((codec_dir / "codec.json").read_text(encoding="utf-8"))
    assert payload["compression"]["latent_reduction_ratio"] == pytest.approx(row["latent_reduction_ratio"])
    assert payload["compression"]["actual_compression_ratio"] == pytest.approx(row["actual_compression_ratio"])
