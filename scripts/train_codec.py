from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec.workflow import run_codec_smoke


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ERA5 codec smoke pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run on synthetic smoke data")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.smoke_test or config.get("data", {}).get("source", "synthetic") == "synthetic":
        run_codec_smoke(config)
        return
    raise NotImplementedError("real-data codec training is not implemented yet")


if __name__ == "__main__":
    main()
