#!/usr/bin/env python3
"""Create a deterministic, leakage-safe ML training sample manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.data.sample_manifest import (  # noqa: E402
    DEFAULT_SAMPLE_SIZES,
    write_training_sample_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the manifest preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ml/training_sample_manifest.json"),
        help="Manifest path (default: outputs/ml/training_sample_manifest.json).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Nested sample sizes; each must be a positive multiple of 16.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the manifest and print a compact provenance summary."""
    args = build_parser().parse_args(argv)
    try:
        path = write_training_sample_manifest(
            args.output,
            sizes=tuple(args.sizes),
            seed=args.seed,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"prepare_ml_sample_manifest: error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_path": str(path),
                "manifest_sha256": payload["integrity"]["manifest_sha256"],
                "seed": payload["seed"],
                "sizes": [int(size) for size in args.sizes],
                "train_only": payload["train_only"],
                "temporal_embargo_hours": payload["temporal_embargo_hours"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
