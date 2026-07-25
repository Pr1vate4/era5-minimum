#!/usr/bin/env python3
"""Dry-run or fetch the pinned CRA5-159 checkpoint outside the repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.cra5.provenance import (
    CRA5_CHECKPOINT_URL,
    checkpoint_cache_path,
    fetch_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and verify the checkpoint; default mode performs no network access.",
    )
    parser.add_argument("--url", default=CRA5_CHECKPOINT_URL, help="Checkpoint URL.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"External cache directory (default: {checkpoint_cache_path().parent}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = fetch_checkpoint(
            url=args.url,
            cache_dir=args.cache_dir,
            download=args.download,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
