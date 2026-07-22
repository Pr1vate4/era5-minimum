"""Deprecated compatibility wrapper for :mod:`download_era5`."""

from __future__ import annotations

import sys
from typing import Sequence

from download_era5 import main as unified_main


def main(argv: Sequence[str] | None = None) -> int:
    print("DEPRECATION: use 'python scripts/download_era5.py ...' instead.", file=sys.stderr)
    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
