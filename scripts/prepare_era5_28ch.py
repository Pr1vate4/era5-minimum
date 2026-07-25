#!/usr/bin/env python3
"""Backward-compatible entry point for the canonical 28-channel CLI.

Use ``scripts/data/prepare_era5_28ch.py`` in new commands.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "data" / "prepare_era5_28ch.py"), run_name="__main__")
