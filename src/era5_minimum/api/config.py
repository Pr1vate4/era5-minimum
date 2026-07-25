"""Cache-related runtime configuration for ERA5 layer serving."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CacheSettings:
    """
    SSoT settings:

    ERA5_LAYER_CACHE_ENABLED=true
    ERA5_LAYER_CACHE_MAX_ENTRIES=32
    ERA5_LAYER_CACHE_MAX_BYTES=268435456
    """

    enabled: bool = True
    max_entries: int | None = 32
    max_bytes: int | None = 268_435_456

    @classmethod
    def from_env(cls) -> "CacheSettings":
        return cls(
            enabled=_env_bool("ERA5_LAYER_CACHE_ENABLED", True),
            max_entries=_env_int("ERA5_LAYER_CACHE_MAX_ENTRIES", 32),
            max_bytes=_env_int("ERA5_LAYER_CACHE_MAX_BYTES", 268_435_456),
        )