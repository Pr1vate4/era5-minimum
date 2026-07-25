"""Thread-safe in-memory LRU cache for prepared ERA5 weather layers."""

from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from era5_minimum.api.config import CacheSettings
from era5_minimum.metrics.prometheus import (
    CACHE_BYTES,
    CACHE_ENTRIES,
    CACHE_EVICTIONS_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    LAYER_PREPARE_DURATION_SECONDS,
    RESPONSE_POINTS,
    RESPONSE_SIZE_BYTES,
    ZARR_READ_DURATION_SECONDS,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in real runtime
    np = None


@dataclass(frozen=True)
class _CacheEntry:
    payload: dict[str, Any]
    size: int


def estimate_payload_size(obj: Any) -> int:
    """
    Conservative recursive estimate of in-memory payload size.

    The real provider currently returns JSON-serializable dicts with nested
    lists, so this estimator is intentionally generic.
    """

    seen: set[int] = set()

    def walk(o: Any) -> int:
        if isinstance(o, (dict, list, tuple, set, frozenset)):
            oid = id(o)
            if oid in seen:
                return 0
            seen.add(oid)

        if np is not None:
            if isinstance(o, np.ndarray):
                return int(o.nbytes + sys.getsizeof(o))
            if isinstance(o, np.generic):
                return int(o.nbytes + sys.getsizeof(o))

        if isinstance(o, bytes):
            return sys.getsizeof(o)
        if isinstance(o, bytearray):
            return sys.getsizeof(o)
        if isinstance(o, memoryview):
            return sys.getsizeof(o)
        if isinstance(o, str):
            return sys.getsizeof(o)
        if isinstance(o, (int, float, bool, type(None))):
            return sys.getsizeof(o)

        if isinstance(o, dict):
            return sys.getsizeof(o) + sum(walk(k) + walk(v) for k, v in o.items())

        if isinstance(o, (list, tuple, set, frozenset)):
            return sys.getsizeof(o) + sum(walk(item) for item in o)

        if hasattr(o, "__dict__"):
            return sys.getsizeof(o) + walk(o.__dict__)

        if hasattr(o, "__slots__"):
            size = sys.getsizeof(o)
            slots = getattr(o, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if hasattr(o, slot):
                    size += walk(getattr(o, slot))
            return size

        try:
            return sys.getsizeof(o)
        except TypeError:
            return 0

    return max(0, walk(obj))


class WeatherDataService:
    """
    Cache layer between API endpoints and WeatherDataProvider.

    Cache key:
        (
            dataset_id,
            variable,
            timestamp,
            level,
            mode,
            target_width,
            target_height,
            stride,
            format,
        )
    """

    def __init__(self, provider: Any, settings: CacheSettings | None = None) -> None:
        self.provider = provider
        self.settings = settings or CacheSettings.from_env()

        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = 0

        self._max_entries = self.settings.max_entries
        if self._max_entries is not None and self._max_entries < 0:
            self._max_entries = None

        self._max_bytes = self.settings.max_bytes
        if self._max_bytes is not None and self._max_bytes <= 0:
            self._max_bytes = None

        self._provider_name = self._resolve_provider_name(provider)
        self._dataset_id = self._resolve_dataset_id(provider)

    @staticmethod
    def _resolve_provider_name(provider: Any) -> str:
        if getattr(provider, "is_mock", False):
            return "mock"
        if "mock" in type(provider).__name__.lower():
            return "mock"
        return "zarr"

    @staticmethod
    def _resolve_dataset_id(provider: Any) -> str:
        try:
            meta = provider.dataset_metadata()
            return str(meta.get("id", "unknown"))
        except Exception:
            return "unknown"

    def clear_cache(self) -> None:
        """Manual invalidation entrypoint required by SSoT."""
        with self._lock:
            entries = len(self._cache)
            size = self._bytes

            self._cache.clear()
            self._bytes = 0

            if entries:
                CACHE_ENTRIES.dec(entries)
            if size:
                CACHE_BYTES.dec(size)

    def get_layer(
        self,
        variable: str,
        timestamp: str,
        level: int | None = None,
        mode: str = "original",
        target_width: int = 360,
        target_height: int = 180,
        stride: int = 1,
        response_format: str = "json",
    ) -> dict[str, Any]:
        """Backward-compatible helper returning only payload."""
        payload, _ = self.get_layer_with_status(
            variable=variable,
            timestamp=timestamp,
            level=level,
            mode=mode,
            target_width=target_width,
            target_height=target_height,
            stride=stride,
            response_format=response_format,
        )
        return payload

    def get_layer_with_status(
        self,
        variable: str,
        timestamp: str,
        level: int | None = None,
        mode: str = "original",
        target_width: int = 360,
        target_height: int = 180,
        stride: int = 1,
        response_format: str = "json",
    ) -> tuple[dict[str, Any], str]:
        """
        Return `(payload, cache_status)` where cache_status is "hit" or "miss".
        """

        start = time.perf_counter()

        variable_group = "surface" if level is None else "pressure"
        labels = self._base_labels(
            mode=mode,
            response_format=response_format,
            variable_group=variable_group,
        )
        points = max(0, int(target_width)) * max(0, int(target_height))

        if not self.settings.enabled:
            payload = self._read_and_observe_provider(
                variable=variable,
                timestamp=timestamp,
                level=level,
                target_width=target_width,
                target_height=target_height,
                variable_group=variable_group,
            )
            size = estimate_payload_size(payload)

            LAYER_PREPARE_DURATION_SECONDS.labels(**labels, status="miss").observe(
                time.perf_counter() - start
            )
            RESPONSE_SIZE_BYTES.labels(**labels).observe(size)
            RESPONSE_POINTS.labels(**labels).observe(points)

            return payload, "miss"

        key = (
            self._dataset_id,
            variable,
            timestamp,
            level,
            mode,
            target_width,
            target_height,
            stride,
            response_format,
        )

        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)

                CACHE_HITS_TOTAL.labels(**labels).inc()
                LAYER_PREPARE_DURATION_SECONDS.labels(**labels, status="hit").observe(
                    time.perf_counter() - start
                )
                RESPONSE_SIZE_BYTES.labels(**labels).observe(entry.size)
                RESPONSE_POINTS.labels(**labels).observe(points)

                return entry.payload, "hit"

        CACHE_MISSES_TOTAL.labels(**labels).inc()

        payload = self._read_and_observe_provider(
            variable=variable,
            timestamp=timestamp,
            level=level,
            target_width=target_width,
            target_height=target_height,
            variable_group=variable_group,
        )
        size = estimate_payload_size(payload)

        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                payload = existing.payload
                size = existing.size
            else:
                self._store_unlocked(key=key, payload=payload, size=size)

        LAYER_PREPARE_DURATION_SECONDS.labels(**labels, status="miss").observe(
            time.perf_counter() - start
        )
        RESPONSE_SIZE_BYTES.labels(**labels).observe(size)
        RESPONSE_POINTS.labels(**labels).observe(points)

        return payload, "miss"

    def _base_labels(
        self,
        *,
        mode: str,
        response_format: str,
        variable_group: str,
    ) -> dict[str, str]:
        return {
            "provider": self._provider_name,
            "mode": mode,
            "format": response_format,
            "variable_group": variable_group,
        }

    def _read_and_observe_provider(
        self,
        *,
        variable: str,
        timestamp: str,
        level: int | None,
        target_width: int,
        target_height: int,
        variable_group: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return self.provider.layer(
                variable,
                timestamp,
                level,
                target_width,
                target_height,
            )
        finally:
            ZARR_READ_DURATION_SECONDS.labels(
                provider=self._provider_name,
                variable_group=variable_group,
            ).observe(time.perf_counter() - start)

    def _store_unlocked(
        self,
        *,
        key: tuple,
        payload: dict[str, Any],
        size: int,
    ) -> None:
        old = self._cache.get(key)
        if old is not None:
            del self._cache[key]
            self._bytes -= old.size
            CACHE_ENTRIES.dec()
            CACHE_BYTES.dec(old.size)

        self._cache[key] = _CacheEntry(payload=payload, size=size)
        self._bytes += size

        CACHE_ENTRIES.inc()
        CACHE_BYTES.inc(size)

        self._evict_unlocked()

    def _evict_unlocked(self) -> None:
        while self._cache:
            too_many = (
                self._max_entries is not None
                and len(self._cache) > self._max_entries
            )
            too_large = (
                self._max_bytes is not None
                and self._bytes > self._max_bytes
            )

            if not (too_many or too_large):
                break

            _, evicted = self._cache.popitem(last=False)

            self._bytes -= evicted.size
            CACHE_EVICTIONS_TOTAL.inc()
            CACHE_ENTRIES.dec()
            CACHE_BYTES.dec(evicted.size)