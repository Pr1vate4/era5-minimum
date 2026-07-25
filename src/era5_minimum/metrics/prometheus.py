"""Prometheus metrics for ERA5 layer cache and layer preparation."""

from prometheus_client import Counter, Gauge, Histogram


CACHE_HITS_TOTAL = Counter(
    "era5_layer_cache_hits_total",
    "Number of ERA5 layer cache hits.",
    ["provider", "mode", "format", "variable_group"],
)

CACHE_MISSES_TOTAL = Counter(
    "era5_layer_cache_misses_total",
    "Number of ERA5 layer cache misses.",
    ["provider", "mode", "format", "variable_group"],
)

CACHE_EVICTIONS_TOTAL = Counter(
    "era5_layer_cache_evictions_total",
    "Number of ERA5 layer cache evictions caused by LRU limits.",
)

CACHE_ENTRIES = Gauge(
    "era5_layer_cache_entries",
    "Current number of entries in the ERA5 layer cache.",
)

CACHE_BYTES = Gauge(
    "era5_layer_cache_bytes",
    "Current estimated size of the ERA5 layer cache in bytes.",
)

ZARR_READ_DURATION_SECONDS = Histogram(
    "era5_zarr_read_duration_seconds",
    "Time spent reading a layer from Zarr or an equivalent provider.",
    ["provider", "variable_group"],
)

LAYER_PREPARE_DURATION_SECONDS = Histogram(
    "era5_layer_prepare_duration_seconds",
    "Total time spent preparing an ERA5 layer response.",
    ["provider", "mode", "format", "variable_group", "status"],
)

RESPONSE_SIZE_BYTES = Histogram(
    "era5_layer_response_size_bytes",
    "Estimated ERA5 layer response size in bytes.",
    ["provider", "mode", "format", "variable_group"],
    buckets=(
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
        268435456,
    ),
)

RESPONSE_POINTS = Histogram(
    "era5_layer_response_points",
    "Number of grid points in the ERA5 layer response.",
    ["provider", "mode", "format", "variable_group"],
    buckets=(
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
    ),
)