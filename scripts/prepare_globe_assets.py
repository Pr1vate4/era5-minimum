"""Prepare browser-sized globe textures and numeric layers from local arrays."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SURFACE_CHANNELS = {"t2m", "mslp", "u10", "v10", "wind10", "tp6h", "sst", "tcwv", "tcc"}
PRESSURE_CHANNELS = {"T", "U", "V", "Z", "Q"}
PRESSURE_LEVELS = {1000, 925, 850, 700}
GRIDS = {"0p25", "0p5"}
MODES = {"original", "reconstruction", "absolute-error"}
NO_DATA_VALUE = np.float32(-3.4028235e38)

PALETTES: dict[str, list[str]] = {
    "temperature": ["#172554", "#2563EB", "#22D3EE", "#10B981", "#FACC15", "#F97316", "#DC2626"],
    "pressure": ["#5B21B6", "#2563EB", "#BAE6FD", "#F8FAFC", "#FDBA74", "#DC2626"],
    "wind": ["#172554", "#2563EB", "#BAE6FD", "#F8FAFC", "#FDBA74", "#DC2626"],
    "speed": ["#ECFEFF", "#67E8F9", "#22C55E", "#FACC15", "#F97316", "#B91C1C"],
    "precipitation": ["#F8FAFC", "#BAE6FD", "#38BDF8", "#2563EB", "#7E22CE", "#BE123C"],
    "humidity": ["#F0FDFA", "#99F6E4", "#2DD4BF", "#0EA5E9", "#1D4ED8", "#312E81"],
    "cloud": ["#334155", "#64748B", "#94A3B8", "#CBD5E1", "#F8FAFC"],
    "geopotential": ["#0F766E", "#22C55E", "#A3E635", "#FACC15", "#F97316", "#991B1B"],
    "error": ["#FFF7ED", "#FED7AA", "#FB923C", "#EF4444", "#991B1B"],
}

DEFAULT_UNITS = {
    "t2m": "K",
    "mslp": "Pa",
    "u10": "m/s",
    "v10": "m/s",
    "wind10": "m/s",
    "tp6h": "mm/6h",
    "sst": "K",
    "tcwv": "kg/m2",
    "tcc": "1",
    "T": "K",
    "U": "m/s",
    "V": "m/s",
    "Z": "m2/s2",
    "Q": "kg/kg",
}


@dataclass(frozen=True)
class LoadedField:
    values: np.ndarray
    latitude_order: str
    longitude_range: str
    unit: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an equirectangular globe texture, Float32 values, and manifest entry.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Local .npy, .npz, or Zarr input.")
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Generate a deterministic visual demo field (never labelled ERA5).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output globe directory.")
    parser.add_argument("--channel", required=True, choices=sorted(SURFACE_CHANNELS | PRESSURE_CHANNELS))
    parser.add_argument("--timestamp", required=True, help="ISO-8601 timestamp.")
    parser.add_argument("--grid", required=True, choices=sorted(GRIDS))
    parser.add_argument("--level", type=int, choices=sorted(PRESSURE_LEVELS))
    parser.add_argument("--mode", choices=sorted(MODES), default="original")
    parser.add_argument("--array-key", help="Array key for .npz or variable name for Zarr.")
    parser.add_argument("--unit", help="Physical unit stored in the input.")
    parser.add_argument("--run-id")
    parser.add_argument("--train-frames", type=int)
    parser.add_argument("--compression-ratio", type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--normalization-mean", type=float)
    parser.add_argument("--normalization-std", type=float)
    parser.add_argument(
        "--public-prefix",
        default="data/globe",
        help="Public URL prefix written to manifest (default: data/globe).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    try:
        loaded = (
            generate_demo_field(args.channel, args.grid, args.mode, args.unit)
            if args.demo
            else load_input_field(
                args.input,
                args.channel,
                args.timestamp,
                args.level,
                args.array_key,
                args.unit,
            )
        )
        frame = write_assets(args, loaded)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Prepared frame: {frame['id']}")
    print(f"Texture: {frame['textureUrl']}")
    print(f"Values: {frame['valuesUrl']}")
    print(f"Manifest: {args.output / 'manifest.json'}")
    if args.demo:
        print("Source: demo (not ERA5)")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    timestamp = args.timestamp.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("--timestamp must be a valid ISO-8601 value") from error

    if args.channel in PRESSURE_CHANNELS and args.level is None:
        raise ValueError(f"--level is required for pressure variable {args.channel}")
    if args.channel in SURFACE_CHANNELS and args.level is not None:
        raise ValueError(f"--level is not applicable to surface channel {args.channel}")
    if args.normalization_std is not None and args.normalization_std <= 0:
        raise ValueError("--normalization-std must be positive")
    if (args.normalization_mean is None) != (args.normalization_std is None):
        raise ValueError("normalization mean and std must be provided together")
    if args.mode != "original" and not args.demo and not args.run_id:
        raise ValueError("--run-id is required for reconstruction and absolute-error assets")


def load_input_field(
    path: Path,
    channel: str,
    timestamp: str,
    level: int | None,
    array_key: str | None,
    unit: str | None,
) -> LoadedField:
    if not path.exists():
        raise ValueError(f"input does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
        return normalize_loaded_array(values, unit or DEFAULT_UNITS[channel])
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = array_key or channel
            if key not in archive:
                raise KeyError(f"array {key!r} not found; available: {', '.join(archive.files)}")
            return normalize_loaded_array(archive[key], unit or DEFAULT_UNITS[channel])
    return load_zarr_field(path, channel, timestamp, level, array_key, unit)


def load_zarr_field(
    path: Path,
    channel: str,
    timestamp: str,
    level: int | None,
    array_key: str | None,
    unit: str | None,
) -> LoadedField:
    try:
        import xarray as xr
    except ImportError as error:
        raise ImportError("xarray and zarr are required to read Zarr inputs") from error

    dataset = xr.open_zarr(path)
    variable = array_key or channel
    if variable not in dataset:
        raise KeyError(f"variable {variable!r} not found in Zarr dataset")
    field = dataset[variable]

    time_dimension = next((name for name in ("time", "timestamp", "valid_time") if name in field.dims), None)
    if time_dimension:
        field = field.sel({time_dimension: np.datetime64(timestamp)}, method=None)

    level_dimension = next(
        (name for name in ("level", "pressure_level", "isobaricInhPa") if name in field.dims),
        None,
    )
    if level is not None:
        if not level_dimension:
            raise ValueError(f"pressure level dimension not found for {channel}")
        field = field.sel({level_dimension: level})

    latitude_name = next((name for name in ("latitude", "lat") if name in field.dims), None)
    longitude_name = next((name for name in ("longitude", "lon") if name in field.dims), None)
    if not latitude_name or not longitude_name:
        raise ValueError("input must expose latitude/longitude dimensions")

    field = field.transpose(latitude_name, longitude_name).load()
    latitudes = np.asarray(field[latitude_name].values)
    longitudes = np.asarray(field[longitude_name].values)
    latitude_order = "north-to-south" if latitudes[0] > latitudes[-1] else "south-to-north"
    longitude_range = "0-360" if float(np.nanmin(longitudes)) >= 0 else "-180-180"
    source_unit = unit or str(field.attrs.get("units") or DEFAULT_UNITS[channel])
    return normalize_loaded_array(
        np.asarray(field.values),
        source_unit,
        latitude_order=latitude_order,
        longitude_range=longitude_range,
    )


def normalize_loaded_array(
    values: np.ndarray,
    unit: str,
    *,
    latitude_order: str = "north-to-south",
    longitude_range: str = "-180-180",
) -> LoadedField:
    array = np.asarray(values, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"selected field must be two-dimensional, got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 4:
        raise ValueError(f"selected field is too small for a globe texture: {array.shape}")
    if not np.isfinite(array).any():
        raise ValueError("selected field contains no finite values")
    return LoadedField(array, latitude_order, longitude_range, unit)


def generate_demo_field(channel: str, grid: str, mode: str, unit: str | None) -> LoadedField:
    width = 720 if grid == "0p5" else 1440
    height = width // 2 + 1
    latitudes = np.linspace(90.0, -90.0, height, dtype=np.float32)
    longitudes = np.linspace(-180.0, 180.0, width, endpoint=False, dtype=np.float32)
    latitude, longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    wave = np.sin(2.5 * lon_rad + 0.7 * np.sin(lat_rad * 3)) * np.cos(lat_rad)
    planetary = np.cos(3 * lon_rad - 1.2 * lat_rad) * np.cos(lat_rad) ** 2

    if channel in {"t2m", "T", "sst"}:
        original = 273.15 + 30 * np.cos(lat_rad) - 8 * np.sin(lat_rad) ** 2 + 5 * wave
    elif channel == "mslp":
        original = 101325 + 2200 * wave + 900 * planetary
    elif channel in {"u10", "U"}:
        original = 25 * np.sin(2 * lat_rad) + 8 * wave
    elif channel in {"v10", "V"}:
        original = 16 * planetary
    elif channel == "wind10":
        original = np.abs(20 * np.sin(2 * lat_rad)) + 6 * (wave + 1)
    elif channel == "tp6h":
        original = np.maximum(0, 8 * (wave + planetary - 0.5)) ** 1.35
    elif channel == "tcwv":
        original = np.maximum(0, 55 * np.cos(lat_rad) ** 2 + 8 * wave)
    elif channel == "tcc":
        original = np.clip(0.45 + 0.35 * wave + 0.2 * planetary, 0, 1)
    elif channel == "Q":
        original = np.maximum(0, 0.018 * np.cos(lat_rad) ** 4 + 0.002 * wave)
    else:
        original = 50000 + 7000 * np.cos(lat_rad) + 1800 * planetary

    reconstruction = original + 0.025 * np.nanstd(original) * (
        np.sin(5 * lon_rad) * np.cos(2 * lat_rad)
    )
    if mode == "reconstruction":
        values = reconstruction
    elif mode == "absolute-error":
        values = np.abs(original - reconstruction)
    else:
        values = original
    return LoadedField(
        np.asarray(values, dtype=np.float32),
        "north-to-south",
        "-180-180",
        unit or DEFAULT_UNITS[channel],
    )


def write_assets(args: argparse.Namespace, loaded: LoadedField) -> dict[str, Any]:
    values = np.asarray(loaded.values, dtype=np.float32)
    finite = np.isfinite(values)
    stats = values[finite]
    minimum = float(np.min(stats))
    maximum = float(np.max(stats))
    mean = float(np.mean(stats, dtype=np.float64))
    timestamp_slug = re.sub(r"[^0-9A-Za-z]+", "-", args.timestamp).strip("-")
    channel_slug = f"{args.channel}{args.level or ''}"
    mode_directory = "error" if args.mode == "absolute-error" else args.mode
    source_directory = "demo" if args.demo else mode_directory
    relative_directory = Path(source_directory) / mode_directory / args.grid
    output_directory = args.output / relative_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{channel_slug}_{timestamp_slug}"
    if args.run_id and args.mode != "original":
        stem += f"_{safe_slug(args.run_id)}"
    texture_path = output_directory / f"{stem}.png"
    values_path = output_directory / f"{stem}.bin"

    write_texture(values, texture_path, args.channel, args.mode, minimum, maximum)
    encoded_values = np.where(finite, values, NO_DATA_VALUE).astype("<f4", copy=False)
    encoded_values.tofile(values_path)

    public_prefix = args.public_prefix.strip("/")
    texture_url = (Path(public_prefix) / relative_directory / texture_path.name).as_posix()
    values_url = (Path(public_prefix) / relative_directory / values_path.name).as_posix()
    source = "demo" if args.demo else ("ERA5" if args.mode == "original" else "model")
    frame_id = "-".join(
        part
        for part in (
            args.mode,
            args.run_id if args.mode != "original" else None,
            channel_slug,
            args.grid,
            timestamp_slug,
        )
        if part
    )
    frame: dict[str, Any] = {
        "id": frame_id,
        "mode": args.mode,
        "channel": args.channel,
        "timestamp": normalize_timestamp(args.timestamp),
        "grid": args.grid,
        "textureUrl": texture_url,
        "valuesUrl": values_url,
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "unit": loaded.unit,
        "source": source,
        "latitudeOrder": loaded.latitude_order,
        "longitudeRange": loaded.longitude_range,
        "valueEncoding": "float32-le",
        "noDataValue": float(NO_DATA_VALUE),
    }
    add_optional_metadata(frame, args)
    update_manifest(args.output / "manifest.json", frame)
    return frame


def write_texture(
    values: np.ndarray,
    output_path: Path,
    channel: str,
    mode: str,
    minimum: float,
    maximum: float,
) -> None:
    try:
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow and matplotlib are required to create globe textures") from error

    palette_name = choose_palette(channel, mode)
    color_map = LinearSegmentedColormap.from_list(palette_name, PALETTES[palette_name], N=256)
    finite = np.isfinite(values)
    denominator = maximum - minimum
    normalized = np.zeros_like(values, dtype=np.float32)
    if math.isfinite(denominator) and denominator > 0:
        normalized[finite] = np.clip((values[finite] - minimum) / denominator, 0, 1)
    rgba = np.asarray(color_map(normalized, bytes=True), dtype=np.uint8)
    rgba[~finite, 3] = 0
    image = Image.fromarray(rgba, mode="RGBA")
    texture_height = max(1, image.width // 2)
    if image.height != texture_height:
        image = image.resize((image.width, texture_height), Image.Resampling.BILINEAR)
    image.save(output_path, optimize=True)


def choose_palette(channel: str, mode: str) -> str:
    if mode == "absolute-error":
        return "error"
    if channel in {"t2m", "T", "sst"}:
        return "temperature"
    if channel == "mslp":
        return "pressure"
    if channel in {"u10", "v10", "U", "V"}:
        return "wind"
    if channel == "wind10":
        return "speed"
    if channel == "tp6h":
        return "precipitation"
    if channel in {"tcwv", "Q"}:
        return "humidity"
    if channel == "tcc":
        return "cloud"
    return "geopotential"


def add_optional_metadata(frame: dict[str, Any], args: argparse.Namespace) -> None:
    if args.level is not None:
        frame["level"] = args.level
    if args.run_id:
        frame["runId"] = args.run_id
    if args.train_frames is not None:
        frame["trainFrames"] = args.train_frames
    if args.compression_ratio is not None:
        frame["compressionRatio"] = args.compression_ratio
    if args.checkpoint:
        frame["checkpoint"] = args.checkpoint
    if args.normalization_mean is not None and args.normalization_std is not None:
        frame["normalization"] = {
            "mean": args.normalization_mean,
            "std": args.normalization_std,
        }


def update_manifest(path: Path, frame: dict[str, Any]) -> None:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
            raise ValueError(f"invalid existing manifest: {path}")
    else:
        manifest = {"version": 1, "frames": []}

    frames = [candidate for candidate in manifest["frames"] if candidate.get("id") != frame["id"]]
    frames.append(frame)
    frames.sort(key=lambda item: (item["timestamp"], item["mode"], item["channel"], item.get("level", 0)))
    manifest["version"] = 1
    manifest["generatedAt"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest["frames"] = frames
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def normalize_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def safe_slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
