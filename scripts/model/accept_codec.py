#!/usr/bin/env python3
"""Run a strict, checkpoint-backed ERA5 codec acceptance pass.

The command is deliberately model-only: it neither trains nor downloads data,
and only writes compact, ignored acceptance artifacts below ``outputs``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from era5_minimum.codec.acceptance import (
    AcceptanceError,
    WeatherCodec,
    compression_metrics,
    evaluate_physical_reconstruction,
    sha256_file,
)
from era5_minimum.codec.resources import measure_runtime_resources
from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.data.era5_28ch import assemble_model_tensor, validate_prepared


EXPECTED_DEMO_TIMESTAMPS = (
    "2020-01-01T00:00:00.000000000",
    "2020-01-01T06:00:00.000000000",
    "2020-01-01T12:00:00.000000000",
    "2020-01-01T18:00:00.000000000",
)
OUTPUT_FILES = (
    "acceptance_report.md",
    "model_info.json",
    "dataset_info.json",
    "run_config.json",
    "metrics.json",
    "metrics_per_timestamp.json",
    "timings.json",
    "resource_usage.json",
    "compression.json",
    "roundtrip.json",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional immutable model config recorded by checksum.")
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timestamp", action="append", help="One timestamp; repeat for several timestamps.")
    group.add_argument("--timestamps", choices=("all",), help="Use all split timestamps in chronological order.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-height", type=int, default=128)
    parser.add_argument("--crop-width", type=int, default=128)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in this acceptance output only.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-bitstream", action="store_true", help="Structural diagnostic only; never accepts the codec.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least one")
    if not args.full_frame and (args.crop_height < 1 or args.crop_width < 1):
        raise ValueError("crop dimensions must be positive")
    output = args.output_dir.resolve()
    project_outputs = (Path.cwd() / "outputs" / "model_acceptance").resolve()
    if project_outputs not in (output, *output.parents):
        raise ValueError("--output-dir must be below outputs/model_acceptance")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"acceptance output exists and is non-empty: {output}; omit reuse or pass --overwrite")


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("bitstreams", "previews", "logs"):
        (output / name).mkdir(exist_ok=True)


def _seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _dataset_info(root: Path, split: str, *, full_frame: bool, crop_height: int, crop_width: int) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    if not (root / f"{split}.zarr").is_dir() or not (root / "static.zarr").is_dir():
        raise FileNotFoundError(f"dataset split/static Zarr stores are missing under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channel_order") != list(CHANNEL_NAMES):
        raise AcceptanceError("dataset manifest channel order differs from official 28-channel contract")
    expected_hash_path = root / "manifest.json.sha256"
    manifest_hash = sha256_file(manifest_path)
    recorded_hash = expected_hash_path.read_text(encoding="utf-8").split()[0] if expected_hash_path.is_file() else None
    if recorded_hash and manifest_hash != recorded_hash:
        raise AcceptanceError("dataset manifest checksum mismatch")
    scope = "full_grid" if full_frame else "crop_smoke"
    return {
        "source": manifest.get("source_uri"),
        "dataset_root": str(root),
        "manifest_sha256": manifest_hash,
        "split": split,
        "declared_split": manifest.get("splits", {}).get(split),
        "timestamps": [],
        "grid": manifest.get("grid_shape"),
        "resolution": manifest.get("resolution"),
        "pressure_levels_hpa": manifest.get("pressure_levels_hpa"),
        "channel_order": list(CHANNEL_NAMES),
        "evaluation_scope": scope,
        "not_full_grid": not full_frame,
        "crop": None if full_frame else {"height": crop_height, "width": crop_width, "origin": [0, 0]},
        "physical_store_unchanged": True,
    }


def _select_timestamps(ds: xr.Dataset, args: argparse.Namespace) -> list[np.datetime64]:
    available = list(ds.time.values)
    if args.timestamp:
        selected: list[np.datetime64] = []
        missing: list[str] = []
        for requested in args.timestamp:
            try:
                requested_value = np.datetime64(requested)
            except ValueError:
                missing.append(requested)
                continue
            match = next((value for value in available if value == requested_value), None)
            if match is None:
                missing.append(requested)
            else:
                selected.append(match)
        if missing:
            raise ValueError(f"requested timestamps are absent from {args.split}: {missing}")
    elif args.timestamps == "all":
        selected = available
    else:
        selected = available[:1]
    selected = sorted(selected)
    return selected[: args.max_samples]


def _load_crop(
    dynamic: xr.Dataset,
    static: xr.Dataset,
    timestamp: np.datetime64,
    *,
    full_frame: bool,
    crop_height: int,
    crop_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = int(dynamic.sizes["latitude"]), int(dynamic.sizes["longitude"])
    if full_frame:
        row_stop, col_stop = height, width
    else:
        row_stop, col_stop = min(crop_height, height), min(crop_width, width)
    frame_ds = dynamic.sel(time=timestamp).isel(latitude=slice(0, row_stop), longitude=slice(0, col_stop))
    physical = assemble_model_tensor(frame_ds.expand_dims(time=[timestamp])).load().values.astype(np.float32, copy=False)
    ocean = static["ocean_mask"].isel(latitude=slice(0, row_stop), longitude=slice(0, col_stop)).load().values.astype(bool, copy=False)
    latitudes = dynamic.latitude.isel(latitude=slice(0, row_stop)).values.astype(np.float64, copy=False)
    if physical.shape[1:] != (len(CHANNEL_NAMES), row_stop, col_stop):
        raise AcceptanceError(f"adapter returned unexpected tensor shape: {physical.shape}")
    return physical, ocean, latitudes


def _checksums(symbols: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(symbols, dtype=np.int32).tobytes()).hexdigest()


def _mean_compression(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"per_timestamp": [], "aggregate": None}
    keys = ("input_bits_float32", "payload_bytes", "header_bytes", "side_information_bytes", "total_bitstream_bytes")
    totals = {key: sum(int(row[key]) for row in rows) for key in keys}
    value_count = sum(int(np.prod(row["original_shape"])) for row in rows)
    total_ratio = totals["input_bits_float32"] / (8 * totals["total_bitstream_bytes"])
    return {
        "per_timestamp": rows,
        "aggregate": {
            **totals,
            "bits_per_value": 8 * totals["total_bitstream_bytes"] / value_count,
            "compression_ratio": total_ratio,
            "target_32x_reached": total_ratio >= 32.0,
            "target_64x_reached": total_ratio >= 64.0,
        },
    }


def _report(*, status: str, blockers: list[str], output: Path, run: dict[str, Any]) -> str:
    lines = [
        "# ERA5-Minimum model/codec acceptance",
        "",
        f"**Status: {status}**",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Evaluation scope: `{run['evaluation_scope']}`",
        f"- Dataset: `{run['dataset_root']}`",
        f"- Checkpoint: `{run['checkpoint']}`",
        f"- Device: `{run['device']}`",
        f"- Timestamp count requested: {run['requested_samples']}",
        "",
        "## Result",
        "",
    ]
    if blockers:
        lines.extend([f"- {item}" for item in blockers])
    else:
        lines.append("- Mandatory checkpoint-backed codec checks completed; see JSON artifacts for exact values.")
    lines.extend([
        "",
        "## Integrity and interpretation",
        "",
        "- The compression ratio uses all bytes in each standalone stream: `32 × T × C × H × W / (8 × total_bitstream_bytes)`.",
        "- Tensor element reduction is not reported as a binary compression ratio.",
        "- The decoder requires the checkpoint whose SHA256 is recorded in `model_info.json`; checkpoint bytes are not charged per sample.",
        "- Crop metrics are explicitly not global-grid metrics.",
        "- No validation-demo statistic is fitted or labelled as a train statistic.",
        "",
        "## Artifacts",
        "",
        *[f"- `{name}`" for name in OUTPUT_FILES],
        "- `bitstreams/` contains standalone per-sample byte streams only when codec serialization ran.",
        "",
    ])
    return "\n".join(lines)


def _write_blocked(output: Path, args: argparse.Namespace, dataset: dict[str, Any] | None, reason: str) -> int:
    run = {
        "run_id": output.name,
        "started_at": _utc_now(),
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "evaluation_scope": "full_grid" if args.full_frame else "crop_smoke",
        "requested_samples": args.max_samples,
        "status": "BLOCKED",
    }
    _json(output / "run_config.json", {**vars(args), "dataset_root": str(args.dataset_root), "checkpoint": str(args.checkpoint), "output_dir": str(output)})
    _json(output / "dataset_info.json", dataset or {"status": "unavailable", "reason": reason})
    _json(output / "model_info.json", {"status": "unavailable", "checkpoint": str(args.checkpoint), "reason": reason})
    for name in ("metrics.json", "metrics_per_timestamp.json", "timings.json", "resource_usage.json", "compression.json", "roundtrip.json"):
        _json(output / name, {"status": "not_run", "reason": reason})
    (output / "acceptance_report.md").write_text(_report(status="BLOCKED", blockers=[reason], output=output, run=run), encoding="utf-8")
    print(f"BLOCKED: {reason}", file=sys.stderr)
    return 2


def _write_failure(output: Path, args: argparse.Namespace, reason: str) -> int:
    """Leave a readable report when a loaded model fails an acceptance check."""

    run = {
        "run_id": output.name,
        "started_at": _utc_now(),
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "evaluation_scope": "full_grid" if args.full_frame else "crop_smoke",
        "requested_samples": args.max_samples,
    }
    _json(output / "model_info.json", {"status": "failed_after_checkpoint_load", "checkpoint": str(args.checkpoint), "reason": reason})
    (output / "acceptance_report.md").write_text(_report(status="FAIL", blockers=[reason], output=output, run=run), encoding="utf-8")
    print(f"FAIL: {reason}", file=sys.stderr)
    return 2


def _run(args: argparse.Namespace) -> int:
    _validate_args(args)
    output = args.output_dir.resolve()
    _prepare_output(output)
    _seed(args.seed, args.deterministic)
    dataset: dict[str, Any] | None = None
    try:
        dataset = _dataset_info(args.dataset_root.resolve(), args.split, full_frame=args.full_frame, crop_height=args.crop_height, crop_width=args.crop_width)
        validate_prepared(args.dataset_root)
    except (FileNotFoundError, AcceptanceError, ValueError) as exc:
        return _write_blocked(output, args, dataset, f"dataset prerequisite failed: {exc}")
    if not args.checkpoint.is_file():
        return _write_blocked(output, args, dataset, f"checkpoint is missing: {args.checkpoint}")
    if args.config is not None and not args.config.is_file():
        return _write_blocked(output, args, dataset, f"config is missing: {args.config}")
    if args.dry_run:
        return _write_blocked(output, args, dataset, "dry-run requested: checkpoint was deliberately not loaded and no codec result exists")

    started_at, wall_started = _utc_now(), time.perf_counter()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    try:
        codec = WeatherCodec.load(args.checkpoint, device=args.device)
    except (AcceptanceError, FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        return _write_blocked(output, args, dataset, f"checkpoint is incompatible: {exc}")
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in codec.model.parameters())
    trainable_count = sum(parameter.numel() for parameter in codec.model.parameters() if parameter.requires_grad)
    if trainable_count > 20_000_000:
        return _write_blocked(output, args, dataset, f"trainable parameter limit exceeded: {trainable_count} > 20000000")

    dynamic = xr.open_zarr(args.dataset_root / f"{args.split}.zarr", consolidated=True)
    static = xr.open_zarr(args.dataset_root / "static.zarr", consolidated=True)
    selected = _select_timestamps(dynamic, args)
    dataset["timestamps"] = [str(item) for item in selected]
    if args.split == "validation" and args.timestamps == "all" and len(selected) == 4:
        dataset["expected_demo_timestamps_confirmed"] = tuple(dataset["timestamps"]) == EXPECTED_DEMO_TIMESTAMPS

    model_info = {
        "model_class": type(codec.model).__name__,
        "model_config": {"in_channels": codec.model.in_channels, "latent_channels": codec.model.latent_channels},
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": codec.checkpoint_sha256,
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "git_commit": _git_commit(),
        "total_parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "external_pretraining": "unknown_not_recorded_by_checkpoint",
        "channel_count": len(CHANNEL_NAMES),
        "channel_order": list(CHANNEL_NAMES),
        "normalization": codec.normalization.to_dict(),
        "preprocessing": codec.preprocessing,
        "dtype": "float32",
        "device": str(codec.device),
        "supported_input_dimensions": "rank-4 [B,28,H,W]; decoder crops transpose-convolution output to requested H,W",
        "model_load_seconds": load_seconds,
    }
    if args.config is not None:
        model_info["config"] = {"path": str(args.config), "sha256": sha256_file(args.config)}

    # Warm-up only; it is intentionally absent from all timing aggregates.
    warm_physical, warm_ocean, _ = _load_crop(dynamic, static, selected[0], full_frame=False, crop_height=min(32, args.crop_height), crop_width=min(32, args.crop_width))
    warm_input, _, _ = codec.preprocess(warm_physical, ocean_mask=warm_ocean)
    _ = codec.reconstruct(warm_input)
    del warm_physical, warm_ocean, warm_input
    gc.collect()

    metrics_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for timestamp in selected:
        sample_started = time.perf_counter()
        physical, ocean, latitudes = _load_crop(dynamic, static, timestamp, full_frame=args.full_frame, crop_height=args.crop_height, crop_width=args.crop_width)
        pre_started = time.perf_counter()
        model_input, validity_mask, filled_invalid = codec.preprocess(physical, ocean_mask=ocean)
        preprocess_seconds = time.perf_counter() - pre_started
        if not np.isfinite(model_input).all():
            raise AcceptanceError("preprocessed model input contains NaN or infinity")
        latent_once = codec.encode(model_input)
        latent_twice = codec.encode(model_input)
        deterministic = bool(np.array_equal(latent_once, latent_twice))
        if not deterministic:
            raise AcceptanceError("encoder output is non-deterministic for identical input")
        if args.skip_bitstream:
            blockers.append("--skip-bitstream was used: no codec acceptance is possible")
            break
        result = codec.reconstruct(model_input)
        bitstream_path = output / "bitstreams" / f"{str(timestamp).replace(':', '-').replace('.', '_')}.e5ac"
        bitstream_path.write_bytes(result.bitstream)
        # A new context excludes accidental dependence on encoder-side Python objects.
        independent = WeatherCodec.load(args.checkpoint, device=args.device)
        independent_reconstruction, decoded_symbols, independent_decode_seconds, _ = independent.decompress(bitstream_path.read_bytes())
        exact_symbols = bool(np.array_equal(result.symbols, decoded_symbols))
        exact_reconstruction = bool(np.array_equal(result.reconstruction_normalized, independent_reconstruction))
        if not exact_symbols or not exact_reconstruction:
            raise AcceptanceError("independent decode did not reproduce symbols and reconstruction exactly")
        post_started = time.perf_counter()
        reconstructed_physical = codec.postprocess(independent_reconstruction, ocean_mask=ocean)
        postprocess_seconds = time.perf_counter() - post_started
        physical_metrics = evaluate_physical_reconstruction(
            physical, reconstructed_physical, latitudes=latitudes, ocean_mask=ocean, train_std=codec.normalization.std
        )
        compression = compression_metrics(original_shape=tuple(model_input.shape), bitstream=result.bitstream, header=result.header)
        compression["timestamp"] = str(timestamp)
        compression["original_shape"] = list(model_input.shape)
        compression["latent_shape"] = list(result.latent_shape)
        compression_rows.append(compression)
        before_checksum, after_checksum = _checksums(result.symbols), _checksums(decoded_symbols)
        roundtrip_rows.append({
            "timestamp": str(timestamp), "quantized_symbol_count": int(result.symbols.size), "symbol_dtype": str(result.symbols.dtype),
            "symbol_shape": list(result.symbols.shape), "exact_match": exact_symbols, "mismatch_count": 0 if exact_symbols else int(np.count_nonzero(result.symbols != decoded_symbols)),
            "checksum_before": before_checksum, "checksum_after": after_checksum, "independent_decode": {"passed": exact_reconstruction, "decode_seconds": independent_decode_seconds},
        })
        metrics_rows.append({"timestamp": str(timestamp), "evaluation_scope": dataset["evaluation_scope"], "not_full_grid": dataset["not_full_grid"], **physical_metrics})
        timing_rows.append({
            "timestamp": str(timestamp), "preprocess_seconds": preprocess_seconds, "encode_and_serialize_seconds": result.encode_seconds,
            "decode_seconds": result.decode_seconds, "independent_decode_seconds": independent_decode_seconds, "postprocess_seconds": postprocess_seconds,
            "total_sample_seconds": time.perf_counter() - sample_started, "filled_invalid_model_values": filled_invalid,
            "validity_mask_fraction": float(validity_mask.mean()), "deterministic_encoder": deterministic,
        })
        del physical, ocean, latitudes, model_input, validity_mask, latent_once, latent_twice, result, independent, independent_reconstruction, decoded_symbols, reconstructed_physical
        gc.collect()

    total_seconds = time.perf_counter() - wall_started
    finished_at = _utc_now()
    compression_payload = _mean_compression(compression_rows)
    status = "PASS" if args.full_frame and not blockers else "PASS_WITH_LIMITATIONS"
    if not args.full_frame:
        blockers.append("Acceptance is crop_smoke only; it is not a full-grid/global result.")
    resource = measure_runtime_resources(
        run_id=output.name, operation="model_codec_acceptance", grid_resolution="0.25_degree", dataset_size=len(metrics_rows),
        started_at=started_at, finished_at=finished_at, wall_clock_seconds=total_seconds, trainable_parameter_count=trainable_count,
        total_parameter_count=parameter_count, optimizer_steps=0, examples_seen=0, unique_train_timestamps=0, patches_seen=0,
        encode_seconds=sum(row["encode_and_serialize_seconds"] for row in timing_rows), decode_seconds=sum(row["decode_seconds"] for row in timing_rows),
        bitstream_bytes=sum(row["total_bitstream_bytes"] for row in compression_rows), parameter_limit=20_000_000, max_vram_gb=24,
    )
    _json(output / "model_info.json", model_info)
    _json(output / "dataset_info.json", dataset)
    _json(output / "run_config.json", {**vars(args), "dataset_root": str(args.dataset_root), "checkpoint": str(args.checkpoint), "output_dir": str(output), "started_at": started_at, "finished_at": finished_at})
    _json(output / "metrics_per_timestamp.json", metrics_rows)
    _json(output / "metrics.json", {"scope": dataset["evaluation_scope"], "per_timestamp": metrics_rows, "nrmse_source": "checkpoint train_only normalization std"})
    _json(output / "timings.json", {"warmup_excluded": True, "model_load_seconds": load_seconds, "per_timestamp": timing_rows, "total_seconds": total_seconds})
    _json(output / "resource_usage.json", resource)
    _json(output / "compression.json", compression_payload)
    _json(output / "roundtrip.json", {"per_timestamp": roundtrip_rows, "all_exact_symbol_roundtrips": all(row["exact_match"] for row in roundtrip_rows), "all_independent_decodes": all(row["independent_decode"]["passed"] for row in roundtrip_rows)})
    run = {"run_id": output.name, "evaluation_scope": dataset["evaluation_scope"], "dataset_root": str(args.dataset_root), "checkpoint": str(args.checkpoint), "device": args.device, "requested_samples": len(selected)}
    (output / "acceptance_report.md").write_text(_report(status=status, blockers=blockers, output=output, run=run), encoding="utf-8")
    print(f"{status}: wrote acceptance artifacts to {output}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Run acceptance and preserve a terminal report for a model-level failure."""

    try:
        return _run(args)
    except (AcceptanceError, RuntimeError, KeyError, ValueError, OSError) as exc:
        output = args.output_dir.resolve()
        if output.exists() and output.is_dir():
            return _write_failure(output, args, f"acceptance check failed: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_parse_args(argv))
    except (AcceptanceError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"acceptance failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
