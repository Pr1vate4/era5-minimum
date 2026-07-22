"""Safely download one ERA5 single-level day through the CDS API.

The module is intentionally usable without ``cdsapi``: request validation,
dry-run, archive safety checks, and tests are all offline.  ``cdsapi`` is
imported only immediately before a real CDS request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DATASET = "reanalysis-era5-single-levels"
REQUEST_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_precipitation",
    "total_cloud_cover",
    "total_column_water_vapour",
)
INSTANT_VARIABLES = frozenset({"u10", "v10", "t2m", "msl", "sst", "tcc", "tcwv"})
INSTANT_FILENAME = "data_stream-oper_stepType-instant.nc"
ACCUMULATED_FILENAME = "data_stream-oper_stepType-accum.nc"
TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
NETCDF_CLASSIC_MAGIC = b"CDF"
NETCDF4_MAGIC = b"\x89HDF\r\n\x1a\n"
PRECIPITATION_WARNINGS = (
    "Raw tp is not automatically tp6h; no tp6h was created.",
    "Sparse six-hour timestamps do not create a six-hour precipitation accumulation.",
)


class DownloadError(RuntimeError):
    """Raised when a download cannot be prepared safely."""


def validate_date(value: str) -> str:
    """Validate and normalize an ISO calendar date."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid --date {value!r}; expected a real YYYY-MM-DD date.") from error
    if parsed.isoformat() != value:
        raise ValueError(f"Invalid --date {value!r}; expected YYYY-MM-DD.")
    return parsed.isoformat()


def normalize_times(values: Sequence[str]) -> list[str]:
    """Validate times, remove duplicates, and return chronological order."""
    if not values:
        raise ValueError("At least one --times value is required.")
    invalid = [value for value in values if not TIME_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(
            "Invalid --times value(s) " + ", ".join(repr(value) for value in invalid) + "; expected HH:MM."
        )
    return sorted(set(values))


def build_request(requested_date: str, requested_times: Sequence[str]) -> dict[str, list[str] | str]:
    """Build the exact, versioned CDS request schema used by this project."""
    checked_date = validate_date(requested_date)
    times = normalize_times(requested_times)
    year, month, day = checked_date.split("-")
    return {
        "product_type": ["reanalysis"],
        "variable": list(REQUEST_VARIABLES),
        "year": [year],
        "month": [month],
        "day": [day],
        "time": times,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def dataset_directory(output_root: Path, requested_date: str) -> Path:
    """Return the fixed per-day raw-data directory without creating it."""
    return output_root / f"era5_single_{requested_date.replace('-', '_')}"


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without loading the whole file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum_manifest(directory: Path, names: Sequence[str]) -> dict[str, str]:
    """Write a ``sha256sum -c`` compatible manifest for named files."""
    hashes = {name: sha256_file(directory / name) for name in names}
    lines = [f"{hashes[name]}  {name}" for name in names]
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashes


def _is_netcdf(path: Path) -> bool:
    with path.open("rb") as source:
        header = source.read(8)
    return header.startswith(NETCDF_CLASSIC_MAGIC) or header == NETCDF4_MAGIC


def detect_download_format(path: Path) -> str:
    """Return ``zip`` or ``netcdf`` based on bytes, never file extension alone."""
    if zipfile.is_zipfile(path):
        return "zip"
    if _is_netcdf(path):
        return "netcdf"
    raise DownloadError(f"Downloaded file is neither ZIP nor NetCDF: {path.name}")


def safe_extract_zip(archive: Path, destination: Path) -> list[Path]:
    """Extract a ZIP archive after rejecting traversal, absolute paths, and links."""
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DownloadError(f"Unsafe ZIP member path: {member.filename!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise DownloadError(f"ZIP symbolic links are not allowed: {member.filename!r}")
            target = (destination / member_path).resolve()
            if not target.is_relative_to(root):
                raise DownloadError(f"ZIP member escapes destination: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise DownloadError(f"ZIP member would overwrite a file: {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def _classify_netcdf(path: Path) -> str | None:
    """Classify a small amount of NetCDF metadata without loading data arrays."""
    try:
        import xarray as xr

        with xr.open_dataset(path, cache=False) as dataset:
            variables = set(dataset.data_vars)
            if "tp" in variables and dataset["tp"].attrs.get("GRIB_stepType") == "accum":
                return "accumulated"
            if INSTANT_VARIABLES.issubset(variables):
                return "instant"
    except (OSError, ValueError) as error:
        raise DownloadError(f"Could not inspect downloaded NetCDF {path.name}: {error}") from error
    return None


def discover_netcdf_pair(directory: Path) -> dict[str, Path]:
    """Discover exactly one instant and one accumulated file from file contents."""
    found: dict[str, Path] = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        if not _is_netcdf(path):
            continue
        role = _classify_netcdf(path)
        if role is None:
            raise DownloadError(f"Downloaded NetCDF does not match expected ERA5 pair: {path.name}")
        if role in found:
            raise DownloadError(f"More than one downloaded {role} NetCDF file was found.")
        found[role] = path
    missing = {"instant", "accumulated"} - set(found)
    if missing:
        raise DownloadError("Downloaded files do not contain the expected ERA5 pair: missing " + ", ".join(sorted(missing)))
    return found


def _relative_output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_metadata(
    *,
    requested_date: str,
    requested_times: Sequence[str],
    target: Path,
    archive_detected: bool,
    extracted_files: Sequence[str],
    file_sizes: dict[str, int],
    hashes: dict[str, str],
) -> dict[str, Any]:
    """Create credential-free metadata for a successfully prepared raw pair."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "requested_date": requested_date,
        "requested_times": list(requested_times),
        "requested_variables": list(REQUEST_VARIABLES),
        "dataset": DATASET,
        "output_directory": _relative_output_path(target),
        "archive_detected": archive_detected,
        "extracted_files": list(extracted_files),
        "file_sizes": file_sizes,
        "sha256": hashes,
        "downloader_version": _git_commit(),
        "warnings": list(PRECIPITATION_WARNINGS),
    }


def _cds_client() -> Any:
    """Create a CDS client only for a real download, with a useful install error."""
    try:
        cdsapi = importlib.import_module("cdsapi")
    except ModuleNotFoundError as error:
        raise DownloadError('cdsapi is required for a real download. Install it with: pip install -e ".[download]"') from error
    return cdsapi.Client()


def _prepare_dataset(stage: Path, archive: Path, keep_archive: bool) -> tuple[bool, list[str], dict[str, str]]:
    """Validate downloaded bytes, discover the pair, and write final artifacts in stage."""
    file_format = detect_download_format(archive)
    archive_detected = file_format == "zip"
    if archive_detected:
        source_directory = stage / "extracted"
        safe_extract_zip(archive, source_directory)
        if keep_archive:
            archive.replace(stage / "source_download.zip")
    else:
        source_directory = stage / "direct_netcdf"
        source_directory.mkdir()
        archive.replace(source_directory / "download.nc")

    discovered = discover_netcdf_pair(source_directory)
    destinations = {
        "instant": stage / INSTANT_FILENAME,
        "accumulated": stage / ACCUMULATED_FILENAME,
    }
    for role, destination in destinations.items():
        shutil.move(str(discovered[role]), destination)
    shutil.rmtree(source_directory)

    request_names = [INSTANT_FILENAME, ACCUMULATED_FILENAME, "request.json"]
    hashes = {name: sha256_file(stage / name) for name in request_names[:2]}
    file_sizes = {name: (stage / name).stat().st_size for name in request_names[:2]}
    extracted_files = [INSTANT_FILENAME, ACCUMULATED_FILENAME]
    return archive_detected, extracted_files, hashes | {"request.json": sha256_file(stage / "request.json")}


def _replace_dataset(prepared: Path, target: Path, overwrite: bool) -> None:
    """Publish a complete staged dataset, retaining old data until replacement is ready."""
    if not target.exists():
        os.replace(prepared, target)
        return
    if not overwrite:
        raise FileExistsError(f"Target dataset already exists: {target}. Use --overwrite to replace it.")
    backup = target.parent / f".{target.name}.backup"
    if backup.exists():
        raise DownloadError(f"Refusing overwrite because stale backup exists: {backup}")
    target.rename(backup)
    try:
        os.replace(prepared, target)
    except OSError:
        backup.rename(target)
        raise
    shutil.rmtree(backup)


def download_era5_day(
    *, requested_date: str, requested_times: Sequence[str], output_root: Path, overwrite: bool, keep_archive: bool
) -> Path:
    """Download, validate, stage, and atomically publish one raw ERA5 day."""
    checked_date = validate_date(requested_date)
    times = normalize_times(requested_times)
    request = build_request(checked_date, times)
    target = dataset_directory(output_root, checked_date)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Target dataset already exists: {target}. Use --overwrite to replace it.")

    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=output_root))
    prepared = stage_root / target.name
    prepared.mkdir()
    try:
        (prepared / "request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        archive = stage_root / "source_download.part"
        client = _cds_client()
        result = client.retrieve(DATASET, request)
        result.download(str(archive))
        if not archive.is_file() or archive.stat().st_size == 0:
            raise DownloadError("CDS did not produce a non-empty temporary download file.")
        archive_detected, extracted_files, hashes = _prepare_dataset(prepared, archive, keep_archive)
        file_sizes = {name: (prepared / name).stat().st_size for name in extracted_files}
        metadata = build_metadata(
            requested_date=checked_date,
            requested_times=times,
            target=target,
            archive_detected=archive_detected,
            extracted_files=extracted_files,
            file_sizes=file_sizes,
            hashes={name: hashes[name] for name in extracted_files},
        )
        (prepared / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        write_checksum_manifest(prepared, [*extracted_files, "request.json"])
        _replace_dataset(prepared, target, overwrite)
        return target
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _write_request_only(path: Path, request: dict[str, list[str] | str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Request file already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")


def _dry_run_summary(target: Path, request: dict[str, list[str] | str], keep_archive: bool) -> str:
    planned = [
        "validate request without importing cdsapi or contacting CDS",
        "download into a temporary .part file",
        "detect ZIP or NetCDF bytes and safely discover the instant/accumulated pair",
        "write request.json, metadata.json, and SHA256SUMS.txt only after success",
    ]
    if keep_archive:
        planned.append("retain source_download.zip when CDS returns a ZIP archive")
    return "DRY RUN — no files will be created\n" + f"Target: {target}\n" + "Planned actions:\n- " + "\n- ".join(planned) + "\nRequest:\n" + json.dumps(request, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely download one ERA5 single-level day from CDS.")
    parser.add_argument("--date", required=True, help="Calendar date in YYYY-MM-DD format.")
    parser.add_argument("--times", required=True, nargs="+", help="One or more HH:MM timestamps.")
    parser.add_argument("--output-root", default="data/raw", help="Directory containing per-day raw datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without filesystem, CDS, or network access.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing day only after new staged data is complete.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep source_download.zip when CDS returns a ZIP archive.")
    parser.add_argument("--request-only", type=Path, help="Write the validated request JSON and exit without CDS access.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader CLI. Dry-run never imports ``cdsapi`` or creates paths."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checked_date = validate_date(args.date)
        times = normalize_times(args.times)
        request = build_request(checked_date, times)
        target = dataset_directory(Path(args.output_root), checked_date)
        if args.dry_run:
            print(_dry_run_summary(target, request, args.keep_archive))
            return 0
        if args.request_only is not None:
            _write_request_only(args.request_only, request, args.overwrite)
            print(f"Request written to {args.request_only}")
            return 0
        output = download_era5_day(
            requested_date=checked_date,
            requested_times=times,
            output_root=Path(args.output_root),
            overwrite=args.overwrite,
            keep_archive=args.keep_archive,
        )
    except (DownloadError, FileExistsError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Downloaded ERA5 raw pair to {output}")
    print("WARNING: tp remains raw tp; no tp1h rename or tp6h creation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
