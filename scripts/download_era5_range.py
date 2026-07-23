"""Sequential, resumable ERA5 range downloads built on ``download_era5.py``.

This module deliberately owns only range planning, complete-day checks, and
range metadata.  CDS requests, credentials, ZIP handling, per-day provenance,
and atomic overwrite behavior remain the responsibility of the daily
downloader.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from download_era5 import dataset_directory, download_era5_day, sha256_file, validate_date


HOURLY_TIMES = tuple(f"{hour:02d}:00" for hour in range(24))
REQUIRED_DAILY_FILES = (
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-accum.nc",
    "request.json",
    "metadata.json",
    "SHA256SUMS.txt",
)
REQUIRED_MANIFEST_FILES = REQUIRED_DAILY_FILES[:3]
SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")


class RangeDownloadError(RuntimeError):
    """Raised when a date range or an existing daily dataset is unsafe."""


@dataclass(frozen=True)
class DailyDatasetCheck:
    """Structured result of validating one existing daily directory."""

    complete: bool
    checksum_status: str
    error: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_relative_path(path: Path) -> str:
    """Avoid recording an absolute user path in range artifacts."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name or "."


def date_range(start_date: str, end_date: str) -> list[str]:
    """Return the inclusive, stable sequence of ISO dates for a range."""
    checked_start = validate_date(start_date)
    checked_end = validate_date(end_date)
    start = date.fromisoformat(checked_start)
    end = date.fromisoformat(checked_end)
    if start > end:
        raise ValueError(f"--start-date {checked_start} must not be later than --end-date {checked_end}.")
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


def range_directory(output_root: Path, start_date: str, end_date: str) -> Path:
    """Return the range-metadata directory without creating it."""
    return output_root / "ranges" / f"era5_range_{start_date.replace('-', '_')}_{end_date.replace('-', '_')}"


def _safe_manifest_path(directory: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RangeDownloadError(f"Unsafe SHA256SUMS path: {filename!r}")
    target = (directory / candidate).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise RangeDownloadError(f"SHA256SUMS path escapes daily directory: {filename!r}")
    return target


def verify_checksum_manifest(directory: Path) -> DailyDatasetCheck:
    """Verify a daily ``SHA256SUMS.txt`` with strict relative-path handling."""
    manifest = directory / "SHA256SUMS.txt"
    if not manifest.is_file():
        return DailyDatasetCheck(False, "missing", "Missing SHA256SUMS.txt")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return DailyDatasetCheck(False, "failed", f"Could not read SHA256SUMS.txt: {error}")
    if not lines:
        return DailyDatasetCheck(False, "failed", "SHA256SUMS.txt is empty")

    names: set[str] = set()
    try:
        for line in lines:
            match = SHA256_LINE.fullmatch(line)
            if match is None:
                raise RangeDownloadError(f"Malformed SHA256SUMS line: {line!r}")
            expected_hash, name = match.groups()
            if name in names:
                raise RangeDownloadError(f"Duplicate SHA256SUMS entry: {name!r}")
            names.add(name)
            target = _safe_manifest_path(directory, name)
            if not target.is_file():
                raise RangeDownloadError(f"SHA256SUMS file is missing: {name!r}")
            actual_hash = sha256_file(target)
            if actual_hash.lower() != expected_hash.lower():
                raise RangeDownloadError(f"SHA-256 mismatch for {name!r}")
    except (OSError, RangeDownloadError) as error:
        return DailyDatasetCheck(False, "failed", str(error))

    missing = sorted(set(REQUIRED_MANIFEST_FILES) - names)
    if missing:
        return DailyDatasetCheck(False, "failed", "SHA256SUMS missing required entries: " + ", ".join(missing))
    return DailyDatasetCheck(True, "passed", None)


def check_complete_day(directory: Path) -> DailyDatasetCheck:
    """Require all daily artifacts, parse JSON metadata, then verify checksums."""
    if not directory.is_dir():
        return DailyDatasetCheck(False, "missing", "Daily directory does not exist")
    missing = [name for name in REQUIRED_DAILY_FILES if not (directory / name).is_file()]
    if missing:
        return DailyDatasetCheck(False, "missing", "Missing required file(s): " + ", ".join(missing))
    for name in ("request.json", "metadata.json"):
        try:
            parsed = json.loads((directory / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return DailyDatasetCheck(False, "failed", f"Invalid {name}: {error}")
        if not isinstance(parsed, dict):
            return DailyDatasetCheck(False, "failed", f"Invalid {name}: expected a JSON object")
    return verify_checksum_manifest(directory)


def _new_day_entry(requested_date: str, target: Path) -> dict[str, Any]:
    return {
        "date": requested_date,
        "target_directory": _safe_relative_path(target),
        "status": "planned",
        "started_at": None,
        "finished_at": None,
        "instant_file": None,
        "accumulated_file": None,
        "checksum_status": "not_checked",
        "error": None,
        "action": "download",
    }


def _mark_verified(entry: dict[str, Any], directory: Path, *, status: str, action: str) -> None:
    entry.update(
        {
            "status": status,
            "finished_at": _utc_now(),
            "instant_file": "data_stream-oper_stepType-instant.nc",
            "accumulated_file": "data_stream-oper_stepType-accum.nc",
            "checksum_status": "passed",
            "error": None,
            "action": action,
        }
    )


def _summary(days: Sequence[dict[str, Any]], started_at: str, finished_at: str) -> dict[str, Any]:
    verified = sum(day["status"] in {"verified", "skipped"} for day in days)
    failed = sum(day["status"] == "failed" for day in days)
    skipped = sum(day["status"] == "skipped" for day in days)
    downloaded = sum(day["status"] == "verified" and day["action"] in {"download", "overwrite"} for day in days)
    return {
        "requested_days": len(days),
        "downloaded_days": downloaded,
        "verified_days": verified,
        "skipped_days": skipped,
        "failed_days": failed,
        "completed": failed == 0 and verified == len(days),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def build_range_request(
    *,
    start_date: str,
    end_date: str,
    output_root: Path,
    keep_archive: bool,
    overwrite: bool,
    continue_on_error: bool,
    created_at: str,
) -> dict[str, Any]:
    """Build credential-free metadata for a sequential range request."""
    dates = date_range(start_date, end_date)
    return {
        "start_date": dates[0],
        "end_date": dates[-1],
        "total_days": len(dates),
        "times": list(HOURLY_TIMES),
        "output_root": _safe_relative_path(output_root),
        "keep_archive": keep_archive,
        "overwrite": overwrite,
        "continue_on_error": continue_on_error,
        "created_at": created_at,
        "downloader": "scripts/download_era5_range.py",
    }


def plan_range(
    *, start_date: str, end_date: str, output_root: Path, overwrite: bool
) -> list[dict[str, Any]]:
    """Read existing state and return offline per-day actions without writing files."""
    entries: list[dict[str, Any]] = []
    for requested_date in date_range(start_date, end_date):
        target = dataset_directory(output_root, requested_date)
        entry = _new_day_entry(requested_date, target)
        check = check_complete_day(target)
        if check.complete and not overwrite:
            entry.update({"status": "skipped", "action": "skip-existing", "checksum_status": "passed"})
        elif check.complete:
            entry.update({"status": "planned", "action": "overwrite", "checksum_status": "passed"})
        elif target.exists() and not overwrite:
            entry.update({"status": "failed", "action": "failed", "checksum_status": check.checksum_status, "error": check.error})
        elif target.exists():
            entry.update({"status": "planned", "action": "overwrite", "checksum_status": check.checksum_status})
        entries.append(entry)
    return entries


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


DailyDownloader = Callable[..., Path]


def run_range(
    *,
    start_date: str,
    end_date: str,
    output_root: Path,
    overwrite: bool,
    keep_archive: bool,
    continue_on_error: bool,
    daily_downloader: DailyDownloader = download_era5_day,
) -> dict[str, Any]:
    """Run sequential daily downloads and persist an honest range manifest."""
    dates = date_range(start_date, end_date)
    started_at = _utc_now()
    metadata_dir = range_directory(output_root, dates[0], dates[-1])
    entries = [_new_day_entry(requested_date, dataset_directory(output_root, requested_date)) for requested_date in dates]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        metadata_dir / "range_request.json",
        build_range_request(
            start_date=dates[0],
            end_date=dates[-1],
            output_root=output_root,
            keep_archive=keep_archive,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
            created_at=started_at,
        ),
    )

    for entry in entries:
        requested_date = entry["date"]
        target = dataset_directory(output_root, requested_date)
        entry["started_at"] = _utc_now()
        before = check_complete_day(target)
        if before.complete and not overwrite:
            _mark_verified(entry, target, status="skipped", action="skip-existing")
            continue
        if target.exists() and not before.complete and not overwrite:
            entry.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "checksum_status": before.checksum_status,
                    "error": "Existing daily directory is incomplete or invalid: " + str(before.error),
                    "action": "failed",
                }
            )
            if not continue_on_error:
                break
            continue

        action = "overwrite" if target.exists() else "download"
        try:
            daily_downloader(
                requested_date=requested_date,
                requested_times=HOURLY_TIMES,
                output_root=output_root,
                overwrite=overwrite,
                keep_archive=keep_archive,
            )
            after = check_complete_day(target)
            if not after.complete:
                raise RangeDownloadError("Daily downloader returned an incomplete dataset: " + str(after.error))
            _mark_verified(entry, target, status="verified", action=action)
        except (OSError, RuntimeError, ValueError) as error:
            entry.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "checksum_status": "failed",
                    "error": str(error),
                    "action": "failed",
                }
            )
            if not continue_on_error:
                break

    finished_at = _utc_now()
    manifest = {
        "range_request": build_range_request(
            start_date=dates[0],
            end_date=dates[-1],
            output_root=output_root,
            keep_archive=keep_archive,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
            created_at=started_at,
        ),
        "days": entries,
        "summary": _summary(entries, started_at, finished_at),
    }
    _write_json(metadata_dir / "range_manifest.json", manifest)
    return manifest


def _print_dry_run(entries: Sequence[dict[str, Any]], metadata_dir: Path) -> None:
    print("DRY RUN — no files will be created and no CDS request will be sent")
    print("Hourly timestamps: " + ", ".join(HOURLY_TIMES))
    for entry in entries:
        suffix = f"; error={entry['error']}" if entry["error"] else ""
        print(f"{entry['date']}: status={entry['status']}; action={entry['action']}; target={entry['target_directory']}{suffix}")
    print(f"Range request path: {metadata_dir / 'range_request.json'}")
    print(f"Range manifest path: {metadata_dir / 'range_manifest.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely download an inclusive ERA5 date range one day at a time.")
    parser.add_argument("--start-date", required=True, help="First date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--end-date", required=True, help="Last date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--output-root", default="data/raw", help="Root containing independent daily datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Plan offline without creating files or importing cdsapi.")
    parser.add_argument("--overwrite", action="store_true", help="Forward safe replacement intent to the daily downloader.")
    parser.add_argument("--keep-archive", action="store_true", help="Forward archive retention to the daily downloader.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue later dates after a failed day.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline planning or sequential range downloads without parallel workers."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    try:
        dates = date_range(args.start_date, args.end_date)
        metadata_dir = range_directory(output_root, dates[0], dates[-1])
        if args.dry_run:
            _print_dry_run(
                plan_range(
                    start_date=dates[0], end_date=dates[-1], output_root=output_root, overwrite=args.overwrite
                ),
                metadata_dir,
            )
            return 0
        manifest = run_range(
            start_date=dates[0],
            end_date=dates[-1],
            output_root=output_root,
            overwrite=args.overwrite,
            keep_archive=args.keep_archive,
            continue_on_error=args.continue_on_error,
        )
    except (RangeDownloadError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = manifest["summary"]
    print(
        "Range summary: "
        f"downloaded={summary['downloaded_days']}, verified={summary['verified_days']}, "
        f"skipped={summary['skipped_days']}, failed={summary['failed_days']}, completed={summary['completed']}"
    )
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
