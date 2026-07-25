"""Pinned CRA5 checkpoint provenance and safe, optional fetching."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

CRA5_UPSTREAM_COMMIT = "2b9e06d8ca31f7b27c5039c9b5fc3334a43b4976"
CRA5_159_SIZE = 1_450_747_681
CRA5_159_SHA256 = "36dfdf0458bb9ed9ecfd1dcdbd75bf9d2599f2320041e769d6c766f3d8563a11"
CRA5_CHECKPOINT_FILENAME = "cra5_159v_150k.pth"
CRA5_CHECKPOINT_URL = (
    "https://huggingface.co/taohan10200/CRA5-model/resolve/main/"
    + CRA5_CHECKPOINT_FILENAME
)
CRA5_LICENSE_NOTE = "See the upstream CRA5 repository and model card for applicable license terms."
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "era5-minimum" / "cra5"
CHUNK_SIZE = 1024 * 1024
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class CheckpointValidationError(ValueError):
    """Raised when checkpoint bytes do not match the pinned artifact."""


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Measured checkpoint bytes after streaming validation."""

    path: Path
    size_bytes: int
    sha256: str

    @property
    def size(self) -> int:
        """Compatibility alias for callers that use a short size field."""
        return self.size_bytes


def checkpoint_cache_path(cache_dir: Path | str | None = None) -> Path:
    """Return the external cache path without creating repository files."""
    root = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir).expanduser()
    return root / CRA5_CHECKPOINT_FILENAME


def _is_git_lfs_pointer(prefix: bytes) -> bool:
    return prefix.lstrip().startswith(GIT_LFS_POINTER_PREFIX)


def _measure(path: Path) -> CheckpointRecord:
    if not path.is_file():
        raise CheckpointValidationError(f"Checkpoint file does not exist: {path}")

    digest = hashlib.sha256()
    size = 0
    prefix = bytearray()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                if len(prefix) < 256:
                    prefix.extend(block[: 256 - len(prefix)])
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise CheckpointValidationError(f"Could not read checkpoint {path}: {error}") from error

    if _is_git_lfs_pointer(bytes(prefix)):
        raise CheckpointValidationError(f"Git LFS pointer is not a checkpoint: {path}")
    return CheckpointRecord(path=path, size_bytes=size, sha256=digest.hexdigest())


def validate_checkpoint_file(
    path: Path | str,
    *,
    expected_size: int = CRA5_159_SIZE,
    expected_sha256: str = CRA5_159_SHA256,
) -> CheckpointRecord:
    """Stream a checkpoint and require its exact size and SHA-256 digest."""
    measured = _measure(Path(path))
    if measured.size_bytes != expected_size:
        raise CheckpointValidationError(
            f"Checkpoint size mismatch: expected {expected_size}, got {measured.size_bytes}"
        )
    if measured.sha256.lower() != expected_sha256.lower():
        raise CheckpointValidationError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {measured.sha256}"
        )
    return measured


def _base_manifest(
    *,
    url: str,
    target: Path,
    expected_size: int,
    expected_sha256: str,
    mode: str,
    downloaded: bool,
    verified: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "artifact": CRA5_CHECKPOINT_FILENAME,
        "url": url,
        "upstream_commit": CRA5_UPSTREAM_COMMIT,
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha256,
        "license_note": CRA5_LICENSE_NOTE,
        "cache_path": str(target),
        "mode": mode,
        "downloaded": downloaded,
        "verified": verified,
    }


def _stream_download(url: str, part_path: Path) -> CheckpointRecord:
    digest = hashlib.sha256()
    size = 0
    prefix = bytearray()
    try:
        with request.urlopen(url, timeout=60) as response, part_path.open("wb") as target:
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                if len(prefix) < 256:
                    prefix.extend(block[: 256 - len(prefix)])
                target.write(block)
                digest.update(block)
                size += len(block)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    if _is_git_lfs_pointer(bytes(prefix)):
        part_path.unlink(missing_ok=True)
        raise CheckpointValidationError(f"Git LFS pointer is not a checkpoint: {url}")
    return CheckpointRecord(path=part_path, size_bytes=size, sha256=digest.hexdigest())


def fetch_checkpoint(
    *,
    url: str = CRA5_CHECKPOINT_URL,
    cache_dir: Path | str | None = None,
    download: bool = False,
    expected_size: int = CRA5_159_SIZE,
    expected_sha256: str = CRA5_159_SHA256,
) -> dict[str, Any]:
    """Describe the pinned checkpoint or download it into the external cache.

    The default is deliberately offline.  A download writes only a sibling
    ``.part`` file and promotes it after exact size/hash validation.
    """
    target = checkpoint_cache_path(cache_dir)
    manifest = _base_manifest(
        url=url,
        target=target,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        mode="download" if download else "dry-run",
        downloaded=False,
        verified=False,
    )
    if not download:
        return manifest

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            record = validate_checkpoint_file(
                target,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except CheckpointValidationError:
            record = None
        if record is not None:
            manifest.update(
                {
                    "downloaded": False,
                    "verified": True,
                    "cache_hit": True,
                    "actual_size_bytes": record.size_bytes,
                    "actual_sha256": record.sha256,
                }
            )
            return manifest

    part_path = target.with_name(target.name + ".part")
    try:
        record = _stream_download(url, part_path)
        if record.size_bytes != expected_size:
            raise CheckpointValidationError(
                f"Checkpoint size mismatch: expected {expected_size}, got {record.size_bytes}"
            )
        if record.sha256.lower() != expected_sha256.lower():
            raise CheckpointValidationError(
                f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {record.sha256}"
            )
        os.replace(part_path, target)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    manifest.update(
        {
            "downloaded": True,
            "verified": True,
            "cache_hit": False,
            "actual_size_bytes": record.size_bytes,
            "actual_sha256": record.sha256,
        }
    )
    return manifest
