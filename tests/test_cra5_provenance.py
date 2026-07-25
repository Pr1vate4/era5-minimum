from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from era5_minimum.cra5.provenance import (
    CRA5_159_SHA256,
    CRA5_159_SIZE,
    CRA5_CHECKPOINT_URL,
    CRA5_UPSTREAM_COMMIT,
    CheckpointValidationError,
    checkpoint_cache_path,
    fetch_checkpoint,
    validate_checkpoint_file,
)


def test_validate_checkpoint_streams_size_and_hash(tmp_path: Path) -> None:
    payload = b"checkpoint bytes for an offline test"
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(payload)

    record = validate_checkpoint_file(
        checkpoint,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert record.path == checkpoint
    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_checkpoint_cache_path_is_outside_repository_by_default(tmp_path: Path) -> None:
    path = checkpoint_cache_path(tmp_path)

    assert path == tmp_path / "cra5_159v_150k.pth"
    assert path.parent == tmp_path


def test_fetch_defaults_to_offline_dry_run(tmp_path: Path) -> None:
    result = fetch_checkpoint(cache_dir=tmp_path)

    assert result["mode"] == "dry-run"
    assert result["downloaded"] is False
    assert result["verified"] is False
    assert result["expected_size_bytes"] == CRA5_159_SIZE
    assert result["expected_sha256"] == CRA5_159_SHA256
    assert result["upstream_commit"] == CRA5_UPSTREAM_COMMIT
    assert result["url"] == CRA5_CHECKPOINT_URL
    assert not checkpoint_cache_path(tmp_path).exists()


def test_fetch_downloads_to_part_verifies_and_atomically_replaces(
    tmp_path: Path,
) -> None:
    payload = b"a small valid checkpoint"
    source = tmp_path / "source.pth"
    source.write_bytes(payload)
    cache_dir = tmp_path / "cache"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    result = fetch_checkpoint(
        url=source.as_uri(),
        cache_dir=cache_dir,
        download=True,
        expected_size=len(payload),
        expected_sha256=expected_sha256,
    )

    target = cache_dir / "cra5_159v_150k.pth"
    assert result["mode"] == "download"
    assert result["downloaded"] is True
    assert result["verified"] is True
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".part").exists()


def test_fetch_rejects_mismatched_download_without_promoting_part(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"wrong checkpoint")
    cache_dir = tmp_path / "cache"

    with pytest.raises(CheckpointValidationError, match="size"):
        fetch_checkpoint(
            url=source.as_uri(),
            cache_dir=cache_dir,
            download=True,
            expected_size=999,
            expected_sha256="0" * 64,
        )

    assert not (cache_dir / "cra5_159v_150k.pth").exists()
    assert not (cache_dir / "cra5_159v_150k.pth.part").exists()


def test_validate_checkpoint_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "checkpoint.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 1450747681\n",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointValidationError, match="Git LFS pointer"):
        validate_checkpoint_file(
            pointer,
            expected_size=pointer.stat().st_size,
            expected_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),
        )


def test_fetch_cli_prints_dry_run_json_without_network(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "fetch_cra5_checkpoint.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--cache-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output["mode"] == "dry-run"
    assert output["downloaded"] is False
    assert output["upstream_commit"] == CRA5_UPSTREAM_COMMIT
