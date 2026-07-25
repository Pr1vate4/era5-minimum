from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from era5_minimum.cra5.bridge import (
    BRIDGE_PROTOCOL_VERSION,
    Cra5BridgeError,
    Cra5BridgeRequest,
    run_cra5_bridge,
)
from era5_minimum.data.channel_spec import CHANNEL_NAMES


FAKE_RUNTIME = Path(__file__).parent / "fixtures" / "fake_cra5_runtime.py"


def _request(tmp_path: Path, *, input_path: Path | str = "input.bin") -> Cra5BridgeRequest:
    return Cra5BridgeRequest(
        run_id="bridge-test-1",
        operation="encode_decode",
        input_path=input_path,
        output_path=Path("reconstruction.bin"),
        response_path=Path("response.json"),
        tensor_shape=(1, 28, 4, 8),
        channel_order=CHANNEL_NAMES,
        checkpoint_sha256="a" * 64,
    )


def _command() -> tuple[str, ...]:
    return (sys.executable, str(FAKE_RUNTIME))


def test_bridge_runs_fake_runtime_and_validates_response(tmp_path: Path) -> None:
    (tmp_path / "input.bin").write_bytes(b"encoded-input")

    result = run_cra5_bridge(
        _request(tmp_path),
        command=_command(),
        work_dir=tmp_path,
        timeout_seconds=1.0,
    )

    assert result.protocol_version == BRIDGE_PROTOCOL_VERSION
    assert result.run_id == "bridge-test-1"
    assert result.operation == "encode_decode"
    assert result.tensor_shape == (1, 28, 4, 8)
    assert result.channel_order == CHANNEL_NAMES
    assert result.checkpoint_sha256 == "a" * 64
    assert result.bitstream_sha256 == hashlib.sha256(b"fake-cra5-bitstream-v1").hexdigest()
    assert result.exact_roundtrip is True
    assert result.encode_seconds == 0.01
    assert result.decode_seconds == 0.02
    assert (tmp_path / "reconstruction.bin").read_bytes() == b"encoded-input-decoded"


def test_bridge_rejects_path_traversal_before_running_process(tmp_path: Path) -> None:
    with pytest.raises(Cra5BridgeError, match="under work_dir"):
        run_cra5_bridge(
            _request(tmp_path, input_path="../outside.bin"),
            command=(_command()[0], "-c", "raise SystemExit(99)"),
            work_dir=tmp_path,
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    (
        ("stale", "run_id"),
        ("malformed", "JSON"),
        ("nonzero", "exit code"),
    ),
)
def test_bridge_rejects_invalid_runtime_results(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    (tmp_path / "input.bin").write_bytes(b"encoded-input")
    (tmp_path / ".fake-cra5-mode").write_text(mode, encoding="utf-8")

    with pytest.raises(Cra5BridgeError, match=match):
        run_cra5_bridge(
            _request(tmp_path),
            command=_command(),
            work_dir=tmp_path,
            timeout_seconds=1.0,
        )


def test_bridge_rejects_timeout(tmp_path: Path) -> None:
    (tmp_path / "input.bin").write_bytes(b"encoded-input")
    (tmp_path / ".fake-cra5-mode").write_text("timeout", encoding="utf-8")

    with pytest.raises(Cra5BridgeError, match="timed out"):
        run_cra5_bridge(
            _request(tmp_path),
            command=_command(),
            work_dir=tmp_path,
            timeout_seconds=0.05,
        )
