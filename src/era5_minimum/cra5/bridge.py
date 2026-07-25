"""Validated subprocess bridge for an isolated CRA5 runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from era5_minimum.data.channel_spec import CHANNEL_NAMES

BRIDGE_PROTOCOL_VERSION = "cra5-bridge-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_ALLOWED_OPERATIONS = frozenset({"encode_decode"})


class Cra5BridgeError(RuntimeError):
    """Raised when the external CRA5 process violates the bridge contract."""


@dataclass(frozen=True, slots=True)
class Cra5BridgeRequest:
    """Request manifest passed to the isolated CRA5 process."""

    run_id: str
    operation: str
    input_path: Path | str
    output_path: Path | str
    tensor_shape: tuple[int, ...]
    channel_order: tuple[str, ...]
    checkpoint_sha256: str
    response_path: Path | str = Path("bridge_response.json")
    protocol_version: str = BRIDGE_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request using relative artifact paths."""
        return {
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "operation": self.operation,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "response_path": str(self.response_path),
            "tensor_shape": list(self.tensor_shape),
            "channel_order": list(self.channel_order),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Cra5BridgeRequest":
        """Build a request from a JSON object and validate its schema."""
        try:
            request = cls(
                protocol_version=str(payload["protocol_version"]),
                run_id=str(payload["run_id"]),
                operation=str(payload["operation"]),
                input_path=str(payload["input_path"]),
                output_path=str(payload["output_path"]),
                response_path=str(payload.get("response_path", "bridge_response.json")),
                tensor_shape=tuple(int(value) for value in payload["tensor_shape"]),
                channel_order=tuple(str(value) for value in payload["channel_order"]),
                checkpoint_sha256=str(payload["checkpoint_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Cra5BridgeError(f"invalid bridge request: {error}") from error
        _validate_request(request)
        return request


@dataclass(frozen=True, slots=True)
class Cra5BridgeResponse:
    """Validated response metadata emitted by the isolated CRA5 process."""

    run_id: str
    operation: str
    output_path: Path | str
    bitstream_path: Path | str
    tensor_shape: tuple[int, ...]
    channel_order: tuple[str, ...]
    checkpoint_sha256: str
    bitstream_sha256: str
    exact_roundtrip: bool
    encode_seconds: float
    decode_seconds: float
    protocol_version: str = BRIDGE_PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Cra5BridgeResponse":
        """Build a response from JSON and validate scalar types and values."""
        try:
            response = cls(
                protocol_version=str(payload["protocol_version"]),
                run_id=str(payload["run_id"]),
                operation=str(payload["operation"]),
                output_path=str(payload["output_path"]),
                bitstream_path=str(payload["bitstream_path"]),
                tensor_shape=tuple(int(value) for value in payload["tensor_shape"]),
                channel_order=tuple(str(value) for value in payload["channel_order"]),
                checkpoint_sha256=str(payload["checkpoint_sha256"]),
                bitstream_sha256=str(payload["bitstream_sha256"]),
                exact_roundtrip=payload["exact_roundtrip"],
                encode_seconds=float(payload["encode_seconds"]),
                decode_seconds=float(payload["decode_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Cra5BridgeError(f"invalid bridge response: {error}") from error
        _validate_response(response)
        return response


def run_cra5_bridge(
    request: Cra5BridgeRequest,
    *,
    command: tuple[str, ...],
    work_dir: Path,
    timeout_seconds: float,
) -> Cra5BridgeResponse:
    """Run and validate one isolated CRA5 request.

    The command receives one absolute path to a request JSON file. All paths
    inside the request and response remain relative to ``work_dir``.
    """
    _validate_request(request)
    if not command:
        raise Cra5BridgeError("bridge command must not be empty")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise Cra5BridgeError("timeout_seconds must be finite and positive")

    root = Path(work_dir).expanduser().resolve()
    if not root.is_dir():
        raise Cra5BridgeError(f"work_dir does not exist or is not a directory: {root}")
    input_path = _resolve_artifact(root, request.input_path, field="input_path")
    output_path = _resolve_artifact(root, request.output_path, field="output_path")
    response_path = _resolve_artifact(root, request.response_path, field="response_path")
    if not input_path.is_file():
        raise Cra5BridgeError(f"input_path does not exist under work_dir: {request.input_path}")
    response_path.unlink(missing_ok=True)

    request_payload = request.to_dict()
    request_payload["response_path"] = response_path.relative_to(root).as_posix()
    request_file = root / (
        ".cra5-bridge-request-"
        + hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()[:16]
        + ".json"
    )
    request_file.write_text(
        json.dumps(request_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(
            (*command, str(request_file)),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise Cra5BridgeError(
            f"CRA5 bridge timed out after {timeout_seconds} seconds"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise Cra5BridgeError(
            f"CRA5 bridge process returned non-zero exit code {completed.returncode}{detail}"
        )
    if not response_path.is_file():
        raise Cra5BridgeError(f"CRA5 bridge response is missing: {response_path}")

    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Cra5BridgeError(f"CRA5 bridge response is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise Cra5BridgeError("CRA5 bridge response must be a JSON object")
    response = Cra5BridgeResponse.from_dict(payload)
    if response.protocol_version != request.protocol_version:
        raise Cra5BridgeError("CRA5 bridge response protocol_version does not match request")
    if response.run_id != request.run_id:
        raise Cra5BridgeError("CRA5 bridge response run_id does not match request")
    if response.operation != request.operation:
        raise Cra5BridgeError("CRA5 bridge response operation does not match request")
    if response.tensor_shape != request.tensor_shape:
        raise Cra5BridgeError("CRA5 bridge response tensor_shape does not match request")
    if response.channel_order != request.channel_order:
        raise Cra5BridgeError("CRA5 bridge response channel_order does not match request")
    if response.checkpoint_sha256.lower() != request.checkpoint_sha256.lower():
        raise Cra5BridgeError("CRA5 bridge response checkpoint_sha256 does not match request")

    _resolve_artifact(root, response.output_path, field="response.output_path")
    bitstream_path = _resolve_artifact(root, response.bitstream_path, field="response.bitstream_path")
    resolved_output = _resolve_artifact(root, response.output_path, field="response.output_path")
    if not resolved_output.is_file():
        raise Cra5BridgeError(f"CRA5 reconstruction is missing: {resolved_output}")
    if not bitstream_path.is_file():
        raise Cra5BridgeError(f"CRA5 bitstream is missing: {bitstream_path}")
    measured_sha256 = hashlib.sha256(bitstream_path.read_bytes()).hexdigest()
    if measured_sha256.lower() != response.bitstream_sha256.lower():
        raise Cra5BridgeError(
            "CRA5 bridge bitstream_sha256 does not match serialized bitstream"
        )
    return response


def _validate_request(request: Cra5BridgeRequest) -> None:
    if request.protocol_version != BRIDGE_PROTOCOL_VERSION:
        raise Cra5BridgeError(
            f"unsupported bridge protocol_version: {request.protocol_version!r}"
        )
    if not request.run_id or "/" in request.run_id or "\\" in request.run_id:
        raise Cra5BridgeError("run_id must be a non-empty path-safe string")
    if request.operation not in _ALLOWED_OPERATIONS:
        raise Cra5BridgeError(f"unsupported bridge operation: {request.operation!r}")
    if tuple(request.channel_order) != CHANNEL_NAMES:
        raise Cra5BridgeError("bridge channel_order must match canonical 28-channel order")
    if len(request.tensor_shape) != 4 or any(
        not isinstance(value, int) or value <= 0 for value in request.tensor_shape
    ):
        raise Cra5BridgeError("tensor_shape must contain four positive dimensions")
    if not _SHA256_PATTERN.fullmatch(request.checkpoint_sha256):
        raise Cra5BridgeError("checkpoint_sha256 must be a 64-character hexadecimal digest")


def _validate_response(response: Cra5BridgeResponse) -> None:
    if response.protocol_version != BRIDGE_PROTOCOL_VERSION:
        raise Cra5BridgeError(
            f"unsupported response protocol_version: {response.protocol_version!r}"
        )
    if response.operation not in _ALLOWED_OPERATIONS:
        raise Cra5BridgeError(f"unsupported response operation: {response.operation!r}")
    if tuple(response.channel_order) != CHANNEL_NAMES:
        raise Cra5BridgeError("response channel_order must match canonical 28-channel order")
    if len(response.tensor_shape) != 4 or any(
        not isinstance(value, int) or value <= 0 for value in response.tensor_shape
    ):
        raise Cra5BridgeError("response tensor_shape must contain four positive dimensions")
    if not _SHA256_PATTERN.fullmatch(response.checkpoint_sha256):
        raise Cra5BridgeError("response checkpoint_sha256 is invalid")
    if not _SHA256_PATTERN.fullmatch(response.bitstream_sha256):
        raise Cra5BridgeError("response bitstream_sha256 is invalid")
    if not isinstance(response.exact_roundtrip, bool):
        raise Cra5BridgeError("response exact_roundtrip must be boolean")
    if (
        not math.isfinite(response.encode_seconds)
        or not math.isfinite(response.decode_seconds)
        or response.encode_seconds < 0
        or response.decode_seconds < 0
    ):
        raise Cra5BridgeError("response timings must be finite and non-negative")


def _resolve_artifact(root: Path, value: Path | str, *, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise Cra5BridgeError(f"{field} must resolve under work_dir")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise Cra5BridgeError(f"{field} must resolve under work_dir")
    return resolved
