#!/usr/bin/env python3
"""Deterministic fake CRA5 runtime used by bridge contract tests."""

from __future__ import annotations

import json
import sys
import time
from hashlib import sha256
from pathlib import Path


def main() -> int:
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    work_dir = request_path.parent
    mode_path = work_dir / ".fake-cra5-mode"
    mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else ""

    if mode == "timeout":
        time.sleep(2.0)
        return 0
    if mode == "nonzero":
        print("fake CRA5 runtime failed", file=sys.stderr)
        return 7
    if mode == "malformed":
        response_path = work_dir / request["response_path"]
        response_path.write_text("{not-json", encoding="utf-8")
        return 0

    input_path = work_dir / request["input_path"]
    output_path = work_dir / request["output_path"]
    bitstream_path = work_dir / "bitstream.bin"
    output_path.write_bytes(input_path.read_bytes() + b"-decoded")
    bitstream = b"fake-cra5-bitstream-v1"
    bitstream_path.write_bytes(bitstream)
    response = {
        "protocol_version": request["protocol_version"],
        "run_id": request["run_id"] + ("-stale" if mode == "stale" else ""),
        "operation": request["operation"],
        "output_path": request["output_path"],
        "bitstream_path": "bitstream.bin",
        "tensor_shape": request["tensor_shape"],
        "channel_order": request["channel_order"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "bitstream_sha256": sha256(bitstream).hexdigest(),
        "exact_roundtrip": True,
        "encode_seconds": 0.01,
        "decode_seconds": 0.02,
    }
    response_path = work_dir / request["response_path"]
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
