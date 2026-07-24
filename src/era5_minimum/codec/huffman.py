from __future__ import annotations

from heapq import heappop, heappush
from typing import Mapping


def build_code_lengths(counts: Mapping[int, int]) -> dict[int, int]:
    """Build Huffman code lengths from symbol frequencies."""
    if not counts:
        raise ValueError("counts must not be empty")
    if len(counts) == 1:
        symbol = next(iter(counts))
        return {int(symbol): 1}

    heap: list[tuple[int, int, int | tuple[object, object]]] = []
    serial = 0
    for symbol, count in counts.items():
        heappush(heap, (int(count), serial, int(symbol)))
        serial += 1

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        merged = (left[2], right[2])
        heappush(heap, (left[0] + right[0], serial, merged))
        serial += 1

    lengths: dict[int, int] = {}

    def visit(node: int | tuple[object, object], depth: int) -> None:
        if isinstance(node, tuple):
            visit(node[0], depth + 1)
            visit(node[1], depth + 1)
            return
        lengths[int(node)] = max(depth, 1)

    visit(heap[0][2], 0)
    return lengths


def build_canonical_codes(code_lengths: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    """Build canonical Huffman codes as (code, width) tuples."""
    if not code_lengths:
        raise ValueError("code_lengths must not be empty")

    ordered = sorted((int(length), int(symbol)) for symbol, length in code_lengths.items())
    code = 0
    previous_width = 0
    codes: dict[int, tuple[int, int]] = {}

    for width, symbol in ordered:
        if width < 1:
            raise ValueError("code width must be positive")
        code <<= width - previous_width
        codes[symbol] = (code, width)
        code += 1
        previous_width = width

    return codes
