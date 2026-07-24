# ML-001 Codec Harness Design

## Goal

Build the first honest codec layer for ERA5-Minimum: deterministic quantization, real entropy coding, exact symbol roundtrip, and reproducible artifact metadata.

## Scope

- Input tensor contract: `[batch, channels, height, width]`
- Configurable channel order, with support for the current smoke fixtures and the future 28-channel contest layout
- Train-only normalization contract with stored mean/std checksum and versioned provenance
- Quantization to integer symbols
- Canonical entropy-coded bitstream
- Decode path with exact symbol equality
- Metadata and checksum recording
- Local verification tests

## Selected Approach

Use per-channel scalar quantization followed by canonical Huffman coding over the integer symbol stream.

Why this first:

- simple enough to implement and test quickly
- produces a real serialized bitstream
- makes exact symbol roundtrip easy to verify
- keeps compression metrics honest and separate

## Alternatives Considered

- zlib/deflate wrapper: easier, but too opaque for codec debugging
- rANS/arithmetic coding: better long-term codec foundation, but higher implementation risk for the first step

## Architecture

- `CodecConfig`: version, channel order, grid, quantization parameters, seed, git commit
- `NormalizationSpec`: per-channel mean/std, source manifest, SHA256, and train-only flag
- `Quantizer`: maps float tensors to int symbols and back
- `EntropyCoder`: encodes/decodes integer symbols into a byte stream
- `BitstreamWriter/Reader`: stores header, payload, checksum, and versioned metadata
- `CodecResult`: paths, sizes, tensor ratio, serialized ratio, and roundtrip status

## Data Flow

1. Validate tensor shape, dtype, channel order, and grid config.
2. Apply stored normalization, never refit stats inside the codec run.
3. Quantize each channel deterministically.
4. Flatten symbols in fixed channel-major order.
5. Encode symbols into bytes with canonical Huffman coding.
6. Write bitstream plus metadata.
7. Decode bytes back to symbols.
8. Assert exact symbol equality before any float-space metric is reported.

## Error Handling

- reject unsupported shapes, channel counts, or grid configs
- reject missing or refit normalization statistics
- reject missing or incompatible metadata
- reject unknown codec versions
- reject corrupted or truncated bitstreams
- fail loudly on symbol mismatch after decode

## Testing

- exact symbol roundtrip on synthetic tensors
- metadata completeness and version fields
- normalization provenance and checksum fields
- corruption detection for truncated bitstreams
- separate reporting of tensor ratio and serialized ratio
- deterministic output for fixed seed/config

## Non-Goals

- final 32x/64x performance
- latent probe training
- full evaluator integration
- architecture search

## Assumptions

- current repo fixtures remain the smoke-test input for the first implementation pass
- 28-channel support will be added through config, not hard-coded tensors
- compression ratio must never be conflated with raw tensor element count
