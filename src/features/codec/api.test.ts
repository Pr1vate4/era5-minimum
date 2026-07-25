import { describe, expect, it } from 'vitest'
import { parseCodecJob, parseCodecStatus } from './api'

describe('codec API contract', () => {
  it('rejects a ready status without checkpoint provenance', () => {
    expect(() => parseCodecStatus({ ready: true })).toThrow(/checkpoint/i)
  })

  it('represents an unavailable model explicitly', () => {
    expect(
      parseCodecStatus({
        ready: false,
        checkpoint: null,
        message: 'Checkpoint is not installed',
      }),
    ).toEqual({
      ready: false,
      checkpoint: null,
      modelName: null,
      message: 'Checkpoint is not installed',
    })
  })

  it('keeps serialized and tensor compression ratios separate', () => {
    const job = parseCodecJob({
      id: 'job-1',
      status: 'completed',
      progress: 1,
      metrics: {
        serialized_compression_ratio: 33.2,
        tensor_compression_ratio: 64,
        bitstream_bytes: 1024,
        exact_roundtrip: true,
        encode_seconds: 0.8,
        decode_seconds: 0.4,
      },
      downloads: {
        bitstream: '/api/v1/codec/jobs/job-1/bitstream',
        reconstruction: '/api/v1/codec/jobs/job-1/reconstruction',
      },
    })

    expect(job.metrics?.serializedCompressionRatio).toBe(33.2)
    expect(job.metrics?.tensorCompressionRatio).toBe(64)
    expect(job.metrics?.exactRoundtrip).toBe(true)
    expect(job.downloads?.bitstream).toContain('bitstream')
  })

  it('rejects a completed job without honest serialized metrics', () => {
    expect(() =>
      parseCodecJob({
        id: 'job-2',
        status: 'completed',
        progress: 1,
        metrics: {
          tensor_compression_ratio: 64,
        },
      }),
    ).toThrow(/serialized/i)
  })
})
