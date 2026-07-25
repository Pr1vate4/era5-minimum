import { afterEach, describe, expect, it, vi } from 'vitest'
import * as codecApi from './api'
import { parseCodecJob, parseCodecStatus } from './api'
import * as codecWorkspace from './useCodecWorkspace'
import { shouldPollCodecJob } from './useCodecWorkspace'

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
      previews: {
        original: '/api/v1/codec/jobs/job-1/preview/original.png',
        reconstruction: '/api/v1/codec/jobs/job-1/preview/reconstruction.png',
      },
    })

    expect(job.metrics?.serializedCompressionRatio).toBe(33.2)
    expect(job.metrics?.tensorCompressionRatio).toBe(64)
    expect(job.metrics?.exactRoundtrip).toBe(true)
    expect(job.downloads?.bitstream).toContain('bitstream')
    expect(job.previews?.reconstruction).toContain('preview')
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

  it('polls only real queued or running backend jobs', () => {
    expect(shouldPollCodecJob(null)).toBe(false)
    expect(
      shouldPollCodecJob({
        id: 'job-3',
        status: 'queued',
        progress: 0,
        message: null,
        error: null,
        metrics: null,
        downloads: null,
        previews: null,
      }),
    ).toBe(true)
  })

  it('does not apply the short metadata timeout to a large file upload', () => {
    const timeoutPolicy = (
      codecApi as unknown as {
        getCodecRequestTimeoutMs?: (
          requestKind: 'metadata' | 'upload',
          defaultTimeoutMs: number,
        ) => number | null
      }
    ).getCodecRequestTimeoutMs

    expect(timeoutPolicy).toBeTypeOf('function')
    expect(timeoutPolicy?.('metadata', 15_000)).toBe(15_000)
    expect(timeoutPolicy?.('upload', 15_000)).toBeNull()
  })

  it('keeps polling after a transient status request failure', async () => {
    const startPolling = (
      codecWorkspace as unknown as {
        startCodecPolling?: (options: {
          jobId: string
          getJob: (jobId: string, signal: AbortSignal) => Promise<ReturnType<typeof parseCodecJob>>
          onJob: (job: ReturnType<typeof parseCodecJob>) => void
          onError: (error: unknown) => void
          delayMs: number
        }) => () => void
      }
    ).startCodecPolling

    expect(startPolling).toBeTypeOf('function')
    if (!startPolling) return

    vi.useFakeTimers()
    const completedJob = parseCodecJob({
      id: 'job-poll',
      status: 'completed',
      progress: 1,
      metrics: {
        serialized_compression_ratio: 32,
        tensor_compression_ratio: 64,
        bitstream_bytes: 2048,
        exact_roundtrip: true,
      },
      downloads: {
        bitstream: '/bitstream',
        reconstruction: '/reconstruction',
      },
      previews: {
        original: '/original.png',
        reconstruction: '/reconstruction.png',
      },
    })
    const getJob = vi
      .fn()
      .mockRejectedValueOnce(new Error('temporary network error'))
      .mockResolvedValueOnce(completedJob)
    const onJob = vi.fn()
    const onError = vi.fn()
    const stop = startPolling({
      jobId: 'job-poll',
      getJob,
      onJob,
      onError,
      delayMs: 10,
    })

    await vi.advanceTimersByTimeAsync(10)
    expect(onError).toHaveBeenCalledOnce()
    await vi.advanceTimersByTimeAsync(10)
    expect(getJob).toHaveBeenCalledTimes(2)
    expect(onJob).toHaveBeenCalledWith(completedJob)
    stop()
  })

  it('extracts an immediate failed-job response for the error UI', () => {
    const getFailure = (
      codecWorkspace as unknown as {
        getCodecJobFailure?: (
          job: ReturnType<typeof parseCodecJob>,
        ) => string | null
      }
    ).getCodecJobFailure

    expect(getFailure).toBeTypeOf('function')
    expect(
      getFailure?.({
        id: 'failed-job',
        status: 'failed',
        progress: 0,
        message: null,
        error: 'Checkpoint rejected the input',
        metrics: null,
        downloads: null,
        previews: null,
      }),
    ).toBe('Checkpoint rejected the input')
  })

  it('requires downloadable artifacts and visual previews for completed jobs', () => {
    expect(() =>
      parseCodecJob({
        id: 'job-no-artifacts',
        status: 'completed',
        progress: 1,
        metrics: {
          serialized_compression_ratio: 32,
          bitstream_bytes: 2048,
          exact_roundtrip: true,
        },
      }),
    ).toThrow(/artifacts|downloads|previews/i)
  })
})

afterEach(() => {
  vi.useRealTimers()
})
