import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { parseCodecJob } from './api'
import {
  FrameCompressionButton,
  FrameCompressionNotice,
  FrameCompressionReport,
} from './CurrentFrameCompression'
import {
  createCompressionRunGuard,
  requestCurrentFrameCompression,
} from './useCurrentFrameCompression'

const completedJob = () =>
  parseCodecJob({
    id: 'era5-job',
    status: 'completed',
    progress: 1,
    metrics: {
      serialized_compression_ratio: 33.2,
      tensor_compression_ratio: 32,
      bitstream_bytes: 1024,
      exact_roundtrip: true,
      encode_seconds: 0.8,
      decode_seconds: 0.4,
    },
    downloads: {
      bitstream: '/api/v1/codec/jobs/era5-job/bitstream',
      reconstruction: '/api/v1/codec/jobs/era5-job/reconstruction',
    },
    previews: {
      original: '/api/v1/codec/jobs/era5-job/preview/original.png',
      reconstruction:
        '/api/v1/codec/jobs/era5-job/preview/reconstruction.png',
    },
    source: {
      type: 'era5',
      timestamp: '2020-01-01T18:00:00Z',
      dataset_id: 'weatherbench-validation',
    },
  })

describe('current frame compression UI', () => {
  it('shows a disabled loading state without fake progress', () => {
    const html = renderToStaticMarkup(
      <FrameCompressionButton
        timestamp="2020-01-01T18:00:00Z"
        frameReady
        serviceLoading={false}
        serviceReady
        processing
        onCompress={vi.fn()}
      />,
    )

    expect(html).toContain('disabled')
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('Сжатие кадра')
    expect(html).not.toMatch(/\d+%/)
  })

  it('prevents a second run until the active run releases its guard', () => {
    const guard = createCompressionRunGuard()

    expect(guard.acquire()).toBe(true)
    expect(guard.acquire()).toBe(false)
    guard.release()
    expect(guard.acquire()).toBe(true)
  })

  it('submits the current selection timestamp on every new run', async () => {
    const createEra5Job = vi.fn().mockResolvedValue(completedJob())
    const firstTimestamp = '2020-01-01T18:00:00Z'
    const nextTimestamp = '2020-01-02T06:00:00Z'

    await requestCurrentFrameCompression({ createEra5Job }, firstTimestamp)
    await requestCurrentFrameCompression({ createEra5Job }, nextTimestamp)

    expect(createEra5Job).toHaveBeenNthCalledWith(1, firstTimestamp, undefined)
    expect(createEra5Job).toHaveBeenNthCalledWith(2, nextTimestamp, undefined)
  })

  it('shows an accessible error while keeping retry available', () => {
    const notice = renderToStaticMarkup(
      <FrameCompressionNotice
        processing={false}
        job={null}
        error="Кадр для этого timestamp не найден."
        service={{
          ready: true,
          checkpoint: 'sha256',
          modelName: 'N32',
          message: 'ready',
        }}
        serviceError={null}
        onRetryService={vi.fn()}
      />,
    )
    const button = renderToStaticMarkup(
      <FrameCompressionButton
        timestamp="2020-01-01T18:00:00Z"
        frameReady
        serviceLoading={false}
        serviceReady
        processing={false}
        onCompress={vi.fn()}
      />,
    )

    expect(notice).toContain('role="alert"')
    expect(notice).toContain('Кадр для этого timestamp не найден.')
    expect(notice).not.toContain('Serialized compression')
    expect(button).not.toContain(' disabled=""')
  })

  it('feeds the completed backend job into the existing report', () => {
    const html = renderToStaticMarkup(
      <FrameCompressionReport
        job={completedJob()}
        jobTimestamp="2020-01-02T06:00:00Z"
        codecBaseUrl="http://codec.local"
      />,
    )

    expect(html.match(/data-testid="frame-compression-report"/g)).toHaveLength(1)
    expect(html).toContain('33.2×')
    expect(html).toContain('32.0×')
    expect(html).toContain('1.0 КиБ')
    expect(html).toContain('Encode 0.80 с')
    expect(html).toContain('Decode 0.40 с')
    expect(html).toContain('Подтверждён')
    expect(html).toContain(
      'http://codec.local/api/v1/codec/jobs/era5-job/bitstream',
    )
    expect(html).toContain(
      'http://codec.local/api/v1/codec/jobs/era5-job/reconstruction',
    )
    expect(html).toContain('полный канонический ERA5-кадр')
    expect(html).toContain('01.01.2020')
    expect(html).not.toContain('02.01.2020')
  })
})
