import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { CodecResults } from './CodecResults'
import { parseCodecJob } from './api'

describe('CodecResults', () => {
  it('renders backend previews for the original and reconstructed fields', () => {
    const job = parseCodecJob({
      id: 'job-preview',
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
        bitstream: '/bitstream',
        reconstruction: '/reconstruction',
      },
      previews: {
        original: '/preview/original.png',
        reconstruction: '/preview/reconstruction.png',
      },
    })

    const html = renderToStaticMarkup(
      <CodecResults
        job={job}
        sourceDescription="frame.npz · 4.0 КиБ"
        resolveDownload={(path) => `http://codec.local${path}`}
      />,
    )

    expect(html.match(/<img/g)).toHaveLength(2)
    expect(html).toContain('http://codec.local/preview/original.png')
    expect(html).toContain('http://codec.local/preview/reconstruction.png')
    expect(html).toContain('Исходное поле ERA5')
    expect(html).toContain('Восстановленное поле ERA5')
  })
})
