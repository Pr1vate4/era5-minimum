import type {
  CodecClient,
  CodecDownloads,
  CodecJob,
  CodecJobStatus,
  CodecMetrics,
  CodecServiceStatus,
  CodecTargetRatio,
} from './types'

type JsonRecord = Record<string, unknown>

export function parseCodecStatus(input: unknown): CodecServiceStatus {
  const value = asRecord(input, 'Codec status')
  const ready = asBoolean(value.ready, 'ready')
  const checkpoint = nullableString(value.checkpoint, 'checkpoint')
  const modelName = nullableString(value.model_name ?? value.modelName, 'model_name')
  const message =
    optionalString(value.message) ??
    (ready ? 'Модель готова к обработке ERA5.' : 'Модель пока недоступна.')

  if (ready && !checkpoint) {
    throw new Error('Codec status marked ready without checkpoint provenance')
  }

  return { ready, checkpoint, modelName, message }
}

export function parseCodecJob(input: unknown): CodecJob {
  const value = asRecord(input, 'Codec job')
  const id = asNonEmptyString(value.id, 'id')
  const status = asJobStatus(value.status)
  const progress = clamp(asFiniteNumber(value.progress ?? 0, 'progress'), 0, 1)
  const metrics = value.metrics == null ? null : parseMetrics(value.metrics)
  const downloads = value.downloads == null ? null : parseDownloads(value.downloads)

  if (status === 'completed' && !metrics) {
    throw new Error('Completed codec job must contain serialized metrics')
  }

  return {
    id,
    status,
    progress,
    message: optionalString(value.message),
    error: optionalString(value.error),
    metrics,
    downloads,
  }
}

export function createCodecClient({
  baseUrl,
  timeoutMs,
}: {
  baseUrl: string
  timeoutMs: number
}): CodecClient {
  const request = async (
    path: string,
    init: RequestInit = {},
    externalSignal?: AbortSignal,
  ) => {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)
    const abort = () => controller.abort()
    externalSignal?.addEventListener('abort', abort, { once: true })

    try {
      const response = await fetch(joinUrl(baseUrl, path), {
        ...init,
        signal: controller.signal,
        headers: {
          Accept: 'application/json',
          ...init.headers,
        },
      })
      if (!response.ok) {
        const details = (await response.text()).trim()
        throw new Error(`Codec API: HTTP ${response.status}${details ? ` · ${details}` : ''}`)
      }
      return (await response.json()) as unknown
    } finally {
      window.clearTimeout(timeoutId)
      externalSignal?.removeEventListener('abort', abort)
    }
  }

  return {
    getStatus: async (signal) =>
      parseCodecStatus(await request('/api/v1/codec/status', {}, signal)),
    createJob: async (file, targetRatio, signal) => {
      const body = new FormData()
      body.set('file', file)
      body.set('target_ratio', String(targetRatio))
      return parseCodecJob(
        await request('/api/v1/codec/jobs', { method: 'POST', body }, signal),
      )
    },
    getJob: async (jobId, signal) =>
      parseCodecJob(
        await request(`/api/v1/codec/jobs/${encodeURIComponent(jobId)}`, {}, signal),
      ),
  }
}

export function isCodecTargetRatio(value: number): value is CodecTargetRatio {
  return value === 32 || value === 64
}

function parseMetrics(input: unknown): CodecMetrics {
  const value = asRecord(input, 'Codec metrics')
  const serializedCompressionRatio = asPositiveNumber(
    value.serialized_compression_ratio,
    'serialized_compression_ratio',
  )
  const bitstreamBytes = asPositiveNumber(value.bitstream_bytes, 'bitstream_bytes')

  return {
    serializedCompressionRatio,
    tensorCompressionRatio: nullableFiniteNumber(
      value.tensor_compression_ratio,
      'tensor_compression_ratio',
    ),
    bitstreamBytes,
    exactRoundtrip: asBoolean(value.exact_roundtrip, 'exact_roundtrip'),
    encodeSeconds: nullableFiniteNumber(value.encode_seconds, 'encode_seconds'),
    decodeSeconds: nullableFiniteNumber(value.decode_seconds, 'decode_seconds'),
  }
}

function parseDownloads(input: unknown): CodecDownloads {
  const value = asRecord(input, 'Codec downloads')
  return {
    bitstream: asNonEmptyString(value.bitstream, 'downloads.bitstream'),
    reconstruction: asNonEmptyString(
      value.reconstruction,
      'downloads.reconstruction',
    ),
  }
}

function asJobStatus(input: unknown): CodecJobStatus {
  if (
    input === 'queued' ||
    input === 'running' ||
    input === 'completed' ||
    input === 'failed'
  ) {
    return input
  }
  throw new Error(`Invalid codec job status: ${String(input)}`)
}

function asRecord(input: unknown, label: string): JsonRecord {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error(`${label} must be an object`)
  }
  return input as JsonRecord
}

function asBoolean(input: unknown, label: string) {
  if (typeof input !== 'boolean') throw new Error(`${label} must be a boolean`)
  return input
}

function asFiniteNumber(input: unknown, label: string) {
  if (typeof input !== 'number' || !Number.isFinite(input)) {
    throw new Error(`${label} must be a finite number`)
  }
  return input
}

function asPositiveNumber(input: unknown, label: string) {
  const value = asFiniteNumber(input, label)
  if (value <= 0) throw new Error(`${label} must be positive`)
  return value
}

function nullableFiniteNumber(input: unknown, label: string) {
  return input == null ? null : asFiniteNumber(input, label)
}

function nullableString(input: unknown, label: string) {
  if (input == null) return null
  return asNonEmptyString(input, label)
}

function optionalString(input: unknown) {
  return typeof input === 'string' && input.trim() ? input.trim() : null
}

function asNonEmptyString(input: unknown, label: string) {
  if (typeof input !== 'string' || !input.trim()) {
    throw new Error(`${label} must be a non-empty string`)
  }
  return input.trim()
}

function joinUrl(baseUrl: string, path: string) {
  return `${baseUrl.trim().replace(/\/$/, '')}${path}`
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}
