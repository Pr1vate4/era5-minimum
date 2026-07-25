export type CodecTargetRatio = 32 | 64

export type CodecServiceStatus = {
  ready: boolean
  checkpoint: string | null
  modelName: string | null
  message: string
}

export type CodecJobStatus = 'queued' | 'running' | 'completed' | 'failed'

export type CodecMetrics = {
  serializedCompressionRatio: number
  tensorCompressionRatio: number | null
  bitstreamBytes: number
  exactRoundtrip: boolean
  encodeSeconds: number | null
  decodeSeconds: number | null
}

export type CodecDownloads = {
  bitstream: string
  reconstruction: string
}

export type CodecPreviews = {
  original: string
  reconstruction: string
}

export type CodecJob = {
  id: string
  status: CodecJobStatus
  progress: number
  message: string | null
  error: string | null
  metrics: CodecMetrics | null
  downloads: CodecDownloads | null
  previews: CodecPreviews | null
}

export type CodecClient = {
  getStatus: (signal?: AbortSignal) => Promise<CodecServiceStatus>
  createJob: (
    file: File,
    targetRatio: CodecTargetRatio,
    signal?: AbortSignal,
  ) => Promise<CodecJob>
  getJob: (jobId: string, signal?: AbortSignal) => Promise<CodecJob>
}
