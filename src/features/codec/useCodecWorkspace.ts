import { useCallback, useEffect, useMemo, useState } from 'react'
import { useAppSettings } from '../../hooks/useAppSettings'
import { createCodecClient } from './api'
import { validateCodecFile } from './fileValidation'
import type {
  CodecJob,
  CodecServiceStatus,
  CodecTargetRatio,
} from './types'

const POLL_INTERVAL_MS = 1200

export function useCodecWorkspace() {
  const { settings } = useAppSettings()
  const [file, setFileState] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [targetRatio, setTargetRatio] = useState<CodecTargetRatio>(32)
  const [service, setService] = useState<CodecServiceStatus | null>(null)
  const [serviceLoading, setServiceLoading] = useState(true)
  const [serviceError, setServiceError] = useState<string | null>(null)
  const [job, setJob] = useState<CodecJob | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [statusRevision, setStatusRevision] = useState(0)
  const pollingJobId = shouldPollCodecJob(job) ? job.id : null

  const client = useMemo(
    () =>
      createCodecClient({
        baseUrl: settings.services.codecBaseUrl,
        timeoutMs: settings.api.timeoutMs,
      }),
    [settings.api.timeoutMs, settings.services.codecBaseUrl],
  )

  useEffect(() => {
    const controller = new AbortController()
    setServiceLoading(true)
    setServiceError(null)
    client
      .getStatus(controller.signal)
      .then(setService)
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        setService(null)
        setServiceError(toMessage(error, 'Codec API недоступен.'))
      })
      .finally(() => {
        if (!controller.signal.aborted) setServiceLoading(false)
      })
    return () => controller.abort()
  }, [client, statusRevision])

  useEffect(() => {
    if (!pollingJobId) return
    return startCodecPolling({
      jobId: pollingJobId,
      getJob: client.getJob,
      onJob: (nextJob) => {
        setJob(nextJob)
        setRunError(getCodecJobFailure(nextJob))
      },
      onError: (error) => {
        setRunError(toMessage(error, 'Не удалось получить статус задачи. Повторяем запрос…'))
      },
      delayMs: POLL_INTERVAL_MS,
    })
  }, [client, pollingJobId])

  const setFile = useCallback((nextFile: File | null) => {
    setJob(null)
    setRunError(null)
    setFileState(nextFile)
    setFileError(nextFile ? validateCodecFile(nextFile) : null)
  }, [])

  const submit = useCallback(async () => {
    if (!file) {
      setFileError('Выберите ERA5-файл.')
      return
    }
    const validationError = validateCodecFile(file)
    if (validationError) {
      setFileError(validationError)
      return
    }
    if (!service?.ready) {
      setRunError(service?.message ?? 'Модель ещё не готова.')
      return
    }
    setRunError(null)
    setJob(null)
    setSubmitting(true)
    try {
      const nextJob = await client.createJob(file, targetRatio)
      setJob(nextJob)
      setRunError(getCodecJobFailure(nextJob))
    } catch (error) {
      setJob(null)
      setRunError(toMessage(error, 'Не удалось запустить сжатие.'))
    } finally {
      setSubmitting(false)
    }
  }, [client, file, service, targetRatio])

  return {
    file,
    fileError,
    setFile,
    targetRatio,
    setTargetRatio,
    service,
    serviceLoading,
    serviceError,
    retryService: () => setStatusRevision((value) => value + 1),
    job,
    runError,
    submit,
    processing: submitting || shouldPollCodecJob(job),
  }
}

export function shouldPollCodecJob(
  job: CodecJob | null,
): job is CodecJob & { status: 'queued' | 'running' } {
  return job?.status === 'queued' || job?.status === 'running'
}

export function getCodecJobFailure(job: CodecJob) {
  if (job.status !== 'failed') return null
  return job.error ?? job.message ?? 'Обработка завершилась с ошибкой.'
}

export function startCodecPolling({
  jobId,
  getJob,
  onJob,
  onError,
  delayMs,
}: {
  jobId: string
  getJob: (jobId: string, signal: AbortSignal) => Promise<CodecJob>
  onJob: (job: CodecJob) => void
  onError: (error: unknown) => void
  delayMs: number
}) {
  let active = true
  let controller: AbortController | null = null
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  const schedule = () => {
    if (!active) return
    timeoutId = globalThis.setTimeout(() => void poll(), delayMs)
  }

  const poll = async () => {
    controller = new AbortController()
    try {
      const nextJob = await getJob(jobId, controller.signal)
      if (!active) return
      onJob(nextJob)
      if (shouldPollCodecJob(nextJob)) schedule()
    } catch (error) {
      if (!active || controller.signal.aborted) return
      onError(error)
      schedule()
    }
  }

  schedule()

  return () => {
    active = false
    controller?.abort()
    if (timeoutId !== null) globalThis.clearTimeout(timeoutId)
  }
}

function toMessage(error: unknown, fallback: string) {
  if (error instanceof DOMException && error.name === 'AbortError') {
    return 'Запрос превысил допустимое время.'
  }
  return error instanceof Error ? error.message : fallback
}
