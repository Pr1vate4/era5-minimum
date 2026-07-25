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
  const [runError, setRunError] = useState<string | null>(null)
  const [statusRevision, setStatusRevision] = useState(0)

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
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => {
      client
        .getJob(job.id, controller.signal)
        .then((nextJob) => {
          setJob(nextJob)
          if (nextJob.status === 'failed') {
            setRunError(nextJob.error ?? 'Обработка завершилась с ошибкой.')
          }
        })
        .catch((error: unknown) => {
          if (!controller.signal.aborted) {
            setRunError(toMessage(error, 'Не удалось получить статус задачи.'))
          }
        })
    }, POLL_INTERVAL_MS)
    return () => {
      controller.abort()
      window.clearTimeout(timeoutId)
    }
  }, [client, job])

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
    setJob({
      id: 'upload',
      status: 'queued',
      progress: 0,
      message: 'Загрузка ERA5-файла…',
      error: null,
      metrics: null,
      downloads: null,
    })
    try {
      setJob(await client.createJob(file, targetRatio))
    } catch (error) {
      setJob(null)
      setRunError(toMessage(error, 'Не удалось запустить сжатие.'))
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
    processing: job?.status === 'queued' || job?.status === 'running',
  }
}

function toMessage(error: unknown, fallback: string) {
  if (error instanceof DOMException && error.name === 'AbortError') {
    return 'Запрос превысил допустимое время.'
  }
  return error instanceof Error ? error.message : fallback
}
