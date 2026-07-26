import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAppSettings } from '../../hooks/useAppSettings'
import { createCodecClient } from './api'
import {
  getCodecJobFailure,
  shouldPollCodecJob,
  startCodecPolling,
} from './useCodecWorkspace'
import type { CodecJob, CodecServiceStatus } from './types'
import type { CodecClient } from './types'

const POLL_INTERVAL_MS = 1200

export function useCurrentFrameCompression(
  selectedTimestamp: string | undefined,
  enabled: boolean,
) {
  const { settings } = useAppSettings()
  const [service, setService] = useState<CodecServiceStatus | null>(null)
  const [serviceLoading, setServiceLoading] = useState(enabled)
  const [serviceError, setServiceError] = useState<string | null>(null)
  const [job, setJob] = useState<CodecJob | null>(null)
  const [jobTimestamp, setJobTimestamp] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statusRevision, setStatusRevision] = useState(0)
  const requestController = useRef<AbortController | null>(null)
  const activeJobId = useRef<string | null>(null)
  const runGuard = useRef(createCompressionRunGuard())
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
    if (!enabled) {
      setService(null)
      setServiceLoading(false)
      setServiceError(null)
      return
    }
    const controller = new AbortController()
    setServiceLoading(true)
    setServiceError(null)
    client
      .getStatus(controller.signal)
      .then(setService)
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        setService(null)
        setServiceError(toMessage(caught, 'Сервис сжатия сейчас недоступен.'))
      })
      .finally(() => {
        if (!controller.signal.aborted) setServiceLoading(false)
      })
    return () => controller.abort()
  }, [client, enabled, statusRevision])

  useEffect(() => {
    if (!pollingJobId) return
    return startCodecPolling({
      jobId: pollingJobId,
      getJob: client.getJob,
      onJob: (nextJob) => {
        if (activeJobId.current !== nextJob.id) return
        setJob(nextJob)
        setError(getCodecJobFailure(nextJob))
        if (!shouldPollCodecJob(nextJob)) runGuard.current.release()
      },
      onError: (caught) => {
        if (activeJobId.current !== pollingJobId) return
        setError(toMessage(caught, 'Не удалось получить статус сжатия.'))
      },
      delayMs: POLL_INTERVAL_MS,
    })
  }, [client, pollingJobId])

  useEffect(
    () => () => {
      requestController.current?.abort()
      runGuard.current.release()
    },
    [],
  )

  const compressCurrentFrame = useCallback(async () => {
    if (!selectedTimestamp || submitting || shouldPollCodecJob(job)) return
    if (!service?.ready) {
      setError(service?.message ?? 'Сервис сжатия сейчас недоступен.')
      return
    }
    if (!runGuard.current.acquire()) return

    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller
    setSubmitting(true)
    setError(null)
    setJob(null)
    setJobTimestamp(selectedTimestamp)
    activeJobId.current = null
    let keepLockedForPolling = false
    try {
      const nextJob = await requestCurrentFrameCompression(
        client,
        selectedTimestamp,
        controller.signal,
      )
      if (controller.signal.aborted) return
      activeJobId.current = nextJob.id
      keepLockedForPolling = shouldPollCodecJob(nextJob)
      setJob(nextJob)
      setError(getCodecJobFailure(nextJob))
    } catch (caught) {
      if (!controller.signal.aborted) {
        setError(toMessage(caught, 'Не удалось запустить сжатие текущего кадра.'))
      }
    } finally {
      if (!keepLockedForPolling) runGuard.current.release()
      if (requestController.current === controller) {
        requestController.current = null
      }
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }, [client, job, selectedTimestamp, service, submitting])

  return {
    service,
    serviceLoading,
    serviceError,
    retryService: () => setStatusRevision((value) => value + 1),
    job,
    jobTimestamp,
    error,
    processing: submitting || shouldPollCodecJob(job),
    compressCurrentFrame,
  }
}

export function requestCurrentFrameCompression(
  client: Pick<CodecClient, 'createEra5Job'>,
  timestamp: string,
  signal?: AbortSignal,
) {
  return client.createEra5Job(timestamp, signal)
}

export function createCompressionRunGuard() {
  let locked = false
  return {
    acquire() {
      if (locked) return false
      locked = true
      return true
    },
    release() {
      locked = false
    },
  }
}

function toMessage(error: unknown, fallback: string) {
  if (error instanceof DOMException && error.name === 'AbortError') return fallback
  return error instanceof Error && error.message.trim() ? error.message : fallback
}
