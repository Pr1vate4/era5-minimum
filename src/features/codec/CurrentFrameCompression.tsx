import {
  AlertTriangle,
  FileArchive,
  LoaderCircle,
  RefreshCw,
} from 'lucide-react'
import { buildApiUrl } from '../../app/settings'
import { CodecResults } from './CodecResults'
import type { CodecJob, CodecServiceStatus } from './types'

export function FrameCompressionButton({
  timestamp,
  frameReady,
  serviceLoading,
  serviceReady,
  processing,
  onCompress,
}: {
  timestamp: string | undefined
  frameReady: boolean
  serviceLoading: boolean
  serviceReady: boolean
  processing: boolean
  onCompress: () => void | Promise<void>
}) {
  const disabled =
    !timestamp ||
    !frameReady ||
    serviceLoading ||
    !serviceReady ||
    processing

  return (
    <button
      type="button"
      disabled={disabled}
      aria-busy={processing}
      title={getButtonTitle({
        timestamp,
        frameReady,
        serviceLoading,
        serviceReady,
        processing,
      })}
      onClick={() => void onCompress()}
      className="ui-button-subtle ui-focus-ring inline-flex h-10 min-w-0 items-center justify-center gap-2 rounded-xl border px-3 text-[12px] font-bold disabled:cursor-not-allowed disabled:opacity-50"
    >
      {processing || serviceLoading ? (
        <LoaderCircle className="h-4 w-4 shrink-0 animate-spin" aria-hidden="true" />
      ) : (
        <FileArchive className="h-4 w-4 shrink-0" aria-hidden="true" />
      )}
      <span>
        {processing
          ? 'Сжатие кадра…'
          : serviceLoading
            ? 'Проверка N32…'
            : 'Сжать текущий кадр'}
      </span>
    </button>
  )
}

export function FrameCompressionNotice({
  processing,
  job,
  error,
  service,
  serviceError,
  onRetryService,
}: {
  processing: boolean
  job: CodecJob | null
  error: string | null
  service: CodecServiceStatus | null
  serviceError: string | null
  onRetryService: () => void
}) {
  if (processing) {
    return (
      <div
        className="mx-3 mt-3 flex items-center gap-2.5 rounded-xl border border-[var(--accent-border)] bg-[var(--accent-soft)] px-3.5 py-2.5 text-[12px] font-semibold text-[var(--accent-text)] sm:mx-5 lg:mx-6"
        role="status"
        aria-live="polite"
      >
        <LoaderCircle className="h-4 w-4 shrink-0 animate-spin" aria-hidden="true" />
        <span>{job?.message ?? 'Подготовка полного ERA5-кадра и сжатие N32…'}</span>
      </div>
    )
  }

  const serviceUnavailable = !service?.ready
  const message =
    error ??
    serviceError ??
    (serviceUnavailable && service ? service.message : null)
  if (!message) return null

  const canRetryService = !error && serviceUnavailable
  return (
    <div
      className="mx-3 mt-3 flex flex-wrap items-center gap-2.5 rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-2.5 text-[12px] text-rose-800 sm:mx-5 lg:mx-6"
      role="alert"
    >
      <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
      <span className="min-w-0 flex-1 font-semibold">{message}</span>
      {canRetryService ? (
        <button
          type="button"
          onClick={onRetryService}
          className="ui-focus-ring inline-flex items-center gap-1.5 rounded-lg px-2 py-1 font-bold hover:bg-rose-100"
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          Повторить проверку
        </button>
      ) : null}
    </div>
  )
}

export function FrameCompressionReport({
  job,
  jobTimestamp,
  codecBaseUrl,
}: {
  job: CodecJob
  jobTimestamp: string | null
  codecBaseUrl: string
}) {
  if (job.status !== 'completed') return null

  const sourceTimestamp = job.source?.timestamp ?? jobTimestamp
  const sourceDetails = [
    'ERA5 Zarr',
    sourceTimestamp ? formatUtcTimestamp(sourceTimestamp) : 'timestamp не указан',
    job.source?.datasetId,
    'N32',
  ].filter(Boolean)

  return (
    <div
      className="border-t border-[var(--border)] bg-[var(--background-subtle)] px-3 py-5 sm:px-5 lg:px-6"
      data-testid="frame-compression-report"
    >
      <div className="mb-4 flex flex-col gap-1">
        <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-[var(--accent-text)]">
          Результат сжатия N32
        </p>
        <h3 className="text-[17px] font-bold text-[var(--text-primary)]">
          Кадр {sourceTimestamp ? formatUtcTimestamp(sourceTimestamp) : 'без timestamp'}
        </h3>
        <p className="text-[12px] leading-5 text-[var(--text-muted)]">
          Восстановлен полный канонический ERA5-кадр float32 [1, 28, 360, 720].
          Его можно скачать как NPZ. Глобус остаётся на исходном слое, потому что
          текущий renderer принимает подготовленную текстуру одного параметра, а
          не полный 28-канальный tensor.
        </p>
      </div>

      <CodecResults
        job={job}
        sourceDescription={sourceDetails.join(' · ')}
        resolveDownload={(path) => resolveArtifactUrl(codecBaseUrl, path)}
      />
    </div>
  )
}

function getButtonTitle({
  timestamp,
  frameReady,
  serviceLoading,
  serviceReady,
  processing,
}: {
  timestamp: string | undefined
  frameReady: boolean
  serviceLoading: boolean
  serviceReady: boolean
  processing: boolean
}) {
  if (processing) return 'Сжатие выбранного ERA5-кадра уже выполняется'
  if (!timestamp) return 'Сначала выберите доступную дату и время'
  if (!frameReady) return 'Дождитесь загрузки выбранного ERA5-кадра'
  if (serviceLoading) return 'Проверяем доступность модели N32'
  if (!serviceReady) return 'Сервис сжатия N32 недоступен'
  return `Сжать ERA5-кадр ${formatUtcTimestamp(timestamp)} моделью N32`
}

function resolveArtifactUrl(baseUrl: string, path: string) {
  return /^(https?:)?\/\//i.test(path) ? path : buildApiUrl(baseUrl, path)
}

function formatUtcTimestamp(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'UTC',
    timeZoneName: 'short',
  }).format(date)
}
