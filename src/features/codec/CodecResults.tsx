import {
  Download,
  FileArchive,
  Gauge,
  Layers3,
  ShieldCheck,
} from 'lucide-react'
import type { CodecJob } from './types'

export function CodecProgress({ job }: { job: CodecJob }) {
  const percent = Math.round(job.progress * 100)
  return (
    <div className="rounded-[18px] border border-[var(--accent-border)] bg-[var(--accent-soft)] p-4" aria-live="polite">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-[13px] font-bold text-[var(--accent-text)]">
            {job.status === 'queued' ? 'Задача в очереди' : 'Модель обрабатывает поле'}
          </p>
          <p className="mt-1 text-[12px] text-[var(--text-muted)]">
            {job.message ?? 'Кодирование, квантование и восстановление…'}
          </p>
        </div>
        <span className="text-[20px] font-black text-[var(--accent)]">{percent}%</span>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/70">
        <div
          className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}

export function CodecResults({
  job,
  sourceDescription,
  resolveDownload,
}: {
  job: CodecJob
  sourceDescription: string
  resolveDownload: (path: string) => string
}) {
  if (!job.metrics || !job.downloads || !job.previews) return null
  const { downloads, metrics, previews } = job

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <ResultMetric
          icon={FileArchive}
          label="Serialized compression"
          value={`${metrics.serializedCompressionRatio.toFixed(1)}×`}
          caption="По полному bitstream в байтах"
        />
        <ResultMetric
          icon={Layers3}
          label="Tensor ratio"
          value={
            metrics.tensorCompressionRatio == null
              ? 'Нет данных'
              : `${metrics.tensorCompressionRatio.toFixed(1)}×`
          }
          caption="Отдельно от файлового сжатия"
        />
        <ResultMetric
          icon={Gauge}
          label="Bitstream"
          value={formatBytes(metrics.bitstreamBytes)}
          caption="Включая headers и side information"
        />
        <ResultMetric
          icon={ShieldCheck}
          label="Exact roundtrip"
          value={metrics.exactRoundtrip ? 'Подтверждён' : 'Не пройден'}
          caption="Равенство квантованных символов"
        />
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <ArtifactPanel
          eyebrow="Исходное состояние"
          title="Исходное поле ERA5"
          caption={sourceDescription}
          previewSrc={resolveDownload(previews.original)}
          previewAlt="Исходное поле ERA5"
        />
        <ArtifactPanel
          eyebrow="Восстановленное состояние"
          title="Восстановленное поле ERA5"
          caption={`Encode ${formatSeconds(metrics.encodeSeconds)} · Decode ${formatSeconds(metrics.decodeSeconds)}`}
          previewSrc={resolveDownload(previews.reconstruction)}
          previewAlt="Восстановленное поле ERA5"
          actions={
            <div className="flex flex-wrap gap-2">
              <DownloadLink href={resolveDownload(downloads.bitstream)}>
                Bitstream
              </DownloadLink>
              <DownloadLink href={resolveDownload(downloads.reconstruction)}>
                Реконструкция
              </DownloadLink>
            </div>
          }
        />
      </div>
    </div>
  )
}

function ResultMetric({
  icon: Icon,
  label,
  value,
  caption,
}: {
  icon: typeof Gauge
  label: string
  value: string
  caption: string
}) {
  return (
    <div className="ui-panel p-4">
      <span className="ui-accent-icon inline-flex h-9 w-9 items-center justify-center rounded-xl">
        <Icon className="h-4 w-4" aria-hidden="true" />
      </span>
      <p className="mt-4 text-[11px] font-bold uppercase tracking-[0.12em] text-[var(--text-muted)]">
        {label}
      </p>
      <p className="mt-1 text-[24px] font-black tracking-[-0.03em] text-[var(--text-primary)]">
        {value}
      </p>
      <p className="mt-1 text-[11px] leading-5 text-[var(--text-muted)]">{caption}</p>
    </div>
  )
}

function ArtifactPanel({
  eyebrow,
  title,
  caption,
  previewSrc,
  previewAlt,
  actions,
}: {
  eyebrow: string
  title: string
  caption: string
  previewSrc: string
  previewAlt: string
  actions?: React.ReactNode
}) {
  return (
    <div className="ui-panel overflow-hidden">
      <div className="aspect-[2/1] overflow-hidden border-b border-[var(--border)] bg-[var(--background-subtle)]">
        <img
          src={previewSrc}
          alt={previewAlt}
          loading="lazy"
          decoding="async"
          className="h-full w-full object-cover"
        />
      </div>
      <div className="flex min-h-[124px] flex-col justify-between gap-4 p-4">
        <div className="min-w-0">
          <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-[var(--text-muted)]">
            {eyebrow}
          </p>
          <p className="mt-1 truncate text-[15px] font-bold text-[var(--text-primary)]">{title}</p>
          <p className="mt-1 text-[11px] leading-5 text-[var(--text-muted)]">{caption}</p>
        </div>
        {actions}
      </div>
    </div>
  )
}

function DownloadLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a className="ui-action ui-action-secondary" href={href} download>
      <Download className="h-4 w-4" aria-hidden="true" />
      {children}
    </a>
  )
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} Б`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} КиБ`
  return `${(value / 1024 ** 2).toFixed(1)} МиБ`
}

function formatSeconds(value: number | null) {
  return value == null ? '—' : `${value.toFixed(2)} с`
}
