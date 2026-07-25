import { Expand } from 'lucide-react'
import { useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { resolveDataAssetPath, toFiniteNumber } from '../data/resultsSelectors'
import type { Reconstruction, ReconstructionStats } from '../types'
import { EmptyState } from './EmptyState'
import { FilterBar, SegmentedControl, SelectField } from './common/Controls'

type ViewMode = 'panels' | 'compare'

export function ReconstructionViewer({
  reconstructions,
  grid,
  compressionRatio,
}: {
  reconstructions: Reconstruction[]
  grid?: string
  compressionRatio?: number
}) {
  const [searchParams, setSearchParams] = useSearchParams()
  const fullscreenRef = useRef<HTMLDivElement>(null)
  const [slide, setSlide] = useState(50)
  const channels = Array.from(new Set(reconstructions.map((item) => item.channel)))
  const requestedChannel = searchParams.get('channel')
  const channel = requestedChannel && channels.includes(requestedChannel) ? requestedChannel : (channels[0] ?? '')
  const channelEntries = reconstructions.filter((item) => item.channel === channel)
  const compressionOptions = Array.from(
    new Set(
      channelEntries
        .map((item) => toFiniteNumber(item.compression_ratio) ?? compressionRatio)
        .filter((value): value is number => value !== undefined),
    ),
  )
  const requestedCompression = toFiniteNumber(searchParams.get('compression') ?? undefined)
  const compression =
    requestedCompression !== undefined && compressionOptions.includes(requestedCompression)
      ? requestedCompression
      : compressionOptions[0]
  const gridOptions = Array.from(
    new Set(
      channelEntries
        .map((item) => item.grid ?? grid)
        .filter((value): value is string => Boolean(value)),
    ),
  )
  const requestedGrid = searchParams.get('grid')
  const selectedGrid =
    requestedGrid && gridOptions.includes(requestedGrid) ? requestedGrid : gridOptions[0]
  const levelOptions = Array.from(
    new Set(
      channelEntries.map((item) => {
        const level = toFiniteNumber(item.level_hpa) ?? parseLevel(item.channel)
        return level === undefined ? 'surface' : String(level)
      }),
    ),
  )
  const requestedLevel = searchParams.get('level')
  const level =
    requestedLevel && levelOptions.includes(requestedLevel) ? requestedLevel : levelOptions[0]
  const filteredEntries = channelEntries.filter((item) => {
    const itemCompression = toFiniteNumber(item.compression_ratio) ?? compressionRatio
    const itemGrid = item.grid ?? grid
    const itemLevel = toFiniteNumber(item.level_hpa) ?? parseLevel(item.channel)
    const normalizedLevel = itemLevel === undefined ? 'surface' : String(itemLevel)

    if (compression !== undefined && itemCompression !== compression) return false
    if (selectedGrid && itemGrid !== selectedGrid) return false
    if (level && normalizedLevel !== level) return false
    return true
  })
  const timestamps = filteredEntries.map((item) => item.timestamp)
  const requestedTimestamp = searchParams.get('timestamp')
  const timestamp =
    requestedTimestamp && timestamps.includes(requestedTimestamp)
      ? requestedTimestamp
      : (timestamps[0] ?? '')
  const requestedView = searchParams.get('view')
  const view: ViewMode = requestedView === 'compare' ? 'compare' : 'panels'
  const entry = filteredEntries.find((item) => item.timestamp === timestamp)

  const updateParam = (key: string, value: string, defaultValue?: string) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (!value || value === defaultValue) next.delete(key)
        else next.set(key, value)
        return next
      },
      { replace: true },
    )
  }

  if (!entry) {
    return (
      <EmptyState
        title="Реконструкции отсутствуют"
        message="В results.json нет записей reconstructions с каналом, временем и путями к изображениям."
      />
    )
  }

  const originalImage = resolveDataAssetPath(entry.original_image)
  const reconstructedImage = resolveDataAssetPath(entry.reconstructed_image)
  const diffImage = resolveDataAssetPath(entry.diff_image)
  const relativeErrorImage = resolveDataAssetPath(entry.relative_error_image)
  const levelHpa = toFiniteNumber(entry.level_hpa) ?? parseLevel(entry.channel)

  const openFullscreen = () => {
    void fullscreenRef.current?.requestFullscreen?.()
  }

  return (
    <div className="space-y-4">
      <FilterBar>
        <SelectField
          label="Канал"
          value={channel}
          onChange={(value) => {
            const nextTimestamp = reconstructions.find((item) => item.channel === value)?.timestamp
            setSearchParams(
              (current) => {
                const next = new URLSearchParams(current)
                if (value === channels[0]) next.delete('channel')
                else next.set('channel', value)
                if (nextTimestamp) next.set('timestamp', nextTimestamp)
                else next.delete('timestamp')
                return next
              },
              { replace: true },
            )
          }}
          options={channels.map((value) => ({ value, label: value }))}
        />
        <SelectField
          label="Дата и время"
          value={timestamp}
          onChange={(value) => updateParam('timestamp', value, timestamps[0])}
          options={timestamps.map((value) => ({
            value,
            label: formatTimestamp(value),
          }))}
        />
        <SelectField
          label="Уровень"
          value={level ?? (levelHpa ? String(levelHpa) : 'surface')}
          onChange={(value) => updateParam('level', value, levelOptions[0])}
          disabled={levelOptions.length <= 1}
          options={levelOptions.map((value) => ({
            value,
            label: value === 'surface' ? 'Приземный канал' : `${value} hPa`,
          }))}
        />
        <SelectField
          label="Сетка"
          value={selectedGrid ?? 'unknown'}
          onChange={(value) => updateParam('grid', value, gridOptions[0])}
          disabled={gridOptions.length <= 1}
          options={
            gridOptions.length
              ? gridOptions.map((value) => ({ value, label: value }))
              : [{ value: 'unknown', label: 'Нет данных' }]
          }
        />
        <SelectField
          label="Сжатие"
          value={String(compression ?? 'unknown')}
          onChange={(value) => updateParam('compression', value, String(compressionOptions[0]))}
          disabled={compressionOptions.length <= 1}
          options={
            compressionOptions.length
              ? compressionOptions.map((value) => ({ value: String(value), label: `${value}×` }))
              : [{ value: 'unknown', label: 'Нет данных' }]
          }
        />
        <SelectField
          label="Цветовая шкала"
          value="artifact"
          onChange={() => undefined}
          disabled
          options={[{ value: 'artifact', label: 'Задана артефактом' }]}
        />
        <SegmentedControl
          label="Режим"
          value={view}
          onChange={(value) => updateParam('view', value, 'panels')}
          options={[
            { value: 'panels', label: 'Три поля' },
            { value: 'compare', label: 'До / после' },
          ]}
        />
      </FilterBar>

      <div ref={fullscreenRef} className="rounded-2xl border border-slate-200 bg-white p-4 fullscreen:overflow-auto">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-[16px] font-semibold text-slate-900">
              {entry.channel} · {formatTimestamp(entry.timestamp)}
            </h2>
            <p className="mt-1 text-[12px] text-slate-500">
              Изображения загружены из путей, указанных в results.json.
            </p>
          </div>
          <button
            type="button"
            onClick={openFullscreen}
            className="ui-button-ghost ui-focus-ring inline-flex h-9 items-center gap-2 rounded-lg px-3 text-[12px] font-semibold"
          >
            <Expand className="h-4 w-4" aria-hidden="true" />
            Полный экран
          </button>
        </div>

        {view === 'panels' ? (
          <div className="grid gap-3 lg:grid-cols-3">
            <ImagePanel key={`original-${originalImage}`} label="Оригинальное поле" src={originalImage} alt={`Оригинал ${entry.channel}`} />
            <ImagePanel
              key={`reconstructed-${reconstructedImage}`}
              label="Восстановленное поле"
              src={reconstructedImage}
              alt={`Реконструкция ${entry.channel}`}
            />
            <ImagePanel key={`diff-${diffImage}`} label="Абсолютная ошибка" src={diffImage} alt={`Ошибка ${entry.channel}`} />
          </div>
        ) : originalImage && reconstructedImage ? (
          <div>
            <div className="relative h-[min(58vh,520px)] overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
              <img src={originalImage} alt={`Оригинал ${entry.channel}`} className="h-full w-full object-contain" />
              <img
                src={reconstructedImage}
                alt={`Реконструкция ${entry.channel}`}
                className="absolute inset-0 h-full w-full object-contain"
                style={{ clipPath: `inset(0 ${100 - slide}% 0 0)` }}
              />
              <div
                className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_0_1px_rgba(15,23,42,0.35)]"
                style={{ left: `${slide}%` }}
              />
            </div>
            <label className="mt-4 block text-[12px] font-medium text-slate-600">
              Граница сравнения: {slide}%
              <input
                type="range"
                min="0"
                max="100"
                value={slide}
                onChange={(event) => setSlide(Number(event.target.value))}
                className="mt-2 block w-full"
                style={{ accentColor: 'var(--accent-active)' }}
              />
            </label>
          </div>
        ) : (
          <EmptyState
            title="Сравнение недоступно"
            message="Для режима «до / после» нужны пути original_image и reconstructed_image."
          />
        )}

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <StatsCard title="Оригинал" stats={entry.original_stats} unit={entry.unit} />
          <StatsCard title="Реконструкция" stats={entry.reconstructed_stats} unit={entry.unit} />
          <StatsCard title="Абсолютная ошибка" stats={entry.error_stats} unit={entry.unit} />
        </div>

        {!relativeErrorImage ? (
          <p className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-[12px] text-slate-500">
            Карта относительной ошибки отсутствует в results.json, поэтому она не подменяется вычисленной или случайной картинкой.
          </p>
        ) : (
          <div className="mt-4">
            <ImagePanel
              label="Относительная ошибка"
              src={relativeErrorImage}
              alt={`Относительная ошибка ${entry.channel}`}
            />
          </div>
        )}
      </div>
    </div>
  )
}

function ImagePanel({ label, src, alt }: { label: string; src?: string; alt: string }) {
  const [failed, setFailed] = useState(false)

  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      {src && !failed ? (
        <img
          src={src}
          alt={alt}
          onError={() => setFailed(true)}
          className="h-[300px] w-full object-contain"
        />
      ) : (
        <div className="flex h-[300px] items-center justify-center px-6 text-center text-[12px] text-slate-500">
          Изображение не указано или недоступно
        </div>
      )}
      <figcaption className="border-t border-slate-200 px-3 py-2 text-[12px] font-medium text-slate-600">
        {label}
      </figcaption>
    </figure>
  )
}

function StatsCard({
  title,
  stats,
  unit,
}: {
  title: string
  stats?: ReconstructionStats
  unit?: string
}) {
  const values = [
    ['min', toFiniteNumber(stats?.min)],
    ['max', toFiniteNumber(stats?.max)],
    ['mean', toFiniteNumber(stats?.mean)],
  ] as const
  const hasValues = values.some(([, value]) => value !== undefined)

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <h3 className="text-[12px] font-semibold text-slate-700">{title}</h3>
      {hasValues ? (
        <dl className="mt-2 grid grid-cols-3 gap-2">
          {values.map(([label, value]) => (
            <div key={label}>
              <dt className="text-[10px] uppercase text-slate-400">{label}</dt>
              <dd className="mt-1 text-[12px] font-semibold text-slate-800">
                {value === undefined ? '—' : `${value.toFixed(3)}${unit ? ` ${unit}` : ''}`}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="mt-2 text-[11px] text-slate-400">Статистика отсутствует</p>
      )}
    </div>
  )
}

function parseLevel(channel: string) {
  const match = /^[TUVZQ](\d{3,4})$/i.exec(channel)
  return match ? Number(match[1]) : undefined
}

function formatTimestamp(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('ru-RU', {
        dateStyle: 'medium',
        timeStyle: 'short',
        timeZone: 'UTC',
      }).format(date)
}
