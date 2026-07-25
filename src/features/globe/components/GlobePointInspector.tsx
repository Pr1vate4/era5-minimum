import { getDisplayUnit, getGlobeParameterConfig, toDisplayValue } from '../data/globeParameterConfig'
import type { GlobeFrameAsset, GlobeGridPoint } from '../types/globe'

type GlobePointInspectorProps = {
  frame: GlobeFrameAsset | undefined
  point: GlobeGridPoint | null
  valuesLoading: boolean
  valuesError: string | null
}

export function GlobePointInspector({
  frame,
  point,
  valuesLoading,
  valuesError,
}: GlobePointInspectorProps) {
  if (!point) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-[12px] leading-5 text-slate-500">
        Нажмите на глобус, чтобы получить координаты ближайшей ячейки и числовое значение.
      </div>
    )
  }

  const configuration = frame ? getGlobeParameterConfig(frame.channel) : undefined
  const displayValue =
    frame && point.value !== undefined
      ? toDisplayValue(frame.channel, point.value, frame.unit)
      : undefined
  const normalizedValue =
    frame?.normalization && point.value !== undefined
      ? (point.value - frame.normalization.mean) / frame.normalization.std
      : undefined

  return (
    <div>
      <dl className="grid gap-2.5 text-[12px]">
        <InfoRow label="Широта" value={formatCoordinate(point.latitude, 'N', 'S')} />
        <InfoRow label="Долгота" value={formatCoordinate(point.longitude, 'E', 'W')} />
        <InfoRow label="Ячейка сетки" value={`i=${point.row}, j=${point.column}`} />
        <InfoRow
          label={configuration?.shortName ?? 'Значение'}
          value={
            valuesLoading
              ? 'Загрузка…'
              : displayValue === undefined
                ? 'Нет данных'
                : `${formatNumber(displayValue)} ${frame ? getDisplayUnit(frame.channel, frame.unit) : ''}`
          }
        />
        {normalizedValue !== undefined && Number.isFinite(normalizedValue) ? (
          <InfoRow label="Нормализованное значение" value={formatNumber(normalizedValue)} />
        ) : null}
        {frame ? <InfoRow label="Время" value={formatTimestamp(frame.timestamp)} /> : null}
        {frame ? <InfoRow label="Сетка" value={frame.grid === '0p25' ? '0.25°' : '0.5°'} /> : null}
        {frame?.level ? <InfoRow label="Уровень" value={`${frame.level} hPa`} /> : null}
      </dl>

      {!frame?.valuesUrl ? (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-[11px] leading-4 text-amber-800">
          Числовой слой для этой текстуры не подготовлен.
        </p>
      ) : null}
      {valuesError ? (
        <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-[11px] leading-4 text-rose-700">
          Не удалось прочитать числовой слой: {valuesError}
        </p>
      ) : null}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 border-b border-slate-100 pb-2 last:border-0 last:pb-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-semibold text-slate-800">{value}</dd>
    </div>
  )
}

function formatCoordinate(value: number, positive: string, negative: string) {
  return `${Math.abs(value).toFixed(2)}° ${value >= 0 ? positive : negative}`
}

function formatNumber(value: number) {
  const absolute = Math.abs(value)
  if (absolute !== 0 && absolute < 0.001) return value.toExponential(3)
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 3 })
}

function formatTimestamp(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return `${new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'UTC',
  }).format(date)} UTC`
}
