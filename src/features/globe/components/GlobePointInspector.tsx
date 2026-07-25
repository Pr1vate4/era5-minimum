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
      <div className="rounded-xl border border-dashed border-[#D0D5DD] bg-[#F8FAFC] px-4 py-5 text-[14px] leading-6 text-[#667085]">
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
      <dl>
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

      {valuesError ? (
        <p className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-3 text-[12px] leading-5 text-rose-700">
          Не удалось прочитать числовой слой: {valuesError}
        </p>
      ) : null}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid min-h-11 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-t border-[#EAECF0] py-3 first:border-t-0">
      <dt className="text-[13px] font-medium text-[#667085]">{label}</dt>
      <dd className="text-right text-[15px] font-semibold tabular-nums text-[#101828]">{value}</dd>
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
