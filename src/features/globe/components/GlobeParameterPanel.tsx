import { useEffect, useState } from 'react'
import { getDisplayUnit, getGlobeParameterConfig, toDisplayValue } from '../data/globeParameterConfig'
import type { GlobeFrameAsset, GlobeGridPoint } from '../types/globe'
import { GlobePointInspector } from './GlobePointInspector'

type InspectorTab = 'parameter' | 'point'

type GlobeParameterPanelProps = {
  frame: GlobeFrameAsset | undefined
  point: GlobeGridPoint | null
  valuesLoading: boolean
  valuesError: string | null
}

export function GlobeParameterPanel({
  frame,
  point,
  valuesLoading,
  valuesError,
}: GlobeParameterPanelProps) {
  const [tab, setTab] = useState<InspectorTab>('parameter')
  const configuration = getGlobeParameterConfig(frame?.channel ?? 't2m')

  useEffect(() => {
    if (point) setTab('point')
  }, [point])

  return (
    <aside className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 xl:min-h-[430px]">
      <div className="flex rounded-lg bg-slate-100 p-0.5" role="tablist" aria-label="Инспектор глобуса">
        <TabButton active={tab === 'parameter'} onClick={() => setTab('parameter')}>
          О параметре
        </TabButton>
        <TabButton active={tab === 'point'} onClick={() => setTab('point')}>
          Значение в точке
        </TabButton>
      </div>

      <div className="mt-4">
        {tab === 'parameter' ? (
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
              Выбранный параметр
            </p>
            <div className="mt-2 flex items-start justify-between gap-3">
              <h3 className="text-[16px] font-semibold leading-5 text-[#101828]">
                {configuration.shortName}
                {frame?.level ? ` · ${frame.level} hPa` : ''}
              </h3>
              <span className="shrink-0 rounded-md bg-blue-50 px-2 py-1 font-mono text-[11px] font-semibold text-blue-700">
                {frame?.channel ?? configuration.channel}
              </span>
            </div>
            <p className="mt-2 text-[12px] leading-5 text-[#667085]">
              {frame?.mode === 'absolute-error'
                ? 'Абсолютная разница между ERA5 и восстановлением модели.'
                : configuration.description}
            </p>

            <dl className="mt-4 grid gap-2.5 text-[12px]">
              <InfoRow label="Минимум" value={formatStatistic(frame, frame?.min)} />
              <InfoRow label="Максимум" value={formatStatistic(frame, frame?.max)} />
              <InfoRow label="Среднее" value={formatStatistic(frame, frame?.mean)} />
              <InfoRow
                label="Единицы"
                value={frame ? getDisplayUnit(frame.channel, frame.unit) : configuration.displayUnit}
              />
              <InfoRow
                label="Сетка"
                value={frame ? (frame.grid === '0p25' ? '0.25°' : '0.5°') : 'Нет данных'}
              />
              <InfoRow label="Время" value={frame ? formatTimestamp(frame.timestamp) : 'Нет данных'} />
              {frame?.level ? <InfoRow label="Уровень" value={`${frame.level} hPa`} /> : null}
              <InfoRow label="Источник" value={formatSource(frame)} />
            </dl>

            {frame?.source === 'demo' ? (
              <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-4 text-amber-800">
                Визуальный демо-слой. Не является данными ERA5.
              </p>
            ) : null}
          </div>
        ) : (
          <GlobePointInspector
            frame={frame}
            point={point}
            valuesLoading={valuesLoading}
            valuesError={valuesError}
          />
        )}
      </div>
    </aside>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: string
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`min-h-8 flex-1 rounded-md px-2 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
        active ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'
      }`}
    >
      {children}
    </button>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-2 last:border-0 last:pb-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-semibold text-slate-800">{value}</dd>
    </div>
  )
}

function formatStatistic(frame: GlobeFrameAsset | undefined, value: number | undefined) {
  if (!frame || value === undefined || !Number.isFinite(value)) return 'Нет данных'
  const converted = toDisplayValue(frame.channel, value, frame.unit)
  const absolute = Math.abs(converted)
  const formatted =
    absolute !== 0 && absolute < 0.001
      ? converted.toExponential(3)
      : converted.toLocaleString('ru-RU', { maximumFractionDigits: 2 })
  return `${formatted} ${getDisplayUnit(frame.channel, frame.unit)}`
}

function formatSource(frame: GlobeFrameAsset | undefined) {
  if (!frame) return 'Нет данных'
  if (frame.source === 'ERA5') return 'ERA5'
  if (frame.source === 'model') {
    return frame.mode === 'absolute-error' ? 'ERA5 − модель' : 'Модель'
  }
  return 'Демо-данные'
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
