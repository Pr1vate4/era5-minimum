import { useEffect, useState } from 'react'
import { getDisplayUnit, getGlobeParameterConfig, toDisplayValue } from '../data/globeParameterConfig'
import type {
  GlobeDisplayMode,
  GlobeFrameAsset,
  GlobeGridPoint,
} from '../types/globe'
import { GlobePointInspector } from './GlobePointInspector'

type InspectorTab = 'parameter' | 'point'

type GlobeParameterPanelProps = {
  displayMode: GlobeDisplayMode
  frame: GlobeFrameAsset | undefined
  cloudFrame: GlobeFrameAsset | undefined
  cloudLoading: boolean
  cloudError: string | null
  onCloudRetry?: () => void
  point: GlobeGridPoint | null
  valuesLoading: boolean
  valuesError: string | null
}

export function GlobeParameterPanel({
  displayMode,
  frame,
  cloudFrame,
  cloudLoading,
  cloudError,
  onCloudRetry,
  point,
  valuesLoading,
  valuesError,
}: GlobeParameterPanelProps) {
  const [tab, setTab] = useState<InspectorTab>('parameter')
  const configuration = getGlobeParameterConfig(frame?.channel ?? 't2m')

  useEffect(() => {
    if (displayMode === 'earth') setTab('parameter')
    else if (point) setTab('point')
  }, [displayMode, point])

  if (displayMode === 'earth') {
    return (
      <aside className="min-w-0 max-w-full overflow-hidden rounded-2xl border border-[#E4E7EC] bg-white p-5 shadow-[0_1px_2px_rgba(16,24,40,0.03)] xl:min-h-[580px]">
        <p className="text-[12px] font-semibold uppercase tracking-[0.04em] text-[#667085]">
          Режим просмотра
        </p>
        <h3 className="mt-3 text-[19px] font-semibold leading-6 tracking-[-0.01em] text-[#101828]">
          Обычная Земля с облачностью
        </h3>
        <p className="mt-3 text-[14px] leading-6 text-[#667085]">
          Естественная поверхность планеты. Белый облачный слой построен из общей
          облачности TCC для выбранного момента времени.
        </p>

        <dl className="mt-5">
          <InfoRow label="Источник поверхности" value="Blue Marble" />
          <InfoRow
            label="Источник облачности"
            value={cloudFrame?.source === 'demo' ? 'Демо TCC' : 'ERA5 TCC'}
          />
          <InfoRow
            label="Дата и время"
            value={cloudFrame ? formatTimestamp(cloudFrame.timestamp) : 'Нет данных'}
          />
          <InfoRow
            label="Сетка облачности"
            value={cloudFrame ? formatGrid(cloudFrame.grid) : 'Нет данных'}
          />
          <InfoRow label="Научный параметр" value="Не выбран" />
        </dl>

        {cloudLoading ? (
          <p
            className="mt-5 rounded-xl border border-blue-200 bg-blue-50 px-3.5 py-3 text-[12px] leading-5 text-blue-800"
            aria-live="polite"
          >
            Загрузка облачности для выбранного времени…
          </p>
        ) : null}
        {cloudError ? (
          <div
            className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-3 text-[12px] leading-5 text-amber-900"
            role="status"
            aria-live="polite"
          >
            <p>Облачность временно недоступна: {cloudError}</p>
            {onCloudRetry ? (
              <button
                type="button"
                onClick={onCloudRetry}
                className="mt-2 font-semibold text-amber-950 underline decoration-amber-400 underline-offset-2"
              >
                Повторить
              </button>
            ) : null}
          </div>
        ) : null}

        <p className="mt-5 rounded-xl border border-[#E4E7EC] bg-[#F8FAFC] px-3.5 py-3 text-[12px] leading-5 text-[#475467]">
          Облачный слой показывает степень покрытия облаками по данным TCC. Это не
          спутниковая фотография и не прогноз в реальном времени.
        </p>
      </aside>
    )
  }

  return (
    <aside className="min-w-0 max-w-full overflow-hidden rounded-2xl border border-[#E4E7EC] bg-white p-5 shadow-[0_1px_2px_rgba(16,24,40,0.03)] xl:min-h-[580px]">
      <div
        className="flex rounded-xl border border-[#E4E7EC] bg-[#F2F4F7] p-1"
        role="tablist"
        aria-label="Инспектор глобуса"
      >
        <TabButton active={tab === 'parameter'} onClick={() => setTab('parameter')}>
          О параметре
        </TabButton>
        <TabButton active={tab === 'point'} onClick={() => setTab('point')}>
          Значение в точке
        </TabButton>
      </div>

      <div className="mt-5">
        {tab === 'parameter' ? (
          <div>
            <p className="text-[12px] font-semibold uppercase tracking-[0.04em] text-[#667085]">
              Выбранный параметр
            </p>
            <div className="mt-3 flex items-start justify-between gap-3">
              <h3 className="text-[19px] font-semibold leading-6 tracking-[-0.01em] text-[#101828]">
                {configuration.shortName}
                {frame?.level ? ` · ${frame.level} hPa` : ''}
              </h3>
              <span className="inline-flex shrink-0 items-center rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-[12px] font-semibold text-blue-700">
                {frame?.channel ?? configuration.channel}
              </span>
            </div>
            <p className="mt-3 text-[14px] font-normal leading-6 text-[#667085]">
              {frame?.mode === 'absolute-error'
                ? 'Абсолютная разница между ERA5 и восстановлением модели.'
                : configuration.description}
            </p>

            <dl className="mt-5">
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
              <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-3 text-[12px] leading-5 text-amber-800">
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
      className={`min-h-10 min-w-0 flex-1 rounded-lg px-2.5 text-[13px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-100 ${
        active
          ? 'bg-white text-blue-700 shadow-sm'
          : 'text-[#667085] hover:bg-white/60 hover:text-[#344054]'
      }`}
    >
      {children}
    </button>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-h-11 items-center justify-between gap-3 border-t border-[#EAECF0] py-3 first:border-t-0">
      <dt className="text-[13px] font-medium text-[#667085]">{label}</dt>
      <dd className="text-right text-[15px] font-semibold tabular-nums text-[#101828]">{value}</dd>
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

function formatGrid(grid: GlobeFrameAsset['grid']) {
  return grid === '0p25' ? '0.25°' : '0.5°'
}
