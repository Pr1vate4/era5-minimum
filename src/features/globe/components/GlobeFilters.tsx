import {
  ChevronDown,
  CloudRain,
  Droplets,
  Gauge,
  Globe2,
  Thermometer,
  Waves,
  Wind,
} from 'lucide-react'
import { isPressureChannel } from '../data/globeParameterConfig'
import type { GlobeSelection } from '../hooks/useGlobeSelection'
import {
  NO_PARAMETER_VALUE,
  type GlobeChannel,
  type GlobeMode,
} from '../types/globe'

type GlobeFiltersProps = {
  selection: GlobeSelection
  availableModes: GlobeMode[]
  autoRotate: boolean
  onAutoRotateChange: (value: boolean) => void
}

const modeLabels: Record<GlobeMode, string> = {
  original: 'Оригинал',
  reconstruction: 'Реконструкция',
  'absolute-error': 'Абсолютная ошибка',
}

export function GlobeFilters({
  selection,
  availableModes,
  autoRotate,
  onAutoRotateChange,
}: GlobeFiltersProps) {
  const isEarthMode = selection.displayMode === 'earth'
  const SelectedIcon = isEarthMode ? Globe2 : getParameterIcon(selection.channel)
  const surfaceOptions = selection.channelOptions.filter((option) => option.group === 'surface')
  const pressureOptions = selection.channelOptions.filter((option) => option.group === 'pressure')
  const showResearchFilters = !isEarthMode && selection.mode !== 'original'
  const showLevelFilter = !isEarthMode && isPressureChannel(selection.channel)
  const primaryGridClass = showLevelFilter
    ? 'xl:grid-cols-[minmax(240px,1.4fr)_minmax(220px,1.1fr)_160px_160px_180px]'
    : 'xl:grid-cols-[minmax(240px,1.4fr)_minmax(220px,1.1fr)_160px_180px]'

  return (
    <div className="ui-filter-surface border-b px-5 py-4 lg:px-6 lg:py-5">
      {!isEarthMode && availableModes.length > 1 ? (
        <fieldset className="mb-4">
          <legend className="mb-2 text-[12px] font-semibold leading-4 text-[var(--text-secondary)]">
            Режим слоя
          </legend>
          <div className="inline-grid min-h-11 max-w-full grid-flow-col rounded-xl border border-[var(--border)] bg-[var(--background-subtle)] p-1 shadow-[var(--shadow-soft)]">
            {availableModes.map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => selection.setMode(mode)}
                aria-pressed={selection.mode === mode}
                className={`ui-focus-ring min-h-9 rounded-lg border px-3 text-[13px] font-semibold transition-colors sm:px-4 ${
                  selection.mode === mode
                    ? 'ui-selected'
                    : 'border-transparent text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-secondary)]'
                }`}
              >
                {modeLabels[mode]}
              </button>
            ))}
          </div>
        </fieldset>
      ) : null}

      {showResearchFilters ? (
        <div className="mb-5 grid grid-cols-[minmax(0,1fr)] gap-4 border-b border-[var(--border)] pb-5 md:grid-cols-2 xl:grid-cols-4">
          <SelectControl
            label="Запуск"
            value={selection.research.runId ?? ''}
            onChange={selection.setRunId}
            options={selection.researchOptions.runIds.map((value) => ({
              value,
              label: value,
            }))}
          />
          <SelectControl
            label="Обучающие кадры"
            value={String(selection.research.trainFrames ?? '')}
            onChange={(value) => selection.setTrainFrames(Number(value))}
            options={selection.researchOptions.trainFrames.map((value) => ({
              value: String(value),
              label: value.toLocaleString('ru-RU'),
            }))}
          />
          <SelectControl
            label="Сжатие"
            value={String(selection.research.compressionRatio ?? '')}
            onChange={(value) => selection.setCompressionRatio(Number(value))}
            options={selection.researchOptions.compressionRatios.map((value) => ({
              value: String(value),
              label: `${value}×`,
            }))}
          />
          <SelectControl
            label="Checkpoint"
            value={selection.research.checkpoint ?? ''}
            onChange={selection.setCheckpoint}
            options={selection.researchOptions.checkpoints.map((value) => ({
              value,
              label: value,
            }))}
          />
        </div>
      ) : null}

      <div className={`grid grid-cols-[minmax(0,1fr)] items-end gap-4 md:grid-cols-2 ${primaryGridClass}`}>
        <label className="grid min-w-0 gap-2">
          <span className="text-[12px] font-semibold leading-4 text-[var(--text-secondary)]">Параметр</span>
          <span className="relative">
            <SelectedIcon
              className="pointer-events-none absolute left-3.5 top-1/2 h-[18px] w-[18px] -translate-y-1/2 text-[var(--accent)]"
              aria-hidden="true"
            />
            <select
              value={isEarthMode ? NO_PARAMETER_VALUE : selection.channel}
              onChange={(event) => {
                const value = event.target.value
                if (value === NO_PARAMETER_VALUE) {
                  selection.setDisplayMode('earth')
                  return
                }
                selection.setChannel(value as GlobeChannel)
              }}
              aria-label="Параметр отображения глобуса"
              className={`${selectClassName} w-full pl-10 pr-10`}
            >
              <option value={NO_PARAMETER_VALUE}>
                Без параметров — обычная Земля
              </option>
              <optgroup label="Приземные поля">
                {surfaceOptions.map((option) => (
                  <option key={option.channel} value={option.channel} disabled={!option.available}>
                    {option.shortName} · {option.channel}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Атмосферные поля">
                {pressureOptions.map((option) => (
                  <option key={option.channel} value={option.channel} disabled={!option.available}>
                    {option.shortName} · {option.channel}
                  </option>
                ))}
              </optgroup>
            </select>
            <ChevronDown
              className="pointer-events-none absolute right-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-muted)]"
              aria-hidden="true"
            />
          </span>
        </label>

        <SelectControl
          label={isEarthMode ? 'Дата и время облачности' : 'Дата и время'}
          value={selection.timestamp}
          onChange={selection.setTimestamp}
          options={selection.timestampOptions.map((timestamp) => ({
            value: timestamp,
            label: formatTimestamp(timestamp),
          }))}
        />

        <fieldset
          className="grid gap-2"
          title={
            isEarthMode
              ? 'Сетка используется для отображения облачности TCC.'
              : undefined
          }
        >
          <legend className="text-[12px] font-semibold leading-4 text-[var(--text-secondary)]">
            {isEarthMode ? 'Сетка облачности' : 'Сетка'}
          </legend>
          <div className="grid h-11 grid-cols-2 rounded-xl border border-[var(--border)] bg-[var(--background-subtle)] p-1 shadow-[var(--shadow-soft)]">
            {selection.gridOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                disabled={!option.available}
                title={option.available ? undefined : 'Данные не подготовлены'}
                onClick={() => selection.setGrid(option.value)}
                aria-pressed={option.available && selection.grid === option.value}
                className={`ui-focus-ring rounded-lg border px-2.5 text-[14px] font-semibold transition-colors ${
                  option.available && selection.grid === option.value
                    ? 'ui-selected shadow-[0_2px_8px_rgba(14,165,233,0.08)]'
                    : 'border-transparent text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-secondary)] disabled:cursor-not-allowed disabled:opacity-40'
                }`}
              >
                {option.value === '0p25' ? '0.25°' : '0.5°'}
              </button>
            ))}
          </div>
        </fieldset>

        {showLevelFilter ? (
          <SelectControl
            label="Уровень"
            value={String(selection.level ?? '')}
            onChange={(value) => selection.setLevel(Number(value) as 1000 | 925 | 850 | 700)}
            options={selection.levelOptions.map((level) => ({
              value: String(level),
              label: `${level} hPa`,
            }))}
          />
        ) : null}

        <div className="grid min-w-0 gap-2">
          <span className="text-[12px] font-semibold leading-4 text-[var(--text-secondary)]">Автовращение</span>
          <button
            type="button"
            role="switch"
            aria-checked={autoRotate}
            aria-label="Медленно вращать глобус автоматически"
            title="Медленно вращать глобус автоматически"
            onClick={() => onAutoRotateChange(!autoRotate)}
            className={`ui-switch ui-focus-ring flex h-11 w-full items-center justify-between rounded-xl px-3.5 text-[14px] font-semibold ${
              autoRotate ? 'ui-switch--on' : ''
            }`}
          >
            <span>{autoRotate ? 'Включено' : 'Выключено'}</span>
            <span className="ui-switch__track" aria-hidden="true">
              <span className="ui-switch__thumb" />
            </span>
          </button>
        </div>
      </div>
    </div>
  )
}

type SelectControlProps = {
  label: string
  value: string
  options: Array<{ value: string; label: string }>
  onChange: (value: string) => void
}

const selectClassName =
  'ui-field h-11 min-w-0 appearance-none rounded-xl px-3.5 pr-10 text-[14px] font-medium'

function SelectControl({ label, value, options, onChange }: SelectControlProps) {
  return (
    <label className="grid min-w-0 gap-2">
      <span className="text-[12px] font-semibold leading-4 text-[var(--text-secondary)]">{label}</span>
      <span className="relative">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={options.length <= 1}
          className={`${selectClassName} w-full`}
        >
          {options.length ? (
            options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))
        ) : (
          <option value="">Нет данных</option>
        )}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-muted)]"
          aria-hidden="true"
        />
      </span>
    </label>
  )
}

function formatTimestamp(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return `${new Intl.DateTimeFormat('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'UTC',
  }).format(date)} UTC`
}

function getParameterIcon(channel: GlobeChannel) {
  if (channel === 't2m' || channel === 'sst' || channel === 'T') return Thermometer
  if (channel === 'mslp' || channel === 'Z') return Gauge
  if (channel === 'tp6h' || channel === 'tcc') return CloudRain
  if (channel === 'tcwv' || channel === 'Q') return Droplets
  if (channel === 'wind10' || channel === 'u10' || channel === 'v10' || channel === 'U' || channel === 'V') return Wind
  return Waves
}
