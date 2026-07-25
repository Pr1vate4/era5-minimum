import {
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
    <div className="border-b border-[#E4E7EC] bg-[#F8FAFC] px-5 py-4 lg:px-6 lg:py-5">
      {!isEarthMode && availableModes.length > 1 ? (
        <fieldset className="mb-4">
          <legend className="mb-2 text-[12px] font-semibold leading-4 text-[#344054]">
            Режим слоя
          </legend>
          <div className="inline-grid min-h-11 max-w-full grid-flow-col rounded-xl border border-[#D0D5DD] bg-white p-1 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
            {availableModes.map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => selection.setMode(mode)}
                aria-pressed={selection.mode === mode}
                className={`min-h-9 rounded-lg px-3 text-[13px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-100 sm:px-4 ${
                  selection.mode === mode
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-[#667085] hover:bg-[#F2F4F7] hover:text-[#344054]'
                }`}
              >
                {modeLabels[mode]}
              </button>
            ))}
          </div>
        </fieldset>
      ) : null}

      {showResearchFilters ? (
        <div className="mb-5 grid grid-cols-[minmax(0,1fr)] gap-4 border-b border-[#E4E7EC] pb-5 md:grid-cols-2 xl:grid-cols-4">
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
          <span className="text-[12px] font-semibold leading-4 text-[#344054]">Параметр</span>
          <span className="relative">
            <SelectedIcon
              className="pointer-events-none absolute left-3.5 top-1/2 h-[18px] w-[18px] -translate-y-1/2 text-blue-600"
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
              className={`${selectClassName} w-full pl-10`}
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
          <legend className="text-[12px] font-semibold leading-4 text-[#344054]">
            {isEarthMode ? 'Сетка облачности' : 'Сетка'}
          </legend>
          <div className="grid h-11 grid-cols-2 rounded-xl border border-[#D0D5DD] bg-white p-1 shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
            {selection.gridOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                disabled={!option.available}
                title={option.available ? undefined : 'Данные не подготовлены'}
                onClick={() => selection.setGrid(option.value)}
                aria-pressed={option.available && selection.grid === option.value}
                className={`rounded-lg px-2.5 text-[14px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-100 ${
                  option.available && selection.grid === option.value
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-[#667085] hover:bg-[#F2F4F7] hover:text-[#344054] disabled:cursor-not-allowed disabled:opacity-40'
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
          <span className="text-[12px] font-semibold leading-4 text-[#344054]">Автовращение</span>
          <button
            type="button"
            role="switch"
            aria-checked={autoRotate}
            aria-label="Медленно вращать глобус автоматически"
            title="Медленно вращать глобус автоматически"
            onClick={() => onAutoRotateChange(!autoRotate)}
            className={`flex h-11 w-full items-center justify-between rounded-xl border px-3.5 text-[14px] font-semibold shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-100 ${
              autoRotate
                ? 'border-blue-600 bg-blue-600 text-white shadow-[0_2px_6px_rgba(37,99,235,0.22)]'
                : 'border-[#D0D5DD] bg-white text-[#475467] hover:border-[#98A2B3] hover:bg-[#F9FAFB]'
            }`}
          >
            <span>{autoRotate ? 'Включено' : 'Выключено'}</span>
            <span
              className={`relative inline-flex h-[22px] w-10 shrink-0 rounded-full transition-colors ${
                autoRotate ? 'bg-white/25' : 'bg-[#D0D5DD]'
              }`}
              aria-hidden="true"
            >
              <span
                className={`absolute top-0.5 h-[18px] w-[18px] rounded-full shadow-sm transition-transform ${
                  autoRotate
                    ? 'translate-x-5 bg-white'
                    : 'translate-x-0.5 bg-[#98A2B3]'
                }`}
              />
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
  'h-11 min-w-0 rounded-xl border border-[#D0D5DD] bg-white px-3.5 text-[14px] font-medium text-[#101828] shadow-[0_1px_2px_rgba(16,24,40,0.04)] outline-none transition hover:border-[#98A2B3] focus:border-blue-500 focus:ring-4 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-[#F2F4F7] disabled:text-[#98A2B3]'

function SelectControl({ label, value, options, onChange }: SelectControlProps) {
  return (
    <label className="grid min-w-0 gap-2">
      <span className="text-[12px] font-semibold leading-4 text-[#344054]">{label}</span>
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
