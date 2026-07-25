import {
  CloudRain,
  Droplets,
  Gauge,
  Thermometer,
  Waves,
  Wind,
} from 'lucide-react'
import { isPressureChannel } from '../data/globeParameterConfig'
import type { GlobeSelection } from '../hooks/useGlobeSelection'
import type { GlobeChannel, GlobeMode } from '../types/globe'

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
  const SelectedIcon = getParameterIcon(selection.channel)
  const surfaceOptions = selection.channelOptions.filter((option) => option.group === 'surface')
  const pressureOptions = selection.channelOptions.filter((option) => option.group === 'pressure')
  const showResearchFilters = selection.mode !== 'original'

  return (
    <div className="border-y border-[#E4E7EC] bg-slate-50/65 px-4 py-3 sm:px-5">
      {availableModes.length > 1 ? (
        <fieldset className="mb-3">
          <legend className="mb-1.5 text-[11px] font-semibold text-slate-500">Режим слоя</legend>
          <div className="inline-flex max-w-full rounded-lg border border-slate-300 bg-slate-100 p-0.5">
            {availableModes.map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => selection.setMode(mode)}
                aria-pressed={selection.mode === mode}
                className={`min-h-8 rounded-md px-2.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 sm:px-3 sm:text-[12px] ${
                  selection.mode === mode
                    ? 'bg-white text-blue-700 shadow-sm'
                    : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                {modeLabels[mode]}
              </button>
            ))}
          </div>
        </fieldset>
      ) : null}

      {showResearchFilters ? (
        <div className="mb-3 grid gap-2.5 border-b border-slate-200 pb-3 sm:grid-cols-2 xl:grid-cols-4">
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

      <div className="grid items-end gap-2.5 sm:grid-cols-2 lg:grid-cols-[minmax(180px,1.25fr)_minmax(190px,1fr)_auto_auto_auto]">
        <label className="grid min-w-0 gap-1">
          <span className="text-[11px] font-semibold text-slate-500">Параметр</span>
          <span className="relative">
            <SelectedIcon
              className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-blue-600"
              aria-hidden="true"
            />
            <select
              value={selection.channel}
              onChange={(event) => selection.setChannel(event.target.value as GlobeChannel)}
              className={`${selectClassName} w-full pl-8`}
            >
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
          label="Дата и время"
          value={selection.timestamp}
          onChange={selection.setTimestamp}
          options={selection.timestampOptions.map((timestamp) => ({
            value: timestamp,
            label: formatTimestamp(timestamp),
          }))}
        />

        <fieldset className="grid gap-1">
          <legend className="text-[11px] font-semibold text-slate-500">Сетка</legend>
          <div className="inline-flex h-9 rounded-lg border border-slate-300 bg-slate-100 p-0.5">
            {selection.gridOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                disabled={!option.available}
                title={option.available ? undefined : 'Данные не подготовлены'}
                onClick={() => selection.setGrid(option.value)}
                aria-pressed={selection.grid === option.value}
                className={`rounded-md px-2.5 text-[12px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                  selection.grid === option.value
                    ? 'border border-blue-200 bg-blue-50 text-blue-700 shadow-sm'
                    : 'text-slate-500 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-35'
                }`}
              >
                {option.value === '0p25' ? '0.25°' : '0.5°'}
              </button>
            ))}
          </div>
        </fieldset>

        {isPressureChannel(selection.channel) ? (
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

        <div className="grid gap-1">
          <span className="text-[11px] font-semibold text-slate-500">Автовращение</span>
          <button
            type="button"
            role="switch"
            aria-checked={autoRotate}
            aria-label="Медленно вращать глобус автоматически"
            title="Медленно вращать глобус автоматически"
            onClick={() => onAutoRotateChange(!autoRotate)}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-slate-300 bg-white px-2.5 text-[12px] font-semibold text-slate-600 transition-colors hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <span
              className={`relative h-5 w-9 rounded-full transition-colors ${
                autoRotate ? 'bg-blue-600' : 'bg-slate-300'
              }`}
              aria-hidden="true"
            >
              <span
                className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                  autoRotate ? 'translate-x-[18px]' : 'translate-x-0.5'
                }`}
              />
            </span>
            <span>{autoRotate ? 'Вкл' : 'Выкл'}</span>
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
  'h-9 min-w-0 rounded-lg border border-slate-300 bg-white px-2.5 text-[12px] font-medium text-slate-700 outline-none transition-colors focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400'

function SelectControl({ label, value, options, onChange }: SelectControlProps) {
  return (
    <label className="grid min-w-0 gap-1">
      <span className="text-[11px] font-semibold text-slate-500">{label}</span>
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
