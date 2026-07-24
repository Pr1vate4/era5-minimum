export type ResourceKey =
  | 'gpuCount'
  | 'peakVramGb'
  | 'paramsMillions'
  | 'optimizerSteps'
  | 'gpuHours'
  | 'probeParamsMillions'
  | 'probeSteps'

export type ResourceLimit = {
  key: ResourceKey
  label: string
  limit: number
  unit: string
}

export const resourceLimits: ResourceLimit[] = [
  { key: 'gpuCount', label: 'Количество GPU', limit: 1, unit: 'GPU' },
  { key: 'peakVramGb', label: 'Пиковый VRAM', limit: 24, unit: 'GB' },
  { key: 'paramsMillions', label: 'Обучаемые параметры', limit: 20, unit: 'млн' },
  { key: 'optimizerSteps', label: 'Шаги оптимизатора', limit: 50_000, unit: 'шагов' },
  { key: 'gpuHours', label: 'Вычислительный бюджет', limit: 48, unit: 'GPU-часов' },
  { key: 'probeParamsMillions', label: 'Параметры probe', limit: 2, unit: 'млн' },
  { key: 'probeSteps', label: 'Шаги probe', limit: 5_000, unit: 'шагов' },
]
