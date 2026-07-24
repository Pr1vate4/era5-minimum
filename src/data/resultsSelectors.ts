import type {
  DashboardResults,
  DataEfficiencyPoint,
  NumericValue,
  PerChannel,
  RateDistortionPoint,
  Reconstruction,
  SpectralPoint,
} from '../types'

export function toFiniteNumber(value: NumericValue | undefined): number | undefined {
  if (value === undefined || value === null || value === '') return undefined

  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

export function formatNumber(value: number | undefined, digits = 2) {
  return value === undefined ? 'Нет данных' : value.toLocaleString('ru-RU', { maximumFractionDigits: digits })
}

export function selectOverviewData(results: DashboardResults) {
  const criteriaCount = results.criteria.length
  const passedCriteria = results.criteria.filter((criterion) => criterion.pass === true).length
  const bitstreamBytes = toFiniteNumber(results.compression.bitstream_bytes)

  return {
    meta: results.meta,
    targetRatio: toFiniteNumber(results.compression.target_ratio),
    actualRatio: toFiniteNumber(results.compression.actual_ratio),
    bitstreamBytes,
    bitstreamAvailable: bitstreamBytes !== undefined && bitstreamBytes > 0,
    roundtripExact: results.compression.roundtrip_exact,
    overallNrmse: toFiniteNumber(results.scores.overall_nrmse),
    surfaceNrmse: toFiniteNumber(results.scores.surface_nrmse),
    pressureNrmse: toFiniteNumber(results.scores.pressure_nrmse),
    criteriaCount,
    passedCriteria,
    allCriteriaPassed: criteriaCount > 0 && passedCriteria === criteriaCount,
  }
}

export function selectCriteriaData(results: DashboardResults) {
  return {
    criteria: results.criteria.slice(),
    targetRatio: toFiniteNumber(results.compression.target_ratio),
    actualRatio: toFiniteNumber(results.compression.actual_ratio),
    bitstreamBytes: toFiniteNumber(results.compression.bitstream_bytes),
    roundtripExact: results.compression.roundtrip_exact,
  }
}

export type DataEfficiencyChartPoint = {
  n_samples: number
  overall_nrmse: number
  surface_nrmse?: number
  pressure_nrmse?: number
  reference_nrmse?: number
  ci_low?: number
  ci_high?: number
  pass?: boolean
}

export function selectDataEfficiencyData(results: DashboardResults): DataEfficiencyChartPoint[] {
  return results.data_efficiency
    .flatMap((point: DataEfficiencyPoint): DataEfficiencyChartPoint[] => {
      const sampleCount = toFiniteNumber(point.n_samples)
      const overallNrmse = toFiniteNumber(point.overall_nrmse)

      if (sampleCount === undefined || sampleCount <= 0 || overallNrmse === undefined) {
        return []
      }

      return [{
        n_samples: sampleCount,
        overall_nrmse: overallNrmse,
        surface_nrmse: toFiniteNumber(point.surface_nrmse),
        pressure_nrmse: toFiniteNumber(point.pressure_nrmse),
        reference_nrmse: toFiniteNumber(point.reference_nrmse),
        ci_low: toFiniteNumber(point.ci_low),
        ci_high: toFiniteNumber(point.ci_high),
        pass: point.pass,
      }]
    })
    .sort((left, right) => left.n_samples - right.n_samples)
}

export type RateDistortionChartPoint = {
  compression_ratio: number
  overall_nrmse: number
  surface_nrmse?: number
  pressure_nrmse?: number
  psnr?: number
  reference_nrmse?: number
}

export function selectRateDistortionData(results: DashboardResults): RateDistortionChartPoint[] {
  return results.rate_distortion
    .flatMap((point: RateDistortionPoint): RateDistortionChartPoint[] => {
      const compressionRatio = toFiniteNumber(point.compression_ratio)
      const overallNrmse = toFiniteNumber(point.overall_nrmse)

      if (compressionRatio === undefined || overallNrmse === undefined) {
        return []
      }

      return [{
        compression_ratio: compressionRatio,
        overall_nrmse: overallNrmse,
        surface_nrmse: toFiniteNumber(point.surface_nrmse),
        pressure_nrmse: toFiniteNumber(point.pressure_nrmse),
        psnr: toFiniteNumber(point.psnr),
        reference_nrmse: toFiniteNumber(point.reference_nrmse),
      }]
    })
    .sort((left, right) => left.compression_ratio - right.compression_ratio)
}

export type ChannelMetric = {
  code: string
  group: 'surface' | 'pressure' | 'unknown'
  name?: string
  unit?: string
  levelHpa?: number
  variable?: 'T' | 'U' | 'V' | 'Z' | 'Q'
  rmse?: number
  nrmse?: number
  psnr?: number
  deltaVsReferencePct?: number
  pass?: boolean
}

function parseChannelIdentity(channel: PerChannel) {
  const match = /^([TUVZQ])(\d{3,4})$/i.exec(channel.code)
  const explicitLevel = toFiniteNumber(channel.level_hpa)

  return {
    levelHpa: explicitLevel ?? (match ? Number(match[2]) : undefined),
    variable: match?.[1].toUpperCase() as ChannelMetric['variable'],
  }
}

export function selectChannelMetrics(results: DashboardResults): ChannelMetric[] {
  return results.per_channel.map((channel) => {
    const identity = parseChannelIdentity(channel)

    return {
      code: channel.code,
      group: channel.group === 'surface' || channel.group === 'pressure' ? channel.group : 'unknown',
      name: channel.name,
      unit: channel.unit,
      levelHpa: identity.levelHpa,
      variable: identity.variable,
      rmse: toFiniteNumber(channel.rmse),
      nrmse: toFiniteNumber(channel.nrmse),
      psnr: toFiniteNumber(channel.psnr),
      deltaVsReferencePct: toFiniteNumber(channel.delta_vs_reference_pct),
      pass: channel.pass,
    }
  })
}

export function selectReconstructions(results: DashboardResults): Reconstruction[] {
  return results.reconstructions
    .filter((item) => Boolean(item.channel && item.timestamp))
    .map((item) => ({ ...item }))
}

export function resolveDataAssetPath(path: string | undefined) {
  if (!path) return undefined
  if (/^https?:\/\//i.test(path)) return path

  const normalized = path.replace(/^\/+/, '')
  const dataPath = normalized.startsWith('data/') ? normalized : `data/${normalized}`
  return `${import.meta.env.BASE_URL}${dataPath}`
}

export type SpectralChartPoint = {
  wavenumber: number
  reference_energy: number
  model_energy: number
  relative_error_pct: number
  channel?: string
  grid?: string
  compression_ratio?: number
}

export function selectSpectralData(results: DashboardResults): SpectralChartPoint[] {
  return results.spectral
    .flatMap((point: SpectralPoint): SpectralChartPoint[] => {
      const wavenumber = toFiniteNumber(point.wavenumber)
      const referenceEnergy = toFiniteNumber(point.reference_energy)
      const modelEnergy = toFiniteNumber(point.model_energy)
      const relativeError =
        referenceEnergy !== undefined && referenceEnergy !== 0 && modelEnergy !== undefined
          ? (Math.abs(modelEnergy - referenceEnergy) / Math.abs(referenceEnergy)) * 100
          : undefined

      if (
        wavenumber === undefined ||
        wavenumber <= 0 ||
        referenceEnergy === undefined ||
        referenceEnergy <= 0 ||
        modelEnergy === undefined ||
        modelEnergy <= 0 ||
        relativeError === undefined
      ) {
        return []
      }

      return [{
        wavenumber,
        reference_energy: referenceEnergy,
        model_energy: modelEnergy,
        relative_error_pct: relativeError,
        channel: point.channel,
        grid: point.grid,
        compression_ratio: toFiniteNumber(point.compression_ratio),
      }]
    })
    .sort((left, right) => left.wavenumber - right.wavenumber)
}

export function selectProbeData(results: DashboardResults) {
  const probe = results.probe_forecast

  return {
    nPairs: toFiniteNumber(probe.n_pairs),
    steps: toFiniteNumber(probe.steps),
    paramsMillions: toFiniteNumber(probe.params_millions),
    latentNrmse: toFiniteNumber(probe.latent_nrmse),
    persistenceNrmse: toFiniteNumber(probe.persistence_nrmse),
    referenceProbeNrmse: toFiniteNumber(probe.reference_probe_nrmse),
    improvementVsPersistencePct: toFiniteNumber(probe.improvement_vs_persistence_pct),
    improvementVsReferencePct: toFiniteNumber(probe.improvement_vs_reference_probe_pct),
    pass: probe.pass,
  }
}

export function selectResourcesData(results: DashboardResults) {
  const resources = results.resources

  return {
    gpuCount: toFiniteNumber(resources.gpu_count),
    gpuHours: toFiniteNumber(resources.gpu_hours),
    peakVramGb: toFiniteNumber(resources.peak_vram_gb),
    paramsMillions: toFiniteNumber(resources.params_millions),
    optimizerSteps: toFiniteNumber(resources.optimizer_steps),
    probeParamsMillions: toFiniteNumber(resources.probe_params_millions),
    probeSteps: toFiniteNumber(resources.probe_steps),
    meta: results.meta,
  }
}
