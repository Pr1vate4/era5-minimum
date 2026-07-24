import { useEffect, useState } from 'react'
import type { DashboardResults } from '../types'

const defaultResults: DashboardResults = {
  meta: {
    team: 'Unknown team',
    run_id: 'n/a',
    grid: '0.25deg',
    checkpoint: 'unknown',
    generated_at: new Date().toISOString(),
  },
  compression: {
    target_ratio: 32,
    actual_ratio: 32,
    bitstream_bytes: 0,
    roundtrip_exact: false,
  },
  scores: {
    surface_nrmse: 0,
    pressure_nrmse: 0,
    overall_nrmse: 0,
    delta_vs_reference_pct: 0,
    delta_ci95: [0, 0],
    psnr_delta_db: 0,
    spectral_error_pct: 0,
  },
  criteria: [],
  per_channel: [],
  data_efficiency: [],
  rate_distortion: [],
  spectral: [],
  probe_forecast: {
    n_pairs: 0,
    steps: 0,
    improvement_vs_persistence_pct: 0,
    improvement_vs_reference_probe_pct: 0,
  },
  resources: {
    gpu_hours: 0,
    peak_vram_gb: 0,
    params_millions: 0,
    optimizer_steps: 0,
  },
  reconstructions: [],
}

function normalizeResults(input: Partial<DashboardResults> | null | undefined): DashboardResults {
  return {
    meta: { ...defaultResults.meta, ...(input?.meta ?? {}) },
    compression: { ...defaultResults.compression, ...(input?.compression ?? {}) },
    scores: { ...defaultResults.scores, ...(input?.scores ?? {}) },
    criteria: input?.criteria ?? defaultResults.criteria,
    per_channel: input?.per_channel ?? defaultResults.per_channel,
    data_efficiency: input?.data_efficiency ?? defaultResults.data_efficiency,
    rate_distortion: input?.rate_distortion ?? defaultResults.rate_distortion,
    spectral: input?.spectral ?? defaultResults.spectral,
    probe_forecast: { ...defaultResults.probe_forecast, ...(input?.probe_forecast ?? {}) },
    resources: { ...defaultResults.resources, ...(input?.resources ?? {}) },
    reconstructions: input?.reconstructions ?? defaultResults.reconstructions,
  }
}

export function useResults() {
  const [data, setData] = useState<DashboardResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    fetch('/data/results.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load dashboard data (${response.status})`)
        }

        return response.json() as Promise<Partial<DashboardResults>>
      })
      .then((json) => {
        if (!isMounted) return
        setData(normalizeResults(json))
        setLoading(false)
      })
      .catch((caughtError) => {
        if (!isMounted) return
        setError(caughtError instanceof Error ? caughtError.message : 'Unknown error')
        setLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [])

  return { data, loading, error }
}
