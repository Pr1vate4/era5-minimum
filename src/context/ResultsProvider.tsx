import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { ResultsContext } from '../hooks/useResults'
import type { DashboardResults } from '../types'

const emptyResults: DashboardResults = {
  meta: {},
  compression: {},
  scores: {},
  criteria: [],
  per_channel: [],
  data_efficiency: [],
  rate_distortion: [],
  spectral: [],
  probe_forecast: {},
  resources: {},
  reconstructions: [],
}

function normalizeResults(input: Partial<DashboardResults> | null | undefined): DashboardResults {
  return {
    meta: input?.meta ?? emptyResults.meta,
    compression: input?.compression ?? emptyResults.compression,
    scores: input?.scores ?? emptyResults.scores,
    criteria: Array.isArray(input?.criteria) ? input.criteria : [],
    per_channel: Array.isArray(input?.per_channel) ? input.per_channel : [],
    data_efficiency: Array.isArray(input?.data_efficiency) ? input.data_efficiency : [],
    rate_distortion: Array.isArray(input?.rate_distortion) ? input.rate_distortion : [],
    spectral: Array.isArray(input?.spectral) ? input.spectral : [],
    probe_forecast: input?.probe_forecast ?? emptyResults.probe_forecast,
    resources: input?.resources ?? emptyResults.resources,
    reconstructions: Array.isArray(input?.reconstructions) ? input.reconstructions : [],
    globe: input?.globe,
  }
}

export function ResultsProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<DashboardResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [requestVersion, setRequestVersion] = useState(0)

  const reload = useCallback(() => {
    setRequestVersion((version) => version + 1)
  }, [])

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError(null)

    fetch(`${import.meta.env.BASE_URL}data/results.json`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Не удалось загрузить results.json: HTTP ${response.status}`)
        }

        return response.json() as Promise<Partial<DashboardResults>>
      })
      .then((json) => {
        setData(normalizeResults(json))
        setLoading(false)
      })
      .catch((caughtError: unknown) => {
        if (caughtError instanceof DOMException && caughtError.name === 'AbortError') {
          return
        }

        setError(caughtError instanceof Error ? caughtError.message : 'Неизвестная ошибка загрузки')
        setLoading(false)
      })

    return () => controller.abort()
  }, [requestVersion])

  return (
    <ResultsContext.Provider value={{ data, loading, error, reload }}>
      {children}
    </ResultsContext.Provider>
  )
}
