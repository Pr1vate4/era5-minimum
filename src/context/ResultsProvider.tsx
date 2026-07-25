import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { resolveResultsUrl } from '../app/settings'
import { useAppSettings } from '../hooks/useAppSettings'
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
  const { settings } = useAppSettings()
  const [data, setData] = useState<DashboardResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [requestVersion, setRequestVersion] = useState(0)
  const { resultsUrl, cacheMode, refreshMinutes } = settings.data

  const reload = useCallback(() => {
    setRequestVersion((version) => version + 1)
  }, [])

  useEffect(() => {
    if (refreshMinutes <= 0) return

    const intervalId = window.setInterval(reload, refreshMinutes * 60_000)
    return () => window.clearInterval(intervalId)
  }, [refreshMinutes, reload])

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError(null)

    fetch(resolveResultsUrl(resultsUrl), {
      signal: controller.signal,
      cache: cacheMode,
    })
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
  }, [cacheMode, requestVersion, resultsUrl])

  return (
    <ResultsContext.Provider value={{ data, loading, error, reload }}>
      {children}
    </ResultsContext.Provider>
  )
}
