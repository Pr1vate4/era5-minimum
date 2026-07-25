export type InterfaceDensity = 'comfortable' | 'compact'
export type InterfaceTheme = 'light' | 'dark'
export type InterfaceFontSize = 'small' | 'medium' | 'large'
export type DataCacheMode = 'default' | 'no-store'

export type InterfaceSettings = {
  density: InterfaceDensity
  theme: InterfaceTheme
  fontSize: InterfaceFontSize
}

export type DataSettings = {
  resultsUrl: string
  cacheMode: DataCacheMode
  refreshMinutes: number
}

export type ApiSettings = {
  enabled: boolean
  baseUrl: string
  timeoutMs: number
}

export type ServiceSettings = {
  codecBaseUrl: string
  grafanaUrl: string
}

export type AppSettings = {
  interface: InterfaceSettings
  data: DataSettings
  api: ApiSettings
  services: ServiceSettings
}

export const SETTINGS_STORAGE_KEY = 'era5-minimum:settings:v1'

export const DEFAULT_APP_SETTINGS: AppSettings = {
  interface: {
    density: 'comfortable',
    theme: 'light',
    fontSize: 'medium',
  },
  data: {
    resultsUrl: 'data/results.json',
    cacheMode: 'default',
    refreshMinutes: 0,
  },
  api: {
    enabled: import.meta.env.VITE_WEATHER_API_ENABLED !== 'false',
    baseUrl: (import.meta.env.VITE_WEATHER_API_BASE_URL ?? '').replace(/\/$/, ''),
    timeoutMs: 15_000,
  },
  services: {
    codecBaseUrl: (import.meta.env.VITE_CODEC_API_BASE_URL ?? '').replace(/\/$/, ''),
    grafanaUrl: import.meta.env.VITE_GRAFANA_URL ?? 'http://localhost:3000',
  },
}

export function resolveResultsUrl(path: string) {
  const trimmedPath = path.trim()

  if (/^(https?:)?\/\//i.test(trimmedPath) || trimmedPath.startsWith('/')) {
    return trimmedPath
  }

  return `${import.meta.env.BASE_URL}${trimmedPath.replace(/^\.\//, '')}`
}

export function buildApiUrl(baseUrl: string, path: string) {
  return `${baseUrl.trim().replace(/\/$/, '')}${path}`
}

export function normalizeStoredSettings(input: unknown): AppSettings {
  if (!input || typeof input !== 'object') return DEFAULT_APP_SETTINGS

  const stored = input as Partial<{
    interface: Partial<InterfaceSettings>
    data: Partial<DataSettings>
    api: Partial<ApiSettings>
    services: Partial<ServiceSettings>
  }>
  const storedInterface = stored.interface
  const storedData = stored.data
  const storedApi = stored.api
  const storedServices = stored.services
  const refreshMinutes = storedData?.refreshMinutes
  const timeoutMs = storedApi?.timeoutMs

  return {
    interface: {
      density: storedInterface?.density === 'compact' ? 'compact' : 'comfortable',
      theme: storedInterface?.theme === 'dark' ? 'dark' : 'light',
      fontSize:
        storedInterface?.fontSize === 'small' || storedInterface?.fontSize === 'large'
          ? storedInterface.fontSize
          : 'medium',
    },
    data: {
      resultsUrl:
        typeof storedData?.resultsUrl === 'string' && storedData.resultsUrl.trim()
          ? storedData.resultsUrl.trim()
          : DEFAULT_APP_SETTINGS.data.resultsUrl,
      cacheMode: storedData?.cacheMode === 'no-store' ? 'no-store' : 'default',
      refreshMinutes:
        typeof refreshMinutes === 'number' && [0, 1, 5, 15].includes(refreshMinutes)
          ? refreshMinutes
          : DEFAULT_APP_SETTINGS.data.refreshMinutes,
    },
    api: {
      enabled:
        typeof storedApi?.enabled === 'boolean'
          ? storedApi.enabled
          : DEFAULT_APP_SETTINGS.api.enabled,
      baseUrl:
        typeof storedApi?.baseUrl === 'string'
          ? storedApi.baseUrl.trim().replace(/\/$/, '')
          : DEFAULT_APP_SETTINGS.api.baseUrl,
      timeoutMs:
        typeof timeoutMs === 'number' && [5_000, 15_000, 30_000, 60_000].includes(timeoutMs)
          ? timeoutMs
          : DEFAULT_APP_SETTINGS.api.timeoutMs,
    },
    services: {
      codecBaseUrl:
        typeof storedServices?.codecBaseUrl === 'string'
          ? storedServices.codecBaseUrl.trim().replace(/\/$/, '')
          : DEFAULT_APP_SETTINGS.services.codecBaseUrl,
      grafanaUrl:
        typeof storedServices?.grafanaUrl === 'string' && storedServices.grafanaUrl.trim()
          ? storedServices.grafanaUrl.trim()
          : DEFAULT_APP_SETTINGS.services.grafanaUrl,
    },
  }
}
