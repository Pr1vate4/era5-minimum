import { useEffect, useState, type ReactNode } from 'react'
import {
  DEFAULT_APP_SETTINGS,
  normalizeStoredSettings,
  SETTINGS_STORAGE_KEY,
  type AppSettings,
} from '../app/settings'
import { AppSettingsContext } from '../hooks/useAppSettings'

function readSettings() {
  try {
    const stored = window.localStorage.getItem(SETTINGS_STORAGE_KEY)
    return stored ? normalizeStoredSettings(JSON.parse(stored)) : DEFAULT_APP_SETTINGS
  } catch {
    return DEFAULT_APP_SETTINGS
  }
}

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(readSettings)

  useEffect(() => {
    const root = document.documentElement
    root.dataset.uiDensity = settings.interface.density
    root.dataset.theme = settings.interface.theme
    root.dataset.uiFontSize = settings.interface.fontSize
    root.dataset.reduceMotion = String(settings.interface.reduceMotion)
    root.style.colorScheme = settings.interface.theme

    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
    } catch {
      // The settings still work for the current session if browser storage is unavailable.
    }
  }, [settings])

  const saveSettings = (nextSettings: AppSettings) => {
    setSettings(normalizeStoredSettings(nextSettings))
  }

  const resetSettings = () => {
    setSettings(DEFAULT_APP_SETTINGS)
  }

  return (
    <AppSettingsContext.Provider value={{ settings, saveSettings, resetSettings }}>
      {children}
    </AppSettingsContext.Provider>
  )
}
