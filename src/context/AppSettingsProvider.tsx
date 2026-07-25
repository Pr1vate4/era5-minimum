import { useEffect, useRef, useState, type ReactNode } from 'react'
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
  const transitionTimerRef = useRef<number | null>(null)

  useEffect(() => {
    const root = document.documentElement
    root.dataset.uiDensity = settings.interface.density
    root.dataset.theme = settings.interface.theme
    root.dataset.uiFontSize = settings.interface.fontSize
    root.style.colorScheme = settings.interface.theme

    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
    } catch {
      // The settings still work for the current session if browser storage is unavailable.
    }
  }, [settings])

  useEffect(
    () => () => {
      if (transitionTimerRef.current !== null) {
        window.clearTimeout(transitionTimerRef.current)
      }
      document.documentElement.classList.remove('theme-transition')
    },
    [],
  )

  const commitSettings = (nextSettings: AppSettings) => {
    const normalized = normalizeStoredSettings(nextSettings)
    const themeChanged = normalized.interface.theme !== settings.interface.theme

    if (themeChanged) {
      const root = document.documentElement
      root.classList.add('theme-transition')

      if (transitionTimerRef.current !== null) {
        window.clearTimeout(transitionTimerRef.current)
      }
      transitionTimerRef.current = window.setTimeout(() => {
        root.classList.remove('theme-transition')
        transitionTimerRef.current = null
      }, 280)
    }

    setSettings(normalized)
  }

  const saveSettings = (nextSettings: AppSettings) => {
    commitSettings(nextSettings)
  }

  const resetSettings = () => {
    commitSettings(DEFAULT_APP_SETTINGS)
  }

  return (
    <AppSettingsContext.Provider value={{ settings, saveSettings, resetSettings }}>
      {children}
    </AppSettingsContext.Provider>
  )
}
