import { createContext, useContext } from 'react'
import type { AppSettings } from '../app/settings'

export type AppSettingsState = {
  settings: AppSettings
  saveSettings: (settings: AppSettings) => void
  resetSettings: () => void
}

export const AppSettingsContext = createContext<AppSettingsState | null>(null)

export function useAppSettings() {
  const context = useContext(AppSettingsContext)

  if (!context) {
    throw new Error('useAppSettings must be used inside AppSettingsProvider')
  }

  return context
}
