import { createContext, useContext } from 'react'
import type { DashboardResults } from '../types'

export type ResultsState = {
  data: DashboardResults | null
  loading: boolean
  error: string | null
  reload: () => void
}

export const ResultsContext = createContext<ResultsState | null>(null)

export function useResults() {
  const context = useContext(ResultsContext)

  if (!context) {
    throw new Error('useResults must be used inside ResultsProvider')
  }

  return context
}
