import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { AppSettingsProvider } from './context/AppSettingsProvider'
import { ResultsProvider } from './context/ResultsProvider'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppSettingsProvider>
      <ResultsProvider>
        <App />
      </ResultsProvider>
    </AppSettingsProvider>
  </React.StrictMode>,
)
