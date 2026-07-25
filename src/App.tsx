import { lazy } from 'react'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './layout/AppShell'

const OverviewPage = lazy(() => import('./pages/OverviewPage'))
const CriteriaPage = lazy(() => import('./pages/CriteriaPage'))
const DataEfficiencyPage = lazy(() => import('./pages/DataEfficiencyPage'))
const RateDistortionPage = lazy(() => import('./pages/RateDistortionPage'))
const ChannelsPage = lazy(() => import('./pages/ChannelsPage'))
const ReconstructionsPage = lazy(() => import('./pages/ReconstructionsPage'))
const SpectralPage = lazy(() => import('./pages/SpectralPage'))
const ProbePage = lazy(() => import('./pages/ProbePage'))
const ResourcesPage = lazy(() => import('./pages/ResourcesPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/criteria" element={<CriteriaPage />} />
          <Route path="/data-efficiency" element={<DataEfficiencyPage />} />
          <Route path="/rate-distortion" element={<RateDistortionPage />} />
          <Route path="/channels" element={<ChannelsPage />} />
          <Route path="/reconstructions" element={<ReconstructionsPage />} />
          <Route path="/spectral" element={<SpectralPage />} />
          <Route path="/probe" element={<ProbePage />} />
          <Route path="/resources" element={<ResourcesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}

export default App
