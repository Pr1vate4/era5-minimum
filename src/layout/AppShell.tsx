import { Suspense, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { Sidebar } from '../components/Sidebar'
import { useAppSettings } from '../hooks/useAppSettings'
import { useResults } from '../hooks/useResults'
import { TopHeader } from './TopHeader'

function ScrollToTop() {
  const { pathname } = useLocation()

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0 })
  }, [pathname])

  return null
}

export function AppShell() {
  const { loading, error, reload } = useResults()
  const { settings } = useAppSettings()
  const { pathname } = useLocation()
  const independentPage = pathname === '/settings' || pathname === '/codec'
  const contentSpacing =
    settings.interface.density === 'compact'
      ? 'px-3 py-3 lg:px-4'
      : 'px-5 py-5 lg:px-7'

  return (
    <div className="app-backdrop min-h-screen overflow-x-hidden text-[var(--text-primary)]">
      <ScrollToTop />
      <TopHeader />
      <Sidebar />

      <main className="min-h-[calc(100vh-64px)] overflow-x-hidden pb-20 pt-16 sm:ml-[72px] sm:pb-0">
        <div className={`mx-auto w-full max-w-[1720px] ${contentSpacing}`}>
          {independentPage ? (
            <Suspense fallback={<LoadingSkeleton />}>
              <Outlet />
            </Suspense>
          ) : loading ? (
            <LoadingSkeleton />
          ) : error ? (
            <ErrorState
              title="Не удалось загрузить результаты"
              message={`Проверьте источник «${settings.data.resultsUrl}» в настройках данных и повторите попытку.`}
              details={error}
              onRetry={reload}
            />
          ) : (
            <Suspense fallback={<LoadingSkeleton />}>
              <Outlet />
            </Suspense>
          )}
        </div>
      </main>
    </div>
  )
}
