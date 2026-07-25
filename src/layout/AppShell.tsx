import { Suspense, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { Sidebar } from '../components/Sidebar'
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

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#F7F8FA] text-[#101828]">
      <ScrollToTop />
      <TopHeader />
      <Sidebar />

      <main className="ml-[60px] min-h-[calc(100vh-56px)] overflow-x-hidden pt-14">
        <div className="w-full px-5 py-5 lg:px-7">
          {loading ? (
            <LoadingSkeleton />
          ) : error ? (
            <ErrorState
              title="Не удалось загрузить результаты"
              message="Проверьте доступность public/data/results.json и повторите попытку."
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
