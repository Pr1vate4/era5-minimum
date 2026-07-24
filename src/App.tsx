import { Activity, BarChart3, CircleGauge, Download, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  navigationSections,
  Sidebar,
  type NavigationSectionId,
} from './components/Sidebar'
import { ChannelTable } from './components/ChannelTable'
import { CriteriaPanel } from './components/CriteriaPanel'
import { DataEfficiencyChart } from './components/DataEfficiencyChart'
import { ProbeCard } from './components/ProbeCard'
import { RateDistortionChart } from './components/RateDistortionChart'
import { ReconstructionViewer } from './components/ReconstructionViewer'
import { ResourcesCard } from './components/ResourcesCard'
import { SpectralChart } from './components/SpectralChart'
import { useResults } from './hooks/useResults'
import { LoadingSkeleton } from './components/LoadingSkeleton'
import { EmptyState } from './components/EmptyState'
import { MetricCard } from './components/MetricCard'

function App() {
  const { data, loading, error } = useResults()
  const { activeSection, navigateToSection } = useSectionNavigation(Boolean(data && !loading && !error))

  if (loading) {
    return <LoadingSkeleton />
  }

  if (error || !data) {
    return <EmptyState title="Ошибка загрузки" message="Не удалось загрузить результаты эксперимента из public/data/results.json." />
  }

  const overviewCards = [
    { title: 'Сетка', value: data.meta.grid, icon: <Sparkles className="h-4 w-4" /> },
    { title: 'Уровни', value: '1000 / 925 / 850 / 700 hPa', icon: <LayersIcon className="h-4 w-4" /> },
    { title: 'Фактическое сжатие', value: `${data.compression.actual_ratio.toFixed(1)}×`, icon: <CircleGauge className="h-4 w-4" /> },
    { title: 'Контрольная модель', value: 'VAEformer / CRA5', icon: <BarChart3 className="h-4 w-4" /> },
    { title: 'Run ID', value: data.meta.run_id, icon: <Activity className="h-4 w-4" /> },
  ]
  const activeSectionLabel =
    navigationSections.find((section) => section.id === activeSection)?.label ?? 'Обзор эксперимента'
  const generatedDate = formatGeneratedDate(data.meta.generated_at)

  const exportReport = () => {
    const report = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const downloadUrl = URL.createObjectURL(report)
    const link = document.createElement('a')

    link.href = downloadUrl
    link.download = `${data.meta.run_id}-report.json`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0)
  }

  return (
    <div className="min-h-screen bg-[#F7F8FA] text-[#101828]">
      <Sidebar
        team={data.meta.team}
        runId={data.meta.run_id}
        generatedAt={data.meta.generated_at}
        activeSection={activeSection}
        onNavigate={navigateToSection}
      />

      <header className="fixed inset-x-0 top-0 z-50 flex h-14 items-center justify-between border-b border-[#E4E7EC] bg-white px-4 sm:px-5">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-full border border-[#D0D5DD] bg-white shadow-sm">
            <img
              src="/data/images/logo.png"
              alt="Логотип МетеоКод"
              className="h-full w-full object-cover"
            />
          </div>
          <div className="flex min-w-0 items-center">
            <div className="whitespace-nowrap text-[15px] font-semibold text-black">МетеоКод</div>
            <div className="ml-3 hidden min-w-0 border-l border-[#E4E7EC] pl-3 text-[13px] font-medium text-[#667085] sm:block">
              <span className="block truncate">{activeSectionLabel}</span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <time
            dateTime={data.meta.generated_at}
            className="hidden whitespace-nowrap text-[13px] font-medium text-[#667085] md:block"
          >
            {generatedDate}
          </time>
          <button
            type="button"
            onClick={exportReport}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-[#D0D5DD] bg-white px-3 text-[13px] font-semibold text-[#344054] shadow-sm transition-colors hover:border-[#98A2B3] hover:bg-[#F9FAFB] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#147DCC] focus-visible:ring-offset-2"
          >
            <Download className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">Экспорт отчёта</span>
            <span className="sr-only sm:hidden">Экспорт отчёта</span>
          </button>
        </div>
      </header>

      <div className="pl-[60px] pt-14">
        <main className="p-5">
          <section id="overview" className="mb-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Обзор эксперимента</div>
                <h1 className="mt-2 text-[24px] font-semibold text-slate-900">Сжатие погодных данных ERA5</h1>
                <div className="mt-1 text-[13px] text-slate-500">Автокодировщик с ограниченной выборкой для обучения</div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-5 gap-3">
            {overviewCards.map((card) => (
              <MetricCard key={card.title} title={card.title} value={card.value} icon={card.icon} caption={card.title === 'Фактическое сжатие' ? 'Цель: 32–64×' : undefined} />
            ))}
          </section>

          <section className="mt-3 grid grid-cols-12 gap-3">
            <div id="criteria" className="col-span-5">
              <CriteriaPanel criteria={data.criteria} />
            </div>
            <div id="data-efficiency" className="col-span-3">
              <DataEfficiencyChart data={data.data_efficiency} />
            </div>
            <div id="rate-distortion" className="col-span-4">
              <RateDistortionChart data={data.rate_distortion} />
            </div>
          </section>

          <section className="mt-3 grid grid-cols-12 gap-3">
            <div id="per-channel" className="col-span-7">
              <ChannelTable channels={data.per_channel} />
            </div>
            <div id="reconstruction" className="col-span-5">
              <ReconstructionViewer reconstructions={data.reconstructions} />
            </div>
          </section>

          <section className="mt-3 grid grid-cols-12 gap-3">
            <div id="spectral" className="col-span-5">
              <SpectralChart data={data.spectral} />
            </div>
            <div id="probe-forecast" className="col-span-3">
              <ProbeCard probe={data.probe_forecast} />
            </div>
            <div id="resources" className="col-span-4">
              <ResourcesCard resources={data.resources} />
            </div>
          </section>

          <footer className="mt-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-center text-[11px] text-slate-500">
            Data: ERA5 via WeatherBench 2 · Processing: xarray/Zarr
          </footer>
        </main>
      </div>
    </div>
  )
}

function LayersIcon({ className }: { className?: string }) {
  return <span className={className}>▦</span>
}

function formatGeneratedDate(value: string) {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(date)
}

function getSectionFromHash(): NavigationSectionId {
  const hash = window.location.hash.slice(1)
  const section = navigationSections.find((item) => item.id === hash)

  return section?.id ?? 'overview'
}

function useSectionNavigation(enabled: boolean) {
  const [activeSection, setActiveSection] = useState<NavigationSectionId>(getSectionFromHash)
  const activeSectionRef = useRef(activeSection)

  const setActive = (sectionId: NavigationSectionId, syncHash: boolean) => {
    activeSectionRef.current = sectionId
    setActiveSection(sectionId)

    if (syncHash && window.location.hash !== `#${sectionId}`) {
      window.history.replaceState(null, '', `#${sectionId}`)
    }
  }

  useEffect(() => {
    activeSectionRef.current = activeSection
  }, [activeSection])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const sectionElements = navigationSections
      .map((section) => document.getElementById(section.id))
      .filter((element): element is HTMLElement => element !== null)
    const visibleSections = new Map<NavigationSectionId, number>()
    const initialSection = getSectionFromHash()
    let initialScrollFrame = 0

    setActive(initialSection, false)

    if (window.location.hash) {
      initialScrollFrame = window.requestAnimationFrame(() => {
        document.getElementById(initialSection)?.scrollIntoView({ block: 'start' })
      })
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const sectionId = entry.target.id as NavigationSectionId
          visibleSections.set(sectionId, entry.isIntersecting ? entry.intersectionRatio : 0)
        }

        if (window.scrollY < 24) {
          setActive('overview', true)
          return
        }

        if ((visibleSections.get(activeSectionRef.current) ?? 0) > 0.02) {
          return
        }

        const nextSection = navigationSections
          .map((section) => ({
            id: section.id,
            ratio: visibleSections.get(section.id) ?? 0,
            top: document.getElementById(section.id)?.getBoundingClientRect().top ?? Infinity,
          }))
          .filter((section) => section.ratio > 0.02)
          .sort((left, right) => Math.abs(left.top - 72) - Math.abs(right.top - 72))[0]

        if (nextSection) {
          setActive(nextSection.id, true)
        }
      },
      {
        rootMargin: '-56px 0px -55% 0px',
        threshold: [0, 0.02, 0.25, 0.5, 0.75, 1],
      },
    )

    sectionElements.forEach((element) => observer.observe(element))

    const handleHashChange = () => {
      setActive(getSectionFromHash(), false)
    }
    const handleScroll = () => {
      if (window.scrollY < 24 && activeSectionRef.current !== 'overview') {
        setActive('overview', true)
      }
    }

    window.addEventListener('hashchange', handleHashChange)
    window.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      window.cancelAnimationFrame(initialScrollFrame)
      observer.disconnect()
      window.removeEventListener('hashchange', handleHashChange)
      window.removeEventListener('scroll', handleScroll)
    }
  }, [enabled])

  return {
    activeSection,
    navigateToSection: (sectionId: NavigationSectionId) => setActive(sectionId, false),
  }
}

export default App
