import { Activity, BarChart3, CircleGauge, Download, Sparkles } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
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

  return (
    <div className="min-h-screen bg-[#F7F8FA] text-[#101828]">
      <Sidebar team={data.meta.team} runId={data.meta.run_id} generatedAt={data.meta.generated_at} />

      <header className="fixed inset-x-0 top-0 z-50 flex h-16 items-center justify-between border-b border-[#E4E7EC] bg-[#FFFFFF] px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-full border border-[#D0D5DD] bg-white shadow-sm">
            <img
              src="/data/images/logo.png"
              alt="Логотип МетеоКод"
              className="h-full w-full object-cover"
            />
          </div>
          <div>
            <div className="text-[15px] font-semibold text-[#000000]">МетеоКод</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          
          
        </div>
      </header>

      <div className="pl-[84px] pt-16">
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

export default App
