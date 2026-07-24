import { Activity, BarChart3, CircleGauge, ImageIcon, Radar } from 'lucide-react'
import { useMemo } from 'react'
import { ChannelTable } from './components/ChannelTable'
import { CriteriaPanel } from './components/CriteriaPanel'
import { DataEfficiencyChart } from './components/DataEfficiencyChart'
import { ProbeCard } from './components/ProbeCard'
import { RateDistortionChart } from './components/RateDistortionChart'
import { ReconstructionViewer } from './components/ReconstructionViewer'
import { ResourcesCard } from './components/ResourcesCard'
import { SpectralChart } from './components/SpectralChart'
import { useResults } from './hooks/useResults'

function App() {
  const { data, loading, error } = useResults()

  const mappedSections = useMemo(
    () => [
      { id: 'criteria', label: 'Criteria' },
      { id: 'data-efficiency', label: 'Data efficiency' },
      { id: 'rate-distortion', label: 'Rate-distortion' },
      { id: 'per-channel', label: 'Channels' },
      { id: 'reconstruction', label: 'Reconstruction' },
      { id: 'spectral', label: 'Spectral' },
      { id: 'probe-forecast', label: 'Probe' },
      { id: 'resources', label: 'Resources' },
    ],
    [],
  )

  if (loading) {
    return <div className="flex min-h-screen items-center justify-center text-slate-300">Loading dashboard…</div>
  }

  if (error || !data) {
    return <div className="flex min-h-screen items-center justify-center text-rose-300">Dashboard data could not be loaded.</div>
  }

  return (
    <div className="min-h-screen text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800/60 bg-slate-950/90 backdrop-blur-md">
        <div className="mx-auto max-w-[1480px] px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">ERA5-Minimum Hackathon Demo</p>
              <h1 className="text-3xl font-semibold text-white">Weather compression dashboard</h1>
            </div>
            <nav className="flex gap-2">
              {mappedSections.map((section) => (
                <a key={section.id} href={`#${section.id}`} className="rounded-full border border-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:border-accent hover:text-white">
                  {section.label}
                </a>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1480px] space-y-6 px-6 py-6">
        <section className="grid grid-cols-12 gap-4">
          <div className="col-span-8 rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-soft">
            <div className="flex items-start justify-between gap-6">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">{data.meta.team}</p>
                <h2 className="mt-2 text-4xl font-semibold text-white">{data.meta.run_id}</h2>
                <p className="mt-2 text-sm text-slate-300">Grid: {data.meta.grid} · Checkpoint: {data.meta.checkpoint}</p>
              </div>
              <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
                Exact roundtrip: {data.compression.roundtrip_exact ? 'enabled' : 'disabled'}
              </div>
            </div>
          </div>

          <div className="col-span-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-soft">
            <div className="flex items-center gap-2 text-slate-300">
              <CircleGauge className="h-4 w-4 text-accent" />
              <span className="text-xs uppercase tracking-[0.2em]">Compression ratio</span>
            </div>
            <p className="mt-4 text-5xl font-semibold text-white">{data.compression.actual_ratio.toFixed(1)}×</p>
            <p className="mt-2 text-sm text-slate-300">Target: {data.compression.target_ratio}× · Bitstream: {data.compression.bitstream_bytes.toLocaleString()} bytes</p>
          </div>
        </section>

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-soft">
            <div className="flex items-center gap-2 text-slate-300"><Activity className="h-4 w-4 text-accent" />Overall NRMSE</div>
            <p className="mt-3 text-3xl font-semibold text-white">{data.scores.overall_nrmse.toFixed(3)}</p>
          </div>
          <div className="col-span-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-soft">
            <div className="flex items-center gap-2 text-slate-300"><Radar className="h-4 w-4 text-accent" />Surface / pressure</div>
            <p className="mt-3 text-3xl font-semibold text-white">{data.scores.surface_nrmse.toFixed(3)} / {data.scores.pressure_nrmse.toFixed(3)}</p>
          </div>
          <div className="col-span-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-soft">
            <div className="flex items-center gap-2 text-slate-300"><BarChart3 className="h-4 w-4 text-accent" />PSNR delta</div>
            <p className="mt-3 text-3xl font-semibold text-white">{data.scores.psnr_delta_db.toFixed(2)} dB</p>
          </div>
          <div className="col-span-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-soft">
            <div className="flex items-center gap-2 text-slate-300"><ImageIcon className="h-4 w-4 text-accent" />Spectral error</div>
            <p className="mt-3 text-3xl font-semibold text-white">{data.scores.spectral_error_pct.toFixed(1)}%</p>
          </div>
        </div>

        <CriteriaPanel criteria={data.criteria} />
        <DataEfficiencyChart data={data.data_efficiency} />
        <RateDistortionChart data={data.rate_distortion} />
        <ChannelTable channels={data.per_channel} />
        <ReconstructionViewer reconstructions={data.reconstructions} />
        <SpectralChart data={data.spectral} />
        <ProbeCard probe={data.probe_forecast} />
        <ResourcesCard resources={data.resources} />

        <footer className="rounded-2xl border border-slate-800 bg-slate-900/70 px-5 py-4 text-center text-xs text-slate-400">
          Data: ERA5 via WeatherBench 2 · Processing: xarray/Zarr
        </footer>
      </main>
    </div>
  )
}

export default App
