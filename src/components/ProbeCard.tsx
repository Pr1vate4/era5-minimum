import type { ProbeForecast } from '../types'

export function ProbeCard({ probe }: { probe: ProbeForecast }) {
  return (
    <section id="probe-forecast" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Latent forecast probe</p>
        <h2 className="text-2xl font-semibold text-white">+6h forecast utility</h2>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric label="Pairs" value={probe.n_pairs.toLocaleString()} />
        <Metric label="Steps" value={probe.steps.toLocaleString()} />
        <Metric label="vs Persistence" value={`${probe.improvement_vs_persistence_pct.toFixed(1)}%`} />
        <Metric label="vs Reference probe" value={`${probe.improvement_vs_reference_probe_pct.toFixed(1)}%`} />
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</p>
      <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
    </div>
  )
}
