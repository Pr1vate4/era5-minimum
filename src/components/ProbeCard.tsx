import type { ProbeForecast } from '../types'

export function ProbeCard({ probe }: { probe: ProbeForecast }) {
  return (
    <section id="probe-forecast" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Latent-прогноз</div>
        <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Проверка полезности latent-пространства</h2>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Metric label="Пар" value={probe.n_pairs.toLocaleString()} />
        <Metric label="Шаги" value={probe.steps.toLocaleString()} />
        <Metric label="vs persistence" value={`${probe.improvement_vs_persistence_pct.toFixed(1)}%`} />
        <Metric label="vs reference probe" value={`${probe.improvement_vs_reference_probe_pct.toFixed(1)}%`} />
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className="mt-2 text-[18px] font-semibold text-slate-900">{value}</p>
    </div>
  )
}
