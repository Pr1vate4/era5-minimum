import type { Resources } from '../types'

const limits = {
  gpu_hours: 48,
  peak_vram_gb: 24,
  params_millions: 20,
  optimizer_steps: 50000,
}

export function ResourcesCard({ resources }: { resources: Resources }) {
  const rows = [
    { key: 'gpu_hours', label: 'GPU hours', value: resources.gpu_hours, limit: limits.gpu_hours },
    { key: 'peak_vram_gb', label: 'Peak VRAM', value: resources.peak_vram_gb, limit: limits.peak_vram_gb },
    { key: 'params_millions', label: 'Parameters', value: resources.params_millions, limit: limits.params_millions },
    { key: 'optimizer_steps', label: 'Optimizer steps', value: resources.optimizer_steps, limit: limits.optimizer_steps },
  ]

  return (
    <section id="resources" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Compute footprint</p>
        <h2 className="text-2xl font-semibold text-white">Resources</h2>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {rows.map((row) => {
          const pass = row.value <= row.limit
          return (
            <div key={row.key} className="rounded-xl border border-slate-800 bg-slate-950/80 p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-300">{row.label}</p>
                <span className={`rounded-full border px-2 py-1 text-xs ${pass ? 'border-emerald-500/40 text-emerald-300' : 'border-rose-500/40 text-rose-300'}`}>
                  {pass ? 'under limit' : 'over limit'}
                </span>
              </div>
              <div className="mt-3 flex items-end justify-between">
                <p className="text-3xl font-semibold text-white">{row.value}</p>
                <p className="text-sm text-slate-400">limit {row.limit}</p>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
