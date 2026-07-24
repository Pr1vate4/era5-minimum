import type { Resources } from '../types'

const limits = {
  gpu_hours: 48,
  peak_vram_gb: 24,
  params_millions: 20,
  optimizer_steps: 50000,
}

export function ResourcesCard({ resources }: { resources: Resources }) {
  const rows = [
    { key: 'gpu_hours', label: 'GPU-часы', value: resources.gpu_hours, limit: limits.gpu_hours },
    { key: 'peak_vram_gb', label: 'Пиковый VRAM', value: resources.peak_vram_gb, limit: limits.peak_vram_gb },
    { key: 'params_millions', label: 'Параметры', value: resources.params_millions, limit: limits.params_millions },
    { key: 'optimizer_steps', label: 'Шаги оптимизатора', value: resources.optimizer_steps, limit: limits.optimizer_steps },
  ]

  return (
    <section id="resources" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Ресурсы</div>
        <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Ограничения и потребление</h2>
      </div>

      <div className="space-y-2">
        {rows.map((row) => {
          const percent = Math.min((row.value / row.limit) * 100, 100)
          const tone = percent >= 100 ? 'bg-red-600' : percent >= 85 ? 'bg-amber-500' : 'bg-blue-500'
          return (
            <div key={row.key} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between text-[12px]">
                <span className="text-slate-700">{row.label}</span>
                <span className="text-slate-500">{row.value} / {row.limit}</span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
                <div className={`h-full rounded-full ${tone}`} style={{ width: `${percent}%` }} />
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
