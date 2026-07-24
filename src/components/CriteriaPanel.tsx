import type { Criterion } from '../types'

function statusTone(pass: boolean) {
  return pass
    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
    : 'border-rose-500/40 bg-rose-500/10 text-rose-300'
}

export function CriteriaPanel({ criteria }: { criteria: Criterion[] }) {
  return (
    <section id="criteria" className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Admission criteria</p>
          <h2 className="text-2xl font-semibold text-white">Pass / Fail</h2>
        </div>
        <div className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">
          {criteria.filter((entry) => entry.pass).length}/{criteria.length} passed
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        {criteria.map((item) => (
          <article
            key={item.name}
            className={`rounded-xl border p-4 ${statusTone(item.pass)}`}
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-300">{item.target}</p>
                <h3 className="mt-2 text-lg font-semibold text-white">{item.name}</h3>
              </div>
              <span className="rounded-full border border-current px-2.5 py-1 text-xs font-semibold uppercase">
                {item.pass ? 'PASS' : 'FAIL'}
              </span>
            </div>

            <p className="mt-4 text-3xl font-semibold text-white">
              {typeof item.value === 'number' ? item.value.toFixed(2) : 'required'}
              <span className="ml-1 text-sm text-slate-300">{item.unit}</span>
            </p>
          </article>
        ))}
      </div>
    </section>
  )
}
