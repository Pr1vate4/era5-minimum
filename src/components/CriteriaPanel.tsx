import type { Criterion } from '../types'
import { StatusChip } from './StatusChip'

export function CriteriaPanel({ criteria }: { criteria: Criterion[] }) {
  const passedCount = criteria.filter((entry) => entry.pass).length

  const renderValue = (item: Criterion) => {
    if (item.name === 'Exact roundtrip') {
      return item.pass ? 'Подтверждён' : 'Не подтверждён'
    }

    if (typeof item.value === 'number') {
      return `${item.value.toFixed(2)} ${item.unit}`
    }

    return 'Обязателен'
  }

  return (
    <section id="criteria" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Критерии допуска</div>
          <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Статус эксперимента</h2>
        </div>
        <div className="flex items-center gap-2">
          <div className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-600">{passedCount} из {criteria.length} пройдено</div>
          <StatusChip ok={passedCount === criteria.length} trueLabel="Пройдено" falseLabel="Не пройдено" />
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-200">
        <table className="min-w-full text-[12px]">
          <thead className="bg-slate-50 text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Критерий</th>
              <th className="px-3 py-2 text-left">Порог</th>
              <th className="px-3 py-2 text-left">Факт</th>
              <th className="px-3 py-2 text-left">Статус</th>
            </tr>
          </thead>
          <tbody>
            {criteria.map((item) => (
              <tr key={item.name} className="border-t border-slate-200 bg-white hover:bg-slate-50/80">
                <td className="px-3 py-2 text-slate-900">{item.name}</td>
                <td className="px-3 py-2 text-slate-500">{item.target}</td>
                <td className="px-3 py-2 text-slate-900">{renderValue(item)}</td>
                <td className="px-3 py-2"><StatusChip ok={item.pass} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
