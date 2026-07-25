import type { Criterion } from '../types'
import { EmptyState } from './EmptyState'
import { StatusChip } from './StatusChip'
import { DataTable } from './common/ContentCard'

function renderValue(item: Criterion) {
  if (/exact roundtrip/i.test(item.name) || typeof item.value === 'boolean') {
    return item.pass ? 'Подтверждён' : 'Не подтверждён'
  }

  if (item.value === undefined) return 'Нет данных'

  const numericValue = typeof item.value === 'number' ? item.value : Number(item.value)
  const value = Number.isFinite(numericValue) ? numericValue.toFixed(2) : String(item.value)

  return `${value}${item.unit && item.unit !== 'bool' ? ` ${item.unit}` : ''}`
}

export function CriteriaPanel({ criteria }: { criteria: Criterion[] }) {
  if (criteria.length === 0) {
    return (
      <EmptyState
        title="Критерии отсутствуют"
        message="В results.json нет массива criteria для выбранного режима сжатия."
      />
    )
  }

  return (
    <DataTable>
      <table className="min-w-full text-[12px]">
        <thead className="bg-slate-50 text-slate-500">
          <tr>
            <th className="px-3 py-2.5 text-left font-semibold">Критерий</th>
            <th className="px-3 py-2.5 text-left font-semibold">Допустимое значение</th>
            <th className="px-3 py-2.5 text-left font-semibold">Фактическое значение</th>
            <th className="px-3 py-2.5 text-left font-semibold">Статус</th>
          </tr>
        </thead>
        <tbody>
          {criteria.map((item) => (
            <tr key={item.name} className="border-t border-slate-200 bg-white hover:bg-slate-50/80">
              <td className="px-3 py-3 font-medium text-slate-900">{item.name}</td>
              <td className="px-3 py-3 text-slate-500">{item.target ?? 'Не указан'}</td>
              <td className="px-3 py-3 text-slate-900">{renderValue(item)}</td>
              <td className="px-3 py-3">
                {item.pass === undefined ? (
                  <span className="text-slate-400">Нет данных</span>
                ) : (
                  <StatusChip ok={item.pass} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </DataTable>
  )
}
