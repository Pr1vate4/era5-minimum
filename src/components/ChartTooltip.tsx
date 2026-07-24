import type { TooltipProps } from 'recharts'
import type { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent'

export function ChartTooltip({ active, payload, label }: TooltipProps<ValueType, NameType>) {
  if (!active || !payload?.length) return null

  return (
    <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-[12px] text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.08)]">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{label}</div>
      {payload.map((item) => (
        <div key={item.dataKey?.toString()} className="flex items-center justify-between gap-3">
          <span className="text-slate-600">{item.name}</span>
          <span className="font-semibold text-slate-900">{typeof item.value === 'number' ? item.value.toFixed(3) : item.value}</span>
        </div>
      ))}
    </div>
  )
}
