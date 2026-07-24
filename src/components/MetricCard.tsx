import type { ReactNode } from 'react'

type MetricCardProps = {
  title: string
  value: string
  caption?: string
  icon: ReactNode
}

export function MetricCard({ title, value, caption, icon }: MetricCardProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
        <span className="text-blue-600">{icon}</span>
        {title}
      </div>
      <div className="mt-3 text-[22px] font-semibold text-slate-900">{value}</div>
      {caption ? <div className="mt-1 text-[12px] text-slate-500">{caption}</div> : null}
    </div>
  )
}
