import type { ReactNode } from 'react'

type MetricCardProps = {
  title: string
  value: ReactNode
  caption?: ReactNode
  icon?: ReactNode
}

export function MetricCard({ title, value, caption, icon }: MetricCardProps) {
  return (
    <div className="ui-panel min-h-[150px] p-5">
      <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.16em] text-[var(--text-muted)]">
        {icon ? <span className="text-[var(--accent)]">{icon}</span> : null}
        {title}
      </div>
      <div className="mt-5 text-[25px] font-black tracking-[-0.03em] text-[var(--text-primary)]">{value}</div>
      {caption ? <div className="mt-2 text-[12px] leading-5 text-[var(--text-muted)]">{caption}</div> : null}
    </div>
  )
}
