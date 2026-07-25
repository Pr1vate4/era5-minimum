import type { ReactNode } from 'react'

type ContentCardProps = {
  title?: string
  description?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}

export function ContentCard({ title, description, actions, children, className = '' }: ContentCardProps) {
  return (
    <section className={`ui-surface rounded-[22px] border p-5 shadow-[var(--shadow-raised)] ${className}`}>
      {title || description || actions ? (
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title ? <h2 className="text-[17px] font-bold tracking-[-0.015em] text-[var(--text-primary)]">{title}</h2> : null}
            {description ? <p className="mt-1 text-[12px] leading-5 text-[var(--text-muted)]">{description}</p> : null}
          </div>
          {actions ? <div className="shrink-0">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  )
}

export const ChartCard = ContentCard

export function DataTable({ children }: { children: ReactNode }) {
  return <div className="overflow-auto rounded-xl border border-[var(--border)]">{children}</div>
}
