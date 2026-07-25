import type { ReactNode } from 'react'

type PageHeaderProps = {
  title: string
  description: string
  actions?: ReactNode
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
      <div className="min-w-0">
        <h1 className="text-[24px] font-semibold tracking-tight text-[var(--text-primary)]">{title}</h1>
        <PageDescription>{description}</PageDescription>
      </div>
      {actions ? <div className="shrink-0">{actions}</div> : null}
    </div>
  )
}

export function PageDescription({ children }: { children: ReactNode }) {
  return <p className="mt-1 max-w-3xl text-[13px] leading-5 text-[var(--text-muted)]">{children}</p>
}
