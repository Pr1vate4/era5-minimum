import { Inbox } from 'lucide-react'
import type { ReactNode } from 'react'

export function EmptyState({
  title,
  message,
  action,
}: {
  title: string
  message: string
  action?: ReactNode
}) {
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600">
      <Inbox className="h-5 w-5 text-slate-400" aria-hidden="true" />
      <div className="mt-3 text-base font-semibold text-slate-900">{title}</div>
      <div className="mt-1 max-w-2xl leading-5">{message}</div>
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  )
}
