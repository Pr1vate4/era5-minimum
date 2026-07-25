import { DatabaseZap } from 'lucide-react'

type GlobeEmptyStateProps = {
  message: string
  details?: string
  onSelectAvailable?: () => void
}

export function GlobeEmptyState({
  message,
  details,
  onSelectAvailable,
}: GlobeEmptyStateProps) {
  return (
    <div className="absolute inset-x-4 top-1/2 z-20 mx-auto max-w-md -translate-y-1/2 rounded-2xl border border-slate-200 bg-white/95 p-4 text-center shadow-lg backdrop-blur-sm">
      <DatabaseZap className="mx-auto h-5 w-5 text-slate-400" aria-hidden="true" />
      <h3 className="mt-2 text-[14px] font-semibold text-slate-900">{message}</h3>
      {details ? <p className="mt-1 text-[11px] leading-4 text-slate-500">{details}</p> : null}
      {onSelectAvailable ? (
        <button
          type="button"
          onClick={onSelectAvailable}
          className="mt-3 inline-flex h-8 items-center rounded-lg border border-blue-200 bg-blue-50 px-3 text-[11px] font-semibold text-blue-700 hover:bg-blue-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          Выбрать доступный кадр
        </button>
      ) : null}
    </div>
  )
}
