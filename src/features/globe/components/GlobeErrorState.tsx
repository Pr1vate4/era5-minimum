import { AlertTriangle, RotateCcw } from 'lucide-react'

export function GlobeErrorState({
  message,
  details,
  onRetry,
}: {
  message: string
  details?: string
  onRetry: () => void
}) {
  return (
    <div className="absolute inset-x-4 top-1/2 z-20 mx-auto max-w-md -translate-y-1/2 rounded-2xl border border-rose-200 bg-white/95 p-4 shadow-lg backdrop-blur-sm">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" aria-hidden="true" />
        <div className="min-w-0">
          <h3 className="text-[14px] font-semibold text-slate-900">{message}</h3>
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 inline-flex h-8 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-[11px] font-semibold text-slate-700 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Повторить
          </button>
          {details ? (
            <details className="mt-3 text-[10px] text-slate-500">
              <summary className="cursor-pointer font-semibold">Технические детали</summary>
              <p className="mt-1 break-words">{details}</p>
            </details>
          ) : null}
        </div>
      </div>
    </div>
  )
}
