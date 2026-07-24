import { AlertTriangle, RotateCcw } from 'lucide-react'

type ErrorStateProps = {
  title: string
  message: string
  details?: string
  onRetry?: () => void
}

export function ErrorState({ title, message, details, onRetry }: ErrorStateProps) {
  return (
    <div className="rounded-2xl border border-rose-200 bg-white p-6">
      <div className="flex items-start gap-3">
        <div className="rounded-lg bg-rose-50 p-2 text-rose-600">
          <AlertTriangle className="h-5 w-5" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <h2 className="text-base font-semibold text-slate-900">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">{message}</p>
          {onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="mt-4 inline-flex h-9 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
              Повторить загрузку
            </button>
          ) : null}
          {details ? (
            <details className="mt-4 text-xs text-slate-500">
              <summary className="cursor-pointer font-medium">Технические детали</summary>
              <pre className="mt-2 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-3">{details}</pre>
            </details>
          ) : null}
        </div>
      </div>
    </div>
  )
}
