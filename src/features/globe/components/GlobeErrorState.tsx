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
    <div className="absolute inset-x-4 top-1/2 z-20 mx-auto max-w-md -translate-y-1/2 rounded-2xl border border-rose-200 bg-white/95 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" aria-hidden="true" />
        <div className="min-w-0">
          <h3 className="text-[16px] font-semibold text-[#101828]">{message}</h3>
          <button
            type="button"
            onClick={onRetry}
            className="ui-button-ghost ui-focus-ring mt-4 inline-flex h-10 items-center gap-2 rounded-xl px-4 text-[13px] font-semibold"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Повторить
          </button>
          {details ? (
            <details className="mt-4 text-[12px] leading-5 text-[#667085]">
              <summary className="cursor-pointer font-semibold">Технические детали</summary>
              <p className="mt-1 break-words">{details}</p>
            </details>
          ) : null}
        </div>
      </div>
    </div>
  )
}
