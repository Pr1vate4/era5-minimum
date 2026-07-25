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
    <div className="absolute inset-x-4 top-1/2 z-20 mx-auto max-w-md -translate-y-1/2 rounded-2xl border border-[#E4E7EC] bg-white/95 p-5 text-center shadow-lg backdrop-blur-sm">
      <DatabaseZap className="mx-auto h-6 w-6 text-[#667085]" aria-hidden="true" />
      <h3 className="mt-3 text-[16px] font-semibold text-[#101828]">{message}</h3>
      {details ? <p className="mt-2 text-[13px] leading-5 text-[#667085]">{details}</p> : null}
      {onSelectAvailable ? (
        <button
          type="button"
          onClick={onSelectAvailable}
          className="mt-4 inline-flex h-10 items-center rounded-xl border border-blue-200 bg-blue-50 px-4 text-[13px] font-semibold text-blue-700 hover:bg-blue-100 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-100"
        >
          Выбрать доступный кадр
        </button>
      ) : null}
    </div>
  )
}
