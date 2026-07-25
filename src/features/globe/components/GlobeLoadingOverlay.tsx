export function GlobeLoadingOverlay({ message = 'Загрузка слоя ERA5…' }: { message?: string }) {
  return (
    <div
      className="ui-accent-badge pointer-events-none absolute left-1/2 top-4 z-20 -translate-x-1/2 rounded-full px-3.5 py-2 text-[12px] font-semibold shadow-[var(--shadow-soft)]"
      role="status"
      aria-live="polite"
    >
      <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
      {message}
    </div>
  )
}
