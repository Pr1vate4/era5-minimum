export function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-5" aria-label="Загрузка страницы" aria-busy="true">
      <div className="space-y-2">
        <div className="h-7 w-64 rounded bg-slate-200" />
        <div className="h-4 w-full max-w-xl rounded bg-slate-200" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index} className="h-28 rounded-2xl border border-slate-200 bg-white p-4">
            <div className="h-3 w-24 rounded bg-slate-200" />
            <div className="mt-5 h-7 w-20 rounded bg-slate-200" />
          </div>
        ))}
      </div>
      <div className="h-80 rounded-2xl border border-slate-200 bg-white p-4">
        <div className="h-full rounded-xl bg-slate-100" />
      </div>
    </div>
  )
}
