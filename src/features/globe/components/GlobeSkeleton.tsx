export function GlobeSkeleton() {
  return (
    <section
      className="overflow-hidden rounded-[20px] border border-[#E4E7EC] bg-white shadow-[0_2px_8px_rgba(16,24,40,0.04)]"
      aria-label="Загрузка интерактивного глобуса"
      aria-busy="true"
    >
      <div className="animate-pulse">
        <div className="flex items-center gap-3 px-5 py-4">
          <div className="h-9 w-9 rounded-xl bg-slate-200" />
          <div className="space-y-2">
            <div className="h-4 w-64 max-w-[65vw] rounded bg-slate-200" />
            <div className="h-3 w-80 max-w-[70vw] rounded bg-slate-100" />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 border-y border-slate-200 bg-slate-50 p-4 sm:grid-cols-4">
          {Array.from({ length: 4 }, (_, index) => (
            <div key={index} className="h-9 rounded-lg bg-slate-200" />
          ))}
        </div>
        <div className="grid gap-5 p-5 xl:grid-cols-[minmax(0,1fr)_300px]">
          <div className="mx-auto aspect-square h-[330px] max-w-full rounded-full bg-slate-100 sm:h-[400px]" />
          <div className="h-72 rounded-2xl border border-slate-200 bg-slate-50" />
        </div>
        <div className="border-t border-slate-200 px-5 py-4">
          <div className="h-2.5 rounded-full bg-slate-200" />
        </div>
      </div>
    </section>
  )
}
