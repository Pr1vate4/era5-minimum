export function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600">
      <div className="text-base font-semibold text-slate-900">{title}</div>
      <div className="mt-1">{message}</div>
    </div>
  )
}
