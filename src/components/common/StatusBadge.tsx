type StatusTone = 'success' | 'danger' | 'warning' | 'neutral'

const tones: Record<StatusTone, string> = {
  success: 'border-blue-200 bg-blue-50 text-blue-700',
  danger: 'border-rose-200 bg-rose-50 text-rose-700',
  warning: 'border-amber-200 bg-amber-50 text-amber-700',
  neutral: 'border-slate-200 bg-slate-50 text-slate-600',
}

export function StatusBadge({ label, tone = 'neutral' }: { label: string; tone?: StatusTone }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[12px] font-semibold ${tones[tone]}`}>
      {label}
    </span>
  )
}
