type StatusChipProps = {
  ok: boolean
  trueLabel?: string
  falseLabel?: string
}

export function StatusChip({ ok, trueLabel = 'Пройдено', falseLabel = 'Не пройдено' }: StatusChipProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium ${
        ok
          ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
          : 'border-rose-200 bg-rose-50 text-rose-700'
      }`}
    >
      {ok ? trueLabel : falseLabel}
    </span>
  )
}
