import { ChevronDown, RotateCcw, Search } from 'lucide-react'
import type { ReactNode } from 'react'

export function FilterBar({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end gap-3 rounded-2xl border border-slate-200 bg-white p-3">
      {children}
    </div>
  )
}

export type SelectOption = {
  value: string
  label: string
  disabled?: boolean
}

type SelectFieldProps = {
  label: string
  value: string
  options: SelectOption[]
  onChange: (value: string) => void
  disabled?: boolean
}

export function SelectField({ label, value, options, onChange, disabled }: SelectFieldProps) {
  return (
    <label className="grid min-w-[150px] gap-1.5 text-[11px] font-semibold text-slate-500">
      <span>{label}</span>
      <span className="relative">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          className="h-10 w-full appearance-none rounded-xl border border-slate-200 bg-white/90 px-3 pr-9 text-[13px] font-medium text-slate-700 outline-none transition-colors hover:border-slate-300 hover:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
          aria-hidden="true"
        />
      </span>
    </label>
  )
}

export type SegmentOption<T extends string> = {
  value: T
  label: string
  disabled?: boolean
}

type SegmentedControlProps<T extends string> = {
  label: string
  value: T
  options: SegmentOption<T>[]
  onChange: (value: T) => void
}

export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedControlProps<T>) {
  return (
    <fieldset className="grid gap-1">
      <legend className="text-[11px] font-semibold text-slate-500">{label}</legend>
      <div className="inline-flex h-9 rounded-lg border border-slate-300 bg-slate-50 p-0.5">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            disabled={option.disabled}
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`rounded-md px-3 text-[12px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
              value === option.value
                ? 'bg-white text-blue-700 shadow-sm'
                : 'text-slate-500 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-40'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </fieldset>
  )
}

type SearchInputProps = {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
}

export function SearchInput({ label, value, onChange, placeholder }: SearchInputProps) {
  return (
    <label className="grid min-w-[190px] gap-1 text-[11px] font-semibold text-slate-500">
      <span>{label}</span>
      <span className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          type="search"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="h-9 w-full rounded-lg border border-slate-300 bg-white pl-8 pr-3 text-[13px] font-medium text-slate-700 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
        />
      </span>
    </label>
  )
}

export function CheckboxField({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
}) {
  return (
    <label className="flex h-9 items-center gap-2 text-[12px] font-medium text-slate-600">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
        className="h-4 w-4 rounded border-slate-300 accent-blue-600 disabled:cursor-not-allowed"
      />
      <span className={disabled ? 'text-slate-400' : undefined}>{label}</span>
    </label>
  )
}

export function ResetFiltersButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex h-9 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-[12px] font-semibold text-slate-600 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
      Сбросить
    </button>
  )
}
