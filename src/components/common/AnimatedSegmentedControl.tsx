import type { CSSProperties, ReactNode } from 'react'

export type SegmentedOption<T extends string> = {
  value: T
  label: string
  icon?: ReactNode
  disabled?: boolean
}

type AnimatedSegmentedControlProps<T extends string> = {
  value: T
  options: readonly SegmentedOption<T>[]
  onChange: (value: T) => void
  ariaLabel: string
  disabled?: boolean
}

type SegmentedStyle = CSSProperties & {
  '--segment-count': number
  '--active-index': number
}

export function AnimatedSegmentedControl<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  disabled = false,
}: AnimatedSegmentedControlProps<T>) {
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  )
  const style: SegmentedStyle = {
    '--segment-count': Math.max(options.length, 1),
    '--active-index': selectedIndex,
    gridTemplateColumns: `repeat(${Math.max(options.length, 1)}, minmax(0, 1fr))`,
  }

  return (
    <div
      className="animated-segmented"
      style={style}
      role="group"
      aria-label={ariaLabel}
      aria-disabled={disabled || undefined}
    >
      <span className="animated-segmented__indicator" aria-hidden="true" />
      {options.map((option) => {
        const selected = option.value === value
        const optionDisabled = disabled || option.disabled

        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={selected}
            disabled={optionDisabled}
            data-selected={selected}
            className="animated-segmented__option"
            onClick={() => onChange(option.value)}
          >
            {option.icon ? (
              <span className="animated-segmented__icon" aria-hidden="true">
                {option.icon}
              </span>
            ) : null}
            <span>{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}
