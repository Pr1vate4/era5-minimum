import { useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { NavLink } from 'react-router-dom'
import type { NavigationItem } from '../app/navigationConfig'

type TooltipPosition = {
  left: number
  top: number
}

export function SidebarItem({ item }: { item: NavigationItem }) {
  const tooltipId = useId()
  const linkRef = useRef<HTMLAnchorElement>(null)
  const [tooltipPosition, setTooltipPosition] = useState<TooltipPosition | null>(null)
  const Icon = item.icon

  const showTooltip = () => {
    const rect = linkRef.current?.getBoundingClientRect()

    if (rect) {
      setTooltipPosition({
        left: rect.right + 10,
        top: rect.top + rect.height / 2,
      })
    }
  }

  return (
    <>
      <NavLink
        ref={linkRef}
        to={item.path}
        aria-label={item.title}
        aria-describedby={tooltipPosition ? tooltipId : undefined}
        onMouseEnter={showTooltip}
        onMouseLeave={() => setTooltipPosition(null)}
        onFocus={showTooltip}
        onBlur={() => setTooltipPosition(null)}
        className={({ isActive }) =>
          `ui-focus-ring group relative flex h-11 w-11 shrink-0 items-center justify-center rounded-[14px] border transition-all duration-150 ${
            isActive
              ? 'translate-x-0.5 border-[var(--accent-border)] bg-[var(--accent)] text-white shadow-[0_7px_16px_rgba(23,92,199,0.2)]'
              : 'border-transparent text-[var(--text-muted)] hover:-translate-y-0.5 hover:border-[var(--border)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-secondary)]'
          }`
        }
      >
        <Icon className="h-[19px] w-[19px]" aria-hidden="true" />
      </NavLink>

      {tooltipPosition
        ? createPortal(
            <div
              id={tooltipId}
              role="tooltip"
              className="pointer-events-none fixed z-[70] -translate-y-1/2 whitespace-nowrap rounded-md border border-[var(--border-strong)] bg-[var(--surface-raised)] px-2.5 py-1.5 text-[12px] font-medium text-[var(--text-primary)] shadow-[var(--shadow-raised)]"
              style={tooltipPosition}
            >
              {item.title}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
