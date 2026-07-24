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
          `group relative flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 ${
            isActive
              ? 'border-blue-200 bg-blue-100 text-blue-700'
              : 'border-transparent text-slate-500 hover:border-blue-100 hover:bg-blue-50 hover:text-blue-600'
          }`
        }
      >
        <Icon className="h-[18px] w-[18px]" aria-hidden="true" />
      </NavLink>

      {tooltipPosition
        ? createPortal(
            <div
              id={tooltipId}
              role="tooltip"
              className="pointer-events-none fixed z-[70] -translate-y-1/2 whitespace-nowrap rounded-md border border-slate-200 bg-slate-900 px-2.5 py-1.5 text-[12px] font-medium text-white shadow-sm"
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
