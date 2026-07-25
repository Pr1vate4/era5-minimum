import { navigationItems } from '../app/navigationConfig'
import { SidebarItem } from '../layout/SidebarItem'

export function Sidebar() {
  return (
    <aside className="app-navigation-scroll fixed inset-x-0 bottom-0 z-40 h-16 overflow-x-auto border-t border-[var(--border)] bg-[var(--chrome)]/95 px-2 py-1.5 backdrop-blur-xl sm:inset-x-auto sm:bottom-auto sm:left-0 sm:top-16 sm:h-[calc(100vh-64px)] sm:w-[72px] sm:overflow-y-auto sm:border-r sm:border-t-0 sm:px-2.5 sm:py-4">
      <nav className="flex min-w-max items-center gap-2 sm:min-w-0 sm:flex-col" aria-label="Разделы платформы">
        {navigationItems.map((item) => (
          <SidebarItem key={item.id} item={item} />
        ))}
      </nav>
    </aside>
  )
}
