import { navigationItems } from '../app/navigationConfig'
import { SidebarItem } from '../layout/SidebarItem'

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-16 z-40 h-[calc(100vh-64px)] w-[72px] overflow-y-auto border-r border-[var(--border)] bg-[var(--chrome)]/88 px-2.5 py-4 backdrop-blur-xl">
      <nav className="flex flex-col items-center gap-2" aria-label="Разделы платформы">
        {navigationItems.map((item) => (
          <SidebarItem key={item.id} item={item} />
        ))}
      </nav>
    </aside>
  )
}
