import { navigationItems } from '../app/navigationConfig'
import { SidebarItem } from '../layout/SidebarItem'

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-14 z-40 h-[calc(100vh-56px)] w-[60px] overflow-y-auto border-r border-[#E4E7EC] bg-white px-2 py-3">
      <nav className="flex flex-col items-center gap-1.5" aria-label="Разделы платформы">
        {navigationItems.map((item) => (
          <SidebarItem key={item.id} item={item} />
        ))}
      </nav>
    </aside>
  )
}
