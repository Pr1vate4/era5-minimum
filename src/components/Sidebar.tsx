import { Activity, Gauge, Layers3, Radar, ScanSearch, ServerCog, Sparkles, UploadCloud } from 'lucide-react'

type SidebarProps = {
  team: string
  runId: string
  generatedAt: string
  activeSection: NavigationSectionId
  onNavigate: (sectionId: NavigationSectionId) => void
}

export const navigationSections = [
  { id: 'overview', label: 'Обзор эксперимента', icon: Sparkles },
  { id: 'criteria', label: 'Критерии допуска', icon: CheckCircleIcon },
  { id: 'data-efficiency', label: 'Эффективность данных', icon: Activity },
  { id: 'rate-distortion', label: 'Сжатие и качество', icon: Gauge },
  { id: 'per-channel', label: 'Метрики по каналам', icon: Layers3 },
  { id: 'reconstruction', label: 'Реконструкции', icon: UploadCloud },
  { id: 'spectral', label: 'Спектральный анализ', icon: Radar },
  { id: 'probe-forecast', label: 'Latent-прогноз', icon: ScanSearch },
  { id: 'resources', label: 'Ресурсы', icon: ServerCog },
] as const

export type NavigationSectionId = (typeof navigationSections)[number]['id']

function CheckCircleIcon({ className }: { className?: string }) {
  return <span className={className}>✓</span>
}

export function Sidebar({ activeSection, onNavigate }: SidebarProps) {
  return (
    <aside className="fixed left-0 top-14 z-40 h-[calc(100vh-56px)] w-[60px] overflow-y-auto border-r border-[#E4E7EC] bg-[#FFFFFF] px-2 py-3">
      <div className="flex h-full flex-col justify-between">
        <div>
          

          <nav className="flex flex-col items-center gap-1.5" aria-label="Разделы дашборда">
            {navigationSections.map((section) => {
              const Icon = section.icon
              const isActive = activeSection === section.id

              return (
                <a
                  key={section.id}
                  href={`#${section.id}`}
                  aria-label={section.label}
                  aria-current={isActive ? 'location' : undefined}
                  title={section.label}
                  onClick={() => onNavigate(section.id)}
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#147DCC] focus-visible:ring-offset-2 ${
                    isActive
                      ? 'border-[#B9DDFF] bg-[#EAF5FF] text-[#147DCC]'
                      : 'border-transparent text-[#667085] hover:border-[#D6EAFF] hover:bg-[#F2F8FF] hover:text-[#147DCC]'
                  }`}
                >
                  <Icon className="h-[18px] w-[18px]" />
                </a>
              )
            })}
          </nav>
        </div>
      </div>
    </aside>
  )
}
