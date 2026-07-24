import { Activity, Gauge, Layers3, Radar, ScanSearch, ServerCog, Sparkles, UploadCloud } from 'lucide-react'

type SidebarProps = {
  team: string
  runId: string
  generatedAt: string
}

const sections = [
  { id: 'criteria', label: 'Допуск', icon: CheckCircleIcon },
  { id: 'data-efficiency', label: 'Данные', icon: Activity },
  { id: 'rate-distortion', label: 'Сжатие', icon: Gauge },
  { id: 'per-channel', label: 'Каналы', icon: Layers3 },
  { id: 'reconstruction', label: 'Рекон.', icon: UploadCloud },
  { id: 'spectral', label: 'Спектр', icon: Radar },
  { id: 'probe-forecast', label: 'Прогноз', icon: ScanSearch },
  { id: 'resources', label: 'Ресурсы', icon: ServerCog },
]

function CheckCircleIcon({ className }: { className?: string }) {
  return <span className={className}>✓</span>
}

export function Sidebar({ team, runId, generatedAt }: SidebarProps) {
  return (
    <aside className="fixed left-0 top-16 z-40 h-[calc(100vh-64px)] w-[84px] border-r border-[#E4E7EC] bg-[#FFFFFF] px-3 py-4">
      <div className="flex h-full flex-col justify-between">
        <div>
          

          <nav className="space-y-1.5">
            {sections.map((section) => {
              const Icon = section.icon

              return (
                <a
                  key={section.id}
                  href={`#${section.id}`}
                  aria-label={section.label}
                  title={section.label}
                  className="flex h-10 w-10 items-center justify-center rounded-xl border border-transparent text-[#667085] transition hover:border-[#E4E7EC] hover:bg-[#F7F8FA] hover:text-[#101828]"
                >
                  <Icon className="h-4 w-4" />
                </a>
              )
            })}
          </nav>
        </div>
      </div>
    </aside>
  )
}
