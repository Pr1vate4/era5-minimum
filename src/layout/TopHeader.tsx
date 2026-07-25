import { Download } from 'lucide-react'
import { useResults } from '../hooks/useResults'

function formatGeneratedDate(value: string | undefined) {
  if (!value) return 'Дата не указана'

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  return new Intl.DateTimeFormat('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(date)
}

export function TopHeader() {
  const { data } = useResults()

  const exportReport = () => {
    if (!data) return

    const report = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const downloadUrl = URL.createObjectURL(report)
    const link = document.createElement('a')

    link.href = downloadUrl
    link.download = `${data.meta.run_id ?? 'meteokod'}-report.json`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0)
  }

  return (
    <header className="fixed inset-x-0 top-0 z-50 flex h-14 items-center justify-between gap-4 border-b border-[#E4E7EC] bg-white px-4 sm:px-5">
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="h-8 w-8 shrink-0 overflow-hidden rounded-full border border-slate-300 bg-white shadow-sm">
          <img
            src={`${import.meta.env.BASE_URL}data/images/logo.png`}
            alt="Логотип МетеоКода"
            className="h-full w-full object-cover"
          />
        </div>
        <span className="hidden whitespace-nowrap text-[15px] font-semibold text-slate-950 sm:block">
          МетеоКод
        </span>
      </div>
    </header>
  )
}
