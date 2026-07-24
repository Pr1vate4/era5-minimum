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
        <span className="hidden whitespace-nowrap text-[15px] font-semibold text-black sm:block">
          МетеоКод
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-3">
        <time
          dateTime={data?.meta.generated_at}
          className="hidden whitespace-nowrap text-[13px] font-medium text-slate-500 lg:block"
        >
          {formatGeneratedDate(data?.meta.generated_at)}
        </time>
        <button
          type="button"
          onClick={exportReport}
          disabled={!data}
          className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-[13px] font-semibold text-slate-700 shadow-sm transition-colors duration-150 hover:border-slate-400 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          <span className="hidden md:inline">Экспорт отчёта</span>
          <span className="sr-only md:hidden">Экспорт отчёта</span>
        </button>
      </div>
    </header>
  )
}
