import { Activity, Download, ExternalLink } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useAppSettings } from '../hooks/useAppSettings'
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
  const { settings } = useAppSettings()

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
    <header className="fixed inset-x-0 top-0 z-50 flex h-16 items-center justify-between gap-4 border-b border-[var(--border)] bg-[var(--chrome)]/95 px-4 shadow-[var(--shadow-soft)] backdrop-blur-xl sm:px-5">
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="h-9 w-9 shrink-0 overflow-hidden rounded-[13px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-soft)]">
          <img
            src={`${import.meta.env.BASE_URL}data/images/logo.png`}
            alt="Логотип МетеоКода"
            className="h-full w-full object-cover"
          />
        </div>
        <span className="hidden whitespace-nowrap text-[15px] font-semibold text-[var(--text-primary)] sm:block">
          МетеоКод
        </span>
      </div>
      <div className="flex items-center gap-2">
        {data ? (
          <button
            type="button"
            onClick={exportReport}
            className="ui-button-ghost ui-focus-ring hidden h-10 items-center gap-2 rounded-xl px-3 text-[12px] font-bold lg:inline-flex"
          >
            <Download className="h-4 w-4" />
            Отчёт
          </button>
        ) : null}
        <a
          href={settings.services.grafanaUrl}
          target="_blank"
          rel="noreferrer"
          className="ui-action ui-action-secondary !min-h-10 !px-3"
          aria-label="Открыть Grafana в новой вкладке"
        >
          <Activity className="h-4 w-4" />
          <span className="hidden sm:inline">Grafana</span>
          <ExternalLink className="h-3.5 w-3.5 opacity-60" />
        </a>
        <NavLink to="/codec" className="ui-action ui-action-primary !min-h-10 !px-3">
          <span className="hidden sm:inline">Сжать ERA5</span>
          <span className="sm:hidden">Сжать</span>
        </NavLink>
      </div>
    </header>
  )
}
