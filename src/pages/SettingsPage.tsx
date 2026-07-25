import {
  CheckCircle2,
  ChevronDown,
  Database,
  LoaderCircle,
  Moon,
  Palette,
  PlugZap,
  RotateCcw,
  Save,
  Sun,
  Wifi,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import {
  buildApiUrl,
  DEFAULT_APP_SETTINGS,
  normalizeStoredSettings,
  resolveResultsUrl,
  type ApiSettings,
  type DataSettings,
  type InterfaceSettings,
} from '../app/settings'
import { getNavigationItem } from '../app/navigationConfig'
import { AnimatedSegmentedControl } from '../components/common/AnimatedSegmentedControl'
import { PageHeader } from '../components/common/PageHeader'
import { useAppSettings } from '../hooks/useAppSettings'

const page = getNavigationItem('settings')

type SettingsSection = 'interface' | 'data' | 'api'
type ConnectionState =
  | { status: 'idle'; message: string }
  | { status: 'testing'; message: string }
  | { status: 'success'; message: string }
  | { status: 'error'; message: string }

const sections: Array<{
  id: SettingsSection
  title: string
  description: string
  icon: LucideIcon
}> = [
  {
    id: 'interface',
    title: 'Интерфейс',
    description: 'Тема, размер текста и плотность',
    icon: Palette,
  },
  {
    id: 'data',
    title: 'Данные',
    description: 'Источник отчёта и обновление',
    icon: Database,
  },
  {
    id: 'api',
    title: 'API',
    description: 'Подключение погодных слоёв',
    icon: PlugZap,
  },
]

export default function SettingsPage() {
  const { settings, saveSettings, resetSettings } = useAppSettings()
  const [activeSection, setActiveSection] = useState<SettingsSection>('interface')
  const [interfaceDraft, setInterfaceDraft] = useState(settings.interface)
  const [dataDraft, setDataDraft] = useState(settings.data)
  const [apiDraft, setApiDraft] = useState(settings.api)
  const [connection, setConnection] = useState<ConnectionState>({
    status: 'idle',
    message: 'Соединение ещё не проверялось',
  })

  const resetAll = () => {
    resetSettings()
    setInterfaceDraft(DEFAULT_APP_SETTINGS.interface)
    setDataDraft(DEFAULT_APP_SETTINGS.data)
    setApiDraft(DEFAULT_APP_SETTINGS.api)
    setConnection({ status: 'idle', message: 'Настройки возвращены к исходным' })
  }

  const saveSection = <Section extends keyof typeof settings>(
    section: Section,
    value: (typeof settings)[Section],
  ) => {
    const nextSettings = normalizeStoredSettings({ ...settings, [section]: value })
    saveSettings(nextSettings)

    if (section === 'interface') {
      setInterfaceDraft(nextSettings.interface)
    } else if (section === 'data') {
      setDataDraft(nextSettings.data)
    } else {
      setApiDraft(nextSettings.api)
    }
  }

  const testConnection = async () => {
    const controller = new AbortController()
    const startedAt = performance.now()
    const timeoutId = window.setTimeout(() => controller.abort(), apiDraft.timeoutMs)

    setConnection({ status: 'testing', message: 'Проверяем доступность API…' })

    try {
      const response = await fetch(buildApiUrl(apiDraft.baseUrl, '/api/v1/variables'), {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const variables: unknown = await response.json()
      if (!Array.isArray(variables)) {
        throw new Error('API вернул ответ неожиданного формата')
      }

      const elapsed = Math.round(performance.now() - startedAt)
      setConnection({
        status: 'success',
        message: `Подключено за ${elapsed} мс · доступно параметров: ${variables.length}`,
      })
    } catch (error) {
      const timedOut = error instanceof DOMException && error.name === 'AbortError'
      setConnection({
        status: 'error',
        message: timedOut
          ? `API не ответил за ${apiDraft.timeoutMs / 1000} с`
          : error instanceof Error
            ? error.message
            : 'Не удалось подключиться к API',
      })
    } finally {
      window.clearTimeout(timeoutId)
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title={page.title}
        description="Управление внешним видом панели, источником результатов и подключением ERA5 API."
        actions={
          <button
            type="button"
            onClick={resetAll}
            className="ui-button-ghost ui-focus-ring inline-flex h-10 items-center gap-2 rounded-xl px-3.5 text-[12px] font-semibold"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Сбросить всё
          </button>
        }
      />

      <div className="grid min-h-[520px] overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-soft)] lg:grid-cols-[240px_minmax(0,1fr)]">
        <aside className="border-b border-[var(--border)] bg-[var(--background-subtle)] p-3 lg:border-b-0 lg:border-r">
          <div
            className="grid grid-cols-3 gap-2 lg:grid-cols-1"
            role="tablist"
            aria-label="Разделы настроек"
          >
            {sections.map((section) => {
              const Icon = section.icon
              const active = activeSection === section.id

              return (
                <button
                  key={section.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setActiveSection(section.id)}
                  className={`ui-focus-ring group flex min-w-0 items-center gap-3 rounded-xl border px-3 py-3 text-left transition-colors ${
                    active
                      ? 'ui-selected'
                      : 'border-transparent text-[var(--text-muted)] hover:border-[var(--border)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <span
                    className={`inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                      active ? 'bg-[var(--accent-soft)]' : 'bg-[var(--surface)]'
                    }`}
                  >
                    <Icon className="h-[17px] w-[17px]" aria-hidden="true" />
                  </span>
                  <span className="hidden min-w-0 lg:block">
                    <span className="block text-[13px] font-semibold">{section.title}</span>
                    <span className="mt-0.5 block truncate text-[11px] font-medium text-[var(--text-muted)]">
                      {section.description}
                    </span>
                  </span>
                  <span className="truncate text-[11px] font-semibold lg:hidden">
                    {section.title}
                  </span>
                </button>
              )
            })}
          </div>

          <div className="mt-4 hidden rounded-xl border border-[var(--accent-border)] bg-[var(--accent-soft)] p-3 lg:block">
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--accent-text)]">
              Локальные настройки
            </p>
            <p className="mt-1.5 text-[11px] leading-5 text-[var(--text-muted)]">
              Они сохраняются только в этом браузере и не изменяют научные данные на сервере.
            </p>
          </div>
        </aside>

        <main className="min-w-0 p-4 sm:p-6">
          {activeSection === 'interface' ? (
            <InterfacePanel
              draft={interfaceDraft}
              saved={settings.interface}
              onChange={setInterfaceDraft}
              onSave={() => saveSection('interface', interfaceDraft)}
              onReset={() => setInterfaceDraft(DEFAULT_APP_SETTINGS.interface)}
            />
          ) : null}

          {activeSection === 'data' ? (
            <DataPanel
              draft={dataDraft}
              saved={settings.data}
              onChange={setDataDraft}
              onSave={() => saveSection('data', dataDraft)}
              onReset={() => setDataDraft(DEFAULT_APP_SETTINGS.data)}
            />
          ) : null}

          {activeSection === 'api' ? (
            <ApiPanel
              draft={apiDraft}
              saved={settings.api}
              connection={connection}
              onChange={(next) => {
                setApiDraft(next)
                setConnection({ status: 'idle', message: 'Настройки изменены, запустите проверку' })
              }}
              onSave={() => saveSection('api', apiDraft)}
              onReset={() => {
                setApiDraft(DEFAULT_APP_SETTINGS.api)
                setConnection({ status: 'idle', message: 'Установлены исходные значения' })
              }}
              onTest={testConnection}
            />
          ) : null}
        </main>
      </div>
    </div>
  )
}

type PanelProps<T> = {
  draft: T
  saved: T
  onChange: (settings: T) => void
  onSave: () => void
  onReset: () => void
}

function InterfacePanel({
  draft,
  saved,
  onChange,
  onSave,
  onReset,
}: PanelProps<InterfaceSettings>) {
  const dirty = !settingsEqual(draft, saved)

  return (
    <SettingsPanel
      title="Интерфейс"
      description="Настройте тему, размер текста и плотность панели. Изменения применяются после сохранения."
      icon={Palette}
    >
      <SettingRow
        title="Тема оформления"
        description="Тёмная тема снижает яркость фона, сохраняя исходные цвета научных визуализаций."
      >
        <AnimatedSegmentedControl
          value={draft.theme}
          options={[
            {
              value: 'light',
              label: 'Светлая',
              icon: <Sun className="h-4 w-4" />,
            },
            {
              value: 'dark',
              label: 'Тёмная',
              icon: <Moon className="h-4 w-4" />,
            },
          ]}
          onChange={(theme) => onChange({ ...draft, theme })}
          ariaLabel="Тема оформления"
        />
      </SettingRow>

      <SettingRow
        title="Плотность интерфейса"
        description="Компактный режим уменьшает внешние отступы и освобождает больше места для графиков."
      >
        <AnimatedSegmentedControl
          value={draft.density}
          options={[
            { value: 'comfortable', label: 'Обычная' },
            { value: 'compact', label: 'Компактная' },
          ]}
          onChange={(density) => onChange({ ...draft, density })}
          ariaLabel="Плотность интерфейса"
        />
      </SettingRow>

      <SettingRow
        title="Размер текста"
        description="Меняет размер подписей, кнопок, таблиц и заголовков без замены самого шрифта."
      >
        <AnimatedSegmentedControl
          value={draft.fontSize}
          options={[
            { value: 'small', label: 'Меньше' },
            { value: 'medium', label: 'Обычный' },
            { value: 'large', label: 'Больше' },
          ]}
          onChange={(fontSize) => onChange({ ...draft, fontSize })}
          ariaLabel="Размер текста"
        />
      </SettingRow>

      <SaveBar dirty={dirty} onSave={onSave} onReset={onReset} />
    </SettingsPanel>
  )
}

function DataPanel({ draft, saved, onChange, onSave, onReset }: PanelProps<DataSettings>) {
  const dirty = !settingsEqual(draft, saved)

  return (
    <SettingsPanel
      title="Данные"
      description="Укажите, откуда аналитическая панель получает готовые метрики эксперимента."
      icon={Database}
    >
      <SettingRow
        title="Файл результатов"
        description="Поддерживается путь внутри public или полный HTTP(S)-адрес с разрешённым CORS."
        vertical
      >
        <TextInput
          value={draft.resultsUrl}
          placeholder="data/results.json"
          onChange={(resultsUrl) => onChange({ ...draft, resultsUrl })}
          ariaLabel="Адрес файла результатов"
        />
        <p className="mt-2 break-all rounded-lg bg-slate-50 px-3 py-2 font-mono text-[11px] text-slate-500">
          Будет загружено: {resolveResultsUrl(draft.resultsUrl || DEFAULT_APP_SETTINGS.data.resultsUrl)}
        </p>
      </SettingRow>

      <SettingRow
        title="Кэш браузера"
        description="Отключите кэш, если results.json часто заменяется во время разработки."
      >
        <SelectInput
          value={draft.cacheMode}
          options={[
            { value: 'default', label: 'Использовать кэш' },
            { value: 'no-store', label: 'Всегда свежие данные' },
          ]}
          onChange={(cacheMode) =>
            onChange({ ...draft, cacheMode: cacheMode as DataSettings['cacheMode'] })
          }
          ariaLabel="Режим кэширования данных"
        />
      </SettingRow>

      <SettingRow
        title="Автообновление"
        description="Периодически перечитывает файл без перезагрузки всей страницы."
      >
        <SelectInput
          value={String(draft.refreshMinutes)}
          options={[
            { value: '0', label: 'Выключено' },
            { value: '1', label: 'Каждую минуту' },
            { value: '5', label: 'Каждые 5 минут' },
            { value: '15', label: 'Каждые 15 минут' },
          ]}
          onChange={(refreshMinutes) =>
            onChange({ ...draft, refreshMinutes: Number(refreshMinutes) })
          }
          ariaLabel="Интервал автообновления данных"
        />
      </SettingRow>

      <SaveBar dirty={dirty} onSave={onSave} onReset={onReset} />
    </SettingsPanel>
  )
}

type ApiPanelProps = PanelProps<ApiSettings> & {
  connection: ConnectionState
  onTest: () => void
}

function ApiPanel({
  draft,
  saved,
  connection,
  onChange,
  onSave,
  onReset,
  onTest,
}: ApiPanelProps) {
  const dirty = !settingsEqual(draft, saved)

  return (
    <SettingsPanel
      title="ERA5 API"
      description="Управляет каталогом параметров, временными метками и слоями данных на глобусе."
      icon={PlugZap}
    >
      <SettingRow
        title="Использовать API"
        description="Если выключить, глобус перейдёт на статические файлы из manifest.json."
      >
        <Toggle
          checked={draft.enabled}
          label={draft.enabled ? 'API включён' : 'API выключен'}
          onChange={(enabled) => onChange({ ...draft, enabled })}
        />
      </SettingRow>

      <SettingRow
        title="Базовый адрес"
        description="Оставьте пустым для локального Vite proxy или укажите адрес сервера без завершающего слеша."
        vertical
      >
        <TextInput
          value={draft.baseUrl}
          placeholder="Например, http://localhost:8000"
          onChange={(baseUrl) => onChange({ ...draft, baseUrl })}
          ariaLabel="Базовый адрес API"
        />
      </SettingRow>

      <SettingRow
        title="Тайм-аут запроса"
        description="После этого времени зависший запрос будет отменён."
      >
        <SelectInput
          value={String(draft.timeoutMs)}
          options={[
            { value: '5000', label: '5 секунд' },
            { value: '15000', label: '15 секунд' },
            { value: '30000', label: '30 секунд' },
            { value: '60000', label: '60 секунд' },
          ]}
          onChange={(timeoutMs) => onChange({ ...draft, timeoutMs: Number(timeoutMs) })}
          ariaLabel="Тайм-аут запросов API"
        />
      </SettingRow>

      <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3.5">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <ConnectionStatus state={connection} />
          <button
            type="button"
            onClick={onTest}
            disabled={connection.status === 'testing'}
            className="ui-button-subtle ui-focus-ring inline-flex h-9 shrink-0 items-center justify-center gap-2 rounded-lg px-3 text-[12px] font-semibold disabled:cursor-wait disabled:opacity-60"
          >
            {connection.status === 'testing' ? (
              <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Wifi className="h-4 w-4" aria-hidden="true" />
            )}
            Проверить соединение
          </button>
        </div>
      </div>

      <SaveBar dirty={dirty} onSave={onSave} onReset={onReset} />
    </SettingsPanel>
  )
}

function SettingsPanel({
  title,
  description,
  icon: Icon,
  children,
}: {
  title: string
  description: string
  icon: LucideIcon
  children: ReactNode
}) {
  return (
    <section>
      <div className="mb-5 flex items-start gap-3 border-b border-slate-200 pb-5">
        <span className="ui-accent-icon inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl">
          <Icon className="h-[18px] w-[18px]" aria-hidden="true" />
        </span>
        <div>
          <h2 className="text-[18px] font-semibold text-[var(--text-primary)]">{title}</h2>
          <p className="mt-1 text-[12px] leading-5 text-[var(--text-muted)]">{description}</p>
        </div>
      </div>
      <div className="space-y-1">{children}</div>
    </section>
  )
}

function SettingRow({
  title,
  description,
  vertical = false,
  children,
}: {
  title: string
  description: string
  vertical?: boolean
  children: ReactNode
}) {
  return (
    <div
      className={`border-b border-[var(--border)] py-4 ${
        vertical ? 'grid gap-3' : 'grid gap-3 md:grid-cols-[minmax(0,1fr)_280px] md:items-center'
      }`}
    >
      <div>
        <h3 className="text-[13px] font-semibold text-[var(--text-secondary)]">{title}</h3>
        <p className="mt-1 max-w-xl text-[11px] leading-5 text-[var(--text-muted)]">{description}</p>
      </div>
      <div className={vertical ? 'max-w-2xl' : ''}>{children}</div>
    </div>
  )
}

function Toggle({
  checked,
  label,
  onChange,
}: {
  checked: boolean
  label: string
  onChange: (checked: boolean) => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`ui-switch ui-focus-ring flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2.5 text-[12px] font-semibold ${
        checked ? 'ui-switch--on' : ''
      }`}
    >
      <span>{label}</span>
      <span className="ui-switch__track" aria-hidden="true">
        <span className="ui-switch__thumb" />
      </span>
    </button>
  )
}

function TextInput({
  value,
  placeholder,
  ariaLabel,
  onChange,
}: {
  value: string
  placeholder: string
  ariaLabel: string
  onChange: (value: string) => void
}) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
      spellCheck={false}
      className="ui-field h-11 w-full rounded-xl px-3.5 text-[13px] font-medium placeholder:text-[var(--text-disabled)]"
    />
  )
}

function SelectInput({
  value,
  options,
  ariaLabel,
  onChange,
}: {
  value: string
  options: Array<{ value: string; label: string }>
  ariaLabel: string
  onChange: (value: string) => void
}) {
  return (
    <span className="relative block">
      <select
        value={value}
        aria-label={ariaLabel}
        onChange={(event) => onChange(event.target.value)}
        className="ui-field h-11 w-full appearance-none rounded-xl px-3.5 pr-10 text-[13px] font-medium"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <ChevronDown
        className="pointer-events-none absolute right-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
        aria-hidden="true"
      />
    </span>
  )
}

function ConnectionStatus({ state }: { state: ConnectionState }) {
  const Icon =
    state.status === 'success'
      ? CheckCircle2
      : state.status === 'error'
        ? XCircle
        : state.status === 'testing'
          ? LoaderCircle
          : Wifi
  const color =
    state.status === 'success'
      ? 'text-[var(--accent)]'
      : state.status === 'error'
        ? 'text-rose-600'
        : 'text-slate-500'

  return (
    <div className={`flex min-w-0 items-center gap-2.5 ${color}`}>
      <Icon
        className={`h-[18px] w-[18px] shrink-0 ${state.status === 'testing' ? 'animate-spin' : ''}`}
        aria-hidden="true"
      />
      <div className="min-w-0">
        <p className="text-[11px] font-semibold uppercase tracking-[0.06em]">Состояние API</p>
        <p className="mt-0.5 truncate text-[12px] font-medium">{state.message}</p>
      </div>
    </div>
  )
}

function SaveBar({
  dirty,
  onSave,
  onReset,
}: {
  dirty: boolean
  onSave: () => void
  onReset: () => void
}) {
  return (
    <div className="flex flex-col-reverse justify-between gap-3 pt-5 sm:flex-row sm:items-center">
      <p className={`text-[11px] font-semibold ${dirty ? 'text-amber-700' : 'text-[var(--accent)]'}`}>
        {dirty ? 'Есть несохранённые изменения' : 'Все изменения сохранены'}
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onReset}
          className="ui-button-ghost ui-focus-ring inline-flex h-10 items-center justify-center rounded-xl px-3.5 text-[12px] font-semibold"
        >
          По умолчанию
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={!dirty}
          className="ui-button-primary ui-focus-ring inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-[12px] font-semibold disabled:cursor-not-allowed disabled:border-[var(--border)] disabled:bg-[var(--surface-active)] disabled:text-[var(--text-disabled)]"
        >
          <Save className="h-4 w-4" aria-hidden="true" />
          Сохранить
        </button>
      </div>
    </div>
  )
}

function settingsEqual<T>(left: T, right: T) {
  return JSON.stringify(left) === JSON.stringify(right)
}
