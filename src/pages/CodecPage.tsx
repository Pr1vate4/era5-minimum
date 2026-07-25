import {
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react'
import { getNavigationItem } from '../app/navigationConfig'
import { ContentCard } from '../components/common/ContentCard'
import { PageHeader } from '../components/common/PageHeader'
import { buildApiUrl } from '../app/settings'
import { CodecDropzone } from '../features/codec/CodecDropzone'
import { CodecProgress, CodecResults } from '../features/codec/CodecResults'
import { useCodecWorkspace } from '../features/codec/useCodecWorkspace'
import { useAppSettings } from '../hooks/useAppSettings'
import type { CodecTargetRatio } from '../features/codec/types'

const page = getNavigationItem('codec')

export default function CodecPage() {
  const workspace = useCodecWorkspace()
  const { settings } = useAppSettings()
  const canSubmit =
    Boolean(workspace.file) &&
    !workspace.fileError &&
    workspace.service?.ready === true &&
    !workspace.processing

  return (
    <div className="space-y-5">
      <PageHeader
        title={page.title}
        actions={
          <ServiceBadge
            loading={workspace.serviceLoading}
            ready={workspace.service?.ready === true}
            label={
              workspace.serviceLoading
                ? 'Проверяем модель'
                : workspace.service?.ready
                  ? 'Модель готова'
                  : 'Модель недоступна'
            }
          />
        }
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.55fr)]">
        <ContentCard
          title="Новый запуск"
            description="Подключён профиль N128-equivalent (32 фактических кадра). Фактический serialized ratio сервер вычисляет отдельно по полному bitstream."
          className="ui-panel !rounded-[24px] !p-5"
        >
          <CodecDropzone
            file={workspace.file}
            error={workspace.fileError}
            disabled={workspace.processing}
            onFileChange={workspace.setFile}
          />

          <div className="mt-5 grid gap-4 border-t border-[var(--border)] pt-5 sm:grid-cols-[1fr_auto] sm:items-end">
            <fieldset>
              <legend className="text-[12px] font-bold text-[var(--text-secondary)]">
                Профиль модели
              </legend>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {([32, 64] as CodecTargetRatio[]).map((ratio) => (
                  <button
                    key={ratio}
                    type="button"
                    disabled={workspace.processing || ratio !== 32}
                    onClick={() => workspace.setTargetRatio(ratio)}
                    className={`codec-ratio ui-focus-ring ${
                      workspace.targetRatio === ratio ? 'codec-ratio--active' : ''
                    }`}
                  >
                    <strong>{ratio}×</strong>
                  </button>
                ))}
              </div>
            </fieldset>

            <button
              type="button"
              disabled={!canSubmit}
              onClick={() => void workspace.submit()}
              className="ui-action ui-action-primary min-w-[190px]"
            >
              {workspace.processing ? (
                <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : null}
              {workspace.processing ? 'Обрабатываем…' : 'Запустить сжатие'}
            </button>
          </div>
        </ContentCard>

        <ContentCard
          title="Готовность сервиса"
          description="Фронтенд не подменяет отсутствующую модель демо-результатами."
          className="ui-panel !rounded-[24px] !p-5"
        >
          <ServiceDetails workspace={workspace} />
        </ContentCard>
      </div>

      {workspace.job && workspace.processing ? <CodecProgress job={workspace.job} /> : null}

      {workspace.runError ? (
        <div className="flex items-start gap-3 rounded-[18px] border border-rose-200 bg-rose-50 p-4 text-rose-800" role="alert">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <div>
            <p className="text-[13px] font-bold">
              {workspace.processing ? 'Связь с задачей прервана' : 'Сжатие не выполнено'}
            </p>
            <p className="mt-1 text-[12px] leading-5">{workspace.runError}</p>
          </div>
        </div>
      ) : null}

      {workspace.job?.status === 'completed' && workspace.file ? (
        <ContentCard
          title="Результат"
          description="Размеры и ratios получены из завершённого backend job."
          className="ui-panel !rounded-[24px] !p-5"
        >
          <CodecResults
            job={workspace.job}
            sourceFile={workspace.file}
            resolveDownload={(path) =>
              /^(https?:)?\/\//.test(path)
                ? path
                : buildApiUrl(settings.services.codecBaseUrl, path)
            }
          />
        </ContentCard>
      ) : null}
    </div>
  )
}

function ServiceBadge({
  loading,
  ready,
  label,
}: {
  loading: boolean
  ready: boolean
  label: string
}) {
  return (
    <span className={`service-badge ${ready ? 'service-badge--ready' : ''}`}>
      {loading ? (
        <LoaderCircle className="h-4 w-4 animate-spin" />
      ) : ready ? (
        <CheckCircle2 className="h-4 w-4" />
      ) : (
        <ShieldAlert className="h-4 w-4" />
      )}
      {label}
    </span>
  )
}

function ServiceDetails({ workspace }: { workspace: ReturnType<typeof useCodecWorkspace> }) {
  if (workspace.serviceLoading) {
    return (
      <div className="flex items-center gap-3 text-[13px] text-[var(--text-muted)]">
        <LoaderCircle className="h-5 w-5 animate-spin text-[var(--accent)]" />
        Проверяем codec API и checkpoint…
      </div>
    )
  }
  if (workspace.serviceError) {
    return (
      <div>
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
          <div>
            <p className="text-[13px] font-bold text-[var(--text-primary)]">Нет соединения</p>
            <p className="mt-1 text-[12px] leading-5 text-[var(--text-muted)]">
              {workspace.serviceError}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={workspace.retryService}
          className="ui-action ui-action-secondary mt-4 w-full"
        >
          <RefreshCw className="h-4 w-4" />
          Повторить проверку
        </button>
      </div>
    )
  }
  return (
    <dl className="space-y-3 text-[12px]">
      <ServiceRow label="Статус" value={workspace.service?.message ?? 'Нет данных'} />
      <ServiceRow label="Модель" value={workspace.service?.modelName ?? 'Не объявлена'} />
      <ServiceRow label="Checkpoint" value={workspace.service?.checkpoint ?? 'Не подключён'} />
    </dl>
  )
}

function ServiceRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-[var(--background-subtle)] p-3">
      <dt className="font-bold uppercase tracking-[0.1em] text-[10px] text-[var(--text-muted)]">{label}</dt>
      <dd className="mt-1 break-all font-semibold leading-5 text-[var(--text-secondary)]">{value}</dd>
    </div>
  )
}
