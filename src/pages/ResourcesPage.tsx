import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { ResourcesCard } from '../components/ResourcesCard'
import { ContentCard } from '../components/common/ContentCard'
import { PageHeader } from '../components/common/PageHeader'
import { selectResourcesData, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('resources')

export default function ResourcesPage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Ресурсы недоступны" message="Результаты эксперимента ещё не загружены." />
  }

  const resources = selectResourcesData(data)
  const hasResourceData = [
    resources.gpuCount,
    resources.gpuHours,
    resources.peakVramGb,
    resources.paramsMillions,
    resources.optimizerSteps,
    resources.probeParamsMillions,
    resources.probeSteps,
  ].some((value) => value !== undefined)

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      {hasResourceData ? (
        <ResourcesCard resources={resources} />
      ) : (
        <EmptyState
          title="Фактические ресурсы отсутствуют"
          message="Лимиты известны из технического задания, но без объекта resources нельзя показывать использование."
        />
      )}

      <ContentCard
        title="Окружение запуска"
        description="Неуказанные поля остаются пустыми и не подменяются предположениями."
      >
        <dl className="grid gap-x-8 gap-y-3 text-[13px] sm:grid-cols-2 xl:grid-cols-3">
          <InfoRow label="GPU" value={resources.meta.gpu_name} />
          <InfoRow label="Длительность обучения" value={resources.meta.training_duration} />
          <InfoRow label="Окружение" value={resources.meta.environment} />
          <InfoRow
            label="Внешний pretraining"
            value={
              resources.meta.external_pretraining === undefined
                ? undefined
                : resources.meta.external_pretraining
                  ? 'Использовался'
                  : 'Не использовался'
            }
          />
          <InfoRow label="Сетка" value={resources.meta.grid} />
          <InfoRow
            label="Batch size"
            value={formatOptionalNumber(resources.meta.batch_size)}
          />
          <InfoRow
            label="Уникальные кадры"
            value={formatOptionalNumber(resources.meta.training_samples)}
          />
          <InfoRow
            label="Обработанные примеры"
            value={formatOptionalNumber(resources.meta.processed_examples)}
          />
        </dl>
      </ContentCard>
    </div>
  )
}

function formatOptionalNumber(value: number | string | undefined) {
  const parsed = toFiniteNumber(value)
  return parsed?.toLocaleString('ru-RU')
}

function InfoRow({ label, value }: { label: string; value?: string }) {
  return (
    <div className="border-b border-slate-100 pb-2">
      <dt className="text-[11px] font-semibold text-slate-500">{label}</dt>
      <dd className="mt-1 font-medium text-slate-800">{value ?? 'Нет данных'}</dd>
    </div>
  )
}
