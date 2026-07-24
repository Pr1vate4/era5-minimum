import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { ProbeCard } from '../components/ProbeCard'
import { ContentCard } from '../components/common/ContentCard'
import { FilterBar, SelectField } from '../components/common/Controls'
import { PageHeader } from '../components/common/PageHeader'
import { StatusBadge } from '../components/common/StatusBadge'
import { selectProbeData } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('probe')

export default function ProbePage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Probe недоступен" message="Результаты эксперимента ещё не загружены." />
  }

  const probe = selectProbeData(data)
  const hasProbeData = Object.entries(probe).some(
    ([key, value]) => key !== 'pass' && value !== undefined,
  )

  return (
    <div className="space-y-5">
      <PageHeader
        title={page.title}
        description={page.description}
        actions={
          <StatusBadge
            label={
              probe.pass === undefined
                ? 'Статус не указан'
                : probe.pass
                  ? 'Проверка пройдена'
                  : 'Проверка не пройдена'
            }
            tone={probe.pass === undefined ? 'neutral' : probe.pass ? 'success' : 'danger'}
          />
        }
      />

      <FilterBar>
        <SelectField
          label="Score"
          value="overall"
          onChange={() => undefined}
          disabled
          options={[{ value: 'overall', label: 'Общий score' }]}
        />
        <SelectField
          label="Канал"
          value="all"
          onChange={() => undefined}
          disabled
          options={[{ value: 'all', label: 'Сводный результат' }]}
        />
        <SelectField
          label="Сетка"
          value={data.meta.grid ?? 'unknown'}
          onChange={() => undefined}
          disabled
          options={[{ value: data.meta.grid ?? 'unknown', label: data.meta.grid ?? 'Нет данных' }]}
        />
        <SelectField
          label="Размер выборки"
          value="unknown"
          onChange={() => undefined}
          disabled
          options={[{ value: 'unknown', label: 'Не указан' }]}
        />
      </FilterBar>

      <ContentCard>
        <p className="text-[13px] leading-6 text-slate-600">
          Персистентность предполагает, что через 6 часов состояние атмосферы не изменится.
        </p>
      </ContentCard>

      {hasProbeData ? (
        <ProbeCard probe={probe} />
      ) : (
        <EmptyState
          title="Данные latent-прогноза отсутствуют"
          message="Для страницы нужен объект probe_forecast с фактическими результатами probe-задачи."
        />
      )}

    </div>
  )
}
