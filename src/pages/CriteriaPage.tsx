import { BadgeCheck, FileArchive, Gauge, ShieldCheck } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { getNavigationItem } from '../app/navigationConfig'
import { CriteriaPanel } from '../components/CriteriaPanel'
import { EmptyState } from '../components/EmptyState'
import { MetricCard } from '../components/MetricCard'
import { ContentCard } from '../components/common/ContentCard'
import { FilterBar, SegmentedControl } from '../components/common/Controls'
import { PageHeader } from '../components/common/PageHeader'
import { StatusBadge } from '../components/common/StatusBadge'
import { selectCriteriaData } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('criteria')
type CompressionMode = '32' | '64'

export default function CriteriaPage() {
  const { data } = useResults()
  const [searchParams, setSearchParams] = useSearchParams()

  if (!data) {
    return <EmptyState title="Критерии недоступны" message="Результаты эксперимента ещё не загружены." />
  }

  const criteriaData = selectCriteriaData(data)
  const defaultMode: CompressionMode = criteriaData.targetRatio === 64 ? '64' : '32'
  const requestedMode = searchParams.get('compression')
  const compression: CompressionMode = requestedMode === '64' || requestedMode === '32' ? requestedMode : defaultMode
  const selectedRatio = Number(compression)
  const criteriaMatchSelectedMode = criteriaData.targetRatio === selectedRatio
  const passedCount = criteriaData.criteria.filter((criterion) => criterion.pass === true).length
  const allPassed = criteriaData.criteria.length > 0 && passedCount === criteriaData.criteria.length

  const setCompression = (value: CompressionMode) => {
    setSearchParams({ compression: value }, { replace: true })
  }

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      <FilterBar>
        <SegmentedControl
          label="Режим сжатия"
          value={compression}
          onChange={setCompression}
          options={[
            { value: '32', label: '32×' },
            { value: '64', label: '64×' },
          ]}
        />
        <div className="ml-auto self-center">
          <StatusBadge
            label={
              criteriaMatchSelectedMode
                ? allPassed
                  ? 'Требования пройдены'
                  : 'Есть непройденные требования'
                : 'Данных для режима нет'
            }
            tone={criteriaMatchSelectedMode ? (allPassed ? 'success' : 'danger') : 'neutral'}
          />
        </div>
      </FilterBar>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Выбранный режим"
          value={`${compression}×`}
          caption={
            criteriaData.targetRatio === undefined
              ? 'Целевой режим запуска не указан'
              : `Цель текущего запуска: ${criteriaData.targetRatio}×`
          }
          icon={<Gauge className="h-4 w-4" />}
        />
        <MetricCard
          title="Фактическое сжатие"
          value={
            criteriaData.actualRatio === undefined ? 'Нет данных' : `${criteriaData.actualRatio.toFixed(1)}×`
          }
          icon={<ShieldCheck className="h-4 w-4" />}
        />
        <MetricCard
          title="Настоящий bitstream"
          value={
            criteriaData.bitstreamBytes === undefined
              ? 'Нет данных'
              : criteriaData.bitstreamBytes > 0
                ? 'Доступен'
                : 'Не подтверждён'
          }
          caption={
            criteriaData.bitstreamBytes && criteriaData.bitstreamBytes > 0
              ? `${criteriaData.bitstreamBytes.toLocaleString('ru-RU')} байт`
              : undefined
          }
          icon={<FileArchive className="h-4 w-4" />}
        />
        <MetricCard
          title="Exact roundtrip"
          value={
            criteriaData.roundtripExact === undefined
              ? 'Нет данных'
              : criteriaData.roundtripExact
                ? 'Подтверждён'
                : 'Не подтверждён'
          }
          icon={<BadgeCheck className="h-4 w-4" />}
        />
      </div>

      <ContentCard
        title={`Критерии для режима ${compression}×`}
        description="Таблица показывает только фактические значения из results.json."
      >
        {criteriaMatchSelectedMode ? (
          <CriteriaPanel criteria={criteriaData.criteria} />
        ) : (
          <EmptyState
            title={`Нет результатов для ${compression}×`}
            message={`Текущий results.json относится к режиму ${criteriaData.targetRatio ?? 'с неуказанным target_ratio'}×. Пороговые значения другого режима не подменяются и не пересчитываются.`}
          />
        )}
      </ContentCard>
    </div>
  )
}
