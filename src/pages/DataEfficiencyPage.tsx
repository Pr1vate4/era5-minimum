import { Activity, CheckCircle2, Database, TrendingDown } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { getNavigationItem } from '../app/navigationConfig'
import {
  DataEfficiencyChart,
  type DataEfficiencyMetric,
} from '../components/DataEfficiencyChart'
import { EmptyState } from '../components/EmptyState'
import { MetricCard } from '../components/MetricCard'
import { ChartCard, ContentCard } from '../components/common/ContentCard'
import {
  CheckboxField,
  FilterBar,
  SegmentedControl,
  SelectField,
} from '../components/common/Controls'
import { PageHeader } from '../components/common/PageHeader'
import { selectDataEfficiencyData, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('data-efficiency')
type ScoreMode = 'overall' | 'surface' | 'pressure'

export default function DataEfficiencyPage() {
  const { data } = useResults()
  const [searchParams, setSearchParams] = useSearchParams()

  if (!data) {
    return <EmptyState title="График недоступен" message="Результаты эксперимента ещё не загружены." />
  }

  const points = selectDataEfficiencyData(data)
  const hasSurface = points.some((point) => point.surface_nrmse !== undefined)
  const hasPressure = points.some((point) => point.pressure_nrmse !== undefined)
  const hasReference = points.some((point) => point.reference_nrmse !== undefined)
  const hasConfidenceInterval = points.some(
    (point) => point.ci_low !== undefined && point.ci_high !== undefined,
  )
  const requestedScore = searchParams.get('score')
  const score: ScoreMode =
    requestedScore === 'surface' && hasSurface
      ? 'surface'
      : requestedScore === 'pressure' && hasPressure
        ? 'pressure'
        : 'overall'
  const showReference = searchParams.get('reference') === '1' && hasReference
  const showConfidence = searchParams.get('ci') === '1' && hasConfidenceInterval
  const chartMetric: DataEfficiencyMetric =
    score === 'surface'
      ? 'surface_nrmse'
      : score === 'pressure'
        ? 'pressure_nrmse'
        : 'overall_nrmse'
  const bestPoint = points.reduce<(typeof points)[number] | undefined>(
    (best, point) => (!best || point.overall_nrmse < best.overall_nrmse ? point : best),
    undefined,
  )
  const minimumPassed = points.find((point) => point.pass === true)
  const lastPoint = points[points.length - 1]
  const previousPoint = points[points.length - 2]
  const adjacentImprovement =
    lastPoint && previousPoint ? previousPoint.overall_nrmse - lastPoint.overall_nrmse : undefined

  const updateFlag = (key: string, enabled: boolean) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (enabled) next.set(key, '1')
        else next.delete(key)
        return next
      },
      { replace: true },
    )
  }

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      <ContentCard>
        <p className="text-[15px] font-semibold text-slate-900">
          Какое минимальное количество уникальных погодных кадров достаточно для качественного обучения модели?
        </p>
      </ContentCard>

      <FilterBar>
        <SelectField
          label="Сетка"
          value={data.meta.grid ?? 'unknown'}
          onChange={() => undefined}
          disabled
          options={[{ value: data.meta.grid ?? 'unknown', label: data.meta.grid ?? 'Нет данных' }]}
        />
        <SelectField
          label="Сжатие"
          value={String(toFiniteNumber(data.compression.target_ratio) ?? 'unknown')}
          onChange={() => undefined}
          disabled
          options={[
            {
              value: String(toFiniteNumber(data.compression.target_ratio) ?? 'unknown'),
              label:
                toFiniteNumber(data.compression.target_ratio) === undefined
                  ? 'Нет данных'
                  : `${toFiniteNumber(data.compression.target_ratio)}×`,
            },
          ]}
        />
        <SegmentedControl
          label="Score"
          value={score}
          onChange={(value) => {
            const next = new URLSearchParams(searchParams)
            if (value === 'overall') next.delete('score')
            else next.set('score', value)
            setSearchParams(next, { replace: true })
          }}
          options={[
            { value: 'overall', label: 'Общий' },
            { value: 'surface', label: 'Surface', disabled: !hasSurface },
            { value: 'pressure', label: 'Pressure', disabled: !hasPressure },
          ]}
        />
        <CheckboxField
          label="Доверительный интервал"
          checked={showConfidence}
          disabled={!hasConfidenceInterval}
          onChange={(checked) => updateFlag('ci', checked)}
        />
        <CheckboxField
          label="Референс"
          checked={showReference}
          disabled={!hasReference}
          onChange={(checked) => updateFlag('reference', checked)}
        />
        <CheckboxField label="Порог допуска отсутствует" checked={false} disabled onChange={() => undefined} />
      </FilterBar>

      {points.length === 0 ? (
        <EmptyState
          title="Нет данных об эффективности выборки"
          message="Нужен непустой массив data_efficiency с положительными n_samples и числовым overall_nrmse."
        />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Минимум, прошедший критерии"
              value={minimumPassed ? minimumPassed.n_samples.toLocaleString('ru-RU') : 'Не отмечен'}
              caption="Требуется поле pass у точки"
              icon={<CheckCircle2 className="h-4 w-4" />}
            />
            <MetricCard
              title="Лучший NRMSE"
              value={bestPoint?.overall_nrmse.toFixed(4) ?? 'Нет данных'}
              caption={bestPoint ? `${bestPoint.n_samples.toLocaleString('ru-RU')} кадров` : undefined}
              icon={<Activity className="h-4 w-4" />}
            />
            <MetricCard
              title="Максимальная выборка"
              value={lastPoint?.n_samples.toLocaleString('ru-RU') ?? 'Нет данных'}
              caption="Уникальные временные кадры"
              icon={<Database className="h-4 w-4" />}
            />
            <MetricCard
              title="Изменение на последнем шаге"
              value={adjacentImprovement === undefined ? 'Нет данных' : adjacentImprovement.toFixed(4)}
              caption="Снижение NRMSE относительно предыдущего размера"
              icon={<TrendingDown className="h-4 w-4" />}
            />
          </div>

          <ChartCard
            title="NRMSE в зависимости от размера выборки"
            description="Ось X логарифмическая; нулевые и нечисловые точки исключены селектором."
          >
            <DataEfficiencyChart
              data={points}
              metric={chartMetric}
              showReference={showReference}
              showConfidence={showConfidence}
            />
          </ChartCard>

          <ContentCard title="Автоматический вывод">
            <p className="text-[13px] leading-6 text-slate-600">
              {bestPoint
                ? `В доступной серии минимальный общий NRMSE равен ${bestPoint.overall_nrmse.toFixed(4)} при ${bestPoint.n_samples.toLocaleString('ru-RU')} уникальных кадрах. Определить минимальную прошедшую выборку нельзя без поля pass или отдельного порога для каждой точки.`
                : 'Недостаточно данных для вывода.'}
            </p>
          </ContentCard>
        </>
      )}
    </div>
  )
}
