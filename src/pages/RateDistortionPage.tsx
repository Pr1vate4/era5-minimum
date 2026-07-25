import { CircleGauge, FileArchive, Scale, Target } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { getNavigationItem } from '../app/navigationConfig'
import {
  RateDistortionChart,
  type RateDistortionMetric,
} from '../components/RateDistortionChart'
import { EmptyState } from '../components/EmptyState'
import { MetricCard } from '../components/MetricCard'
import { ChartCard } from '../components/common/ContentCard'
import {
  CheckboxField,
  FilterBar,
  SegmentedControl,
  SelectField,
} from '../components/common/Controls'
import { PageHeader } from '../components/common/PageHeader'
import { selectRateDistortionData, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('rate-distortion')

export default function RateDistortionPage() {
  const { data } = useResults()
  const [searchParams, setSearchParams] = useSearchParams()

  if (!data) {
    return <EmptyState title="Кривая недоступна" message="Результаты эксперимента ещё не загружены." />
  }

  const points = selectRateDistortionData(data)
  const availability: Record<RateDistortionMetric, boolean> = {
    overall_nrmse: points.some((point) => point.overall_nrmse !== undefined),
    surface_nrmse: points.some((point) => point.surface_nrmse !== undefined),
    pressure_nrmse: points.some((point) => point.pressure_nrmse !== undefined),
    psnr: points.some((point) => point.psnr !== undefined),
  }
  const requestedMetric = searchParams.get('metric') as RateDistortionMetric | null
  const metric: RateDistortionMetric =
    requestedMetric && availability[requestedMetric] ? requestedMetric : 'overall_nrmse'
  const hasReference = points.some((point) => point.reference_nrmse !== undefined)
  const showReference = hasReference && searchParams.get('reference') === '1'
  const targetRatio = toFiniteNumber(data.compression.target_ratio)
  const actualRatio = toFiniteNumber(data.compression.actual_ratio)
  const bitstreamBytes = toFiniteNumber(data.compression.bitstream_bytes)
  const nearestTarget =
    targetRatio === undefined
      ? undefined
      : points.reduce<(typeof points)[number] | undefined>(
          (nearest, point) =>
            !nearest ||
            Math.abs(point.compression_ratio - targetRatio) <
              Math.abs(nearest.compression_ratio - targetRatio)
              ? point
              : nearest,
          undefined,
        )

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      <FilterBar>
        <SegmentedControl
          label="Метрика"
          value={metric}
          onChange={(value) => {
            const next = new URLSearchParams(searchParams)
            if (value === 'overall_nrmse') next.delete('metric')
            else next.set('metric', value)
            setSearchParams(next, { replace: true })
          }}
          options={[
            { value: 'overall_nrmse', label: 'Общий NRMSE' },
            { value: 'surface_nrmse', label: 'Surface', disabled: !availability.surface_nrmse },
            { value: 'pressure_nrmse', label: 'Pressure', disabled: !availability.pressure_nrmse },
            { value: 'psnr', label: 'PSNR', disabled: !availability.psnr },
          ]}
        />
        <SelectField
          label="Сетка"
          value={data.meta.grid ?? 'unknown'}
          onChange={() => undefined}
          disabled
          options={[{ value: data.meta.grid ?? 'unknown', label: data.meta.grid ?? 'Нет данных' }]}
        />
        <CheckboxField
          label="Референс"
          checked={showReference}
          disabled={!hasReference}
          onChange={(checked) => {
            const next = new URLSearchParams(searchParams)
            if (checked) next.set('reference', '1')
            else next.delete('reference')
            setSearchParams(next, { replace: true })
          }}
        />
        <CheckboxField label="Порог не указан" checked={false} disabled onChange={() => undefined} />
      </FilterBar>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Целевое сжатие"
          value={targetRatio === undefined ? 'Нет данных' : `${targetRatio}×`}
          icon={<Target className="h-4 w-4" />}
        />
        <MetricCard
          title="Фактическое сжатие"
          value={actualRatio === undefined ? 'Нет данных' : `${actualRatio.toFixed(1)}×`}
          caption="По сериализованному bitstream"
          icon={<CircleGauge className="h-4 w-4" />}
        />
        <MetricCard
          title="Размер bitstream"
          value={bitstreamBytes === undefined ? 'Нет данных' : `${bitstreamBytes.toLocaleString('ru-RU')} Б`}
          icon={<FileArchive className="h-4 w-4" />}
        />
        <MetricCard
          title="Ближайшая к цели точка"
          value={nearestTarget ? `${nearestTarget.compression_ratio}×` : 'Нет данных'}
          caption={
            nearestTarget ? `Общий NRMSE ${nearestTarget.overall_nrmse.toFixed(4)}` : 'Цель не указана'
          }
          icon={<Scale className="h-4 w-4" />}
        />
      </div>

      {points.length === 0 ? (
        <EmptyState
          title="Нет rate–distortion данных"
          message="Нужен непустой массив rate_distortion с числовыми compression_ratio и overall_nrmse."
        />
      ) : (
        <ChartCard
          title="Кривая rate–distortion"
          description="Синий — модель. Бирюзовый референс отображается только при наличии reference_nrmse."
        >
          <RateDistortionChart data={points} metric={metric} showReference={showReference} />
        </ChartCard>
      )}
    </div>
  )
}
