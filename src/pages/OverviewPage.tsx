import {
  Activity,
  BadgeCheck,
  CircleGauge,
  Database,
  FileArchive,
  Layers3,
  Radar,
  Rows3,
} from 'lucide-react'
import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { MetricCard } from '../components/MetricCard'
import { ContentCard } from '../components/common/ContentCard'
import { PageHeader } from '../components/common/PageHeader'
import { StatusBadge } from '../components/common/StatusBadge'
import { formatNumber, selectOverviewData, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('overview')

export default function OverviewPage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Сводка недоступна" message="Результаты эксперимента ещё не загружены." />
  }

  const overview = selectOverviewData(data)
  const trainingSamples = toFiniteNumber(overview.meta.training_samples)
  const runId = overview.meta.run_id ?? 'Run ID не указан'
  const generatedAt = formatDate(overview.meta.generated_at)

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      <ContentCard className="border-blue-100">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-blue-600">
              Эксперимент нейросетевого сжатия ERA5
            </div>
            <h2 className="mt-2 text-[21px] font-semibold text-slate-900">
              {overview.meta.checkpoint ?? runId}
            </h2>
            <p className="mt-1 text-[13px] text-slate-500">
              {runId} · сформировано {generatedAt}
            </p>
          </div>
          <StatusBadge
            label={
              overview.criteriaCount === 0
                ? 'Критерии не загружены'
                : overview.allCriteriaPassed
                  ? 'Все критерии пройдены'
                  : 'Есть непройденные критерии'
            }
            tone={
              overview.criteriaCount === 0
                ? 'neutral'
                : overview.allCriteriaPassed
                  ? 'success'
                  : 'danger'
            }
          />
        </div>
      </ContentCard>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Фактическое сжатие"
          value={overview.actualRatio === undefined ? 'Нет данных' : `${overview.actualRatio.toFixed(1)}×`}
          caption={
            overview.targetRatio === undefined ? 'Цель не указана' : `Целевой режим: ${overview.targetRatio}×`
          }
          icon={<CircleGauge className="h-4 w-4" />}
        />
        <MetricCard
          title="Размер bitstream"
          value={formatBytes(overview.bitstreamBytes)}
          caption={overview.bitstreamAvailable ? 'Сериализованный поток доступен' : 'Bitstream не подтверждён'}
          icon={<FileArchive className="h-4 w-4" />}
        />
        <MetricCard
          title="Exact roundtrip"
          value={
            overview.roundtripExact === undefined
              ? 'Нет данных'
              : overview.roundtripExact
                ? 'Подтверждён'
                : 'Не подтверждён'
          }
          caption="Проверка квантованных символов"
          icon={<BadgeCheck className="h-4 w-4" />}
        />
        <MetricCard
          title="Общий NRMSE"
          value={formatNumber(overview.overallNrmse, 4)}
          caption={`Surface ${formatNumber(overview.surfaceNrmse, 4)} · Pressure ${formatNumber(overview.pressureNrmse, 4)}`}
          icon={<Activity className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Сетка"
          value={overview.meta.grid ?? 'Нет данных'}
          icon={<Radar className="h-4 w-4" />}
        />
        <MetricCard
          title="Каналы по контракту"
          value="28"
          caption={`${data.per_channel.length} каналов имеют метрики в текущем JSON`}
          icon={<Layers3 className="h-4 w-4" />}
        />
        <MetricCard
          title="Уровни давления"
          value="1000 / 925 / 850 / 700"
          caption="hPa"
          icon={<Rows3 className="h-4 w-4" />}
        />
        <MetricCard
          title="Обучающая выборка"
          value={trainingSamples === undefined ? 'Нет данных' : trainingSamples.toLocaleString('ru-RU')}
          caption="Уникальные временные кадры"
          icon={<Database className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.35fr_1fr]">
        <ContentCard title="Итоговый вывод">
          <p className="text-[13px] leading-6 text-slate-600">
            {buildConclusion(overview)}
          </p>
        </ContentCard>
        <ContentCard title="Контрольный ориентир">
          <dl className="grid gap-3 text-[13px]">
            <InfoRow label="Референс" value="VAEformer / CRA5" />
            <InfoRow
              label="Критерии"
              value={`${overview.passedCriteria} из ${overview.criteriaCount || 0} пройдено`}
            />
            <InfoRow label="Команда" value={overview.meta.team ?? 'Нет данных'} />
          </dl>
        </ContentCard>
      </div>
    </div>
  )
}

function buildConclusion(overview: ReturnType<typeof selectOverviewData>) {
  if (overview.actualRatio === undefined) {
    return 'В results.json отсутствует фактический compression ratio, поэтому итог по сжатию сформировать нельзя.'
  }

  const ratioText = `Фактическое сжатие ${overview.actualRatio.toFixed(1)}×`
  const bitstreamText = overview.bitstreamAvailable
    ? 'рассчитано по доступному сериализованному bitstream'
    : 'не подтверждено ненулевым размером bitstream'
  const criteriaText =
    overview.criteriaCount === 0
      ? 'Критерии допуска в файле отсутствуют.'
      : overview.allCriteriaPassed
        ? 'Все доступные критерии допуска пройдены.'
        : 'Часть доступных критериев допуска не пройдена.'

  return `${ratioText} ${bitstreamText}. ${criteriaText}`
}

function formatBytes(value: number | undefined) {
  if (value === undefined) return 'Нет данных'
  if (value < 1024) return `${value.toLocaleString('ru-RU')} Б`
  return `${(value / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} КБ`
}

function formatDate(value: string | undefined) {
  if (!value) return 'дата не указана'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('ru-RU', { dateStyle: 'long', timeZone: 'UTC' }).format(date)
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-2 last:border-0 last:pb-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-semibold text-slate-800">{value}</dd>
    </div>
  )
}
