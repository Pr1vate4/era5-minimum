import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import { chartTheme } from '../data/chartTheme'
import type { selectProbeData } from '../data/resultsSelectors'
import { EmptyState } from './EmptyState'
import { ChartTooltip } from './ChartTooltip'
import { MetricCard } from './MetricCard'

type ProbeData = ReturnType<typeof selectProbeData>

export function ProbeCard({ probe }: { probe: ProbeData }) {
  const comparison = [
    { name: 'Latent-модель', nrmse: probe.latentNrmse },
    { name: 'Персистентность', nrmse: probe.persistenceNrmse },
    { name: 'Референсные признаки', nrmse: probe.referenceProbeNrmse },
  ].filter((entry): entry is { name: string; nrmse: number } => entry.nrmse !== undefined)

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Обучающие пары"
          value={probe.nPairs?.toLocaleString('ru-RU') ?? 'Нет данных'}
        />
        <MetricCard
          title="Шаги probe"
          value={probe.steps?.toLocaleString('ru-RU') ?? 'Нет данных'}
        />
        <MetricCard
          title="Параметры probe"
          value={probe.paramsMillions === undefined ? 'Нет данных' : `${probe.paramsMillions} млн`}
        />
        <MetricCard
          title="NRMSE latent-модели"
          value={probe.latentNrmse?.toFixed(4) ?? 'Нет данных'}
        />
        <MetricCard
          title="NRMSE персистентности"
          value={probe.persistenceNrmse?.toFixed(4) ?? 'Нет данных'}
        />
        <MetricCard
          title="NRMSE референсного probe"
          value={probe.referenceProbeNrmse?.toFixed(4) ?? 'Нет данных'}
        />
        <MetricCard
          title="Улучшение к персистентности"
          value={
            probe.improvementVsPersistencePct === undefined
              ? 'Нет данных'
              : `${probe.improvementVsPersistencePct.toFixed(1)}%`
          }
        />
        <MetricCard
          title="Улучшение к референсному probe"
          value={
            probe.improvementVsReferencePct === undefined
              ? 'Нет данных'
              : `${probe.improvementVsReferencePct.toFixed(1)}%`
          }
        />
      </div>

      {comparison.length >= 2 ? (
        <div className="h-[320px] rounded-2xl border border-slate-200 bg-white p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparison} margin={{ top: 12, right: 12, bottom: 24, left: 4 }}>
              <CartesianGrid vertical={false} stroke={chartTheme.grid} strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill: chartTheme.axis, fontSize: 11 }} />
              <YAxis tick={{ fill: chartTheme.axis, fontSize: 11 }} />
              <ChartTooltip />
              <Bar dataKey="nrmse" name="NRMSE" fill={chartTheme.model} radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyState
          title="Сравнительный график недоступен"
          message="Для него нужны как минимум два NRMSE: latent-модели, персистентности или probe на референсных признаках."
        />
      )}
    </div>
  )
}
