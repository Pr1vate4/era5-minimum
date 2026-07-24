import { Activity, CheckCircle2, Waves } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { MetricCard } from '../components/MetricCard'
import { SpectralChart } from '../components/SpectralChart'
import { ChartCard, ContentCard, DataTable } from '../components/common/ContentCard'
import {
  CheckboxField,
  FilterBar,
  SelectField,
} from '../components/common/Controls'
import { PageHeader } from '../components/common/PageHeader'
import { StatusBadge } from '../components/common/StatusBadge'
import { selectSpectralData, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('spectral')
type RangeMode = 'all' | '1-8' | '9+'

export default function SpectralPage() {
  const { data } = useResults()
  const [searchParams, setSearchParams] = useSearchParams()

  if (!data) {
    return <EmptyState title="Спектр недоступен" message="Результаты эксперимента ещё не загружены." />
  }

  const points = selectSpectralData(data)
  const channels = Array.from(
    new Set(points.map((point) => point.channel).filter((value): value is string => Boolean(value))),
  )
  const requestedChannel = searchParams.get('channel')
  const channel = requestedChannel && channels.includes(requestedChannel) ? requestedChannel : 'all'
  const requestedRange = searchParams.get('range')
  const range: RangeMode =
    requestedRange === '1-8' || requestedRange === '9+' ? requestedRange : 'all'
  const showRelative = searchParams.get('relative') === '1'
  const filtered = points.filter((point) => {
    if (channel !== 'all' && point.channel !== channel) return false
    if (range === '1-8' && point.wavenumber > 8) return false
    if (range === '9+' && point.wavenumber < 9) return false
    return true
  })
  const spectralError = toFiniteNumber(data.scores.spectral_error_pct)
  const spectralCriterion = data.criteria.find((criterion) => /spectral/i.test(criterion.name))

  const updateParam = (key: string, value: string, defaultValue?: string) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (!value || value === defaultValue) next.delete(key)
        else next.set(key, value)
        return next
      },
      { replace: true },
    )
  }

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      <FilterBar>
        <SelectField
          label="Канал"
          value={channel}
          onChange={(value) => updateParam('channel', value, 'all')}
          disabled={channels.length === 0}
          options={
            channels.length
              ? [{ value: 'all', label: 'Все каналы' }, ...channels.map((value) => ({ value, label: value }))]
              : [{ value: 'all', label: 'Сводный спектр' }]
          }
        />
        <SelectField
          label="Диапазон волновых чисел"
          value={range}
          onChange={(value) => updateParam('range', value, 'all')}
          options={[
            { value: 'all', label: 'Весь диапазон' },
            { value: '1-8', label: '1–8' },
            { value: '9+', label: '9 и выше' },
          ]}
        />
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
        <CheckboxField
          label="Относительная ошибка"
          checked={showRelative}
          onChange={(checked) => updateParam('relative', checked ? '1' : '', '')}
        />
      </FilterBar>

      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard
          title="Spectral error"
          value={spectralError === undefined ? 'Нет данных' : `${spectralError.toFixed(2)}%`}
          icon={<Activity className="h-4 w-4" />}
        />
        <MetricCard
          title="Допустимый порог"
          value={spectralCriterion?.target ?? 'Не указан'}
          icon={<Waves className="h-4 w-4" />}
        />
        <MetricCard
          title="Статус"
          value={
            spectralCriterion?.pass === undefined
              ? 'Нет данных'
              : spectralCriterion.pass
                ? 'Пройдено'
                : 'Не пройдено'
          }
          icon={<CheckCircle2 className="h-4 w-4" />}
        />
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          title="Спектральные точки отсутствуют"
          message="Нужны положительные wavenumber, reference_energy и model_energy для выбранных фильтров."
        />
      ) : (
        <>
          <ChartCard
            title="Спектральная энергия"
            description="Обе оси логарифмические. Синий — модель, бирюзовый — эталон."
            actions={
              spectralCriterion?.pass === undefined ? null : (
                <StatusBadge
                  label={spectralCriterion.pass ? 'Критерий пройден' : 'Критерий не пройден'}
                  tone={spectralCriterion.pass ? 'success' : 'danger'}
                />
              )
            }
          >
            <SpectralChart data={filtered} />
          </ChartCard>

          {showRelative ? (
            <ContentCard
              title="Относительная ошибка по волновым числам"
              description="Вычислена только из доступных reference_energy и model_energy."
            >
              <DataTable>
                <table className="min-w-full text-[12px]">
                  <thead className="bg-slate-50 text-slate-500">
                    <tr>
                      <th className="px-3 py-2 text-left">Волновое число</th>
                      <th className="px-3 py-2 text-left">Относительная ошибка</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((point) => (
                      <tr key={point.wavenumber} className="border-t border-slate-200">
                        <td className="px-3 py-2 text-slate-700">{point.wavenumber}</td>
                        <td className="px-3 py-2 font-medium text-slate-900">
                          {point.relative_error_pct.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </DataTable>
            </ContentCard>
          ) : null}
        </>
      )}

      <ContentCard title="Интерпретация">
        <p className="text-[13px] leading-6 text-slate-600">
          {spectralError === undefined
            ? 'Итоговый spectral error отсутствует, поэтому сделать вывод о прохождении порога нельзя.'
            : spectralCriterion?.pass === true
              ? `Зафиксированный spectral error ${spectralError.toFixed(2)}% соответствует доступному критерию допуска.`
              : spectralCriterion?.pass === false
                ? `Зафиксированный spectral error ${spectralError.toFixed(2)}% не соответствует доступному критерию допуска.`
                : `Зафиксированный spectral error равен ${spectralError.toFixed(2)}%, но статус или порог в results.json не указан.`}
        </p>
      </ContentCard>
    </div>
  )
}
