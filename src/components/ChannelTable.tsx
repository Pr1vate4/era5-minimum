import { ArrowDown, ArrowUp } from 'lucide-react'
import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ChannelMetric } from '../data/resultsSelectors'
import { EmptyState } from './EmptyState'
import { DataTable } from './common/ContentCard'
import {
  FilterBar,
  ResetFiltersButton,
  SearchInput,
  SelectField,
} from './common/Controls'
import { StatusBadge } from './common/StatusBadge'

type GroupFilter = 'all' | 'surface' | 'pressure'
type VariableFilter = 'all' | 'T' | 'U' | 'V' | 'Z' | 'Q'
type LevelFilter = 'all' | '1000' | '925' | '850' | '700'
type SortKey = 'code' | 'rmse' | 'nrmse' | 'psnr'
type SortDirection = 'asc' | 'desc'

const validGroups: GroupFilter[] = ['all', 'surface', 'pressure']
const validVariables: VariableFilter[] = ['all', 'T', 'U', 'V', 'Z', 'Q']
const validLevels: LevelFilter[] = ['all', '1000', '925', '850', '700']
const validSorts: SortKey[] = ['code', 'rmse', 'nrmse', 'psnr']

function getValidParam<T extends string>(
  value: string | null,
  allowed: readonly T[],
  fallback: T,
): T {
  return value !== null && allowed.includes(value as T) ? (value as T) : fallback
}

function numericValue(channel: ChannelMetric, key: Exclude<SortKey, 'code'>) {
  return channel[key]
}

function formatMetric(value: number | undefined, digits: number, unit = '') {
  return value === undefined ? '—' : `${value.toFixed(digits)}${unit}`
}

export function ChannelTable({ channels }: { channels: ChannelMetric[] }) {
  const [searchParams, setSearchParams] = useSearchParams()
  const group = getValidParam(searchParams.get('group'), validGroups, 'all')
  const variable = getValidParam(searchParams.get('variable'), validVariables, 'all')
  const level = getValidParam(searchParams.get('level'), validLevels, 'all')
  const sort = getValidParam(searchParams.get('sort'), validSorts, 'code')
  const direction: SortDirection = searchParams.get('dir') === 'desc' ? 'desc' : 'asc'
  const query = searchParams.get('q') ?? ''

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

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    const next = channels.filter((channel) => {
      if (group !== 'all' && channel.group !== group) return false
      if (variable !== 'all' && channel.variable !== variable) return false
      if (level !== 'all' && channel.levelHpa !== Number(level)) return false
      if (
        normalizedQuery &&
        !channel.code.toLowerCase().includes(normalizedQuery) &&
        !channel.name?.toLowerCase().includes(normalizedQuery)
      ) {
        return false
      }
      return true
    })

    return next.slice().sort((left, right) => {
      let comparison = 0

      if (sort === 'code') {
        comparison = left.code.localeCompare(right.code)
      } else {
        const leftValue = numericValue(left, sort)
        const rightValue = numericValue(right, sort)
        if (leftValue === undefined) comparison = 1
        else if (rightValue === undefined) comparison = -1
        else comparison = leftValue - rightValue
      }

      return direction === 'asc' ? comparison : -comparison
    })
  }, [channels, direction, group, level, query, sort, variable])

  const nrmseValues = filtered
    .map((channel) => channel.nrmse)
    .filter((value): value is number => value !== undefined)
  const minNrmse = nrmseValues.length ? Math.min(...nrmseValues) : 0
  const maxNrmse = nrmseValues.length ? Math.max(...nrmseValues) : 0

  const changeSort = (key: SortKey) => {
    const nextDirection = sort === key && direction === 'asc' ? 'desc' : 'asc'
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('sort', key)
        if (nextDirection === 'asc') next.delete('dir')
        else next.set('dir', nextDirection)
        return next
      },
      { replace: true },
    )
  }

  const resetFilters = () => setSearchParams({}, { replace: true })

  return (
    <div className="space-y-4">
      <FilterBar>
        <SearchInput
          label="Поиск по каналу"
          value={query}
          onChange={(value) => updateParam('q', value)}
          placeholder="Например, t2m или Q925"
        />
        <SelectField
          label="Группа"
          value={group}
          onChange={(value) => updateParam('group', value, 'all')}
          options={[
            { value: 'all', label: 'Все' },
            { value: 'surface', label: 'Приземные' },
            { value: 'pressure', label: 'Атмосферные' },
          ]}
        />
        <SelectField
          label="Переменная"
          value={variable}
          onChange={(value) => updateParam('variable', value, 'all')}
          options={validVariables.map((value) => ({
            value,
            label: value === 'all' ? 'Все T / U / V / Z / Q' : value,
          }))}
        />
        <SelectField
          label="Уровень"
          value={level}
          onChange={(value) => updateParam('level', value, 'all')}
          options={validLevels.map((value) => ({
            value,
            label: value === 'all' ? 'Все уровни' : `${value} hPa`,
          }))}
        />
        <ResetFiltersButton
          onClick={resetFilters}
          disabled={!query && group === 'all' && variable === 'all' && level === 'all' && sort === 'code'}
        />
      </FilterBar>

      {filtered.length === 0 ? (
        <EmptyState
          title="Каналы не найдены"
          message="В доступной части results.json нет каналов, соответствующих выбранным фильтрам."
          action={<ResetFiltersButton onClick={resetFilters} />}
        />
      ) : (
        <DataTable>
          <table className="min-w-[1080px] w-full text-[12px]">
            <thead className="sticky top-0 bg-slate-50 text-slate-500">
              <tr>
                <SortableHeader label="Код" sortKey="code" activeSort={sort} direction={direction} onSort={changeSort} />
                <th className="px-3 py-2.5 text-left font-semibold">Полное название</th>
                <th className="px-3 py-2.5 text-left font-semibold">Группа</th>
                <th className="px-3 py-2.5 text-left font-semibold">Уровень</th>
                <th className="px-3 py-2.5 text-left font-semibold">Единица</th>
                <SortableHeader label="RMSE" sortKey="rmse" activeSort={sort} direction={direction} onSort={changeSort} />
                <SortableHeader label="NRMSE" sortKey="nrmse" activeSort={sort} direction={direction} onSort={changeSort} />
                <SortableHeader label="PSNR" sortKey="psnr" activeSort={sort} direction={direction} onSort={changeSort} />
                <th className="px-3 py-2.5 text-left font-semibold">Δ к референсу</th>
                <th className="px-3 py-2.5 text-left font-semibold">Статус</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((channel) => {
                const heatRatio =
                  channel.nrmse === undefined || maxNrmse === minNrmse
                    ? 0
                    : (channel.nrmse - minNrmse) / (maxNrmse - minNrmse)

                return (
                  <tr key={channel.code} className="border-t border-slate-200 bg-white hover:bg-slate-50/80">
                    <td className="px-3 py-3 font-mono font-semibold text-slate-900">{channel.code}</td>
                    <td className="px-3 py-3 text-slate-700">{channel.name ?? 'Нет данных'}</td>
                    <td className="px-3 py-3 text-slate-600">
                      {channel.group === 'surface'
                        ? 'Приземный'
                        : channel.group === 'pressure'
                          ? 'Атмосферный'
                          : 'Не указана'}
                    </td>
                    <td className="px-3 py-3 text-slate-600">
                      {channel.levelHpa ? `${channel.levelHpa} hPa` : '—'}
                    </td>
                    <td className="px-3 py-3 text-slate-600">{channel.unit ?? '—'}</td>
                    <td className="px-3 py-3 tabular-nums text-slate-700">
                      {formatMetric(channel.rmse, 4)}
                    </td>
                    <td
                      className="px-3 py-3 font-semibold tabular-nums text-slate-800"
                      style={{
                        backgroundColor:
                          channel.nrmse === undefined
                            ? undefined
                            : `rgba(59, 130, 246, ${0.07 + heatRatio * 0.2})`,
                      }}
                    >
                      {formatMetric(channel.nrmse, 4)}
                    </td>
                    <td className="px-3 py-3 tabular-nums text-slate-700">
                      {formatMetric(channel.psnr, 2, ' dB')}
                    </td>
                    <td className="px-3 py-3 tabular-nums text-slate-600">
                      {formatMetric(channel.deltaVsReferencePct, 2, '%')}
                    </td>
                    <td className="px-3 py-3">
                      {channel.pass === undefined ? (
                        <StatusBadge label="Порог не задан" />
                      ) : (
                        <StatusBadge
                          label={channel.pass ? 'Пройдено' : 'Не пройдено'}
                          tone={channel.pass ? 'success' : 'danger'}
                        />
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </DataTable>
      )}
      <p className="text-[12px] text-slate-500">
        Показано {filtered.length} из {channels.length} каналов, присутствующих в results.json.
      </p>
    </div>
  )
}

function SortableHeader({
  label,
  sortKey,
  activeSort,
  direction,
  onSort,
}: {
  label: string
  sortKey: SortKey
  activeSort: SortKey
  direction: SortDirection
  onSort: (key: SortKey) => void
}) {
  const active = sortKey === activeSort

  return (
    <th className="px-3 py-2.5 text-left font-semibold">
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className="ui-focus-ring inline-flex items-center gap-1 rounded"
      >
        {label}
        {active ? (
          direction === 'asc' ? (
            <ArrowUp className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <ArrowDown className="h-3.5 w-3.5" aria-hidden="true" />
          )
        ) : null}
      </button>
    </th>
  )
}
