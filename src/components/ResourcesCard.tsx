import type { selectResourcesData } from '../data/resultsSelectors'
import { resourceLimits, type ResourceKey } from '../data/resourceLimits'
import { StatusBadge } from './common/StatusBadge'

type ResourcesData = ReturnType<typeof selectResourcesData>

export function ResourcesCard({ resources }: { resources: ResourcesData }) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      {resourceLimits.map((limit) => {
        const value = resources[limit.key as ResourceKey]
        const percent = value === undefined ? undefined : (value / limit.limit) * 100
        const tone =
          percent === undefined
            ? 'bg-slate-300'
            : percent > 100
              ? 'bg-rose-500'
              : percent >= 85
                ? 'bg-amber-500'
                : 'bg-blue-500'

        return (
          <div key={limit.key} className="rounded-2xl border border-slate-200 bg-white p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-[13px] font-semibold text-slate-800">{limit.label}</h2>
                <p className="mt-1 text-[12px] text-slate-500">
                  {value === undefined ? 'Нет фактического значения' : `${formatResource(value)} ${limit.unit}`}
                  {' / '}
                  лимит {formatResource(limit.limit)} {limit.unit}
                </p>
              </div>
              <StatusBadge
                label={
                  percent === undefined
                    ? 'Нет данных'
                    : percent > 100
                      ? 'Лимит превышен'
                      : percent >= 85
                        ? 'Близко к лимиту'
                        : 'В пределах лимита'
                }
                tone={
                  percent === undefined
                    ? 'neutral'
                    : percent > 100
                      ? 'danger'
                      : percent >= 85
                        ? 'warning'
                        : 'success'
                }
              />
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200">
              <div
                className={`h-full rounded-full ${tone}`}
                style={{ width: `${Math.min(percent ?? 0, 100)}%` }}
              />
            </div>
            <p className="mt-2 text-right text-[11px] font-medium text-slate-500">
              {percent === undefined ? '—' : `${percent.toFixed(1)}%`}
            </p>
          </div>
        )
      })}
    </div>
  )
}

function formatResource(value: number) {
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 2 })
}
