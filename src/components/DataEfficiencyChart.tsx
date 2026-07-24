import { CartesianGrid, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { DataEfficiencyPoint } from '../types'
import { ChartTooltip } from './ChartTooltip'

const BLUE = '#3B82F6'

export function DataEfficiencyChart({ data }: { data: DataEfficiencyPoint[] }) {
  return (
    <section id="data-efficiency" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Эффективность данных</div>
        <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Зависимость качества от объёма данных</h2>
      </div>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#E5E7EB" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="n_samples" scale="log" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <YAxis stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <ChartTooltip />
            <Line type="monotone" dataKey="overall_nrmse" stroke={BLUE} strokeWidth={2.2} dot={{ r: 3, fill: BLUE }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
