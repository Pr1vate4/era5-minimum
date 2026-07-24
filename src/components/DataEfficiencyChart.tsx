import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { DataEfficiencyPoint } from '../types'

export function DataEfficiencyChart({ data }: { data: DataEfficiencyPoint[] }) {
  return (
    <section id="data-efficiency" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Learning curve</p>
        <h2 className="text-2xl font-semibold text-white">Data efficiency</h2>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#334155" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="n_samples"
              domain={['dataMin', 'dataMax']}
              scale="log"
              tickFormatter={(value) => `${value}`}
              stroke="#94a3b8"
            />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 12 }}
              labelFormatter={(label) => `n_samples: ${label}`}
              formatter={(value: number) => [value.toFixed(3), 'overall_nrmse']}
            />
            <Line type="monotone" dataKey="overall_nrmse" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
