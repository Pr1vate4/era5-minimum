import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { RateDistortionPoint } from '../types'
import { ChartTooltip } from './ChartTooltip'

const BLUE = '#3B82F6'

export function RateDistortionChart({ data }: { data: RateDistortionPoint[] }) {
  const threshold = 0.05

  return (
    <section id="rate-distortion" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Сжатие и качество</div>
        <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Кривая сжатие — качество</h2>
      </div>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#E5E7EB" strokeDasharray="3 3" />
            <XAxis dataKey="compression_ratio" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <YAxis stroke="#64748B" domain={[0, 0.09]} tick={{ fill: '#64748B', fontSize: 11 }} />
            <ChartTooltip />
            <ReferenceLine y={threshold} stroke="#DC2626" strokeDasharray="5 5" />
            <Line type="monotone" dataKey="overall_nrmse" stroke={BLUE} strokeWidth={2.2} dot={{ r: 4, fill: BLUE }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
