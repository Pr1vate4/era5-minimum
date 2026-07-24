import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { RateDistortionPoint } from '../types'

export function RateDistortionChart({ data }: { data: RateDistortionPoint[] }) {
  const threshold = 0.05

  return (
    <section id="rate-distortion" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Compression quality</p>
        <h2 className="text-2xl font-semibold text-white">Rate–distortion</h2>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="compression_ratio" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={[0, 0.09]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 12 }}
              formatter={(value: number) => [value.toFixed(3), 'overall_nrmse']}
            />
            <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'admission threshold', position: 'insideTopRight' }} />
            <Line type="monotone" dataKey="overall_nrmse" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
