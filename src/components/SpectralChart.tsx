import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SpectralPoint } from '../types'

export function SpectralChart({ data }: { data: SpectralPoint[] }) {
  return (
    <section id="spectral" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Frequency analysis</p>
        <h2 className="text-2xl font-semibold text-white">Spectral error</h2>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#334155" strokeDasharray="3 3" />
            <XAxis dataKey="wavenumber" type="number" stroke="#94a3b8" scale="log" />
            <YAxis stroke="#94a3b8" scale="log" />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 12 }}
              formatter={(value: number) => [value.toFixed(3), 'energy']}
            />
            <Line type="monotone" dataKey="reference_energy" stroke="#94a3b8" strokeWidth={1.8} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="model_energy" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
