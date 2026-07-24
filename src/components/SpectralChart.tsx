import { CartesianGrid, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { SpectralPoint } from '../types'
import { ChartTooltip } from './ChartTooltip'

const BLUE = '#3B82F6'
const TEAL = '#14B8A6'

export function SpectralChart({ data }: { data: SpectralPoint[] }) {
  return (
    <section id="spectral" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Спектральный анализ</div>
        <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Спектральная ошибка</h2>
      </div>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid vertical={false} stroke="#E5E7EB" strokeDasharray="3 3" />
            <XAxis dataKey="wavenumber" type="number" scale="log" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <YAxis scale="log" stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <ChartTooltip />
            <Line type="monotone" dataKey="reference_energy" stroke={TEAL} strokeWidth={1.5} strokeDasharray="6 4" dot={{ r: 2, fill: TEAL }} />
            <Line type="monotone" dataKey="model_energy" stroke={BLUE} strokeWidth={2.2} dot={{ r: 2, fill: BLUE }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
