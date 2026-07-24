import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import { chartTheme } from '../data/chartTheme'
import type { SpectralChartPoint } from '../data/resultsSelectors'
import { ChartTooltip } from './ChartTooltip'

export function SpectralChart({ data }: { data: SpectralChartPoint[] }) {
  return (
    <div className="h-[360px] min-h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid vertical={false} stroke={chartTheme.grid} strokeDasharray="3 3" />
          <XAxis
            dataKey="wavenumber"
            type="number"
            scale="log"
            domain={['dataMin', 'dataMax']}
            allowDataOverflow
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
          />
          <YAxis
            type="number"
            scale="log"
            domain={['dataMin', 'dataMax']}
            allowDataOverflow
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
          />
          <ChartTooltip />
          <Line
            type="monotone"
            dataKey="reference_energy"
            name="Эталон"
            stroke={chartTheme.reference}
            strokeWidth={1.8}
            strokeDasharray="6 4"
            dot={{ r: 2.5, fill: chartTheme.reference }}
          />
          <Line
            type="monotone"
            dataKey="model_energy"
            name="Модель"
            stroke={chartTheme.model}
            strokeWidth={2.4}
            dot={{ r: 2.5, fill: chartTheme.model }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
