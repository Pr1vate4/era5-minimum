import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import { chartTheme } from '../data/chartTheme'
import type { RateDistortionChartPoint } from '../data/resultsSelectors'
import { ChartTooltip } from './ChartTooltip'

export type RateDistortionMetric =
  | 'overall_nrmse'
  | 'surface_nrmse'
  | 'pressure_nrmse'
  | 'psnr'

const metricLabels: Record<RateDistortionMetric, string> = {
  overall_nrmse: 'Общий NRMSE',
  surface_nrmse: 'Surface NRMSE',
  pressure_nrmse: 'Pressure NRMSE',
  psnr: 'PSNR',
}

export function RateDistortionChart({
  data,
  metric = 'overall_nrmse',
  showReference = false,
}: {
  data: RateDistortionChartPoint[]
  metric?: RateDistortionMetric
  showReference?: boolean
}) {
  return (
    <div className="h-[360px] min-h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid vertical={false} stroke={chartTheme.grid} strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="compression_ratio"
            domain={['dataMin', 'dataMax']}
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
            tickFormatter={(value: number) => `${value}×`}
          />
          <YAxis
            domain={['auto', 'auto']}
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
            tickFormatter={(value: number) => value.toFixed(metric === 'psnr' ? 1 : 3)}
          />
          <ChartTooltip />
          {showReference && metric === 'overall_nrmse' ? (
            <Line
              type="monotone"
              dataKey="reference_nrmse"
              name="Референс"
              connectNulls
              stroke={chartTheme.reference}
              strokeWidth={1.8}
              strokeDasharray="6 4"
              dot={{ r: 2.5, fill: chartTheme.reference }}
            />
          ) : null}
          <Line
            type="monotone"
            dataKey={metric}
            name={metricLabels[metric]}
            connectNulls
            stroke={chartTheme.model}
            strokeWidth={2.4}
            dot={{ r: 4, fill: chartTheme.model }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
