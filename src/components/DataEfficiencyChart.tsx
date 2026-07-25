import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import type { DataEfficiencyChartPoint } from '../data/resultsSelectors'
import { chartTheme } from '../data/chartTheme'
import { ChartTooltip } from './ChartTooltip'

export type DataEfficiencyMetric = 'overall_nrmse' | 'surface_nrmse' | 'pressure_nrmse'

const metricLabels: Record<DataEfficiencyMetric, string> = {
  overall_nrmse: 'Общий NRMSE',
  surface_nrmse: 'Surface NRMSE',
  pressure_nrmse: 'Pressure NRMSE',
}

export function DataEfficiencyChart({
  data,
  metric = 'overall_nrmse',
  showReference = false,
  showConfidence = false,
}: {
  data: DataEfficiencyChartPoint[]
  metric?: DataEfficiencyMetric
  showReference?: boolean
  showConfidence?: boolean
}) {
  const ticks = data.map((point) => point.n_samples)

  return (
    <div className="h-[360px] min-h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid vertical={false} stroke={chartTheme.grid} strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="n_samples"
            scale="log"
            domain={['dataMin', 'dataMax']}
            ticks={ticks}
            allowDataOverflow
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
            tickFormatter={(value: number) => value.toLocaleString('ru-RU')}
          />
          <YAxis
            domain={['auto', 'auto']}
            stroke={chartTheme.axis}
            tick={{ fill: chartTheme.axis, fontSize: 11 }}
            tickFormatter={(value: number) => value.toFixed(3)}
          />
          <ChartTooltip />
          {showConfidence ? (
            <>
              <Line
                type="monotone"
                dataKey="ci_low"
                name="Нижняя граница CI"
                connectNulls
                stroke={chartTheme.reference}
                strokeWidth={1.2}
                strokeDasharray="3 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="ci_high"
                name="Верхняя граница CI"
                connectNulls
                stroke={chartTheme.reference}
                strokeWidth={1.2}
                strokeDasharray="3 4"
                dot={false}
              />
            </>
          ) : null}
          {showReference ? (
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
            dot={{ r: 3.5, fill: chartTheme.model }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
