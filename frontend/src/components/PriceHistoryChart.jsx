import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  ReferenceLine,
} from 'recharts'
import { fmtDate } from '../utils/format'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'
const TOOLTIP_STYLE = {
  backgroundColor: '#141720',
  border: '1px solid #1e2330',
  borderRadius: 6,
  fontSize: 11,
  fontFamily: '"IBM Plex Mono", monospace',
}

/**
 * Multi-series line chart for price history.
 *
 * `dates` is a list of ISO date strings, `series` is `{ key: { name, values, color } }`.
 * Data is reshaped into Recharts' row-per-point format internally.
 */
export default function PriceHistoryChart({ dates, series, height = 420, yLabel, dot = false, referenceLines = [] }) {
  const data = dates.map((date, i) => {
    const row = { date }
    Object.entries(series).forEach(([key, s]) => {
      row[key] = s.values[i]
    })
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={CHART_GRID} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })}
          tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          minTickGap={32}
        />
        <YAxis
          tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          width={48}
          label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 } : undefined}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(d) => fmtDate(d)}
          formatter={(value, name) => [value?.toFixed ? value.toFixed(2) : value, series[name]?.name || name]}
        />
        <Legend
          wrapperStyle={{ fontSize: 10, fontFamily: 'IBM Plex Mono', color: '#94a3b8' }}
          formatter={(value) => series[value]?.name || value}
        />
        <ReferenceLine y={0} stroke={CHART_GRID} />
        {referenceLines.map((line, i) => (
          <ReferenceLine
            key={i}
            y={line.y}
            stroke={line.color}
            strokeDasharray="3 3"
            strokeWidth={1.5}
            label={{ value: line.label, position: 'right', fill: line.color, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        ))}
        {Object.entries(series).map(([key, s]) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            name={key}
            stroke={s.color}
            strokeWidth={s.strokeWidth ?? 1.5}
            strokeDasharray={s.dashed ? '4 3' : undefined}
            strokeOpacity={s.opacity ?? 1}
            legendType={s.legendType || 'line'}
            dot={dot}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
