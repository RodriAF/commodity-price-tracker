import {
  CartesianGrid,
  Line,
  ComposedChart,
  ReferenceLine,
  Area,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fmtDate } from '../utils/format'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'

/**
 * Single-series ratio chart with a dashed mean line and shaded +/-1σ /
 * +/-2σ bands — replicates the `go.Figure` + `add_hrect` ratio explorer on
 * the Input Cost Analytics page.
 */
export default function RatioExplorerChart({ dates, values, mean, std, height = 300 }) {
  const data = dates.map((date, i) => ({ date, value: values[i] }))
  const numericValues = values.filter((v) => v !== null && v !== undefined)

  const band1 = std ? [mean - std, mean + std] : null
  const band2 = std ? [mean - 2 * std, mean + 2 * std] : null

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
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
          domain={
            band2
              ? [Math.min(band2[0], ...numericValues), Math.max(band2[1], ...numericValues)]
              : ['auto', 'auto']
          }
          label={{ value: 'Ratio (trend only)', angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          labelFormatter={(d) => fmtDate(d)}
          formatter={(value, name) => [Number(value).toFixed(3), name]}
        />

        {band2 && (
          <Area
            dataKey={() => band2}
            stroke="none"
            fill="rgba(245, 158, 11, 1)"
            fillOpacity={0.03}
            isAnimationActive={false}
            legendType="none"
          />
        )}
        {band1 && (
          <Area
            dataKey={() => band1}
            stroke="none"
            fill="rgba(34, 197, 94, 1)"
            fillOpacity={0.05}
            isAnimationActive={false}
            legendType="none"
          />
        )}

        {mean !== null && mean !== undefined && (
          <ReferenceLine
            y={mean}
            stroke="#475569"
            strokeDasharray="5 3"
            strokeWidth={1.5}
            label={{ value: `Mean ${mean.toFixed(3)}`, position: 'insideTopRight', fill: '#64748b', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        )}

        <Line
          type="monotone"
          dataKey="value"
          name="Ratio"
          stroke="rgba(56,189,248,1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
