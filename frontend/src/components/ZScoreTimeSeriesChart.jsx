import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtDate, fmtZScore } from '../utils/format'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'

function barColor(z) {
  const abs = Math.abs(z)
  if (abs > 2) return 'rgba(239, 68, 68, 1)'
  if (abs > 1) return 'rgba(245, 158, 11, 1)'
  return '#1e2330'
}

/**
 * Per-date z-score deviation chart — replicates the `go.Bar` z-score chart
 * on the Analysis page for a single selected asset, with dotted reference
 * bands at +/-1 and +/-2 standard deviations.
 */
export default function ZScoreTimeSeriesChart({ dates, values, height = 180 }) {
  const data = dates.map((date, i) => ({ date, z: values[i] }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
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
          width={40}
          label={{ value: 'Standard Dev (σ)', angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          labelFormatter={(d) => fmtDate(d)}
          formatter={(value) => [fmtZScore(value), 'Z']}
        />
        <ReferenceLine y={0} stroke={CHART_GRID} />
        <ReferenceLine y={2} stroke="rgba(239, 68, 68, 0.3)" strokeDasharray="2 2" />
        <ReferenceLine y={-2} stroke="rgba(239, 68, 68, 0.3)" strokeDasharray="2 2" />
        <ReferenceLine y={1} stroke="rgba(245, 158, 11, 0.2)" strokeDasharray="2 2" />
        <ReferenceLine y={-1} stroke="rgba(245, 158, 11, 0.2)" strokeDasharray="2 2" />
        <Bar dataKey="z" isAnimationActive={false}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={barColor(entry.z ?? 0)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
