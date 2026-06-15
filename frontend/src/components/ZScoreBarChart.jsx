import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtZScore } from '../utils/format'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'

function barColor(z) {
  const abs = Math.abs(z)
  if (abs > 2) return 'rgba(239, 68, 68, 1)'
  if (abs > 1) return 'rgba(245, 158, 11, 1)'
  return '#22477a'
}

/**
 * Horizontal z-score deviation bars — replicates the `go.Bar` z-score chart
 * used on both the Analysis and Input Cost Analytics pages.
 *
 * `data` is `[{ label, z }]`.
 */
export default function ZScoreBarChart({ data, height = 260, layout = 'vertical' }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout} margin={{ top: 8, right: 16, bottom: 8, left: layout === 'horizontal' ? 100 : 0 }}>
        <CartesianGrid stroke={CHART_GRID} horizontal={layout === 'vertical'} vertical={layout === 'horizontal'} />
        {layout === 'vertical' ? (
          <>
            <XAxis
              dataKey="label"
              tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: CHART_GRID }}
              tickLine={{ stroke: CHART_GRID }}
              angle={-35}
              textAnchor="end"
              height={70}
              interval={0}
            />
            <YAxis
              tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: CHART_GRID }}
              tickLine={{ stroke: CHART_GRID }}
              label={{ value: 'Z-Score (σ)', angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 }}
            />
          </>
        ) : (
          <>
            <XAxis
              type="number"
              tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: CHART_GRID }}
              tickLine={{ stroke: CHART_GRID }}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: CHART_GRID }}
              tickLine={{ stroke: CHART_GRID }}
              width={96}
            />
          </>
        )}
        <Tooltip
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          formatter={(value) => [fmtZScore(value), 'Z-Score']}
        />
        <ReferenceLine {...(layout === 'vertical' ? { y: 0 } : { x: 0 })} stroke={CHART_GRID} />
        <ReferenceLine {...(layout === 'vertical' ? { y: 2 } : { x: 2 })} stroke="rgba(239, 68, 68, 0.5)" strokeDasharray="3 3" />
        <ReferenceLine {...(layout === 'vertical' ? { y: -2 } : { x: -2 })} stroke="rgba(239, 68, 68, 0.5)" strokeDasharray="3 3" />
        <ReferenceLine {...(layout === 'vertical' ? { y: 1 } : { x: 1 })} stroke="rgba(245, 158, 11, 0.5)" strokeDasharray="3 3" />
        <ReferenceLine {...(layout === 'vertical' ? { y: -1 } : { x: -1 })} stroke="rgba(245, 158, 11, 0.5)" strokeDasharray="3 3" />
        <Bar dataKey="z" isAnimationActive={false}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={barColor(entry.z)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
