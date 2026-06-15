import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'

/**
 * Builds histogram bins from a flat array of numeric values.
 */
function buildBins(values, binCount = 40) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const width = (max - min) / binCount || 1

  const bins = Array.from({ length: binCount }, (_, i) => ({
    x0: min + i * width,
    x1: min + (i + 1) * width,
    count: 0,
  }))

  values.forEach((v) => {
    let idx = Math.floor((v - min) / width)
    if (idx >= binCount) idx = binCount - 1
    if (idx < 0) idx = 0
    bins[idx].count += 1
  })

  return bins.map((b) => ({ ...b, label: b.x0.toFixed(1) }))
}

/**
 * Percentile distribution histogram — replicates the `go.Histogram` charts
 * on the Analysis (current price vs. history) and Forecasting (forecast vs.
 * history) pages, with vertical reference lines for the current value and
 * an optional secondary marker (median or forecast).
 */
export default function HistogramChart({
  values,
  current,
  secondary,
  secondaryLabel = 'Median',
  currentLabel = 'Current',
  height = 220,
  xLabel,
}) {
  if (!values || values.length === 0) return null

  const bins = buildBins(values)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={bins} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={CHART_GRID} vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          interval={Math.floor(bins.length / 8)}
          label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -2, fill: CHART_AXIS, fontSize: 10 } : undefined}
        />
        <YAxis
          tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          width={40}
          label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          formatter={(value) => [value, 'Observations']}
        />
        <Bar dataKey="count" fill="#1e2f3e" stroke="rgba(56, 189, 248, 0.3)" isAnimationActive={false} />
        {secondary !== undefined && secondary !== null && (
          <ReferenceLine
            x={bins.find((b) => secondary >= b.x0 && secondary <= b.x1)?.label}
            stroke="#475569"
            strokeDasharray="3 3"
            label={{ value: secondaryLabel, position: 'top', fill: '#475569', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        )}
        {current !== undefined && current !== null && (
          <ReferenceLine
            x={bins.find((b) => current >= b.x0 && current <= b.x1)?.label}
            stroke="rgba(245, 158, 11, 1)"
            strokeWidth={2}
            label={{ value: currentLabel, position: 'top', fill: 'rgba(245, 158, 11, 1)', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  )
}
