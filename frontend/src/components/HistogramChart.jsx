import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useLanguage } from '../i18n/LanguageContext'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'
const AXIS_TICK = { fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }
const BIN_COUNT = 40

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

/**
 * Builds histogram bins from a flat array of numeric values.
 *
 * `extra` are values that must be visible on the chart even if they fall
 * outside the historical range (e.g. a forecast above the historical max):
 * the bin range is widened to include them so their marker is always drawn.
 */
function buildBins(values, extra = []) {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (!isNum(v)) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  extra.filter(isNum).forEach((v) => {
    if (v < min) min = v
    if (v > max) max = v
  })

  const width = (max - min) / BIN_COUNT || 1
  const bins = Array.from({ length: BIN_COUNT }, (_, idx) => ({
    idx,
    x0: min + idx * width,
    x1: min + (idx + 1) * width,
    count: 0,
  }))

  const indexOf = (v) => Math.min(BIN_COUNT - 1, Math.max(0, Math.floor((v - min) / width)))
  values.forEach((v) => {
    if (isNum(v)) bins[indexOf(v)].count += 1
  })

  // Decimals so that the ~8 visible tick labels are distinct and readable
  // (a fixed toFixed(1) produced duplicated labels for low-priced series).
  const tickSpacing = width * 5
  const decimals = tickSpacing >= 10 ? 0 : tickSpacing >= 1 ? 1 : tickSpacing >= 0.1 ? 2 : 3

  return { bins, indexOf, decimals }
}

/**
 * Reference-line label drawn INSIDE the plot area (top), so it can never be
 * clipped by the chart edges. `row` stacks several labels vertically so they
 * don't overlap when two markers are close; `align="end"` writes the text to
 * the left of the line (used for lines in the right half of the chart).
 */
function LineLabel({ viewBox, text, color, row = 0, align = 'start' }) {
  if (!viewBox) return null
  const dx = align === 'end' ? -6 : 6
  return (
    <text
      x={viewBox.x + dx}
      y={viewBox.y + 11 + row * 13}
      fill={color}
      fontSize={9}
      fontFamily="IBM Plex Mono"
      textAnchor={align}
      stroke="#141720"
      strokeWidth={3}
      paintOrder="stroke"
    >
      {text}
    </text>
  )
}

/**
 * Percentile distribution histogram — replicates the `go.Histogram` charts
 * on the Analysis (current price vs. history) and Forecasting (forecast vs.
 * history) pages, with vertical reference lines for the current value and
 * an optional secondary marker (median or current price).
 */
export default function HistogramChart({
  values,
  current,
  secondary,
  secondaryLabel,
  currentLabel,
  height = 240,
  xLabel,
}) {
  const { t } = useLanguage()
  if (!values || values.length === 0) return null

  const { bins, indexOf, decimals } = buildBins(values, [current, secondary])
  const fmt = (v) => v.toFixed(decimals)

  const sideFor = (idx) => (idx > bins.length * 0.55 ? 'end' : 'start')
  const hasSecondary = isNum(secondary)
  const hasCurrent = isNum(current)
  const secondaryIdx = hasSecondary ? indexOf(secondary) : null
  const currentIdx = hasCurrent ? indexOf(current) : null

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={bins} margin={{ top: 8, right: 20, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={CHART_GRID} vertical={false} />
        <XAxis
          dataKey="idx"
          tick={AXIS_TICK}
          tickFormatter={(i) => (bins[i] ? fmt(bins[i].x0) : '')}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          interval={4}
          height={xLabel ? 44 : 28}
          label={xLabel ? { value: xLabel, position: 'insideBottom', offset: 2, fill: CHART_AXIS, fontSize: 10 } : undefined}
        />
        <YAxis
          tick={AXIS_TICK}
          axisLine={{ stroke: CHART_GRID }}
          tickLine={{ stroke: CHART_GRID }}
          width={52}
          allowDecimals={false}
          // Headroom above the tallest bar so the marker labels have free space.
          domain={[0, (dataMax) => Math.max(4, Math.ceil(dataMax * 1.3))]}
          label={{ value: t('common.frequency'), angle: -90, position: 'insideLeft', fill: CHART_AXIS, fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: '"IBM Plex Mono", monospace' }}
          labelFormatter={(i) => (bins[i] ? `${fmt(bins[i].x0)} – ${fmt(bins[i].x1)}` : '')}
          formatter={(value) => [value, t('common.observations')]}
        />
        <Bar dataKey="count" fill="#1e2f3e" stroke="rgba(56, 189, 248, 0.3)" isAnimationActive={false} />
        {hasSecondary && (
          <ReferenceLine
            x={secondaryIdx}
            stroke="#64748b"
            strokeDasharray="3 3"
            label={<LineLabel text={secondaryLabel || t('common.current')} color="#94a3b8" row={0} align={sideFor(secondaryIdx)} />}
          />
        )}
        {hasCurrent && (
          <ReferenceLine
            x={currentIdx}
            stroke="rgba(245, 158, 11, 1)"
            strokeWidth={2}
            label={<LineLabel text={currentLabel || t('common.current')} color="rgba(245, 158, 11, 1)" row={1} align={sideFor(currentIdx)} />}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  )
}
