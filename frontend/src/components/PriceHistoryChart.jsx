import { useState } from 'react'
import {
  Brush,
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
import { ChartZoomControls, useYZoom } from './ChartZoomControls'

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

  const { yMin, setYMin, yMax, setYMax, domain, active, reset } = useYZoom()
  const [brushKey, setBrushKey] = useState(0)

  const resetAll = () => {
    reset()
    setBrushKey((k) => k + 1) // remounts Brush, snapping it back to the full range
  }

  return (
    <div>
      <ChartZoomControls yMin={yMin} setYMin={setYMin} yMax={yMax} setYMax={setYMax} active={active} onResetAll={resetAll} />
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: referenceLines.length > 0 ? 14 : 8, right: referenceLines.length > 0 ? 100 : 20, bottom: 4, left: 0 }}>
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
            domain={domain}
            allowDataOverflow
            tick={{ fill: CHART_AXIS, fontSize: 9, fontFamily: 'IBM Plex Mono' }}
            axisLine={{ stroke: CHART_GRID }}
            tickLine={{ stroke: CHART_GRID }}
            width={yLabel ? 64 : 52}
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
          <ReferenceLine key="zero-line" y={0} stroke={CHART_GRID} />
          {referenceLines.map((line, i) => (
            <ReferenceLine
              key={`ref-${i}`}
              y={line.y}
              stroke={line.color}
              strokeDasharray="3 3"
              strokeWidth={1.5}
              // Always show the threshold, extending the Y axis if the data never reaches it
              ifOverflow="extendDomain"
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
          <Brush
            key={brushKey}
            dataKey="date"
            height={26}
            travellerWidth={8}
            gap={4}
            stroke="#475569"
            fill="#141720"
            tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })}
            className="[&_text]:fill-slate-500 [&_text]:text-[9px] [&_text]:font-mono"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
