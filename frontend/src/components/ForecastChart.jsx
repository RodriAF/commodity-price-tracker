import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  Legend,
  XAxis,
  YAxis,
} from 'recharts'
import { fmtDate } from '../utils/format'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'

/**
 * Forecast chart — replicates the Plotly figure on the Forecasting page:
 * the historical series, a "today" marker, optional individual-model
 * overlays (dotted), a GARCH 90% volatility band (shaded area, future
 * dates only), and the prominent ensemble line+markers.
 *
 * All series share a single combined date axis (`dates` = history dates +
 * future dates). Each series array must be pre-aligned to that axis with
 * `null` for points outside its domain.
 */
export default function ForecastChart({ dates, todayDate, historical, ensemble, models = [], garchBand, height = 480, yLabel }) {
  const data = dates.map((date, i) => {
    const row = { date, historical: historical[i], ensemble: ensemble[i] }
    models.forEach((m) => {
      row[m.key] = m.values[i]
    })
    if (garchBand) {
      const lower = garchBand.lower[i]
      const upper = garchBand.upper[i]
      row.garch_range = lower === null || upper === null ? null : [lower, upper]
    }
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
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
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          labelFormatter={(d) => fmtDate(d)}
          formatter={(value, name) => {
            if (name === 'garch_range' || value === null) return null
            const label = name === 'historical' ? 'Historical' : name === 'ensemble' ? 'Ensemble' : models.find((m) => m.key === name)?.name || name
            return [Number(value).toFixed(2), label]
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'IBM Plex Mono', color: '#94a3b8' }} />

        {garchBand && (
          <Area
            dataKey="garch_range"
            stroke="none"
            fill="rgba(248,113,113,0.06)"
            name="GARCH 90% band"
            isAnimationActive={false}
            connectNulls={false}
          />
        )}

        <Line
          type="monotone"
          dataKey="historical"
          name="Historical"
          stroke="#475569"
          strokeWidth={2.5}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />

        {models.map((m) => (
          <Line
            key={m.key}
            type="monotone"
            dataKey={m.key}
            name={m.name}
            stroke={m.color}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            strokeOpacity={0.6}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}

        <Line
          type="monotone"
          dataKey="ensemble"
          name="Ensemble"
          stroke="rgba(245, 158, 11, 1)"
          strokeWidth={3.5}
          dot={{ r: 4, fill: 'rgba(245, 158, 11, 1)', stroke: '#0d0f12', strokeWidth: 1.5 }}
          isAnimationActive={false}
          connectNulls
        />

        {todayDate && (
          <ReferenceLine
            x={todayDate}
            stroke="#475569"
            strokeDasharray="2 2"
            label={{ value: 'Today', position: 'top', fill: '#475569', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
