import { useState } from 'react'
import {
  Area,
  Brush,
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
import { ChartZoomControls, useYZoom } from './ChartZoomControls'
import { useLanguage } from '../i18n/LanguageContext'

const CHART_GRID = '#1e2330'
const CHART_AXIS = '#475569'
const ENSEMBLE_COLOR = 'rgba(245, 158, 11, 1)'

// Series that are drawn but must not show up as plain numbers in the tooltip.
const HIDDEN_IN_TOOLTIP = new Set(['band95', 'band80', 'garch_lo', 'garch_hi'])

/**
 * Forecast chart.
 *
 * Draws the commodity's own history, a "last observation" marker, the
 * ensemble forecast with an 80% / 95% prediction-interval fan, the NAIVE
 * "last value" baseline (grey, dashed) it has to beat, optional individual-model
 * overlays (dotted) and, for daily/weekly series, the GARCH 90% volatility band
 * as red dashed lines (future dates only).
 *
 * All series share a single combined date axis (`dates` = history dates +
 * future dates). Each series array must be pre-aligned to that axis with
 * `null` for points outside its domain. `bands` is
 * `{ b80: {lower, upper}, b95: {lower, upper} }` (arrays aligned to the axis;
 * the fan opens from the last observation); `baseline` is an aligned array.
 * Both are optional, so the chart still renders for forecasts that were
 * produced before intervals / baselines existed.
 */
export default function ForecastChart({
  dates,
  todayDate,
  historical,
  ensemble,
  baseline,
  bands,
  models = [],
  garchBand,
  height = 480,
  yLabel,
}) {
  const { t } = useLanguage()
  const data = dates.map((date, i) => {
    const row = { date, historical: historical[i], ensemble: ensemble[i] }
    if (baseline) row.baseline = baseline[i]
    models.forEach((m) => {
      row[m.key] = m.values[i]
    })
    if (bands) {
      const pair = (b) => {
        const lower = b.lower[i]
        const upper = b.upper[i]
        return lower === null || lower === undefined || upper === null || upper === undefined ? null : [lower, upper]
      }
      row.band95 = pair(bands.b95)
      row.band80 = pair(bands.b80)
    }
    if (garchBand) {
      row.garch_lo = garchBand.lower[i] ?? null
      row.garch_hi = garchBand.upper[i] ?? null
    }
    return row
  })

  const { yMin, setYMin, yMax, setYMax, domain, active, reset } = useYZoom()
  const [brushKey, setBrushKey] = useState(0)

  const resetAll = () => {
    reset()
    setBrushKey((k) => k + 1)
  }

  const labelFor = (name) => {
    if (name === 'historical') return t('forecast.historical')
    if (name === 'ensemble') return t('forecast.ensemble')
    if (name === 'baseline') return t('forecast.naive')
    return models.find((m) => m.key === name)?.name || name
  }

  return (
    <div>
      <ChartZoomControls yMin={yMin} setYMin={setYMin} yMax={yMax} setYMax={setYMax} active={active} onResetAll={resetAll} />
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 28, right: 24, bottom: 4, left: 0 }}>
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
          contentStyle={{ backgroundColor: '#141720', border: '1px solid #1e2330', borderRadius: 6, fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          labelFormatter={(d) => fmtDate(d)}
          formatter={(value, name) => {
            if (HIDDEN_IN_TOOLTIP.has(name) || value === null || value === undefined) return null
            return [Number(value).toFixed(2), labelFor(name)]
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'IBM Plex Mono', color: '#94a3b8' }} />

        {/* Prediction-interval fan: wide/light 95% under narrow/darker 80% */}
        {bands && (
          <Area
            dataKey="band95"
            stroke="none"
            fill="rgba(245,158,11,0.10)"
            name={t('forecast.band95')}
            isAnimationActive={false}
            connectNulls={false}
          />
        )}
        {bands && (
          <Area
            dataKey="band80"
            stroke="none"
            fill="rgba(245,158,11,0.22)"
            name={t('forecast.band80')}
            isAnimationActive={false}
            connectNulls={false}
          />
        )}

        {/* GARCH 90% volatility band: dashed lines, so it stays distinct from the fan */}
        {garchBand && (
          <Line
            dataKey="garch_hi"
            name={t('forecast.garchBand')}
            stroke="rgba(248,113,113,0.7)"
            strokeWidth={1}
            strokeDasharray="2 3"
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />
        )}
        {garchBand && (
          <Line
            dataKey="garch_lo"
            stroke="rgba(248,113,113,0.7)"
            strokeWidth={1}
            strokeDasharray="2 3"
            dot={false}
            legendType="none"
            isAnimationActive={false}
            connectNulls={false}
          />
        )}

        <Line
          type="monotone"
          dataKey="historical"
          name={t('forecast.historical')}
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

        {/* The baseline every forecast has to beat: "the price stays where it is" */}
        {baseline && (
          <Line
            type="monotone"
            dataKey="baseline"
            name={t('forecast.naive')}
            stroke="#94a3b8"
            strokeWidth={2}
            strokeDasharray="7 4"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        )}

        <Line
          type="monotone"
          dataKey="ensemble"
          name={t('forecast.ensemble')}
          stroke={ENSEMBLE_COLOR}
          strokeWidth={3.5}
          dot={{ r: 3.5, fill: ENSEMBLE_COLOR, stroke: '#0d0f12', strokeWidth: 1.5 }}
          isAnimationActive={false}
          connectNulls
        />

        {todayDate && (
          <ReferenceLine
            x={todayDate}
            stroke="#475569"
            strokeDasharray="2 2"
            label={{ value: t('forecast.lastObs'), position: 'top', fill: '#475569', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
          />
        )}

        <Brush
          key={brushKey}
          dataKey="date"
          height={26}
          travellerWidth={8}
          gap={4}
          stroke="#475569"
          fill="#141720"
          tickFormatter={(d) => fmtDate(d, { month: 'short', year: '2-digit' })}
          startIndex={Math.max(0, data.length - Math.round(data.length * 0.45))}
          className="[&_text]:fill-slate-500 [&_text]:text-[9px] [&_text]:font-mono"
        />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
