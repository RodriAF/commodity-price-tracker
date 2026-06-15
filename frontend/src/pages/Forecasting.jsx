import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import KpiCard from '../components/KpiCard'
import ForecastChart from '../components/ForecastChart'
import HistogramChart from '../components/HistogramChart'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { fmtNumber, fmtPercent, fmtDate, changeColorClass } from '../utils/format'

const MODEL_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f87171', '#facc15', '#fb923c']

const CONFIDENCE_STYLE = {
  high: 'text-pos',
  medium: 'text-warn',
  low: 'text-neg',
}

export default function Forecasting() {
  const { data: available, loading: availLoading, error: availError } = useApiData(() => api.forecastAvailable(), [])

  const [commodity, setCommodity] = useState(null)
  const [selectedModels, setSelectedModels] = useState([])

  useEffect(() => {
    if (available && !commodity && available.commodities.length > 0) {
      setCommodity(available.commodities[0].key)
    }
  }, [available, commodity])

  const { data, loading, error } = useApiData(
    () => (commodity ? api.forecast(commodity) : Promise.resolve(null)),
    [commodity],
  )

  // Default the model overlay selection to the top-3 ranked models whenever
  // the commodity (and therefore the model list) changes.
  useEffect(() => {
    if (data) setSelectedModels(data.top_models)
  }, [data])

  if (availLoading) return <LoadingState label="Loading available forecasts…" />
  if (availError) return <ErrorState message={availError.detail || availError.message} />
  if (!available || available.commodities.length === 0) {
    return (
      <div>
        <PageHeader subtitle="Multi-Horizon Forecasting" />
        <InfoState message="No forecast data available. Run the forecasting pipeline to generate predictions." />
      </div>
    )
  }

  const toggleModel = (method) => {
    setSelectedModels((prev) => (prev.includes(method) ? prev.filter((m) => m !== method) : [...prev, method]))
  }

  return (
    <div>
      <PageHeader
        subtitle="Multi-Horizon Forecasting"
        action={
          <div>
            <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1 text-right">Commodity</p>
            <select
              value={commodity ?? ''}
              onChange={(e) => setCommodity(e.target.value)}
              className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body"
            >
              {available.commodities.map((c) => (
                <option key={c.key} value={c.key}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
        }
      />

      {loading && <LoadingState label="Loading forecast…" />}
      {error && <ErrorState message={error.detail || error.message} />}

      {data && <ForecastBody data={data} selectedModels={selectedModels} toggleModel={toggleModel} />}
    </div>
  )
}

function ForecastBody({ data, selectedModels, toggleModel }) {
  // ---------------------------------------------------------------- //
  // Build the combined date axis (history + future) and aligned series //
  // ---------------------------------------------------------------- //
  const histDates = data.history.dates
  const futureDates = data.future_dates
  const n = histDates.length
  const dates = [...histDates, ...futureDates]

  const historical = [...data.history.values, ...futureDates.map(() => null)]

  const bridge = (futureValues) =>
    dates.map((_, i) => {
      if (i < n - 1) return null
      if (i === n - 1) return data.current_price
      return futureValues[i - n] ?? null
    })

  const ensemble = bridge(data.ensemble.predictions)

  const modelColorMap = {}
  data.ranked_models.forEach((m, i) => {
    modelColorMap[m.method] = MODEL_COLORS[i % MODEL_COLORS.length]
  })

  const models = data.ranked_models
    .filter((m) => selectedModels.includes(m.method))
    .map((m) => ({
      key: m.key,
      name: m.method,
      color: modelColorMap[m.method],
      values: bridge(m.predictions),
    }))

  const garchBand = data.garch_band
    ? {
        lower: dates.map((_, i) => (i < n ? null : data.garch_band.lower[i - n] ?? null)),
        upper: dates.map((_, i) => (i < n ? null : data.garch_band.upper[i - n] ?? null)),
      }
    : null

  const confClass = CONFIDENCE_STYLE[data.confidence] || 'text-muted'

  return (
    <>
      {/* Forecast summary */}
      <p className="section-title">Forecast Summary</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard label="Current Price" value={fmtNumber(data.current_price)} delta={data.unit} />
        <KpiCard
          label={`Next ${data.period_label}`}
          value={fmtNumber(data.next_period.value)}
          delta={fmtPercent(data.next_period.change_pct)}
          deltaValue={data.next_period.change_pct}
        />
        <KpiCard
          label="Model Confidence"
          value={data.confidence.toUpperCase()}
          valueClassName={confClass}
          delta={`avg MAPE ${fmtNumber(data.avg_mape, 1)}%`}
        />
        <KpiCard
          label="Historical Position"
          value={data.current_percentile !== null ? `${Math.round(data.current_percentile)}th pct` : '—'}
        />
      </div>

      {/* Forecast visualization */}
      <p className="section-title">Forecast Visualisation</p>
      <div className="card">
        <div className="flex flex-wrap gap-2 mb-4">
          {data.ranked_models.map((m) => (
            <button
              key={m.method}
              onClick={() => toggleModel(m.method)}
              className={`font-mono text-[0.68rem] px-3 py-1 rounded-full border transition-colors ${
                selectedModels.includes(m.method)
                  ? 'border-border text-heading bg-border-soft'
                  : 'border-border text-faint hover:text-muted'
              }`}
              style={selectedModels.includes(m.method) ? { color: modelColorMap[m.method] } : undefined}
            >
              {m.method}
              {m.is_top && ' ★'}
            </button>
          ))}
        </div>

        <ForecastChart
          dates={dates}
          todayDate={data.last_date}
          historical={historical}
          ensemble={ensemble}
          models={models}
          garchBand={garchBand}
          yLabel={data.unit}
          height={460}
        />

        <p className="font-mono text-[0.7rem] text-subtle mt-2">
          ★ top-3 models by hold-out MAPE · shaded band = GARCH 90% volatility interval (future periods only)
        </p>
      </div>

      {/* Period breakdown */}
      <p className="section-title">Period-by-Period Forecast</p>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {data.ensemble.predictions.map((pred, i) => {
          const chg = data.current_price ? ((pred - data.current_price) / data.current_price) * 100 : null
          return (
            <div key={i} className="card">
              <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.1em] mb-1">
                {data.period_label} {i + 1} · {fmtDate(futureDates[i], { month: 'short', year: '2-digit' })}
              </p>
              <p className="font-mono text-[1.3rem] font-semibold text-heading">{fmtNumber(pred)}</p>
              <p className={`font-mono text-[0.78rem] mt-1 ${changeColorClass(chg)}`}>{fmtPercent(chg)} vs. current</p>
            </div>
          )
        })}
      </div>

      {/* Model performance */}
      <p className="section-title">Model Performance Comparison</p>
      <ModelPerformanceTable models={data.ranked_models} />

      {/* Historical context */}
      {data.histogram && (
        <>
          <p className="section-title">Historical Context</p>
          <div className="card">
            <HistogramChart
              values={data.histogram.values}
              current={data.histogram.forecast}
              secondary={data.histogram.current}
              secondaryLabel="Current"
              currentLabel={`Forecast (${Math.round(data.histogram.forecast_percentile ?? 0)}th pct)`}
              xLabel={data.unit}
              height={220}
            />
          </div>
        </>
      )}

      {/* Methodology */}
      <p className="section-title">Methodology</p>
      <Methodology />
    </>
  )
}

function ModelPerformanceTable({ models }) {
  if (!models || models.length === 0) {
    return <InfoState message="No individual model metrics available." />
  }

  const bestMape = Math.min(...models.map((m) => m.mape))

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">Model</th>
            <th className="text-right px-4 py-3">MAPE</th>
            <th className="text-right px-4 py-3">MAE</th>
            <th className="text-left px-4 py-3 w-1/2">Relative Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const barWidth = Math.max(4, Math.min(100, (1 - (m.mape - bestMape) / (bestMape + 1)) * 100))
            return (
              <tr key={m.key} className="border-b border-panel hover:bg-border-soft/40">
                <td className="px-4 py-2 text-body">
                  {m.method}
                  {m.is_top && <span className="text-warn ml-1">★</span>}
                </td>
                <td className="px-4 py-2 text-right font-mono text-heading">{fmtNumber(m.mape, 2)}%</td>
                <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(m.mae, 3)}</td>
                <td className="px-4 py-2">
                  <div className="h-2 bg-border rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${m.is_top ? 'bg-warn' : 'bg-faint'}`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Methodology() {
  return (
    <div className="card text-sm text-muted leading-relaxed space-y-3">
      <p>
        Each forecast combines several independent models, each suited to a different aspect of commodity price
        behaviour. <strong className="text-body">ARIMA</strong> and <strong className="text-body">SARIMA</strong>{' '}
        capture autoregressive trend and seasonal patterns; <strong className="text-body">Ridge regression</strong>{' '}
        models the relationship between a commodity's price and lagged values of related series;{' '}
        <strong className="text-body">GARCH</strong> models time-varying volatility and is used to derive the shaded
        90% confidence band shown on the chart.
      </p>
      <p>
        The <strong className="text-body">Ensemble</strong> forecast is a weighted blend of the individual models,
        with weights informed by each model's historical hold-out accuracy. <strong className="text-body">MAPE</strong>{' '}
        (Mean Absolute Percentage Error) and <strong className="text-body">MAE</strong> (Mean Absolute Error) are
        computed on a held-out portion of history and used both to rank models and to set the overall confidence
        label (high / medium / low).
      </p>
      <p>
        Forecasts are recomputed on every pipeline run and reflect only information available up to the last observed
        date shown on the chart. They are statistical projections, not investment advice, and accuracy degrades the
        further out the horizon extends.
      </p>
    </div>
  )
}
