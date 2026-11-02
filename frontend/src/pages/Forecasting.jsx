import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import KpiCard from '../components/KpiCard'
import ForecastChart from '../components/ForecastChart'
import HistogramChart from '../components/HistogramChart'
import TopBar from '../components/layout/TopBar'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { Rich, useLanguage } from '../i18n/LanguageContext'
import {
  fmtNumber,
  fmtPercent,
  fmtDate,
  fmtRatioPct,
  fmtSignedRatioPct,
  changeColorClass,
} from '../utils/format'

const MODEL_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f87171', '#facc15', '#fb923c', '#2dd4bf', '#e879f9']

const CONFIDENCE_STYLE = {
  high: 'text-pos',
  medium: 'text-warn',
  low: 'text-neg',
}

export default function Forecasting() {
  const { t, tDyn } = useLanguage()
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

  if (availLoading) return <LoadingState label={t('forecast.loadingAvailable')} />
  if (availError) return <ErrorState message={availError.detail || availError.message} />
  if (!available || available.commodities.length === 0) {
    return (
      <div>
        <PageHeader subtitle={t('forecast.subtitle')} />
        <InfoState message={t('forecast.none')} />
      </div>
    )
  }

  const toggleModel = (method) => {
    setSelectedModels((prev) => (prev.includes(method) ? prev.filter((m) => m !== method) : [...prev, method]))
  }

  return (
    <div>
      <TopBar />
      <PageHeader
        subtitle={t('forecast.subtitle')}
        action={
          <div>
            <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1 text-right">{t('forecast.commodity')}</p>
            <select
              value={commodity ?? ''}
              onChange={(e) => setCommodity(e.target.value)}
              className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body"
            >
              {available.commodities.map((c) => (
                <option key={c.key} value={c.key}>
                  {tDyn(c.name)}
                </option>
              ))}
            </select>
          </div>
        }
      />

      {loading && <LoadingState label={t('forecast.loading')} />}
      {error && <ErrorState message={error.detail || error.message} />}

      {data && <ForecastBody data={data} selectedModels={selectedModels} toggleModel={toggleModel} />}
    </div>
  )
}

function ForecastBody({ data, selectedModels, toggleModel }) {
  const { t, tDyn, tOr } = useLanguage()
  const unit = tDyn(data.unit)
  const periodLabel = tDyn(data.period_label)

  // Evaluation fields only exist for forecasts produced by the rolling-origin
  // pipeline. Everything below degrades gracefully when they are missing
  // (older forecast rows, or a backend that has not been updated yet).
  const ev = data.evaluation ?? {}
  const hasEval = ev.skill_vs_naive != null
  const horizon = data.horizon ?? data.ensemble.predictions.length

  // ---------------------------------------------------------------- //
  // Build the combined date axis (history + future) and aligned series //
  // ---------------------------------------------------------------- //
  const histDates = data.history.dates
  const futureDates = data.future_dates
  const n = histDates.length
  const dates = [...histDates, ...futureDates]

  const historical = [...data.history.values, ...futureDates.map(() => null)]

  // Series that continue from the last real observation: they start at the
  // current price so forecasts visually connect to the history.
  const bridge = (futureValues) =>
    dates.map((_, i) => {
      if (i < n - 1) return null
      if (i === n - 1) return data.current_price
      return futureValues?.[i - n] ?? null
    })

  const ensemble = bridge(data.ensemble.predictions)
  const baseline = data.baseline?.predictions ? bridge(data.baseline.predictions) : null

  // Prediction-interval fan: zero width at the last observation, opening afterwards.
  const bands =
    data.intervals?.['80'] && data.intervals?.['95']
      ? {
          b80: { lower: bridge(data.intervals['80'].lower), upper: bridge(data.intervals['80'].upper) },
          b95: { lower: bridge(data.intervals['95'].lower), upper: bridge(data.intervals['95'].upper) },
        }
      : null

  const modelColorMap = {}
  data.ranked_models.forEach((m, i) => {
    modelColorMap[m.method] = MODEL_COLORS[i % MODEL_COLORS.length]
  })

  // The naive baseline has its own dedicated line, so it is not an overlay option.
  const overlayModels = data.ranked_models.filter((m) => !m.is_baseline)

  const models = overlayModels
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
  const beatsNaive = hasEval && ev.skill_vs_naive > 0
  const skillClass = !hasEval ? 'text-muted' : beatsNaive ? 'text-pos' : 'text-neg'

  const lo80 = data.next_period.lower80
  const hi80 = data.next_period.upper80
  const range80 = lo80 != null && hi80 != null ? `${fmtNumber(lo80)} – ${fmtNumber(hi80)}` : null

  const periodDate = (i) =>
    fmtDate(
      futureDates[i],
      data.frequency === 'daily' || data.frequency === 'weekly'
        ? { day: '2-digit', month: 'short', year: '2-digit' }
        : { month: 'short', year: 'numeric' },
    )

  const confidenceCard = (
    <KpiCard
      label={t('forecast.kpi.confidence')}
      value={tOr(`forecast.confidence.${data.confidence}`, data.confidence.toUpperCase())}
      valueClassName={confClass}
      delta={
        hasEval
          ? beatsNaive
            ? t('forecast.kpi.beatsNaive')
            : t('forecast.kpi.noBeatNaive')
          : t('forecast.kpi.avgMape', { n: fmtNumber(data.avg_mape, 1) })
      }
      deltaValue={hasEval ? (beatsNaive ? 1 : -1) : undefined}
    />
  )

  return (
    <>
      {/* Forecast summary */}
      <p className="section-title">{t('forecast.summary')}</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label={t('forecast.kpi.price')}
          value={fmtNumber(data.current_price)}
          delta={t('forecast.kpi.asOf', {
            unit,
            date: fmtDate(data.last_date, { day: '2-digit', month: 'short' }),
          })}
        />
        <KpiCard
          label={t('forecast.kpi.next', { period: periodLabel })}
          value={fmtNumber(data.next_period.value)}
          delta={fmtPercent(data.next_period.change_pct)}
          deltaValue={data.next_period.change_pct}
        />
        {range80 ? (
          <KpiCard label={t('forecast.kpi.range80')} value={range80} delta={t('forecast.kpi.range80Sub')} />
        ) : (
          confidenceCard
        )}
        <KpiCard
          label={t('forecast.kpi.position')}
          value={data.current_percentile !== null ? t('pct.short', { n: Math.round(data.current_percentile) }) : '—'}
          delta={data.data_quality?.n_obs ? t('forecast.kpi.obs', { n: data.data_quality.n_obs }) : undefined}
        />
      </div>

      {/* Backtest evaluation */}
      {hasEval && (
        <>
          <p className="section-title">{t('forecast.eval.title')}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {confidenceCard}
            <KpiCard
              label={t('forecast.kpi.skill')}
              value={fmtSignedRatioPct(ev.skill_vs_naive)}
              valueClassName={skillClass}
              delta={t('forecast.kpi.skillSub')}
            />
            <KpiCard
              label={t('forecast.kpi.mae')}
              value={fmtNumber(ev.mae, 2)}
              delta={
                ev.naive_mae != null
                  ? t('forecast.kpi.maeSub', { naive: fmtNumber(ev.naive_mae, 2), n: ev.n_folds ?? '—' })
                  : t('forecast.kpi.maeSubNoNaive', { n: ev.n_folds ?? '—' })
              }
            />
            <KpiCard
              label={t('forecast.kpi.coverage')}
              value={fmtRatioPct(ev.coverage_80)}
              delta={t('forecast.kpi.coverageSub', { c95: fmtRatioPct(ev.coverage_95) })}
            />
          </div>

          {!beatsNaive && (
            <div className="card mt-4 text-sm text-muted border-l-[3px] border-l-warn">
              <strong className="text-body">{t('forecast.naiveWarning.title')}</strong> {t('forecast.naiveWarning.body')}
            </div>
          )}
        </>
      )}

      {/* Forecast visualization */}
      <p className="section-title">{t('forecast.visualisation')}</p>
      <div className="card">
        <div className="flex flex-wrap gap-2 mb-4">
          {overlayModels.map((m) => (
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
          baseline={baseline}
          bands={bands}
          models={models}
          garchBand={garchBand}
          yLabel={unit}
          height={500}
        />

        <p className="font-mono text-[0.7rem] text-subtle mt-2">
          {t('forecast.legendNote')}
          {garchBand ? t('forecast.legendGarch') : ''}
        </p>
      </div>

      {/* Period breakdown */}
      <p className="section-title">{t('forecast.periodByPeriod')}</p>
      <PeriodTable data={data} periodDate={periodDate} />

      {/* Model performance */}
      <p className="section-title">{t('forecast.performance')}</p>
      <ModelPerformanceTable models={data.ranked_models} />

      {/* Historical context */}
      {data.histogram && (
        <>
          <p className="section-title">{t('forecast.context')}</p>
          <div className="card">
            <HistogramChart
              values={data.histogram.values}
              current={data.histogram.forecast}
              secondary={data.histogram.current}
              secondaryLabel={t('common.current')}
              currentLabel={t('forecast.forecastPct', { n: Math.round(data.histogram.forecast_percentile ?? 0) })}
              xLabel={unit}
              height={240}
            />
          </div>
        </>
      )}

      {/* Methodology */}
      <p className="section-title">{t('forecast.methodology')}</p>
      <Methodology horizon={horizon} />
    </>
  )
}

function PeriodTable({ data, periodDate }) {
  const { t } = useLanguage()
  const i80 = data.intervals?.['80']
  const i95 = data.intervals?.['95']
  const hasBaseline = Array.isArray(data.baseline?.predictions)
  const range = (iv, i) =>
    iv?.lower?.[i] != null && iv?.upper?.[i] != null ? `${fmtNumber(iv.lower[i])} – ${fmtNumber(iv.upper[i])}` : '—'

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">{t('forecast.th.step')}</th>
            <th className="text-left px-4 py-3">{t('forecast.th.date')}</th>
            <th className="text-right px-4 py-3">{t('forecast.th.forecast')}</th>
            <th className="text-right px-4 py-3">{t('forecast.th.vsCurrent')}</th>
            {i80 && <th className="text-right px-4 py-3">{t('forecast.th.range80')}</th>}
            {i95 && <th className="text-right px-4 py-3">{t('forecast.th.range95')}</th>}
            {hasBaseline && <th className="text-right px-4 py-3">{t('forecast.th.naive')}</th>}
          </tr>
        </thead>
        <tbody>
          {data.ensemble.predictions.map((pred, i) => {
            const chg = data.current_price ? ((pred - data.current_price) / data.current_price) * 100 : null
            return (
              <tr key={i} className="border-b border-panel hover:bg-border-soft/40">
                <td className="px-4 py-2 font-mono text-muted">T+{i + 1}</td>
                <td className="px-4 py-2 text-muted">{periodDate(i)}</td>
                <td className="px-4 py-2 text-right font-mono font-semibold text-heading">{fmtNumber(pred)}</td>
                <td className={`px-4 py-2 text-right font-mono ${changeColorClass(chg)}`}>{fmtPercent(chg)}</td>
                {i80 && <td className="px-4 py-2 text-right font-mono text-muted">{range(i80, i)}</td>}
                {i95 && <td className="px-4 py-2 text-right font-mono text-faint">{range(i95, i)}</td>}
                {hasBaseline && (
                  <td className="px-4 py-2 text-right font-mono text-faint">{fmtNumber(data.baseline.predictions[i])}</td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ModelPerformanceTable({ models }) {
  const { t } = useLanguage()

  if (!models || models.length === 0) {
    return <InfoState message={t('forecast.noMetrics')} />
  }

  const hasRmae = models.some((m) => m.rmae != null)
  const hasDirAcc = models.some((m) => m.directional_accuracy != null)
  const hasWeight = models.some((m) => m.weight != null)

  const mapes = models.map((m) => m.mape).filter((v) => v != null)
  const bestMape = mapes.length ? Math.min(...mapes) : 0
  const rmaes = models.map((m) => m.rmae).filter((v) => v != null)
  const bestRmae = rmaes.length ? Math.min(...rmaes) : null

  // Bar = how close each model is to the best one (best = full width). Uses the
  // error relative to naive when available, MAPE for older forecast rows.
  const barFor = (m) => {
    if (hasRmae && m.rmae != null && bestRmae != null && m.rmae > 0) {
      return Math.max(4, Math.min(100, (bestRmae / m.rmae) * 100))
    }
    if (m.mape == null) return 4
    return Math.max(4, Math.min(100, (1 - (m.mape - bestMape) / (bestMape + 1)) * 100))
  }

  const badge = 'ml-2 font-mono text-[0.6rem] uppercase tracking-wide text-faint border border-border rounded px-1.5 py-0.5'

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">{t('forecast.th.model')}</th>
            {hasRmae && (
              <th className="text-right px-4 py-3" title={t('forecast.th.vsNaiveHint')}>
                {t('forecast.th.vsNaive')}
              </th>
            )}
            <th className="text-right px-4 py-3">MAE</th>
            <th className="text-right px-4 py-3">MAPE</th>
            {hasDirAcc && (
              <th className="text-right px-4 py-3" title={t('forecast.th.dirAccHint')}>
                {t('forecast.th.dirAcc')}
              </th>
            )}
            {hasWeight && <th className="text-right px-4 py-3">{t('forecast.th.weight')}</th>}
            <th className="text-left px-4 py-3 w-1/4">{t('forecast.th.accuracy')}</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const beats = m.rmae != null && m.rmae < 1
            return (
              <tr key={m.key} className="border-b border-panel hover:bg-border-soft/40">
                <td className="px-4 py-2 text-body">
                  {m.method}
                  {m.is_top && <span className="text-warn ml-1">★</span>}
                  {m.is_baseline && <span className={badge}>{t('forecast.badge.baseline')}</span>}
                  {m.role === 'benchmark' && <span className={badge}>{t('forecast.badge.benchmark')}</span>}
                </td>
                {hasRmae && (
                  <td
                    className={`px-4 py-2 text-right font-mono ${
                      m.rmae == null || m.is_baseline ? 'text-muted' : beats ? 'text-pos' : 'text-neg'
                    }`}
                  >
                    {m.rmae != null ? m.rmae.toFixed(2) : '—'}
                  </td>
                )}
                <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(m.mae, 3)}</td>
                <td className="px-4 py-2 text-right font-mono text-heading">
                  {m.mape != null ? `${fmtNumber(m.mape, 2)}%` : '—'}
                </td>
                {hasDirAcc && (
                  <td className="px-4 py-2 text-right font-mono text-muted">
                    {m.directional_accuracy != null ? `${Math.round(m.directional_accuracy)}%` : '—'}
                  </td>
                )}
                {hasWeight && (
                  <td className="px-4 py-2 text-right font-mono text-muted">
                    {m.weight != null ? fmtRatioPct(m.weight) : '—'}
                  </td>
                )}
                <td className="px-4 py-2">
                  <div className="h-2 bg-border rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${m.is_top ? 'bg-warn' : 'bg-faint'}`}
                      style={{ width: `${barFor(m)}%` }}
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

function Methodology({ horizon }) {
  const { t } = useLanguage()

  return (
    <div className="card text-sm text-muted leading-relaxed space-y-3">
      {['forecast.method.p1', 'forecast.method.p2', 'forecast.method.p3', 'forecast.method.p4'].map((key) => (
        <p key={key}>
          <Rich text={t(key, { horizon })} strongClassName="text-body" />
        </p>
      ))}
    </div>
  )
}
