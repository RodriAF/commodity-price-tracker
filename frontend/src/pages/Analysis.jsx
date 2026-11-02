import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import TopBar from '../components/layout/TopBar'
import { useLanguage } from '../i18n/LanguageContext'
import PriceHistoryChart from '../components/PriceHistoryChart'
import ZScoreTimeSeriesChart from '../components/ZScoreTimeSeriesChart'
import HistogramChart from '../components/HistogramChart'
import CorrelationHeatmap from '../components/CorrelationHeatmap'
import AssetStatCard from '../components/AssetStatCard'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { fmtNumber } from '../utils/format'

// Line colours per selected asset (same order as the cards). 12 distinct colours so
// that selecting many assets doesn't recycle colours too early.
const ACCENT_COLORS = [
  '#f59e0b', '#22c55e', '#38bdf8', '#a78bfa', '#f87171', '#34d399',
  '#fb923c', '#e879f9', '#facc15', '#2dd4bf', '#818cf8', '#f472b6',
]
const RANGE_OPTIONS = ['3M', '6M', '1Y', '2Y', 'All']

export default function Analysis() {
  const { t, tDyn, tCat, ord } = useLanguage()
  const { data: commData, loading: commLoading, error: commError } = useApiData(() => api.commodities(), [])

  const [sector, setSector] = useState('all')
  const [selectedAssets, setSelectedAssets] = useState([])
  const [normalize, setNormalize] = useState(false)
  const [showMA, setShowMA] = useState(true)
  const [showZ, setShowZ] = useState(true)
  const [range, setRange] = useState('2Y')

  // Sector id -> pool of commodity keys, e.g. { all: [...], crop: [...], ... }
  // (ids are language-independent; labels are translated at render time)
  const sectorOptions = useMemo(() => {
    if (!commData) return { all: [] }
    const all = commData.commodities.map((c) => c.key)
    const options = { all }
    Object.entries(commData.categories).forEach(([cat, keys]) => {
      const filtered = keys.filter((k) => all.includes(k))
      if (filtered.length > 0) options[cat] = filtered
    })
    return options
  }, [commData])

  const sectorLabel = (id) =>
    id === 'all'
      ? t('common.all')
      : tCat(id, id.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()))

  const pool = sectorOptions[sector] || []

  // Default selection: first 3 assets of the chosen sector.
  useEffect(() => {
    if (pool.length > 0) {
      setSelectedAssets(pool.slice(0, Math.min(3, pool.length)))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sector, commData])

  const commodityName = (key) => tDyn(commData?.commodities.find((c) => c.key === key)?.name || key.replace(/_/g, ' '))

  const toggleAsset = (key) => {
    setSelectedAssets((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]))
  }

  const { data, loading, error } = useApiData(
    () =>
      selectedAssets.length > 0
        ? api.analysis({ commodities: selectedAssets.join(','), range, normalize })
        : Promise.resolve(null),
    [selectedAssets, range, normalize],
  )

  if (commLoading) return <LoadingState label={t('analysis.loadingConfig')} />
  if (commError) return <ErrorState message={commError.detail || commError.message} />

  return (
    <div>
      <TopBar />
      <PageHeader subtitle={t('analysis.subtitle')} />

      {/* Chart configuration */}
      <div className="card flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">{t('analysis.sector')}</p>
            <select
              value={sector}
              onChange={(e) => setSector(e.target.value)}
              className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body"
            >
              {Object.keys(sectorOptions).map((id) => (
                <option key={id} value={id}>
                  {sectorLabel(id)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-4 ml-auto">
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />
              {t('analysis.normalize')}
            </label>
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={showMA} onChange={(e) => setShowMA(e.target.checked)} />
              {t('analysis.ma')}
            </label>
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={showZ} onChange={(e) => setShowZ(e.target.checked)} />
              {t('analysis.zscore')}
            </label>
          </div>
        </div>

        <div>
          <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">{t('analysis.assets')}</p>
          <div className="flex flex-wrap gap-2">
            {pool.map((key) => (
              <button
                key={key}
                onClick={() => toggleAsset(key)}
                className={`font-mono text-[0.7rem] px-3 py-1 rounded-full border transition-colors ${
                  selectedAssets.includes(key)
                    ? 'border-border text-heading bg-border-soft'
                    : 'border-border text-faint hover:text-muted'
                }`}
              >
                {commodityName(key)}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">{t('analysis.window')}</p>
          <div className="flex gap-2">
            {RANGE_OPTIONS.map((opt) => (
              <button
                key={opt}
                onClick={() => setRange(opt)}
                className={`font-mono text-[0.7rem] px-3 py-1 rounded border transition-colors ${
                  range === opt ? 'border-warn text-warn' : 'border-border text-faint hover:text-muted'
                }`}
              >
                {t(`analysis.range.${opt}`)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {selectedAssets.length === 0 && <InfoState message={t('analysis.awaiting')} />}

      {loading && <LoadingState label={t('analysis.loading')} />}
      {error && <ErrorState message={error.detail || error.message} />}

      {data && (() => {
        // `data` can briefly lag behind `selectedAssets` while a new request
        // for the analysis endpoint is in flight (e.g. useApiData keeps
        // showing the previous response during loading). Restricting to the
        // keys that actually exist in `data.series`/`data.current` avoids
        // crashing the render (and blanking the whole page) when toggling
        // an asset that hasn't arrived in the response yet.
        const availableAssets = selectedAssets.filter((c) => data.series[c] && data.current[c])

        return (
        <>
          {/* Current market values */}
          <p className="section-title">{t('analysis.currentValues')}</p>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {availableAssets.map((c, idx) => (
              <AssetStatCard
                key={c}
                lineColor={ACCENT_COLORS[idx % ACCENT_COLORS.length]}
                name={data.series[c].name}
                unit={data.series[c].unit}
                current={data.current[c].value}
                zscore={data.current[c].zscore}
                percentile={data.current[c].percentile}
              />
            ))}
          </div>

          {/* Primary time series chart */}
          <p className="section-title">{t('analysis.trends')}</p>
          <div className="card">
            <PriceHistoryChart
              dates={data.dates}
              series={availableAssets.reduce((acc, c, idx) => {
                const color = ACCENT_COLORS[idx % ACCENT_COLORS.length]
                acc[c] = { name: tDyn(data.series[c].name), values: data.series[c].values, color, strokeWidth: 2 }
                if (showMA && data.series[c].ma) {
                  acc[`${c}_ma`] = {
                    name: t('analysis.maSuffix', { name: tDyn(data.series[c].name) }),
                    values: data.series[c].ma,
                    color,
                    dashed: true,
                    opacity: 0.45,
                    strokeWidth: 1,
                    legendType: 'none',
                  }
                }
                return acc
              }, {})}
              height={420}
              yLabel={normalize ? t('analysis.yNormalized') : t('analysis.yPrice')}
            />
            {normalize && (
              <p className="font-mono text-[0.7rem] text-subtle mt-2">
                {t('analysis.normalizeNote')}
              </p>
            )}
          </div>

          {/* Z-score deviation (single asset) */}
          {showZ && availableAssets.length === 1 && data.series[availableAssets[0]].zscore && (
            <>
              <p className="section-title">{t('analysis.zSection')}</p>
              <div className="card">
                <ZScoreTimeSeriesChart dates={data.dates} values={data.series[availableAssets[0]].zscore} />
                <p className="font-mono text-[0.7rem] text-subtle mt-2">
                  {t('analysis.zNote')}
                </p>
              </div>
            </>
          )}

          {/* Historical frequency distribution (single asset) */}
          {data.histogram && (
            <>
              <p className="section-title">{t('analysis.distribution')}</p>
              <div className="card">
                <HistogramChart
                  values={data.histogram.values}
                  current={data.histogram.current}
                  secondary={data.histogram.median}
                  secondaryLabel={t('analysis.median')}
                  currentLabel={t('analysis.currentPct', { n: Math.round(data.histogram.percentile ?? 0) })}
                  xLabel={data.histogram.unit ? tDyn(data.histogram.unit) : t('analysis.quotedPrice')}
                  height={240}
                />
              </div>
              <PercentileMessage histogram={data.histogram} sinceYear={data.histogram.since_year} />
            </>
          )}

          {/* Correlation matrix (2+ assets) */}
          {data.correlation && (
            <>
              <p className="section-title">{t('analysis.correlation')}</p>
              <div className="card">
                <CorrelationHeatmap labels={data.correlation.labels.map(tDyn)} matrix={data.correlation.matrix} />
              </div>
              <CorrelationPairs correlation={{ ...data.correlation, labels: data.correlation.labels.map(tDyn) }} />
            </>
          )}

          {/* Descriptive stats summary */}
          <p className="section-title">{t('analysis.summary')}</p>
          <StatsTable stats={data.stats} />
        </>
        )
      })()}
    </div>
  )
}

function PercentileMessage({ histogram, sinceYear }) {
  const { t, ord } = useLanguage()
  const pctile = histogram.percentile ?? 0
  const params = { p: ord(pctile), year: sinceYear }
  let msg
  if (pctile > 85) {
    msg = t('analysis.pctHigh', params)
  } else if (pctile < 15) {
    msg = t('analysis.pctLow', params)
  } else {
    msg = t('analysis.pctNormal', params)
  }
  return (
    <p className="font-mono text-[0.82rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint">
      {msg}
    </p>
  )
}

function CorrelationPairs({ correlation }) {
  const { t } = useLanguage()
  const { labels, matrix } = correlation
  const pairs = []
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      pairs.push({ a: labels[i], b: labels[j], r: matrix[i][j] })
    }
  }
  const positive = pairs.filter((p) => p.r > 0).sort((a, b) => b.r - a.r).slice(0, 3)
  const negative = pairs.filter((p) => p.r < 0).sort((a, b) => a.r - b.r).slice(0, 3)

  if (pairs.length === 0) return null

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
      <div>
        <p className="font-mono text-[0.7rem] text-faint uppercase tracking-[0.1em] mb-2">
          {t('analysis.corrPositive')}
        </p>
        {positive.map((p, i) => (
          <p key={i} className="font-mono text-[0.8rem] text-muted my-1">
            {p.a} &amp; {p.b} <span className="text-pos font-semibold">{p.r > 0 ? '+' : ''}{p.r.toFixed(3)}</span>
          </p>
        ))}
      </div>
      <div>
        <p className="font-mono text-[0.7rem] text-faint uppercase tracking-[0.1em] mb-2">
          {t('analysis.corrInverse')}
        </p>
        {negative.map((p, i) => (
          <p key={i} className="font-mono text-[0.8rem] text-muted my-1">
            {p.a} &amp; {p.b} <span className="text-neg font-semibold">{p.r.toFixed(3)}</span>
          </p>
        ))}
      </div>
    </div>
  )
}

function StatsTable({ stats }) {
  const { t, tDyn } = useLanguage()
  const pctileColor = (val) => {
    if (val === null || val === undefined) return 'text-subtle'
    if (val > 80) return 'text-neg font-semibold'
    if (val < 20) return 'text-pos font-semibold'
    return 'text-warn'
  }

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">{t('analysis.th.asset')}</th>
            <th className="text-left px-4 py-3">{t('common.freqShort')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.current')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.pctile')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.mean')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.vol')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.min')}</th>
            <th className="text-right px-4 py-3">{t('analysis.th.max')}</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.key} className="border-b border-panel hover:bg-border-soft/40">
              <td className="px-4 py-2 text-body">{tDyn(s.name)}</td>
              <td className="px-4 py-2 text-muted font-mono text-xs">{s.frequency}</td>
              <td className="px-4 py-2 text-right font-mono text-heading">{fmtNumber(s.current)}</td>
              <td className={`px-4 py-2 text-right font-mono ${pctileColor(s.percentile)}`}>
                {s.percentile !== null ? `${s.percentile.toFixed(1)}%` : '—'}
              </td>
              <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(s.mean)}</td>
              <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(s.std)}</td>
              <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(s.min)}</td>
              <td className="px-4 py-2 text-right font-mono text-muted">{fmtNumber(s.max)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
