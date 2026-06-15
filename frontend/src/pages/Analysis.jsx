import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import PriceHistoryChart from '../components/PriceHistoryChart'
import ZScoreTimeSeriesChart from '../components/ZScoreTimeSeriesChart'
import HistogramChart from '../components/HistogramChart'
import CorrelationHeatmap from '../components/CorrelationHeatmap'
import AssetStatCard from '../components/AssetStatCard'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { fmtNumber } from '../utils/format'

const ACCENT_COLORS = ['#f59e0b', '#22c55e', '#38bdf8', '#a78bfa', '#f87171', '#34d399']
const RANGE_OPTIONS = ['3M', '6M', '1Y', '2Y', 'All']

export default function Analysis() {
  const { data: commData, loading: commLoading, error: commError } = useApiData(() => api.commodities(), [])

  const [sector, setSector] = useState('All')
  const [selectedAssets, setSelectedAssets] = useState([])
  const [normalize, setNormalize] = useState(false)
  const [showMA, setShowMA] = useState(true)
  const [showZ, setShowZ] = useState(true)
  const [range, setRange] = useState('2Y')

  // Sector -> pool of commodity keys, e.g. { All: [...], Crop: [...], ... }
  const sectorOptions = useMemo(() => {
    if (!commData) return { All: [] }
    const all = commData.commodities.map((c) => c.key)
    const options = { All: all }
    Object.entries(commData.categories).forEach(([cat, keys]) => {
      const filtered = keys.filter((k) => all.includes(k))
      if (filtered.length > 0) {
        options[cat.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())] = filtered
      }
    })
    return options
  }, [commData])

  const pool = sectorOptions[sector] || []

  // Default selection: first 3 assets of the chosen sector.
  useEffect(() => {
    if (pool.length > 0) {
      setSelectedAssets(pool.slice(0, Math.min(3, pool.length)))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sector, commData])

  const commodityName = (key) => commData?.commodities.find((c) => c.key === key)?.name || key.replace(/_/g, ' ')

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

  if (commLoading) return <LoadingState label="Loading configuration…" />
  if (commError) return <ErrorState message={commError.detail || commError.message} />

  return (
    <div>
      <PageHeader subtitle="Market Analysis" />

      {/* Chart configuration */}
      <div className="card flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">Market Sector</p>
            <select
              value={sector}
              onChange={(e) => setSector(e.target.value)}
              className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body"
            >
              {Object.keys(sectorOptions).map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-4 ml-auto">
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />
              Normalize (Base 100)
            </label>
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={showMA} onChange={(e) => setShowMA(e.target.checked)} />
              Moving Average
            </label>
            <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
              <input type="checkbox" checked={showZ} onChange={(e) => setShowZ(e.target.checked)} />
              Z-Score
            </label>
          </div>
        </div>

        <div>
          <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">Assets</p>
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
          <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-1">Observation Window</p>
          <div className="flex gap-2">
            {RANGE_OPTIONS.map((opt) => (
              <button
                key={opt}
                onClick={() => setRange(opt)}
                className={`font-mono text-[0.7rem] px-3 py-1 rounded border transition-colors ${
                  range === opt ? 'border-warn text-warn' : 'border-border text-faint hover:text-muted'
                }`}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      </div>

      {selectedAssets.length === 0 && <InfoState message="Awaiting asset selection from the configuration menu." />}

      {loading && <LoadingState label="Loading analysis…" />}
      {error && <ErrorState message={error.detail || error.message} />}

      {data && (
        <>
          {/* Current market values */}
          <p className="section-title">Current Market Values</p>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {selectedAssets.slice(0, 5).map((c) => (
              <AssetStatCard
                key={c}
                name={data.series[c].name}
                unit={data.series[c].unit}
                current={data.current[c].value}
                zscore={data.current[c].zscore}
                percentile={data.current[c].percentile}
              />
            ))}
          </div>

          {/* Primary time series chart */}
          <p className="section-title">Historical Price Trends</p>
          <div className="card">
            <PriceHistoryChart
              dates={data.dates}
              series={selectedAssets.reduce((acc, c, idx) => {
                const color = ACCENT_COLORS[idx % ACCENT_COLORS.length]
                acc[c] = { name: data.series[c].name, values: data.series[c].values, color, strokeWidth: 2 }
                if (showMA && data.series[c].ma) {
                  acc[`${c}_ma`] = {
                    name: `${data.series[c].name} MA`,
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
              yLabel={normalize ? 'Normalized Index (Base 100)' : 'Asset Price'}
            />
            {normalize && (
              <p className="font-mono text-[0.7rem] text-subtle mt-2">
                Timeseries data is synthetically rebased to an index of 100 for proper scale comparison.
              </p>
            )}
          </div>

          {/* Z-score deviation (single asset) */}
          {showZ && selectedAssets.length === 1 && data.series[selectedAssets[0]].zscore && (
            <>
              <p className="section-title">Statistical Deviation (Z-Score)</p>
              <div className="card">
                <ZScoreTimeSeriesChart dates={data.dates} values={data.series[selectedAssets[0]].zscore} />
                <p className="font-mono text-[0.7rem] text-subtle mt-2">
                  |z| &gt; 2: Standard Deviation Threshold Breached | Rolling statistical logic enforced via DuckDB metrics.
                </p>
              </div>
            </>
          )}

          {/* Historical frequency distribution (single asset) */}
          {data.histogram && (
            <>
              <p className="section-title">Historical Frequency Distribution</p>
              <div className="card">
                <HistogramChart
                  values={data.histogram.values}
                  current={data.histogram.current}
                  secondary={data.histogram.median}
                  secondaryLabel="Median Value"
                  currentLabel={`Current (${Math.round(data.histogram.percentile ?? 0)}th pct)`}
                  xLabel={data.histogram.unit || 'Quoted Price'}
                  height={220}
                />
              </div>
              <PercentileMessage histogram={data.histogram} sinceYear={data.histogram.since_year} />
            </>
          )}

          {/* Correlation matrix (2+ assets) */}
          {data.correlation && (
            <>
              <p className="section-title">Pearson Correlation Architecture</p>
              <div className="card">
                <CorrelationHeatmap labels={data.correlation.labels} matrix={data.correlation.matrix} />
              </div>
              <CorrelationPairs correlation={data.correlation} />
            </>
          )}

          {/* Descriptive stats summary */}
          <p className="section-title">Descriptive Data Summary</p>
          <StatsTable stats={data.stats} />
        </>
      )}
    </div>
  )
}

function PercentileMessage({ histogram, sinceYear }) {
  const pctile = histogram.percentile ?? 0
  let msg
  if (pctile > 85) {
    msg = `Observation flagged: value is at the ${Math.round(pctile)}th percentile, marking an elevated position above 85% of recorded history since ${sinceYear}.`
  } else if (pctile < 15) {
    msg = `Observation flagged: value is at the ${Math.round(pctile)}th percentile, marking a depressed position below 85% of recorded history since ${sinceYear}.`
  } else {
    msg = `Normal distribution range: currently at the ${Math.round(pctile)}th percentile, well within expected standard deviation profiles.`
  }
  return (
    <p className="font-mono text-[0.82rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint">
      {msg}
    </p>
  )
}

function CorrelationPairs({ correlation }) {
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
          Primary Positive Correlation Base
        </p>
        {positive.map((p, i) => (
          <p key={i} className="font-mono text-[0.8rem] text-muted my-1">
            {p.a} &amp; {p.b} <span className="text-pos font-semibold">{p.r > 0 ? '+' : ''}{p.r.toFixed(3)}</span>
          </p>
        ))}
      </div>
      <div>
        <p className="font-mono text-[0.7rem] text-faint uppercase tracking-[0.1em] mb-2">
          Primary Inverse Correlation Base
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
            <th className="text-left px-4 py-3">Asset Identifier</th>
            <th className="text-left px-4 py-3">Freq</th>
            <th className="text-right px-4 py-3">Current Value</th>
            <th className="text-right px-4 py-3">Pctile Limit</th>
            <th className="text-right px-4 py-3">Mean Avg</th>
            <th className="text-right px-4 py-3">Volatility</th>
            <th className="text-right px-4 py-3">Min Low</th>
            <th className="text-right px-4 py-3">Max High</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.key} className="border-b border-panel hover:bg-border-soft/40">
              <td className="px-4 py-2 text-body">{s.name}</td>
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
