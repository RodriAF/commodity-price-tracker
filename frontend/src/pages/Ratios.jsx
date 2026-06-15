import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import PriceHistoryChart from '../components/PriceHistoryChart'
import ZScoreBarChart from '../components/ZScoreBarChart'
import RatioExplorerChart from '../components/RatioExplorerChart'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { fmtZScore } from '../utils/format'

const REGIME_COLOR = {
  high: 'text-neg border-t-neg',
  normal: 'text-warn border-t-warn',
  low: 'text-pos border-t-pos',
}

const BADGE_CLASS = {
  pressure: 'bg-[rgba(239,68,68,0.15)] text-neg',
  elevated: 'bg-[rgba(245,158,11,0.12)] text-warn',
  normal: 'bg-[rgba(34,197,94,0.10)] text-pos',
}

const TREND_ICON = { up: ['↑', 'text-neg'], down: ['↓', 'text-pos'], flat: ['—', 'text-faint'] }

const COST_INDEX_COLORS = {
  energy_input_cost_index: '#f59e0b',
  fertilizer_cost_index: '#22c55e',
}

export default function Ratios() {
  const [pair, setPair] = useState(null)
  const [showTechnical, setShowTechnical] = useState(false)

  const { data, loading, error } = useApiData(() => api.ratios(pair), [pair])

  // Once the available ratio pairs are known, default to the first one.
  useEffect(() => {
    if (data && !pair && data.ratio_pairs?.length > 0) {
      setPair(data.ratio_pairs[0])
    }
  }, [data, pair])

  if (loading) return <LoadingState label="Loading input cost analytics…" />
  if (error) return <ErrorState message={error.detail || error.message} />
  if (!data) return null

  const { regime, cost_index_history, margin_pressure, zscore_bars, ratio_pairs, ratio_explorer } = data

  return (
    <div>
      <PageHeader subtitle="Input Cost Analytics" />

      {/* Section 1 — Current cost environment */}
      <p className="section-title">Current Input Cost Environment</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(regime).map(([key, r]) => {
          const colorClass = REGIME_COLOR[r.level] || REGIME_COLOR.normal
          const [textClass, borderClass] = colorClass.split(' ')
          return (
            <div key={key} className={`card border-t-[3px] ${borderClass}`}>
              <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-[0.1em] mb-1">{r.label}</p>
              <p className={`text-[1.3rem] font-semibold mb-1 ${textClass}`}>{r.display_label}</p>
              {r.index_value !== null && (
                <>
                  <p className={`font-mono text-[1.7rem] font-semibold ${textClass}`}>{r.index_value.toFixed(1)}</p>
                  <p className="font-mono text-[0.72rem] text-faint mb-1">Index (100 = historical avg)</p>
                </>
              )}
              {r.delta_3m !== null && (
                <p className="font-mono text-[0.75rem] text-muted">
                  {r.delta_3m >= 0 ? '+' : ''}
                  {r.delta_3m.toFixed(1)} pts vs. 3 months ago
                </p>
              )}
              <p className="text-[0.78rem] text-subtle leading-relaxed mt-2">{r.description}</p>
            </div>
          )
        })}
      </div>

      {/* Section 2 — Historical cost index */}
      {cost_index_history && (
        <>
          <p className="section-title">Input Cost Index — Historical Trend</p>
          <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint mb-4">
            Each series is normalised so <strong>100 = its own long-run average</strong>. Values above 110 indicate an
            above-average cost environment; below 90 a below-average one. Both series are directly comparable on this
            scale.
          </p>
          <div className="card">
            <PriceHistoryChart
              dates={cost_index_history.dates}
              series={Object.fromEntries(
                Object.entries(cost_index_history.series).map(([col, values]) => [
                  col,
                  {
                    name: col.replace('_cost_index', '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
                    values,
                    color: COST_INDEX_COLORS[col] || '#94a3b8',
                    strokeWidth: 2.5,
                  },
                ]),
              )}
              referenceLines={[
                { y: 110, color: 'rgba(239,68,68,0.6)', label: 'High cost (110)' },
                { y: 100, color: '#475569', label: 'Average (100)' },
                { y: 90, color: 'rgba(34,197,94,0.6)', label: 'Low cost (90)' },
              ]}
              yLabel="Index (100 = avg)"
              height={320}
            />
          </div>
        </>
      )}

      {/* Section 3 — Margin pressure by crop */}
      <p className="section-title">Margin Pressure by Crop</p>
      <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint mb-4">
        For each crop, the table shows whether current input costs are compressing margins relative to historical
        norms. <strong>Status</strong> is derived from how far the crop-to-input ratio deviates from its own history.
        The trend arrow shows the direction over the last 3 periods.
      </p>

      {margin_pressure.length > 0 ? (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
                <th className="text-left px-4 py-3">Crop</th>
                <th className="text-left px-4 py-3">Margin Status</th>
                <th className="text-center px-4 py-3">Trend</th>
                <th className="text-left px-4 py-3">Key Driver</th>
              </tr>
            </thead>
            <tbody>
              {margin_pressure.map((r) => {
                const [icon, iconColor] = TREND_ICON[r.trend] || TREND_ICON.flat
                const otherDrivers = r.drivers
                  .filter((d) => d.input !== r.worst_input)
                  .map((d) => `${d.label} (${fmtZScore(d.z, 1)})`)
                  .join(', ')
                return (
                  <tr key={r.crop} className="border-b border-panel hover:bg-border-soft/40">
                    <td className="px-4 py-2 text-body font-semibold">{r.crop_label}</td>
                    <td className="px-4 py-2">
                      <span className={`inline-block font-mono text-[0.68rem] font-semibold px-2 py-1 rounded uppercase tracking-wide ${BADGE_CLASS[r.level]}`}>
                        {r.status}
                      </span>
                    </td>
                    <td className={`px-4 py-2 text-center text-base ${iconColor}`}>{icon}</td>
                    <td className="px-4 py-2 font-mono text-[0.78rem] text-muted">
                      Worst driver: {r.worst_input_label} ({fmtZScore(r.worst_z, 1)})
                      {otherDrivers && <div className="text-[0.72rem] text-faint mt-0.5">{otherDrivers}</div>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <InfoState message="No margin data available — not enough historical data to compute z-scores." />
      )}

      {/* Section 4 — Technical detail (collapsed by default) */}
      <p className="section-title">Technical Detail</p>
      <button
        onClick={() => setShowTechnical((v) => !v)}
        className="font-mono text-[0.7rem] text-muted uppercase tracking-wide px-3 py-1.5 border border-border rounded hover:text-heading mb-4"
      >
        {showTechnical ? 'Hide' : 'Show'} Z-Scores &amp; Ratio Explorer
      </button>

      {showTechnical && (
        <div className="flex flex-col gap-6">
          <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint">
            Z-scores measure how far each ratio deviates from its own rolling historical average.{' '}
            <strong>+z: crop price elevated relative to input cost (favourable margins).</strong>{' '}
            <strong>−z: input cost elevated relative to crop price (margin pressure).</strong> The ratio value itself
            is not comparable across pairs due to different units.
          </p>

          {zscore_bars.length > 0 && (
            <div className="card">
              <ZScoreBarChart data={zscore_bars} height={320} />
              <p className="font-mono text-[0.7rem] text-subtle mt-2">
                |z| &gt; 2 = extreme · |z| &gt; 1 = notable · normal range
              </p>
            </div>
          )}

          {ratio_pairs.length > 0 ? (
            <div>
              <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.08em] mb-2">
                Select crop / input pair
              </p>
              <select
                value={pair ?? ''}
                onChange={(e) => setPair(e.target.value)}
                className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body mb-4"
              >
                {ratio_pairs.map((p) => (
                  <option key={p} value={p}>
                    {p.replace('_to_', ' / ').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </option>
                ))}
              </select>

              {ratio_explorer && (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                  <div className="card lg:col-span-3">
                    <RatioExplorerChart
                      dates={ratio_explorer.dates}
                      values={ratio_explorer.values}
                      mean={ratio_explorer.mean}
                      std={ratio_explorer.std}
                    />
                  </div>
                  <RatioExplorerSidePanel explorer={ratio_explorer} />
                </div>
              )}
            </div>
          ) : (
            <InfoState message="No profitability ratio data available." />
          )}
        </div>
      )}
    </div>
  )
}

function RatioExplorerSidePanel({ explorer }) {
  const z = explorer.zscore ?? 0
  const level = Math.abs(z) > 2 ? 'extreme' : Math.abs(z) > 1 ? 'notable' : 'normal'
  const color = { extreme: 'text-neg', notable: 'text-warn', normal: 'text-pos' }[level]
  const pctile = explorer.percentile

  let interpretation = 'Within normal historical range.'
  if (pctile !== null) {
    if (pctile > 85) interpretation = 'Crop price historically strong vs. input cost.'
    else if (pctile < 15) interpretation = 'Input cost historically elevated vs. crop price — margin pressure.'
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="card">
        <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-wide mb-1">Z-Score</p>
        <p className={`font-mono text-[1.6rem] font-semibold ${color}`}>{fmtZScore(z)}</p>
        <p className="font-mono text-[0.75rem] text-faint mt-1">
          Percentile: {pctile !== null ? `${Math.round(pctile)}th` : '—'}
        </p>
      </div>
      <div className="card">
        <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-wide mb-2">Statistics</p>
        <div className="font-mono text-[0.78rem] text-muted leading-loose">
          <p>Mean:&nbsp; {explorer.mean?.toFixed(3) ?? '—'}</p>
          <p>Std:&nbsp;&nbsp; {explorer.std?.toFixed(3) ?? '—'}</p>
          <p>Min:&nbsp;&nbsp; {explorer.min?.toFixed(3) ?? '—'}</p>
          <p>Max:&nbsp;&nbsp; {explorer.max?.toFixed(3) ?? '—'}</p>
        </div>
      </div>
      <p className="font-mono text-[0.74rem] text-subtle leading-relaxed">{interpretation}</p>
    </div>
  )
}
