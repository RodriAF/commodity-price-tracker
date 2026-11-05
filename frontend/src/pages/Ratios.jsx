import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import PriceHistoryChart from '../components/PriceHistoryChart'
import ZScoreBarChart from '../components/ZScoreBarChart'
import RatioExplorerChart from '../components/RatioExplorerChart'
import TopBar from '../components/layout/TopBar'
import { LoadingState, ErrorState, InfoState } from '../components/StatusStates'
import { Rich, useLanguage } from '../i18n/LanguageContext'
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

const TREND_ICON = { up: ['↑', 'text-neg'], down: ['↓', 'text-pos'], flat: ['→', 'text-faint'] }

const COST_INDEX_COLORS = {
  energy_input_cost_index: '#f59e0b',
  fertilizer_cost_index: '#22c55e',
}

const prettify = (s) => s.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())

export default function Ratios() {
  const { t, tDyn, tOr } = useLanguage()
  const [pair, setPair] = useState(null)

  const { data, loading, error } = useApiData(() => api.ratios(pair), [pair])

  // Once the available ratio pairs are known, default to the first one.
  useEffect(() => {
    if (data && !pair && data.ratio_pairs?.length > 0) {
      setPair(data.ratio_pairs[0])
    }
  }, [data, pair])

  if (loading && !data) return <LoadingState label={t('ratios.loading')} />
  if (error) return <ErrorState message={error.detail || error.message} />
  if (!data) return null

  const { regime, cost_index_history, margin_pressure, zscore_bars, ratio_pairs, ratio_explorer } = data

  return (
    <div>
      <TopBar />
      <PageHeader subtitle={t('ratios.subtitle')} />

      {/* Section 1 — Current cost environment */}
      <p className="section-title">{t('ratios.environment')}</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(regime).map(([key, r]) => {
          const colorClass = REGIME_COLOR[r.level] || REGIME_COLOR.normal
          const [textClass, borderClass] = colorClass.split(' ')
          return (
            <div key={key} className={`card border-t-[3px] ${borderClass}`}>
              <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-[0.1em] mb-1">{tDyn(r.label)}</p>
              <p className={`text-[1.3rem] font-semibold mb-1 ${textClass}`}>{tDyn(r.display_label)}</p>
              {r.index_value !== null && (
                <>
                  <p className={`font-mono text-[1.7rem] font-semibold ${textClass}`}>{r.index_value.toFixed(1)}</p>
                  <p className="font-mono text-[0.72rem] text-faint mb-1">{t('ratios.indexNote')}</p>
                </>
              )}
              {r.delta_3m !== null && (
                <p className="font-mono text-[0.75rem] text-muted">
                  {t('ratios.delta3m', { n: `${r.delta_3m >= 0 ? '+' : ''}${r.delta_3m.toFixed(1)}` })}
                </p>
              )}
              <p className="text-[0.78rem] text-subtle leading-relaxed mt-2">{tDyn(r.description)}</p>
            </div>
          )
        })}
      </div>

      {/* Section 2 — Historical cost index */}
      {cost_index_history && (
        <>
          <p className="section-title">{t('ratios.historyTitle')}</p>
          <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint mb-4">
            <Rich text={t('ratios.historyNote')} />
          </p>
          <div className="card">
            <PriceHistoryChart
              dates={cost_index_history.dates}
              series={Object.fromEntries(
                Object.entries(cost_index_history.series).map(([col, values]) => [
                  col,
                  {
                    name: tOr(`ratios.costIndex.${col}`, tDyn(prettify(col.replace('_cost_index', '')))),
                    values,
                    color: COST_INDEX_COLORS[col] || '#94a3b8',
                    strokeWidth: 2.5,
                  },
                ]),
              )}
              referenceLines={[
                { y: 110, color: 'rgba(239,68,68,0.6)', label: t('ratios.refHigh') },
                { y: 100, color: '#475569', label: t('ratios.refAvg') },
                { y: 90, color: 'rgba(34,197,94,0.6)', label: t('ratios.refLow') },
              ]}
              yLabel={t('ratios.yIndex')}
              height={320}
            />
          </div>
        </>
      )}

      {/* Section 3 — Margin pressure by crop */}
      <p className="section-title">{t('ratios.marginTitle')}</p>
      <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint mb-4">
        <Rich text={t('ratios.marginNote')} />
      </p>

      {margin_pressure.length > 0 ? (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
                <th className="text-left px-4 py-3">{t('ratios.th.crop')}</th>
                <th className="text-left px-4 py-3">{t('ratios.th.status')}</th>
                <th className="text-center px-4 py-3">{t('ratios.th.trend')}</th>
                <th className="text-left px-4 py-3">{t('ratios.th.driver')}</th>
              </tr>
            </thead>
            <tbody>
              {margin_pressure.map((r) => {
                const [icon, iconColor] = TREND_ICON[r.trend] || TREND_ICON.flat
                const otherDrivers = r.drivers
                  .filter((d) => d.input !== r.worst_input)
                  .map((d) => `${tDyn(d.label)} (${fmtZScore(d.z, 1)})`)
                  .join(', ')
                return (
                  <tr key={r.crop} className="border-b border-panel hover:bg-border-soft/40">
                    <td className="px-4 py-2 text-body font-semibold">{tDyn(r.crop_label)}</td>
                    <td className="px-4 py-2">
                      <span className={`inline-block font-mono text-[0.68rem] font-semibold px-2 py-1 rounded uppercase tracking-wide ${BADGE_CLASS[r.level]}`}>
                        {tOr(`ratios.status.${r.level}`, r.status)}
                      </span>
                    </td>
                    <td className={`px-4 py-2 text-center ${iconColor}`}>
                      <span className="text-base">{icon}</span>
                      <span className="block font-mono text-[0.62rem] uppercase tracking-wide">
                        {t(`ratios.trend.${TREND_ICON[r.trend] ? r.trend : 'flat'}`)}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-[0.78rem] text-muted">
                      {t('ratios.worstDriver', { label: tDyn(r.worst_input_label), z: fmtZScore(r.worst_z, 1) })}
                      {otherDrivers && <div className="text-[0.72rem] text-faint mt-0.5">{otherDrivers}</div>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <InfoState message={t('ratios.noMargin')} />
      )}

      {/* Section 4 — Ratio explorer (always visible) */}
      <p className="section-title">{t('ratios.selectPair')}</p>
      {ratio_pairs.length > 0 ? (
        <div>
          <select
            value={pair ?? ''}
            onChange={(e) => setPair(e.target.value)}
            className="bg-canvas border border-border rounded px-3 py-1.5 text-sm font-mono text-body mb-4"
          >
            {ratio_pairs.map((p) => (
              <option key={p} value={p}>
                {p
                  .split('_to_')
                  .map((part) => tDyn(prettify(part)))
                  .join(' / ')}
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
        <InfoState message={t('ratios.noRatios')} />
      )}

      {/* Section 5 — Z-Scores (always visible) */}
      <p className="section-title">{t('ratios.zScores')}</p>
      <p className="font-mono text-[0.8rem] text-muted px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint mb-4">
        <Rich text={t('ratios.zNote')} />
      </p>
      {zscore_bars.length > 0 ? (
        <div className="card">
          <ZScoreBarChart
            data={zscore_bars.map((b) => ({ ...b, label: String(b.label).split(' / ').map(tDyn).join(' / ') }))}
            height={340}
          />
          <p className="font-mono text-[0.7rem] text-subtle mt-2">{t('ratios.zBarNote')}</p>
        </div>
      ) : (
        <InfoState message={t('ratios.noRatios')} />
      )}
    </div>
  )
}

function RatioExplorerSidePanel({ explorer }) {
  const { t, ord } = useLanguage()
  const z = explorer.zscore ?? 0
  const level = Math.abs(z) > 2 ? 'extreme' : Math.abs(z) > 1 ? 'notable' : 'normal'
  const color = { extreme: 'text-neg', notable: 'text-warn', normal: 'text-pos' }[level]
  const pctile = explorer.percentile

  let interpretation = t('ratios.interp.normal')
  if (pctile !== null) {
    if (pctile > 85) interpretation = t('ratios.interp.high')
    else if (pctile < 15) interpretation = t('ratios.interp.low')
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="card">
        <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-wide mb-1">{t('ratios.zScore')}</p>
        <p className={`font-mono text-[1.6rem] font-semibold ${color}`}>{fmtZScore(z)}</p>
        <p className="font-mono text-[0.75rem] text-faint mt-1">
          {t('analysis.percentile')}: {pctile !== null ? ord(pctile) : '—'}
        </p>
      </div>
      <div className="card">
        <p className="font-mono text-[0.68rem] text-subtle uppercase tracking-wide mb-2">{t('ratios.statistics')}</p>
        <div className="font-mono text-[0.78rem] text-muted leading-loose">
          <p>{t('ratios.stat.mean')}:&nbsp; {explorer.mean?.toFixed(3) ?? '—'}</p>
          <p>{t('ratios.stat.std')}:&nbsp;&nbsp; {explorer.std?.toFixed(3) ?? '—'}</p>
          <p>{t('ratios.stat.min')}:&nbsp;&nbsp; {explorer.min?.toFixed(3) ?? '—'}</p>
          <p>{t('ratios.stat.max')}:&nbsp;&nbsp; {explorer.max?.toFixed(3) ?? '—'}</p>
        </div>
      </div>
      <p className="font-mono text-[0.74rem] text-subtle leading-relaxed">{interpretation}</p>
    </div>
  )
}
