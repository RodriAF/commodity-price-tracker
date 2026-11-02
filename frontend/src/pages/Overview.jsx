import { useMemo, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import TopBar from '../components/layout/TopBar'
import { useLanguage } from '../i18n/LanguageContext'
import KpiCard from '../components/KpiCard'
import CategoryCard from '../components/CategoryCard'
import SignalsPanel from '../components/SignalsPanel'
import SnapshotTable from '../components/SnapshotTable'
import PriceHistoryChart from '../components/PriceHistoryChart'
import { LoadingState, ErrorState } from '../components/StatusStates'
import { fmtDate, fmtDateTime } from '../utils/format'

// Per-category color ramps, ported from CAT_COLORS in app.py — used to give
// each commodity a distinct but category-coherent line color in the chart.
const CAT_COLORS = {
  energy_input: ['#f59e0b', '#fbbf24', '#d97706', '#b45309', '#78350f'],
  crop: ['#22c55e', '#4ade80', '#16a34a', '#15803d', '#14532d'],
  fertilizer: ['#38bdf8', '#7dd3fc', '#0ea5e9', '#0284c7', '#075985'],
  livestock: ['#ef4444', '#f87171', '#dc2626'],
  index: ['#a855f7', '#c084fc'],
  economic: ['#94a3b8', '#64748b'],
}

export default function Overview() {
  const { t, tCat, tDyn, tOr } = useLanguage()
  const { data, loading, error } = useApiData(() => api.overview(730), [])
  const [activeCategories, setActiveCategories] = useState(null) // null = all categories

  if (loading) return <LoadingState label={t('overview.loading')} />
  if (error) {
    return <ErrorState message={error.detail || error.message || 'Failed to load /api/overview'} />
  }
  if (!data) return null

  const categoryKeys = Object.keys(data.categories)
  const selected = activeCategories ?? categoryKeys

  const toggleCategory = (key) => {
    setActiveCategories((prev) => {
      const current = prev ?? categoryKeys
      if (current.includes(key)) {
        const next = current.filter((k) => k !== key)
        return next.length > 0 ? next : current // never allow an empty selection
      }
      return [...current, key]
    })
  }

  // Build the chart series for the currently selected categories.
  const chartSeries = selected.reduce((acc, catKey) => {
    const colors = CAT_COLORS[catKey] || ['#94a3b8']
    const commodities = data.categories[catKey]?.commodities || []
    commodities.forEach((c, i) => {
      if (data.price_history.series[c.key]) {
        acc[c.key] = {
          name: tDyn(c.name),
          values: data.price_history.series[c.key],
          color: colors[i % colors.length],
          strokeWidth: 1.5,
        }
      }
    })
    return acc
  }, {})

  return (
    <div>
      <TopBar />

      <PageHeader subtitle={t('overview.subtitle')} />

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {/* "Data as of" = newest price observation (bounded by the source's publication lag).
            "Pipeline last run" = when the scheduled flow last finished (independent of the former). */}
        <KpiCard
          label={t('overview.kpi.lastUpdate')}
          value={fmtDate(data.last_update)}
          delta={t('overview.kpi.lastUpdateSub')}
        />
        <KpiCard
          label={t('overview.kpi.pipelineRun')}
          value={data.pipeline?.last_run_at ? fmtDateTime(data.pipeline.last_run_at) : '—'}
          delta={data.pipeline?.status ? tOr(`overview.pipeline.${data.pipeline.status}`, data.pipeline.status.toLowerCase()) : null}
          deltaValue={data.pipeline?.status === 'SUCCESS' ? 1 : data.pipeline?.status ? -1 : 0}
        />
        <KpiCard label={t('overview.kpi.tracked')} value={data.n_commodities} />
        <KpiCard
          label={t('overview.kpi.signals')}
          value={data.active_signals}
          delta={data.extreme_signals > 0 ? t('overview.kpi.extreme', { n: data.extreme_signals }) : null}
          deltaValue={data.extreme_signals > 0 ? -1 : 0}
        />
        <KpiCard label={t('overview.kpi.notable')} value={data.notable_signals} />
      </div>

      {/* Why "Last Update" can lag behind today */}
      <details className="mt-3 group">
        <summary className="cursor-pointer select-none font-mono text-[0.7rem] text-faint hover:text-muted">
          ⓘ {t('overview.lastUpdateNote.summary')}
        </summary>
        <p className="mt-2 px-4 py-3 bg-panel rounded-md border-l-[3px] border-l-faint font-mono text-[0.75rem] leading-relaxed text-muted">
          {t('overview.lastUpdateNote.body')
            .split('`')
            .map((part, i) =>
              i % 2 === 1 ? (
                <code key={i} className="text-body">
                  {part}
                </code>
              ) : (
                <span key={i}>{part}</span>
              ),
            )}
        </p>
      </details>

      <p className="font-mono text-[0.68rem] text-faint mt-2">
        <span aria-hidden="true">ⓘ </span>
        {t('overview.lastUpdateNote')}
      </p>

      {/* Price history chart */}
      <p className="section-title">{t('overview.priceHistory')}</p>
      <div className="card">
        <div className="flex flex-wrap gap-2 mb-4">
          {categoryKeys.map((key) => (
            <button
              key={key}
              onClick={() => toggleCategory(key)}
              className={`font-mono text-[0.68rem] uppercase tracking-wide px-3 py-1 rounded-full border transition-colors ${
                selected.includes(key)
                  ? 'border-border text-heading bg-border-soft'
                  : 'border-border text-faint hover:text-muted'
              }`}
            >
              {tCat(key, data.categories[key].label)}
            </button>
          ))}
        </div>
        <PriceHistoryChart dates={data.price_history.dates} series={chartSeries} height={480} />
      </div>

      {/* Category cards */}
      <p className="section-title">{t('overview.byCategory')}</p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {categoryKeys.map((key) => (
          <CategoryCard
            key={key}
            categoryKey={key}
            label={tCat(key, data.categories[key].label)}
            commodities={data.categories[key].commodities}
          />
        ))}
      </div>

      {/* Active signals */}
      <p className="section-title">{t('overview.signals')}</p>
      <SignalsPanel signals={data.signals} />

      {/* Snapshot table */}
      <p className="section-title">{t('overview.snapshot')}</p>
      <SnapshotTable rows={data.snapshot} />

      <p className="font-mono text-[0.68rem] text-faint mt-3">
        {t('overview.footer', { date: fmtDate(data.last_update), n: data.n_commodities })}
        {data.pipeline?.last_run_at ? t('overview.footerRun', { run: fmtDateTime(data.pipeline.last_run_at) }) : ''}
      </p>
    </div>
  )
}
