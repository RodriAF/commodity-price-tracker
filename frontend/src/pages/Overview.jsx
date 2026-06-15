import { useMemo, useState } from 'react'
import { api } from '../api/client'
import { useApiData } from '../hooks/useApiData'
import PageHeader from '../components/layout/PageHeader'
import KpiCard from '../components/KpiCard'
import CategoryCard from '../components/CategoryCard'
import SignalsPanel from '../components/SignalsPanel'
import SnapshotTable from '../components/SnapshotTable'
import PriceHistoryChart from '../components/PriceHistoryChart'
import { LoadingState, ErrorState } from '../components/StatusStates'
import { fmtDate } from '../utils/format'

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
  const { data, loading, error } = useApiData(() => api.overview(730), [])
  const [activeCategories, setActiveCategories] = useState(null) // null = all categories

  if (loading) return <LoadingState label="Loading market overview…" />
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
          name: c.name,
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
      <PageHeader subtitle="Market Overview" />

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard label="Last Update" value={fmtDate(data.last_update)} />
        <KpiCard label="Commodities Tracked" value={data.n_commodities} />
        <KpiCard
          label="Active Signals"
          value={data.active_signals}
          delta={data.extreme_signals > 0 ? `${data.extreme_signals} extreme` : null}
          deltaValue={data.extreme_signals > 0 ? -1 : 0}
        />
        <KpiCard label="Notable Moves" value={data.notable_signals} />
      </div>

      {/* Price history chart */}
      <p className="section-title">Price History by Category</p>
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
              {data.categories[key].label}
            </button>
          ))}
        </div>
        <PriceHistoryChart dates={data.price_history.dates} series={chartSeries} height={480} />
      </div>

      {/* Category cards */}
      <p className="section-title">Commodities by Category</p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {categoryKeys.map((key) => (
          <CategoryCard
            key={key}
            categoryKey={key}
            label={data.categories[key].label}
            commodities={data.categories[key].commodities}
          />
        ))}
      </div>

      {/* Active signals */}
      <p className="section-title">Active Signals — Statistical Anomalies (|z| &gt; 1)</p>
      <SignalsPanel signals={data.signals} />

      {/* Snapshot table */}
      <p className="section-title">Commodity Snapshot</p>
      <SnapshotTable rows={data.snapshot} />

      <p className="font-mono text-[0.68rem] text-faint mt-3">
        Data source: Central DuckDB Warehouse · {fmtDate(data.last_update)} · {data.n_commodities} series tracked
      </p>
    </div>
  )
}
