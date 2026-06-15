import { fmtNumber, fmtZScore } from '../utils/format'

function percentileColor(pctile) {
  if (pctile === null || pctile === undefined) return '#475569'
  if (pctile > 80) return 'rgba(239, 68, 68, 1)'
  if (pctile < 20) return 'rgba(34, 197, 94, 1)'
  return 'rgba(245, 158, 11, 1)'
}

/**
 * Replicates `.stat-card` from the Analysis page: current value, unit,
 * percentile + z-score, and a percentile progress bar.
 */
export default function AssetStatCard({ name, unit, current, zscore, percentile }) {
  const color = percentileColor(percentile)

  return (
    <div className="card">
      <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.1em] mb-1">{name}</p>
      <p className="font-mono text-[1.4rem] font-semibold text-heading leading-none">
        {current !== null ? fmtNumber(current) : '—'}
      </p>
      <p className="font-mono text-[0.75rem] text-subtle mt-1">{unit}</p>
      <p className="font-mono text-[0.75rem] text-subtle mt-2">
        Percentile:{' '}
        <strong style={{ color }}>{percentile !== null ? `${Math.round(percentile)}th` : '—'}</strong>
        {'  ·  '}Z: <strong className="text-body">{zscore !== null ? fmtZScore(zscore) : '—'}</strong>
      </p>
      <div className="h-1.5 bg-border rounded-full mt-2 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.min(100, Math.max(0, percentile ?? 0))}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}
