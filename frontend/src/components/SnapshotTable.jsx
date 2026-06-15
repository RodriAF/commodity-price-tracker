import { fmtNumber, fmtPercent, fmtZScore, changeColorClass, zScoreColorClass } from '../utils/format'

/**
 * Commodity snapshot table — replicates the styled `st.dataframe` at the
 * bottom of the Overview page (price, % change, z-score, signal — sorted
 * by |z-score| descending by the API).
 */
export default function SnapshotTable({ rows }) {
  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">Commodity</th>
            <th className="text-left px-4 py-3">Category</th>
            <th className="text-left px-4 py-3">Freq</th>
            <th className="text-right px-4 py-3">Price</th>
            <th className="text-right px-4 py-3">Chg %</th>
            <th className="text-right px-4 py-3">Z-Score</th>
            <th className="text-left px-4 py-3">Signal</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b border-panel hover:bg-border-soft/40">
              <td className="px-4 py-2 text-body">{r.name}</td>
              <td className="px-4 py-2 text-muted">{r.category}</td>
              <td className="px-4 py-2 text-muted font-mono text-xs">{r.frequency}</td>
              <td className="px-4 py-2 text-right font-mono text-heading">
                {r.price !== null ? fmtNumber(r.price) : '—'}
              </td>
              <td className={`px-4 py-2 text-right font-mono ${changeColorClass(r.change_pct)}`}>
                {r.change_pct !== null ? fmtPercent(r.change_pct) : '—'}
              </td>
              <td className={`px-4 py-2 text-right font-mono font-semibold ${zScoreColorClass(r.z_score)}`}>
                {r.z_score !== null ? fmtZScore(r.z_score) : '—'}
              </td>
              <td className="px-4 py-2 text-muted">{r.signal ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
