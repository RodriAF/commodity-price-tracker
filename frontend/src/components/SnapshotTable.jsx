import { fmtNumber, fmtPercent, fmtZScore, fmtDate, changeColorClass, zScoreColorClass } from '../utils/format'
import { useLanguage } from '../i18n/LanguageContext'

/**
 * Commodity snapshot table — replicates the styled `st.dataframe` at the
 * bottom of the Overview page (price, % change, z-score, signal — sorted
 * by |z-score| descending by the API).
 */
export default function SnapshotTable({ rows }) {
  const { t, tDyn, tCat } = useLanguage()

  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="font-mono text-[0.65rem] text-faint uppercase tracking-[0.08em] border-b border-border">
            <th className="text-left px-4 py-3">{t('snapshot.commodity')}</th>
            <th className="text-left px-4 py-3">{t('snapshot.category')}</th>
            <th className="text-left px-4 py-3">{t('common.freqShort')}</th>
            <th className="text-right px-4 py-3">{t('snapshot.price')}</th>
            <th className="text-right px-4 py-3">{t('snapshot.asOf')}</th>
            <th className="text-right px-4 py-3" title={t('snapshot.changeHint')}>
              {t('snapshot.change')}
            </th>
            <th className="text-right px-4 py-3">Z-Score</th>
            <th className="text-left px-4 py-3">{t('snapshot.signal')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b border-panel hover:bg-border-soft/40">
              <td className="px-4 py-2 text-body">{tDyn(r.name)}</td>
              <td className="px-4 py-2 text-muted">{tCat(r.category)}</td>
              <td className="px-4 py-2 text-muted font-mono text-xs">{r.frequency}</td>
              <td className="px-4 py-2 text-right font-mono text-heading">
                {r.price !== null ? fmtNumber(r.price) : '—'}
              </td>
              <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                {r.as_of ? fmtDate(r.as_of, { day: '2-digit', month: 'short', year: '2-digit' }) : '—'}
              </td>
              <td className={`px-4 py-2 text-right font-mono ${changeColorClass(r.change_pct)}`}>
                {r.change_pct !== null ? fmtPercent(r.change_pct) : '—'}
              </td>
              <td className={`px-4 py-2 text-right font-mono font-semibold ${zScoreColorClass(r.z_score)}`}>
                {r.z_score !== null ? fmtZScore(r.z_score) : '—'}
              </td>
              <td className="px-4 py-2 text-muted">{r.signal ? tDyn(r.signal) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
