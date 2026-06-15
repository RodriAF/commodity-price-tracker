import { fmtZScore } from '../utils/format'

const LEVEL_BORDER = {
  extreme: 'border-l-neg',
  notable: 'border-l-warn',
}

const LEVEL_TEXT = {
  extreme: 'text-neg',
  notable: 'text-warn',
}

const TAG_STYLE = {
  overvalued: 'bg-[#1c2e1c] text-[#4ade80]',
  undervalued: 'bg-[#2e1c1c] text-[#f87171]',
}

const TAG_LABEL = {
  overvalued: 'ABOVE NORM',
  undervalued: 'BELOW NORM',
}

/**
 * Active anomaly signals list — replicates `.signal-row` from the
 * Overview page, split into two columns on larger screens.
 */
export default function SignalsPanel({ signals }) {
  if (!signals || signals.length === 0) {
    return (
      <div className="card text-subtle font-mono text-[0.85rem]">
        No anomalous signals detected — all commodities within standard statistical bounds.
      </div>
    )
  }

  const half = Math.ceil(signals.length / 2)
  const columns = [signals.slice(0, half), signals.slice(half)]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {columns.map((col, colIdx) => (
        <div key={colIdx} className="flex flex-col gap-2">
          {col.map((s) => (
            <div
              key={s.key}
              className={`flex items-center gap-4 px-4 py-3 rounded-md bg-panel border-l-[3px] ${
                LEVEL_BORDER[s.level] || 'border-l-transparent'
              }`}
            >
              <span className="font-mono text-[0.82rem] text-body flex-1">{s.name}</span>
              <span className={`text-[0.68rem] px-2 py-0.5 rounded uppercase tracking-wide ${TAG_STYLE[s.type]}`}>
                {TAG_LABEL[s.type]}
              </span>
              <span className={`font-mono text-[0.82rem] font-semibold min-w-[60px] text-right ${LEVEL_TEXT[s.level]}`}>
                {fmtZScore(s.z_score)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
