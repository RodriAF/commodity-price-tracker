import { Fragment } from 'react'

/**
 * Pearson correlation heatmap — replicates the Plotly `go.Heatmap` on the
 * Analysis page using a plain CSS grid (no extra chart dependency needed
 * for an N x N matrix).
 *
 * Color scale: -1 -> cool blue, 0 -> neutral panel, +1 -> green, matching
 * the original `[[0,'#1c2e3d'],[0.5,'#1e2330'],[1,'#1e3a2f']]` scale.
 */
function cellColor(value) {
  if (value === null || value === undefined) return '#141720'
  const v = Math.max(-1, Math.min(1, value))
  if (v >= 0) {
    // 0 -> #1e2330, 1 -> #1e3a2f (green tint)
    const t = v
    return `rgb(${30 + t * -1}, ${35 + t * 23}, ${48 + t * (47 - 48)})`
  }
  // 0 -> #1e2330, -1 -> #1c2e3d (blue tint)
  const t = -v
  return `rgb(${30 - t * 2}, ${35 - t * 1}, ${48 + t * 13})`
}

export default function CorrelationHeatmap({ labels, matrix }) {
  if (!labels || !matrix) return null
  const n = labels.length

  return (
    <div className="overflow-x-auto">
      <div
        className="inline-grid gap-[2px]"
        style={{ gridTemplateColumns: `120px repeat(${n}, minmax(64px, 1fr))` }}
      >
        {/* Header row */}
        <div />
        {labels.map((label) => (
          <div
            key={`col-${label}`}
            className="font-mono text-[9px] text-subtle text-center px-1 py-2 truncate"
            title={label}
          >
            {label}
          </div>
        ))}

        {/* Data rows */}
        {labels.map((rowLabel, i) => (
          <Fragment key={`row-${rowLabel}`}>
            <div
              className="font-mono text-[9px] text-subtle flex items-center px-2 truncate"
              title={rowLabel}
            >
              {rowLabel}
            </div>
            {matrix[i].map((value, j) => (
              <div
                key={`cell-${i}-${j}`}
                className="flex items-center justify-center font-mono text-[10px] text-muted py-2 rounded-[2px]"
                style={{ backgroundColor: cellColor(value) }}
                title={`${rowLabel} × ${labels[j]}: ${value ?? '—'}`}
              >
                {value !== null ? value.toFixed(2) : '—'}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
