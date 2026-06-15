/**
 * Small formatting helpers shared across pages — mirrors the f-string
 * formatting used throughout the original Streamlit pages (e.g. `{x:+.2f}%`,
 * `{z:+.2f}σ`).
 */

export const fmtNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(decimals)
}

export const fmtPercent = (value, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${Number(value).toFixed(decimals)}%`
}

export const fmtZScore = (value, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${Number(value).toFixed(decimals)}σ`
}

export const fmtDate = (isoDate, options = { day: '2-digit', month: 'short', year: 'numeric' }) => {
  if (!isoDate) return '—'
  return new Date(isoDate).toLocaleDateString('en-GB', options)
}

/** Tailwind text-color class for a +/- change value. */
export const changeColorClass = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-subtle'
  if (value > 0) return 'text-pos'
  if (value < 0) return 'text-neg'
  return 'text-subtle'
}

/** Tailwind text-color class for a z-score, based on |z| thresholds. */
export const zScoreColorClass = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-subtle'
  const abs = Math.abs(value)
  if (abs > 2) return 'text-neg'
  if (abs > 1) return 'text-warn'
  return 'text-pos'
}

/** 'extreme' / 'notable' / 'normal' classification for |z| thresholds. */
export const zScoreLevel = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return 'normal'
  const abs = Math.abs(value)
  if (abs > 2) return 'extreme'
  if (abs > 1) return 'notable'
  return 'normal'
}
