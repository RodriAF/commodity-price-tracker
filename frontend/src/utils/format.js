/**
 * Small formatting helpers shared across pages — mirrors the f-string
 * formatting used throughout the original Streamlit pages (e.g. `{x:+.2f}%`,
 * `{z:+.2f}σ`).
 */

// Locale used for date formatting. Updated by LanguageProvider when the
// language changes ('es-ES' by default, 'en-GB' for English).
let currentLocale = 'es-ES'
export const setLocale = (locale) => {
  currentLocale = locale
}

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

/**
 * Parse an ISO string. A bare date ("2026-10-20") is a calendar day, so it is
 * built in LOCAL time — `new Date("2026-10-20")` is UTC midnight and would
 * display as the previous day in timezones behind UTC.
 */
const parseIso = (isoDate) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate)
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(isoDate)
}

export const fmtDate = (isoDate, options = { day: '2-digit', month: 'short', year: 'numeric' }) => {
  if (!isoDate) return '—'
  const d = parseIso(isoDate)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleDateString(currentLocale, options)
}

/** Date + time in the viewer's timezone (used for pipeline run timestamps). */
export const fmtDateTime = (isoDateTime) => {
  if (!isoDateTime) return '—'
  const d = new Date(isoDateTime)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString(currentLocale, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

/** 0–1 ratio as a whole-number percentage, e.g. 0.667 -> "67%". */
export const fmtRatioPct = (value, decimals = 0) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(Number(value) * 100).toFixed(decimals)}%`
}

/** 0–1 ratio as a signed percentage, e.g. 0.034 -> "+3.4%". */
export const fmtSignedRatioPct = (value, decimals = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const pct = Number(value) * 100
  return `${pct > 0 ? '+' : ''}${pct.toFixed(decimals)}%`
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
