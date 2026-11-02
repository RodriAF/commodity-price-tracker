import { useState } from 'react'
import { useLanguage } from '../i18n/LanguageContext'

/**
 * Small hook holding the Y-axis zoom state for a chart. `domain` plugs
 * directly into Recharts' <YAxis domain={...}>; leave a bound empty to let
 * that side stay on Recharts' own auto-scaling.
 */
export function useYZoom() {
  const [yMin, setYMin] = useState('')
  const [yMax, setYMax] = useState('')

  const domain = [yMin !== '' ? Number(yMin) : 'auto', yMax !== '' ? Number(yMax) : 'auto']
  const active = yMin !== '' || yMax !== ''
  const reset = () => {
    setYMin('')
    setYMax('')
  }

  return { yMin, setYMin, yMax, setYMax, domain, active, reset }
}

/**
 * Toolbar with "Y min" / "Y max" inputs plus a combined reset button (clears
 * both the Y bounds and, via `onResetAll`, any X-axis brush zoom).
 */
export function ChartZoomControls({ yMin, setYMin, yMax, setYMax, active, onResetAll }) {
  const { t } = useLanguage()

  return (
    <div className="flex flex-wrap items-center gap-3 mb-2 justify-end">
      <span className="font-mono text-[0.62rem] text-subtle uppercase tracking-wide">{t('chart.yZoom')}</span>
      <label className="flex items-center gap-1.5 font-mono text-[0.68rem] text-faint">
        {t('chart.min')}
        <input
          type="number"
          inputMode="decimal"
          value={yMin}
          onChange={(e) => setYMin(e.target.value)}
          placeholder={t('chart.auto')}
          className="w-20 bg-canvas border border-border rounded px-1.5 py-0.5 text-body text-[0.7rem] font-mono focus:outline-none focus:border-faint"
        />
      </label>
      <label className="flex items-center gap-1.5 font-mono text-[0.68rem] text-faint">
        {t('chart.max')}
        <input
          type="number"
          inputMode="decimal"
          value={yMax}
          onChange={(e) => setYMax(e.target.value)}
          placeholder={t('chart.auto')}
          className="w-20 bg-canvas border border-border rounded px-1.5 py-0.5 text-body text-[0.7rem] font-mono focus:outline-none focus:border-faint"
        />
      </label>
      <button
        type="button"
        onClick={onResetAll}
        disabled={!active}
        className="font-mono text-[0.68rem] text-faint hover:text-heading underline underline-offset-2 disabled:opacity-30 disabled:cursor-default disabled:no-underline"
      >
        {t('chart.resetZoom')}
      </button>
    </div>
  )
}
