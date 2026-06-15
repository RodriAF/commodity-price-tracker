import { changeColorClass } from '../utils/format'

/**
 * KPI card — replicates the `.kpi-strip` / `st.metric` styling used across
 * the Overview, Analysis, Ratios and Forecasting pages.
 *
 * `delta` is an optional string (already formatted, e.g. "+2.4%" or
 * "2 extreme"). `deltaValue`, if provided, drives the color (positive /
 * negative / neutral); otherwise the delta is rendered in a neutral tone.
 */
export default function KpiCard({ label, value, delta, deltaValue, valueClassName = '' }) {
  return (
    <div className="card flex flex-col gap-1">
      <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.1em]">{label}</p>
      <p className={`font-mono text-[1.6rem] font-semibold leading-none text-heading ${valueClassName}`}>
        {value}
      </p>
      {delta && (
        <p className={`font-mono text-[0.78rem] ${changeColorClass(deltaValue)}`}>{delta}</p>
      )}
    </div>
  )
}
