import { fmtNumber, fmtPercent, changeColorClass } from '../utils/format'

// Maps category keys to the Tailwind accent colors defined in tailwind.config.js
const CATEGORY_ACCENT = {
  energy_input: 'border-t-cat-energy text-cat-energy',
  crop: 'border-t-cat-crop text-cat-crop',
  fertilizer: 'border-t-cat-fertilizer text-cat-fertilizer',
  livestock: 'border-t-cat-livestock text-cat-livestock',
  index: 'border-t-cat-index text-cat-index',
  economic: 'border-t-cat-economic text-cat-economic',
}

/**
 * Category card — replicates `.category-card` / `.category-header` /
 * `.commodity-row` from the Overview page: a small price + change list
 * grouped by commodity category.
 */
export default function CategoryCard({ categoryKey, label, commodities }) {
  const accent = CATEGORY_ACCENT[categoryKey] || 'border-t-cat-economic text-cat-economic'
  const [borderClass, textClass] = accent.split(' ')

  return (
    <div className={`card border-t-[3px] ${borderClass} h-full`}>
      <div className={`font-mono text-[0.68rem] uppercase tracking-[0.12em] pb-2 mb-2 border-b border-border ${textClass}`}>
        {label}
      </div>

      <div className="flex flex-col">
        {commodities.map((c) => (
          <div
            key={c.key}
            className="flex items-center justify-between py-1 border-b border-border-soft last:border-b-0"
          >
            <span className="text-[0.78rem] text-muted flex-1 truncate pr-2">{c.name}</span>
            <span className="font-mono text-[0.82rem] font-semibold text-heading text-right min-w-[64px]">
              {c.price !== null ? fmtNumber(c.price) : '—'}
            </span>
            <span className={`font-mono text-[0.75rem] text-right min-w-[60px] ml-2 ${changeColorClass(c.change_pct)}`}>
              {c.change_pct !== null ? fmtPercent(c.change_pct) : '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
