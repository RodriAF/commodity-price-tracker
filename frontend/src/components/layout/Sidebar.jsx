import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Overview' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/ratios', label: 'Input Cost Analytics' },
  { to: '/forecasting', label: 'Forecasting' },
]

/**
 * Slim left-hand navigation replacing Streamlit's automatic multi-page
 * sidebar (app.py + pages/*.py).
 */
export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 border-r border-border bg-panel/40 px-4 py-6 hidden md:flex md:flex-col gap-1">
      <div className="px-2 mb-6">
        <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.12em]">
          Commodity Tracker
        </p>
        <p className="text-heading font-semibold mt-1">Agri Markets</p>
      </div>

      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) =>
            [
              'px-3 py-2 rounded-md text-sm font-mono tracking-wide transition-colors',
              isActive
                ? 'bg-border text-heading'
                : 'text-muted hover:text-heading hover:bg-border-soft',
            ].join(' ')
          }
        >
          {item.label}
        </NavLink>
      ))}

      <div className="mt-auto px-2 pt-6">
        <p className="font-mono text-[0.65rem] text-faint leading-relaxed">
          Data source: Central DuckDB Warehouse
        </p>
      </div>
    </aside>
  )
}
