import { NavLink } from 'react-router-dom'
import { useLanguage } from '../../i18n/LanguageContext'

const NAV_ITEMS = [
  { to: '/', labelKey: 'nav.overview' },
  { to: '/analysis', labelKey: 'nav.analysis' },
  { to: '/ratios', labelKey: 'nav.ratios' },
  { to: '/forecasting', labelKey: 'nav.forecasting' },
]

/**
 * Slim left-hand navigation replacing Streamlit's automatic multi-page
 * sidebar (app.py + pages/*.py).
 */
export default function Sidebar() {
  const { t } = useLanguage()

  return (
    <aside className="w-56 shrink-0 border-r border-border bg-panel/40 px-4 py-6 hidden md:flex md:flex-col gap-1">
      <div className="px-2 mb-6">
        <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.12em]">
          {t('sidebar.brand')}
        </p>
        <p className="text-heading font-semibold mt-1">{t('sidebar.brandSub')}</p>
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
          {t(item.labelKey)}
        </NavLink>
      ))}

      <div className="mt-auto px-2 pt-6">
        <p className="font-mono text-[0.65rem] text-faint leading-relaxed">
          {t('common.dataSource')}
        </p>
      </div>
    </aside>
  )
}
