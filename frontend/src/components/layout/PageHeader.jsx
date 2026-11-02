import { useLanguage } from '../../i18n/LanguageContext'

/**
 * Page header — replicates the `.page-header` / `.page-title` /
 * `.page-subtitle` markup repeated at the top of every Streamlit page.
 */
export default function PageHeader({ subtitle, action }) {
  const { t } = useLanguage()

  return (
    <div className="border-b border-[#2a2f3a] pb-5 mb-8 flex items-end justify-between gap-4">
      <div>
        <p className="page-title">{t('app.title')}</p>
        <p className="page-subtitle">{subtitle}</p>
      </div>
      {action}
    </div>
  )
}
