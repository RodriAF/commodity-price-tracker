import { SUPPORTED_LANGS, useLanguage } from '../i18n/LanguageContext'

/** Small ES | EN toggle. */
export default function LanguageSwitcher() {
  const { lang, setLang, t } = useLanguage()

  return (
    <div
      role="group"
      aria-label={t('lang.label')}
      className="inline-flex overflow-hidden rounded-md border border-border font-mono text-[0.7rem] uppercase tracking-wider"
    >
      {SUPPORTED_LANGS.map((code) => (
        <button
          key={code}
          type="button"
          onClick={() => setLang(code)}
          aria-pressed={lang === code}
          className={`px-3 py-1.5 transition-colors ${
            lang === code ? 'bg-border text-heading' : 'text-faint hover:text-muted'
          }`}
        >
          {code}
        </button>
      ))}
    </div>
  )
}
