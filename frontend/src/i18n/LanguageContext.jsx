import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { translations } from './translations'
import { setLocale } from '../utils/format'

export const DEFAULT_LANG = 'es'
export const SUPPORTED_LANGS = ['es', 'en']
const STORAGE_KEY = 'cpt-lang'
const LOCALES = { es: 'es-ES', en: 'en-GB' }

function readStoredLang() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (SUPPORTED_LANGS.includes(stored)) return stored
  } catch {
    // localStorage unavailable (private mode, blocked…) — use the default
  }
  return DEFAULT_LANG
}

/** English ordinal suffix (1st, 2nd, 3rd, 4th, 11th…); Spanish just uses the number. */
function ordinal(lang, n) {
  const v = Math.round(n)
  if (lang !== 'en') return String(v)
  const mod100 = v % 100
  if (mod100 >= 11 && mod100 <= 13) return `${v}th`
  const suffix = { 1: 'st', 2: 'nd', 3: 'rd' }[v % 10] || 'th'
  return `${v}${suffix}`
}

const interpolate = (str, params) =>
  params ? str.replace(/\{(\w+)\}/g, (m, k) => (params[k] !== undefined ? params[k] : m)) : str

const LanguageContext = createContext(null)

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(readStoredLang)

  // Date formatting helpers in utils/format.js read this module-level locale.
  // Set synchronously during render so children format with the right locale.
  setLocale(LOCALES[lang])

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const setLang = useCallback((next) => {
    if (!SUPPORTED_LANGS.includes(next)) return
    setLangState(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // ignore persistence failures
    }
  }, [])

  const value = useMemo(() => {
    const dict = translations[lang]
    const fallback = translations.en

    /** Translate a UI key. Falls back to English, then to the key itself. */
    const t = (key, params) => {
      const raw = dict[key] ?? fallback[key] ?? key
      // `{ord}` is the ordinal form of `{n}` / `{p}` when those are numbers.
      const extra = {}
      if (params && typeof params.n === 'number') extra.ord = ordinal(lang, params.n)
      return interpolate(raw, { ...extra, ...params })
    }

    /**
     * Translate a string that comes from the backend (commodity names, units,
     * descriptions…). Looks for a `dyn.<text>` entry in the current language
     * only; if there is none, the backend text is returned untouched.
     */
    const tDyn = (text) => (text == null ? text : dict[`dyn.${text}`] ?? text)

    /** Translation for `key` if this language defines it, otherwise `fallback` (usually backend text). */
    const tOr = (key, fallback) => dict[key] ?? fallback

    /** Translate a category (key or backend label), falling back to `fallback` / the backend text. */
    const tCat = (keyOrLabel, fallback) => {
      if (!keyOrLabel) return keyOrLabel
      const key = String(keyOrLabel).trim().toLowerCase().replace(/\s+/g, '_')
      return dict[`cat.${key}`] ?? tDyn(fallback ?? keyOrLabel)
    }

    /** Percentile as ordinal in the current language (en: 85th, es: 85). */
    const ord = (n) => ordinal(lang, n)

    return { lang, setLang, t, tDyn, tOr, tCat, ord, locale: LOCALES[lang] }
  }, [lang, setLang])

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useLanguage() {
  const ctx = useContext(LanguageContext)
  if (!ctx) throw new Error('useLanguage must be used inside <LanguageProvider>')
  return ctx
}

/**
 * Renders a translated string containing `**bold**` segments as React nodes.
 * Usage: <Rich text={t('ratios.marginNote')} strongClassName="text-body" />
 */
export function Rich({ text, strongClassName }) {
  return text.split('**').map((part, i) =>
    i % 2 === 1 ? (
      <strong key={i} className={strongClassName}>
        {part}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  )
}
