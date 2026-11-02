import LanguageSwitcher from '../LanguageSwitcher'

/**
 * Top row shared by every page: holds the language switcher (kept here so it
 * is reachable on mobile, where the sidebar is hidden).
 */
export default function TopBar() {
  return (
    <div className="flex justify-end items-center gap-3 mb-6">
      <LanguageSwitcher />
    </div>
  )
}
