/**
 * Shared loading / error / empty-state placeholders so every page handles
 * the "pipeline hasn't run yet" cases (404s from the API) the same way the
 * original pages did with st.error / st.warning / st.info.
 */

export function LoadingState({ label = 'Loading data…' }) {
  return (
    <div className="flex items-center justify-center py-24">
      <p className="font-mono text-sm text-subtle tracking-wide">{label}</p>
    </div>
  )
}

export function ErrorState({ message }) {
  return (
    <div className="card border-l-2 border-l-neg">
      <p className="font-mono text-[0.68rem] text-faint uppercase tracking-[0.1em] mb-1">Error</p>
      <p className="text-sm text-body">{message}</p>
    </div>
  )
}

export function InfoState({ message }) {
  return (
    <div className="card border-l-2 border-l-faint">
      <p className="text-sm text-subtle">{message}</p>
    </div>
  )
}
