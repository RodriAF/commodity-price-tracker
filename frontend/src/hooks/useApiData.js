import { useEffect, useState } from 'react'

/**
 * Tiny data-fetching hook: runs `fetcher()` whenever any value in `deps`
 * changes, exposing { data, loading, error }. `fetcher` should return a
 * Promise (typically one of the `api.*` calls).
 *
 * Kept deliberately minimal — no caching layer. For a production app,
 * consider swapping this for React Query / TanStack Query.
 */
export function useApiData(fetcher, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    fetcher()
      .then((result) => {
        if (!cancelled) setData(result)
      })
      .catch((err) => {
        if (!cancelled) setError(err)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, loading, error }
}
