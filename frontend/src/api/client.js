/**
 * Minimal fetch-based client for the FastAPI backend.
 *
 * The base URL can be overridden via the VITE_API_BASE_URL env var
 * (see .env.example). In local development it defaults to the FastAPI
 * dev server started with `uvicorn main:app --reload --port 8000`.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request(path, params = {}) {
  const url = new URL(`${BASE_URL}/api${path}`)
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, value)
    }
  })

  const res = await fetch(url.toString())

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // response body wasn't JSON — keep statusText
    }
    throw new ApiError(`Request to ${path} failed (${res.status})`, res.status, detail)
  }

  return res.json()
}

export const api = {
  health: () => request('/health'),
  overview: (days) => request('/overview', { days }),
  analysis: ({ commodities, range, normalize }) =>
    request('/analysis', { commodities, range, normalize }),
  ratios: (pair) => request('/ratios', { pair }),
  forecast: (commodity) => request('/forecast', { commodity }),
  forecastAvailable: () => request('/forecast/available'),
  commodities: () => request('/commodities'),
}

export { ApiError }
