/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'monospace'],
        sans: ['"IBM Plex Sans"', 'sans-serif'],
      },
      colors: {
        // Base surfaces — ported from the Streamlit dark theme.
        canvas: 'rgba(13, 15, 18, 1)',
        panel: '#141720',
        border: '#1e2330',
        'border-soft': '#1a1f2e',
        // Text
        heading: '#f1f5f9',
        body: '#e2e8f0',
        muted: '#94a3b8',
        subtle: '#64748b',
        faint: '#475569',
        // Category accent colors (top border / header text per category card)
        cat: {
          energy: 'rgba(245, 158, 11, 1)',
          crop: 'rgba(34, 197, 94, 1)',
          fertilizer: 'rgba(56, 189, 248, 1)',
          livestock: 'rgba(239, 68, 68, 1)',
          index: 'rgba(168, 85, 247, 1)',
          economic: 'rgba(148, 163, 184, 1)',
        },
        // Status colors
        pos: 'rgba(34, 197, 94, 1)',
        neg: 'rgba(239, 68, 68, 1)',
        warn: 'rgba(245, 158, 11, 1)',
      },
    },
  },
  plugins: [],
}
