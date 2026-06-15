import { Route, Routes } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import Overview from './pages/Overview'
import Analysis from './pages/Analysis'
import Ratios from './pages/Ratios'
import Forecasting from './pages/Forecasting'

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 px-6 py-6 md:px-10 md:py-8 max-w-[1400px] mx-auto w-full">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/ratios" element={<Ratios />} />
          <Route path="/forecasting" element={<Forecasting />} />
        </Routes>
      </main>
    </div>
  )
}
