# 🌾 Agricultural Commodity Price Tracker

> **End-to-end automated pipeline for commodity price monitoring, statistical signal detection, multi-model forecasting, and real-time dashboard delivery.**

A production-grade data engineering and AI solution that ingests 22+ commodity series from the Federal Reserve (FRED API), runs daily through an orchestrated pipeline, computes actionable trading signals, generates multi-horizon price forecasts using an ensemble of statistical and ML models, and exposes the results through a REST API consumed by an interactive dashboard.

---

## The Problem

Agricultural commodity markets are inherently volatile. Corn prices spike when fertilizer costs surge. Natural gas drives urea production costs up, squeezing crop margins. Crude oil moves diesel prices, directly impacting logistics and farming operations. For any agribusiness, trader, or cooperative, **the cost of not knowing these dynamics in real time is measured in margin erosion**.

The problem is that this data exists — it's public, published by the Federal Reserve — but it's:

- **Fragmented**: 22+ series spread across different update frequencies (daily, weekly, monthly, quarterly)
- **Noisy**: raw FRED data has gaps, publication lags, and outliers
- **Static**: manual downloads don't scale to daily monitoring
- **Isolated**: no one compares crop prices vs input costs vs macro indicators in one view
- **Backward-looking**: raw prices tell you what happened, not what's likely coming next

This project solves all of it with a fully automated, containerized solution.

---

## The Solution

A **daily automated pipeline** that:

1. **Collects** price data for crops, energy inputs, fertilizers, livestock, and macro indicators from the FRED API — across mixed frequencies, with retry logic and imputation
2. **Validates** each series for minimum data quality (gaps, outliers, imputation rate) before ingesting it
3. **Processes** the data into analytical signals: price change %, rolling moving averages, and frequency-calibrated z-scores
4. **Analyzes** cross-commodity relationships: crop-to-input profitability ratios, cost environment indices, market regime classification, and Pearson correlation matrices
5. **Forecasts** future prices per commodity using an **ensemble of 9 models** (statistical + ML), weighted by historical error, with GARCH-derived volatility bands
6. **Persists** everything to DuckDB — a single embedded analytical warehouse shared by pipeline and API (read-only)
7. **Alerts** via email when any z-score exceeds a configurable threshold, with HTML-formatted tables and market regime context
8. **Serves** all outputs through a FastAPI REST backend consumed by a React dashboard with 4 analytical views

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                          │
│                    Prefect 2.x  (daily cron / serve)               │
│              Scheduling · Retries · Run Logging · Parallelism       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ┌─────────────────▼──────────────────────┐
         │           Task [1/6]: collect           │
         │         AgriculturalCollector           │
         │  FRED API → 22 commodity series         │
         │  Retry (tenacity) · ffill imputation    │
         │  Mixed frequency (D/W/M/Q) alignment    │
         └─────────────────┬──────────────────────┘
                           │
         ┌─────────────────▼──────────────────────┐
         │         Task [2/6]: validate            │
         │           DataValidation                │
         │  Min obs · Gap detection · Outliers     │
         │  Imputation rate · Pydantic Report      │
         └─────────────────┬──────────────────────┘
                           │
         ┌─────────────────▼──────────────────────┐
         │          Task [3/6]: process            │
         │           DataPipeline                  │
         │  Clean · Dedup · Merge · Forward-fill   │
         │  Change % · Rolling MA · Z-Score        │
         │           ↓  Persist                    │
         │        DuckDB: prices table             │
         └─────────────────┬──────────────────────┘
                           │
         ┌─────────────────▼──────────────────────┐
         │         Task [4/6]: analytics           │
         │         CommoditiesAnalytics            │
         │  Profitability ratios · Cost indices    │
         │  Market regime · Correlation matrix     │
         │  Z-Score signals (frequency-aware)      │
         │           ↓  Persist                    │
         │        DuckDB: signals table            │
         └────────────┬───────────┬───────────────┘
                      │           │
         ┌────────────▼──┐   ┌────▼───────────────┐
         │  Task [5/6]   │   │   Task [6/6]        │
         │  forecasting  │   │   alerts            │
         │ CommoditiesF. │   │ SMTP · HTML email   │
         │ 9-model ensem │   │ Z-Score threshold   │
         │ ThreadPool    │   │ Regime context      │
         │ ↓  Persist    │   └─────────────────────┘
         │ DuckDB:       │
         │ forecasts     │
         └───────┬───────┘
                 │
   ┌─────────────▼──────────────────────┐
   │      FastAPI REST Backend          │
   │  /api/overview · /api/analysis     │
   │  /api/ratios   · /api/forecast     │
   │   Read-only DuckDB · Uvicorn       │
   └─────────────┬──────────────────────┘
                 │
   ┌─────────────▼──────────────────────┐
   │       React + Vite Dashboard       │
   │  Overview · Analysis · Ratios      │
   │  Forecasting  ·  Recharts          │
   └────────────────────────────────────┘
```

### Deployment (Docker Compose)

| Service | Role |
|---|---|
| `postgres-db` | Prefect orchestration metadata backend |
| `prefect-server` | Self-hosted Prefect UI + scheduler |
| `pipeline` | Prefect worker — runs `flow.py` daily |
| `backend` | FastAPI + Uvicorn (read-only DuckDB mount) |
| `frontend` | React + Vite served at port 5173 |

---

## Data Coverage

**22 commodity series across 6 categories, all sourced from FRED:**

| Category | Series |
|---|---|
| **Crops** | Corn, Wheat, Soybeans, Rice, Barley |
| **Energy Inputs** | Crude Oil (WTI), Natural Gas, Diesel, Gasoline, Electricity |
| **Fertilizers** | Urea, DAP, Phosphate Rock, Potash, TSP/Fertilizer Materials |
| **Livestock** | Cattle (Beef), Swine, Processed Poultry |
| **Indexes** | Global Food Price Index, Agricultural Raw Materials |
| **Macro** | US Dollar Index, Federal Funds Rate |

Frequencies: `daily` · `weekly` · `monthly` · `quarterly` — all unified in a single analytical frame with frequency-aware z-score calibration.

---

## Forecasting Engine

Nine independent models per commodity, combined into a **weighted ensemble** (exponential error weighting of the top 3 models by MAPE):

**Statistical / Econometric:**
- `ARIMA` — AutoRegressive Integrated Moving Average (auto-order via pmdarima)
- `SARIMA` — Seasonal ARIMA for periodic cycles
- `Exponential Smoothing (Holt)` — Trend-adjusted double exponential smoothing
- `GARCH(1,1)` — Volatility modeling; outputs prediction intervals as confidence bands

**Machine Learning (lag-feature based):**
- `Gradient Boosting Regressor`
- `Random Forest Regressor`
- `XGBoost`
- `Ridge Regression (CV)` — L2 regularized linear baseline
- `LASSO (CV)` — L1 regularized; implicit feature selection on lag features

The ensemble avoids model lock-in: the best-performing models for each commodity and frequency receive higher ensemble weights automatically, based on their in-sample error history.

Forecast horizons are calibrated per frequency: 10 days (daily), 10 weeks (weekly), 5 months (monthly), 4 quarters (quarterly).

---

## Analytics & Signals

**Profitability Ratios:** Crop price / input cost pairs (corn/urea, wheat/dap, soybeans/crude_oil, etc.). Directional signals of margin pressure — rising ratio = improving crop margin; falling = cost squeeze.

**Cost Indices:** Normalized composite cost environment per input category (energy, fertilizer). Enables cross-commodity cost comparison and regime detection.

**Z-Score Signals:** Statistical anomaly detection on a rolling window calibrated per data frequency. A z-score > 2.0 (configurable) triggers an alert. Positive z = price above historical norm; negative = below.

**Market Regime:** Classifies the current environment (e.g. *High Energy Cost Pressure*, *Normal Input Environment*) from cost index ranges. Used as context in alert emails and the dashboard header.

**Correlation Analysis:** Pearson correlation matrix computed across selected commodities for a chosen time window. Identifies price transmission channels (e.g. natural gas → urea → corn margin).

---

## Dashboard

Four analytical pages served by the React frontend:

**Overview** — KPI cards for last price + z-score signal per commodity, market regime banner, tabular snapshot of all series with change %.

**Analysis** — Interactive time-series chart with 30-period MA overlay; z-score bar chart and time-series per commodity; Pearson correlation heatmap for multi-asset selections; distribution histogram with percentile markers; descriptive statistics table.

**Ratios** — Profitability ratio explorer (crop/input pairs over time); composite input cost index chart; z-score time series for ratio signals.

**Forecasting** — Ensemble forecast chart with GARCH confidence bands; individual model comparison table ranked by MAPE; directional change indicator (current → T+1 with % delta).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Orchestration** | Prefect 2.x (self-hosted, Postgres backend) |
| **Data collection** | `fredapi` + `tenacity` (exponential backoff retry) |
| **Data processing** | `pandas`, `numpy` |
| **Validation** | `pydantic` / `pydantic-settings` |
| **Analytics** | `scipy`, `statsmodels` |
| **Forecasting** | `statsmodels` (ARIMA, SARIMA, ES), `arch` (GARCH), `scikit-learn` (GB, RF, Ridge, LASSO), `xgboost`, `pmdarima` |
| **Warehouse** | `DuckDB` 1.1.x (embedded, zero-ops analytical DB) |
| **Orchestration DB** | PostgreSQL 15 (Prefect metadata only) |
| **API** | FastAPI + Uvicorn |
| **Frontend** | React 18, Vite, Tailwind CSS, Recharts |
| **Containerization** | Docker + Docker Compose |
| **Alerts** | SMTP (configurable; falls back to log) |
| **Config** | `.env` + Pydantic Settings (type-safe, validated) |
| **Testing** | `pytest` (pipeline, calculations, forecaster) |

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

### Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Key variables:

```env
# Required
FRED_API_KEY=your_key_here

# Storage
DATA_DIR=data
DAYS_HISTORY=1095

# Alerts (optional — logs instead of sending if not set)
ALERT_ZSCORE_THRESHOLD=2.0
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_user
SMTP_PASSWORD=your_password
ALERT_EMAIL_TO=you@example.com
ALERT_EMAIL_FROM=pipeline@example.com

# Prefect + PostgreSQL
POSTGRES_USER=prefect
POSTGRES_PASSWORD=prefect
POSTGRES_DB=prefect
```

### Run

```bash
# Start the full stack (pipeline + API + Prefect server)
docker compose up -d

# Or run the pipeline directly (no Docker)
pip install -r requirements.txt
python automation/flow.py
```

The pipeline will run on startup and then on schedule (configurable cron or interval in `flow.py`). The FastAPI backend is available at `http://localhost:8000/api`. The React frontend at `http://localhost:5173`.


---

## Design Decisions

**DuckDB as the warehouse** — No server to manage. The pipeline writes; the API reads (read-only flag enforced via env). Zero deployment overhead for an embedded analytical DB that handles DataFrame-native queries.

**Frequency-aware z-scores** — A monthly z-score uses a 12-period rolling window; a daily one uses 365. Computing them with a full-history mean (as many dashboards do) produces a different number than the pipeline's stored value. Here, the pipeline is the single source of truth, and the API reads pre-computed signals.

**Weighted ensemble forecasting** — The ensemble ranks valid forecasting models by MASE when available, falling back to MAPE when MASE cannot be calculated. It selects the top three models and computes their ensemble weights using an exponential weighting scheme based on MAPE. GARCH is excluded from price-forecast aggregation and is used separately for volatility estimates displayed on the dashboard.

**Prefect for orchestration, not Airflow** — Lower operational weight for a single-developer freelance deployment. Self-hosted Prefect server + Postgres is a two-service addition to the compose file, with a full UI and retry/log visibility.

---

## Data Source

All price series are retrieved from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API, maintained by the Federal Reserve Bank of St. Louis.