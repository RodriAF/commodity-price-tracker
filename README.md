# 🌾 Agricultural Commodity Price Tracker

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&logo=fastapi&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Workflow_Orchestration-024DFD?style=flat-square&logo=prefect&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-Analytics-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

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
5. **Forecasts** future prices per commodity on its own observation calendar, with a **pool of models chosen per series type**, a rolling-origin backtest, an always-visible **naive baseline**, and 80% / 95% prediction intervals
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
         │  Mixed frequency (D/W/M/Q) series      │
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
         │  Clean · Dedup · Merge (real obs. only) │
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

## Data Storage Contract

The `prices` table stores **only real observations**: a monthly series has one row per month, a daily series one row per business day. Short gaps *inside* a single series (e.g. a market holiday) are forward-filled by the collector within a frequency-specific limit and flagged `is_imputed = true`. Series are never forward-filled onto each other's calendars in storage. Code that needs aligned series (ratios, correlations, chart lines) aligns them in memory with `utils.alignment.align_wide_frame`; code where it matters (change %, percentiles, statistics, forecasts) reads the raw observations.

**"Data as of" vs "Pipeline last run".** The dashboard shows both. *Data as of* is the date of the newest price observation, which is bounded by the publication lag of the source (FRED publishes the EIA daily series about a week late, on Wednesdays). *Pipeline last run* is when the scheduled flow last finished, read from `pipeline_runs`.

**Upgrading an existing database.** Earlier versions stored forward-filled prices. The first run after upgrading detects this (`schema_meta.data_version < 2`), purges the legacy `prices` rows and per-date momentum signals **after the fresh download is already in memory** (a failed download never empties the database), and rebuilds them from the new data. History is limited to the `DAYS_HISTORY` window (max 3650 days).

## Forecasting Engine

Every commodity is forecast on **its own observation calendar** (business days, weeks, months or quarters) — nothing is forward-filled onto another series' calendar. The model pool depends on the series type, and every model is judged against a **naive "last value" baseline** on a **rolling-origin backtest**.

**Model pool, by series type**

| Series | Models |
|---|---|
| All | Naive (baseline), Drift, damped-trend ETS, Theta, ARIMA (auto-order via pmdarima, parameters refit per fold) |
| Monthly / quarterly | + Seasonal naive, seasonal ETS, SARIMA (when the data shows seasonal structure), Ridge |
| Weekly | + Seasonal naive (52), Ridge, LASSO, Gradient Boosting, Random Forest, XGBoost, GARCH |
| Daily | + Ridge, LASSO, Gradient Boosting, Random Forest, XGBoost, GARCH |

ML models are trained on **returns** with features that only use past information (the target is the change to the *next* period). Prices are modelled in log space, so forecasts cannot go negative. `GARCH(1,1)` only feeds an alternative 90% volatility band; it is never an ensemble price candidate.

**Evaluation.** Models are scored with a rolling-origin backtest (up to 6-8 folds, each refit on the past). Besides MAE / MAPE / MASE, every model gets **rMAE = its MAE / the naive baseline's MAE** on the same folds. Confidence is *high* only when rMAE < 0.85, *medium* when it beats naive (< 1.00) and *low* otherwise — so a forecast that cannot beat "the price stays where it is" is labelled as such on the dashboard.

**Ensemble.** The three best models by rMAE (naive included, so the ensemble shrinks toward "no change" when nothing beats it), weighted by `1 / rMAE²`. The ensemble gets its own backtest metrics, including **skill vs naive** and the **empirical coverage** of its intervals.

**Intervals.** 80% and 95% bands from the historical distribution of k-step-ahead changes of the series (overlapping windows, Student-t quantiles for short histories), rescaled by the recent volatility regime.

**Horizons** are configured per frequency in `config/commodities.json` (`forecast_steps`): 20 days (daily), 12 weeks (weekly), 6 months (monthly), 4 quarters (quarterly).

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

**Forecasting** — Ensemble forecast with an 80% / 95% interval fan, the naive baseline drawn next to it, backtest evaluation (skill vs naive, interval coverage), period-by-period table and a model comparison ranked by error relative to naive.

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
| **Testing** | `pytest` (pipeline, storage, calculations, forecaster, flow adapters, API) |

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

**Weighted ensemble forecasting** — Models are ranked by rMAE (error relative to the naive baseline on a rolling-origin backtest), the top three are blended with weights proportional to `1 / rMAE²`, and GARCH is excluded from price aggregation and used only for a volatility band. See *Forecasting Engine* above.

**Prefect for orchestration, not Airflow** — Lower operational weight for a single-developer freelance deployment. Self-hosted Prefect server + Postgres is a two-service addition to the compose file, with a full UI and retry/log visibility.

---

## Data Source

All price series are retrieved from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API, maintained by the Federal Reserve Bank of St. Louis.