# 🌾 Agricultural Commodity Price Tracker

> A production-grade data pipeline and analytics dashboard for tracking, analyzing, and forecasting agricultural commodity prices sourced from the FRED API.

---

## Overview

This project is an end-to-end data engineering and machine learning system that collects real-time agricultural commodity prices, computes statistical indicators, and delivers interactive multi-page analytics through a Streamlit dashboard. It was designed to reflect the full lifecycle of a production ML system: automated data ingestion, frequency-aware feature engineering, ensemble forecasting, and a polished visualization layer.

The system tracks commodities across six categories — energy inputs, crops, fertilizers, livestock, indices, and economic indicators — and surfaces actionable signals for producers, analysts, and commodity market observers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRED API                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  AgriculturalCol- │
                    │  lector           │  collectors/
                    └─────────┬─────────┘
                              │  raw DataFrame
                    ┌─────────▼─────────┐
                    │  DataPipeline     │  pipeline/data_pipeline.py
                    │  · clean          │
                    │  · merge/dedup    │
                    │  · frequency-     │
                    │    aware metrics  │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐  ┌────▼────┐  ┌─────▼─────────────┐
    │ Commodities    │  │Correla- │  │ CommoditiesForeca- │
    │ Analytics      │  │tion     │  │ ster               │
    │ · ratios       │  │Analysis │  │ · 9 models         │
    │ · cost indices │  └─────────┘  │ · ensemble         │
    │ · z-scores     │               └─────────────────────┘
    │ · regime       │
    └────────────────┘
              │                           │
    ┌─────────▼───────────────────────────▼────┐
    │           data/                          │
    │  commodity_data.csv                      │
    │  commodity_ratios.csv                    │
    │  signals.json                            │
    │  forecasts/forecasts_YYYYMMDD.json       │
    └─────────────────────┬────────────────────┘
                          │
             ┌────────────▼────────────┐
             │   Streamlit Dashboard   │
             │   · Overview            │
             │   · Analysis            │
             │   · Analytics           │
             │   · Forecasting         │
             └─────────────────────────┘
```

---

## Features

### Data Pipeline
- Collects 20+ commodity series from the FRED API across 6 categories
- Handles mixed-frequency data (daily, weekly, monthly, quarterly) with proper resampling and forward-fill alignment
- Frequency-aware metric computation: percentage change, moving averages, and rolling z-scores are all calculated in the commodity's native frequency, then expanded back to the full date index — avoiding the look-ahead bias common in naive daily resampling
- Idempotent: new runs merge with existing data, deduplicate, and recalculate metrics in place

### Analytics Engine
- **Crop Profitability Ratios** — crop price divided by input cost (energy or fertilizer), used as a directional trend indicator for margin pressure
- **Cost Indices** — each commodity normalized to its own historical mean (100 = average); the average across a category produces an energy and fertilizer cost index that is meaningful in absolute terms
- **Rolling Z-Score Signals** — statistical anomaly detection over a 60-observation window; signals fire at |z| > 2.0 (top/bottom ~2.5% of distribution)
- **Market Regime Classification** — classifies current cost environment (high / normal / low) based on index thresholds, serialized to JSON for dashboard consumption

### Forecasting System
Nine models run per commodity, evaluated on a hold-out test set sized to the commodity's native frequency:

| Model | Type | Notes |
|---|---|---|
| ARIMA | Statistical | Auto order selection via `pmdarima` |
| SARIMA | Statistical | Seasonal variant with auto `m` detection |
| Exponential Smoothing | Statistical | Additive trend with SimpleExpSmoothing fallback |
| GARCH(1,1) | Volatility | Rolling one-step-ahead refit; outputs volatility forecast for confidence bands |
| Gradient Boosting | ML | Lag + diff features, no scaler |
| Random Forest | ML | Lag + diff features, no scaler |
| Ridge Regression | ML | Scaled features, cross-validated alpha |
| LASSO Regression | ML | Scaled features, cross-validated alpha |
| XGBoost | ML | 200 estimators, subsampling |

The **ensemble** is an exponential-weighted average of the top 3 models ranked by MASE (falling back to MAPE). A sanity filter rejects any model whose first forecast deviates more than 40% from the last known price.

Confidence is assigned via MASE: `HIGH < 0.8 · MEDIUM < 1.2 · LOW ≥ 1.2`.

### Dashboard (Streamlit, 4 pages)

All pages share a unified dark industrial design system built with IBM Plex Mono / IBM Plex Sans, consistent Plotly theming, and CSS custom properties.

| Page | Question answered | Key components |
|---|---|---|
| **Overview** | What is happening right now? | KPI strip, price history chart by category, commodity snapshot table, live z-score signals |
| **Analysis** | Why is it happening? | Price trends with MA overlay, z-score panel, historical distribution, correlation matrix |
| **Analytics** | What does it cost to produce? | Regime cards, cost index history, profitability z-score bar chart, ratio explorer |
| **Forecasting** | Where is it going? | Ensemble forecast chart with GARCH bands, period-by-period breakdown, model MAPE ranking, historical context |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `fredapi`, `pandas` |
| Statistical models | `statsmodels`, `pmdarima`, `arch` |
| ML models | `scikit-learn`, `xgboost` |
| Dashboard | `streamlit`, `plotly` |
| Config | JSON (single source of truth via `ConfigLoader`) |
| Automation | Python script + optional GitHub Actions / cron |


---

## Setup

### Prerequisites

- Python 3.10+
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
git clone https://github.com/your-username/commodity-tracker.git
cd commodity-tracker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_key_here
DAYS_HISTORY=1095        # 3 years of history minimun
DATA_DIR=data
```

### Running the Pipeline

```bash
# Collect data, compute metrics, generate analytics and forecasts
python automation/run_daily.py
```

### Launching the Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`.

---

## Design Decisions

**Frequency-aware metrics over naive daily resampling.** Many commodity price pipelines resample everything to daily and compute rolling statistics directly. This project instead computes percentage change, moving averages, and z-scores in each commodity's native frequency (e.g., monthly for fertilizers, daily for crude oil), then expands back to the full date index using period mapping. This eliminates artificial signal inflation from interpolated data.

**Config-driven architecture.** All commodity metadata — FRED series IDs, names, units, frequencies, and categories — lives in a single `commodities.json` file. Every module loads from `ConfigLoader`, which implements caching to avoid redundant I/O. Adding a new commodity requires a single JSON entry.

**GARCH for energy commodities.** ARIMA-family models assume constant variance. Energy prices exhibit volatility clustering (large moves follow large moves), which GARCH(1,1) captures. The model outputs a `volatility_forecast` per period alongside price predictions, which the dashboard uses to render 90% confidence bands.

**Ratio z-scores over absolute ratios.** Crop-to-input ratios (e.g., corn / natural gas) are dimensionally inconsistent across pairs and not comparable in absolute terms. The analytics module uses z-scores over a rolling window to normalize each ratio to its own history, making the signal directional and interpretable regardless of unit differences.

---

## Roadmap

- [ ] Add PostgreSQL backend to replace CSV storage
- [ ] Export dashboard as PDF report
- [ ] Docker containerization + GitHub Actions scheduled pipeline

---

## Data Source

All price series are retrieved from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API, maintained by the Federal Reserve Bank of St. Louis. Data is used in accordance with FRED's [Terms of Use](https://fred.stlouisfed.org/legal/).

---
