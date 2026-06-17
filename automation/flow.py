"""
===============================================================================
DAILY COMMODITY TRACKER PIPELINE
Prefect 2.x Orchestration Flow
===============================================================================

Execution Context:
    - Native Prefect flow (@flow), runnable locally without Prefect Cloud.
    - Persists all run data (prices, signals, forecasts, run metadata) to DuckDB.
    - No CSV/JSON side-files are produced.
===============================================================================
"""

import os
import sys
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from prefect import task, flow
from prefect.context import get_run_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collectors.agricultural_collector import AgriculturalCollector
from pipeline.data_pipeline import DataPipeline
from pipeline.calculations import CommoditiesAnalytics, CorrelationAnalysis
from pipeline.predictions import CommoditiesForecaster
from pipeline.validation import validate

from utils.db import log_run, upsert_signals, upsert_forecasts, initialize, check_if_schema_exists
from automation.alerts import send_alert
from config.settings import get_settings


def get_logger() -> logging.Logger:
    """Use Prefect's run logger inside a flow run, fall back to stdlib otherwise."""
    try:
        from prefect import get_run_logger
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# Adapters — bridge wide DataFrames <-> formats expected by other modules
# =============================================================================

def _collector_to_validation_dict(
    series_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Adapts the collector output format to the format expected by validation.py.

    Collector produces:  {commodity: DataFrame(columns=['date', commodity, 'is_imputed'])}
    Validation expects:  {commodity: DataFrame(index=DatetimeIndex, columns=['value', 'is_imputed'])}

    Steps per series:
        1. Parse 'date' column to DatetimeIndex and set as index.
        2. Rename the commodity column to 'value'.
        3. Preserve 'is_imputed' if present.
    """
    adapted = {}
    for commodity, df in series_dict.items():
        if commodity not in df.columns:
            continue
        sub = df.copy()
        sub['date'] = pd.to_datetime(sub['date'])
        sub = sub.set_index('date')
        sub = sub.rename(columns={commodity: 'value'})
        # Keep only value and is_imputed — drop anything else
        cols = ['value'] + (['is_imputed'] if 'is_imputed' in sub.columns else [])
        adapted[commodity] = sub[cols]
    return adapted


def _series_dict_to_wide(series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Converts the validated series dict back to a wide DataFrame.

    Input:  {commodity: DataFrame(index=DatetimeIndex, columns=['value', ...])}
    Output: DataFrame(columns=['date', 'corn', 'wheat', ...])
    """
    if not series_dict:
        return pd.DataFrame(columns=['date'])

    frames = []
    for commodity, sub in series_dict.items():
        s = sub['value'].rename(commodity)
        frames.append(s)

    wide = pd.concat(frames, axis=1)
    wide = wide.reset_index().rename(columns={'index': 'date', 'date': 'date'})
    wide = wide.sort_values('date').reset_index(drop=True)
    return wide


def _regime_to_text(regime: Dict[str, str]) -> str:
    """Format the regime dict as a short human-readable string for logs and alerts."""
    if not regime:
        return "normal"
    return " | ".join(f"{k.title()}: {v}" for k, v in regime.items())


def _signals_to_db_df(signals: List[Dict], run_date: date) -> pd.DataFrame:
    """
    Map the signal dicts produced by CommoditiesAnalytics.generate_signals
    (keys: metric, type, z_score, strength) onto the schema of db.signals
    (run_date, commodity, metric, signal_type, z_score, percentile).
    """
    columns = ['run_date', 'commodity', 'metric', 'signal_type', 'z_score', 'percentile']
    if not signals:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(signals)
    df['run_date'] = run_date
    # 'metric' here is the ratio/index name (e.g. "corn_to_crude_oil") —
    # used as both the commodity identifier and the metric name.
    df['commodity'] = df['metric']
    df['signal_type'] = df['type']
    df['percentile'] = np.nan
    return df[columns]


# Confidence labels are categorical in predictions.py but the `forecasts`
# table stores `confidence` as DOUBLE. We map to a numeric score and keep
# the original label inside metrics_json for full traceability.
_CONFIDENCE_SCORE = {'high': 1.0, 'medium': 0.5, 'low': 0.0}


def _forecasts_to_db_df(all_forecasts: Dict[str, Any], run_date: date) -> pd.DataFrame:
    """
    Flatten the nested forecast results (per commodity -> per model + ensemble)
    into the row-based schema of db.forecasts.
    """
    columns = ['run_date', 'commodity', 'model', 'horizon',
               'predictions_json', 'metrics_json', 'confidence']
    rows = []

    for commodity, data in all_forecasts.items():
        horizon = data.get('horizon')

        for model_key, result in data.get('individual_models', {}).items():
            if 'predictions' not in result:
                continue

            raw_confidence = result.get('confidence', 'low')
            numeric_confidence = _CONFIDENCE_SCORE.get(raw_confidence, 0.0)

            rows.append({
                'run_date': run_date,
                'commodity': commodity,
                'model': result.get('method', model_key),
                'horizon': horizon,
                'predictions_json': json.dumps(result.get('predictions')),
                'metrics_json': json.dumps(result.get('metrics', {})),
                'confidence': numeric_confidence,
            })

        ensemble = data.get('ensemble')
        if ensemble:
            metrics = {
                'avg_mape': ensemble.get('avg_mape'),
                'top_models': ensemble.get('top_models'),
            }

            raw_ens_confidence = ensemble.get('confidence', 'low')
            numeric_ens_confidence = _CONFIDENCE_SCORE.get(raw_ens_confidence, 0.0)
            rows.append({
                'run_date': run_date,
                'commodity': commodity,
                'model': 'ensemble',
                'horizon': horizon,
                'predictions_json': json.dumps(ensemble.get('predictions')),
                'metrics_json': json.dumps(metrics),
                'confidence': numeric_ens_confidence,
            })

    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# PREFECT TASKS
# =============================================================================

@task(retries=0)
def collect(fred_key: str, days_history: int) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Connects to the FRED API to extract raw commodity series.

    Constraint: retries=0. If the source API is down or credentials are
    invalid, the flow fails immediately rather than processing partial data.
    """
    logger = get_logger()
    logger.info("Task [1/6]: Starting Data Collection")

    collector = AgriculturalCollector(fred_key, days_history)
    series_dict, metadata = collector.collect()

    if not series_dict:
        raise ValueError("CRITICAL: No data collected from FRED API.")

    total_records = sum(len(df) for df in series_dict.values())
    logger.info(f"Successfully extracted {len(series_dict)} series, {total_records} records.")
    return series_dict, metadata

@task(retries=2)
def validate_data(
    series_dict: Dict[str, pd.DataFrame], metadata: Dict
) -> pd.DataFrame:
    """
    Adapts the collector format, runs quality checks on each series, 
    and returns a wide DataFrame containing only validated commodities.
    """
    logger = get_logger()
    logger.info("Task [2/6]: Validating Collected Data")

    # Adapt from collector format to validation format
    adapted_dict = _collector_to_validation_dict(series_dict)

    valid_dict, report = validate(adapted_dict, metadata)

    if report.rejected_series:
        logger.warning(f"Rejected series: {list(report.rejected_series.keys())}")
        for name, reason in report.rejected_series.items():
            logger.warning(f"  [{name}]: {reason}")

    for series, series_warnings in report.warnings.items():
        for w in series_warnings:
            logger.info(f"Warning [{series}]: {w}")

    logger.info(
        f"Validation summary: {len(report.valid_series)} passed, "
        f"{len(report.rejected_series)} rejected."
    )

    if not valid_dict:
        raise ValueError("CRITICAL: All series failed validation — nothing to process.")

    return _series_dict_to_wide(valid_dict)

@task(retries=2)
def process(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data, merges with history in DuckDB, recomputes derived
    metrics, and persists the result.
    """
    logger = get_logger()
    logger.info("Task [3/6]: Starting Data Processing")

    pipeline = DataPipeline()
    pipeline.process_and_save(raw_data)

    return pipeline.load_latest()


@task(retries=2)
def analytics(full_data: pd.DataFrame, run_date: date) -> Dict[str, Any]:
    """
    Computes input-to-crop ratios, statistical z-scores, anomaly signals,
    and the current market regime. Signals are persisted to DuckDB.
    """
    logger = get_logger()
    logger.info("Task [4/6]: Executing Commodities Analytics")

    analytics_module = CommoditiesAnalytics(full_data)
    _, analytics_data = analytics_module.calculate_all()

    corr_calc = CorrelationAnalysis(full_data)
    key_corr = corr_calc.key_correlations()
    if not key_corr.empty:
        logger.info("Top Structural Correlations (Crop vs Input):")
        for _, row in key_corr.head(5).iterrows():
            logger.info(f"  {row['crop']:>12} <-> {row['input']:<20} {row['correlation']:+.3f}")

    signals = analytics_data.get('signals', [])
    regime = analytics_data.get('regime', {})

    logger.info(f"Identified Market Regime: {_regime_to_text(regime)}")
    logger.info(f"Total Anomalous Signals Detected: {len(signals)}")

    signals_df = _signals_to_db_df(signals, run_date)
    if not signals_df.empty:
        upsert_signals(signals_df)

    return analytics_data


@task(retries=2)
def forecast_one(commodity: str, full_data: pd.DataFrame, frequency: str, horizon: int = 3) -> Tuple[str, Dict[str, Any]]:
    """
    Runs the full forecasting suite for a single commodity. Designed to be
    mapped across commodities so failures and slow models are isolated.
    """
    logger = get_logger()

    forecaster = CommoditiesForecaster(
        data=full_data, commodity=commodity, frequency=frequency, horizon=horizon
    )

    is_valid, message = forecaster.validate_data()
    if not is_valid:
        logger.warning(f"Target Skipped [{commodity}]: {message}")
        return commodity, {}

    results = forecaster.forecast_all_models()
    if 'error' in results:
        logger.warning(f"Forecast Error [{commodity}]: {results['error']}")
        return commodity, {}

    ensemble = forecaster.create_ensemble(results)
    current_price = float(full_data[commodity].iloc[-1])

    payload = make_serializable({
        'current_price': current_price,
        'frequency': frequency,
        'horizon': horizon,
        'individual_models': results,
        'ensemble': ensemble if ensemble else None,
    })

    if ensemble and 'predictions' in ensemble:
        pred_p1 = ensemble['predictions'][0]
        pct_change = (pred_p1 - current_price) / current_price * 100
        logger.info(
            f"Forecast {commodity:<15} | Current: ${current_price:7.2f} | "
            f"T+1: ${pred_p1:7.2f} ({pct_change:+6.2f}%)"
        )

    return commodity, payload


@task(retries=2)
def forecasting(full_data: pd.DataFrame, metadata: Dict, run_date: date) -> Dict[str, Any]:
    """
    Executes the forecasting suite for all key commodities in parallel and
    persists the results to DuckDB.
    """
    logger = get_logger()
    logger.info("Task [5/6]: Initializing Multi-Horizon Forecasting Engine")

    key_commodities = [
        'corn', 'wheat', 'soybeans',
        'crude_oil', 'natural_gas', 'diesel', 'gasoline',
        'urea', 'dap_fertilizer', 'phosphate', 'potash'
    ]
    active_targets = [c for c in key_commodities if c in full_data.columns]

    all_forecasts: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=min(4, len(active_targets) or 1)) as executor:
        futures = {
            executor.submit(
                forecast_one.fn,  # call the underlying function directly inside the thread pool
                commodity,
                full_data,
                metadata.get(commodity, {}).get('frequency', 'monthly'),
            ): commodity
            for commodity in active_targets
        }

        for future in as_completed(futures):
            commodity = futures[future]
            try:
                name, payload = future.result()
                if payload:
                    all_forecasts[name] = payload
            except Exception as e:
                logger.error(f"Critical failure modeling {commodity}: {e}")

    forecasts_df = _forecasts_to_db_df(all_forecasts, run_date)
    if not forecasts_df.empty:
        upsert_forecasts(forecasts_df)

    return all_forecasts


@task(retries=2)
def alerts(signals: List[Dict], regime: Dict[str, str], run_date: date):
    """
    Dispatches an email alert (or logs it, if SMTP is not configured) when
    any signal exceeds the configured z-score threshold.
    """
    logger = get_logger()
    logger.info("Task [6/6]: Evaluating Alert Dispatch Conditions")

    if not signals:
        logger.info("No actionable signals generated. Dispatch aborted.")
        return

    send_alert(signals, _regime_to_text(regime), run_date.isoformat())
    logger.info("Alert dispatch sequence complete.")


# =============================================================================
# MAIN PREFECT FLOW
# =============================================================================

@flow(name="commodity_tracker_daily", log_prints=True)
def commodity_tracker_daily():
    """
    Primary orchestration wrapper. Loads configuration from Settings,
    runs the pipeline end-to-end, and persists run metadata to DuckDB
    regardless of outcome.
    """
    settings = get_settings()
    logger = get_logger()
    start_time = datetime.now()
    run_date = start_time.date()

    try:
        run_id = str(get_run_context().flow_run.id)
    except Exception:
        run_id = str(uuid.uuid4())

    logger.info("=" * 70)
    logger.info("COMMODITY TRACKER PIPELINE INITIALIZED")
    logger.info(f"Execution Context ID: {run_id}")
    logger.info("=" * 70)

    # SMART SCHEMA CHECK:
    # If the core tables do not exist, initialize the full DuckDB schema.
    # Otherwise, skip initialization to preserve resources and proceed immediately.
    if not check_if_schema_exists():
        logger.info("Database is empty. Initializing database schema for the first time...")
        initialize()
    else:
        logger.info("Database schema detected. Skipping initialization.")

    log_run(run_id=run_id, started_at=start_time, status='RUNNING')

    if not settings.fred_api_key_is_set:
        logger.error("FATAL: FRED_API_KEY is not configured.")
        log_run(run_id=run_id, started_at=start_time, status='FAILED', completed_at=datetime.now())
        raise RuntimeError("FRED_API_KEY is not configured.")

    try:
        raw_data, metadata = collect(settings.fred_api_key, settings.days_history)
        clean_raw = validate_data(raw_data, metadata)
        full_data = process(clean_raw)

        analytics_data = analytics(full_data, run_date)
        signals = analytics_data.get('signals', [])
        regime = analytics_data.get('regime', {})

        forecast_results = forecasting(full_data, metadata, run_date)
        alerts(signals, regime, run_date)

        successful_forecasts = sum(
            1 for v in forecast_results.values() if v.get('ensemble') is not None
        )
        run_summary = json.dumps({
            "signals_detected": len(signals),
            "regime": regime,
            "forecasts_completed": successful_forecasts,
            "total_commodities": len(forecast_results),
        })

        log_run(
            run_id=run_id,
            started_at=start_time,
            status='SUCCESS',
            completed_at=datetime.now(),
            summary_json=run_summary,
        )

        logger.info("=" * 70)
        logger.info(
            f"PIPELINE COMPLETE | Forecast Success Rate: "
            f"{successful_forecasts}/{len(forecast_results)}"
        )
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline execution aborted: {e}", exc_info=True)
        log_run(
            run_id=run_id,
            started_at=start_time,
            status='FAILED',
            completed_at=datetime.now(),
            summary_json=json.dumps({"error_type": type(e).__name__, "message": str(e)}),
        )
        raise


# =============================================================================
# AUTOMATED DEPLOYMENT ORCHESTRATION
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta

    commodity_tracker_daily.serve(
        name="daily-commodity-etl-deployment",
        
        # Prefect 2.x/3.x takes the interval or cron directly
        interval=timedelta(seconds=360), 
        
        # If you wanted to use cron instead, it would look like this:
        # cron="0 1 * * *",
        
        description="Daily automated pipeline..."
    )