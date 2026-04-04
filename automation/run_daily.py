"""
Daily Automation - Updated for Ratios
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import numpy as np
import pandas as pd

# Ensure parent directory is in the path for internal module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collectors.agricultural_collector import AgriculturalCollector
from pipeline.data_pipeline import DataPipeline
from pipeline.calculations import CommoditiesAnalytics, CorrelationAnalysis
from pipeline.predictions import CommoditiesForecaster


def setup_logging():
    """Configure logging with both file and console handlers."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def make_serializable(obj):
    """
    Recursively convert numpy types to native Python types
    to ensure JSON compatibility.
    """
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

    else:
        return obj


def main():
    logger = setup_logging()
    load_dotenv()

    logger.info("=" * 70)
    logger.info("COMMODITY TRACKER - DAILY RUN")
    logger.info("=" * 70)

    # Load environment variables
    fred_key = os.getenv('FRED_API_KEY')
    days_history = int(os.getenv('DAYS_HISTORY', 1095))
    data_dir = os.getenv('DATA_DIR', 'data')

    if not fred_key:
        logger.error("FRED_API_KEY not found in environment variables")
        return 1

    try:
        # ========== STEP 1: COLLECT DATA ==========
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 70)

        collector = AgriculturalCollector(fred_key, days_history)
        raw_data, metadata = collector.collect()

        if raw_data.empty:
            logger.error("No data collected from FRED")
            return 1

        # ========== STEP 2: PROCESS DATA ==========
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: DATA PROCESSING")
        logger.info("=" * 70)

        pipeline = DataPipeline(data_dir)
        filepath = pipeline.process_and_save(raw_data)

        # Load the latest processed data for downstream analysis
        full_data = pipeline.load_latest()


        # ========== STEP 3: COMMODITIES ANALYTICS ==========
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: COMMODITIES ANALYTICS (RATIOS, SIGNALS, REGIME)")
        logger.info("=" * 70)

        analytics = CommoditiesAnalytics(full_data)
        combined, analytics_data = analytics.calculate_all()
        zscores = analytics_data.get('zscores', pd.DataFrame())

        if not zscores.empty:
            combined = pd.concat([combined, zscores], axis=1)

        ratios_filepath = None

        if not combined.empty:
            # Attach date column and persist to disk
            combined.insert(0, 'date', full_data['date'].values)
            ratios_filepath = os.path.join(data_dir, 'commodity_ratios.csv')
            combined.to_csv(ratios_filepath, index=False)
            logger.info(f"  Ratios saved : {ratios_filepath}")
        else:
            logger.warning("  No ratio data produced — check category configuration")

        # Compute and log the top five crop-input correlations
        corr_calc = CorrelationAnalysis(full_data)
        key_corr  = corr_calc.key_correlations()

        if not key_corr.empty:
            logger.info("\n  Top crop-input correlations:")
            for _, row in key_corr.head(5).iterrows():
                logger.info(f"    {row['crop']:>12} <-> {row['input']:<20} {row['correlation']:+.3f}")

        # Retrieve signals and regime classification
        signals = analytics_data.get('signals', [])
        regime  = analytics_data.get('regime', {})

        if signals:
            logger.info(f"\n  Signals detected ({len(signals)} total, showing top 5):")
            for s in signals[:5]:
                direction = "HIGH" if s['type'] == 'overvalued' else "LOW"
                logger.info(f"    [{direction}] {s['metric']:<40} z={s['z_score']:+.2f}")
        else:
            logger.info("  No anomalous signals detected")

        if regime:
            logger.info(f"\n  Market regime: {regime}")

        # Persist signals and regime as JSON for dashboard consumption
        signals_path = os.path.join(data_dir, 'signals.json')
        with open(signals_path, 'w') as f:
            json.dump({
                'date':    datetime.now().isoformat(),
                'signals': signals,
                'regime':  regime
            }, f, indent=2)

        logger.info(f"  Signals saved: {signals_path}")


        # ========== STEP 4: COMMODITIES FORECASTING ==========
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: COMMODITIES MULTI-HORIZON FORECASTING")
        logger.info("=" * 70)

        key_commodities = [
            # Grains and oilseeds
            'corn', 'wheat', 'soybeans',
            # Energy inputs
            'crude_oil', 'natural_gas', 'diesel', 'gasoline',
            # Fertilizers
            'urea', 'dap_fertilizer', 'phosphate', 'potash',
        ]
        # Retain only commodities present in the dataset
        key_commodities = [c for c in key_commodities if c in full_data.columns]

        all_forecasts = {}
        forecast_filepath = None

        if key_commodities:
            for commodity in key_commodities:
                logger.info(f"\n  Forecasting: {commodity}")

                # Retrieve series frequency from metadata; default to monthly
                frequency = metadata.get(commodity, {}).get('frequency', 'monthly')

                try:
                    # Initialise the forecasting engine for this commodity
                    forecaster = CommoditiesForecaster(
                        data=full_data,
                        commodity=commodity,
                        frequency=frequency,
                        horizon=3
                    )

                    # Validate that sufficient data is available before proceeding
                    is_valid, message = forecaster.validate_data()
                    if not is_valid:
                        logger.warning(f"  Skipping {commodity}: {message}")
                        all_forecasts[commodity] = {
                            'error': 'insufficient_data',
                            'message': message
                        }
                        continue

                    # Run all models in the forecasting suite
                    results = forecaster.forecast_all_models()

                    if 'error' in results:
                        logger.warning(f"  Forecast failed for {commodity}: {results['error']}")
                        all_forecasts[commodity] = results
                        continue

                    # Build ensemble prediction from the top-performing models
                    ensemble = forecaster.create_ensemble(results)

                    # Retrieve the most recent observed price
                    current_price = float(full_data[commodity].iloc[-1])

                    # Store serialised results for JSON output
                    all_forecasts[commodity] = make_serializable({
                        'current_price': current_price,
                        'frequency': frequency,
                        'horizon': 3,
                        'individual_models': results,
                        'ensemble': ensemble if ensemble else None,
                        'top_5_models': ensemble['top_models'] if ensemble else []
                    })

                    # Log core forecast metrics for the first forecast period
                    if ensemble and 'predictions' in ensemble:
                        pred_p1 = ensemble['predictions'][0]
                        change_p1 = ((pred_p1 - current_price) / current_price * 100)
                        confidence = ensemble.get('confidence', 'unknown')
                        top_models = ', '.join(ensemble.get('top_models', [])[:3])

                        logger.info(f"  Current price : ${current_price:.2f}")
                        logger.info(f"  Period 1      : ${pred_p1:.2f} ({change_p1:+.2f}%)")
                        logger.info(f"  Confidence    : {confidence.upper()}")
                        logger.info(f"  Top models    : {top_models}")
                    else:
                        logger.warning(f"  Ensemble could not be constructed for {commodity}")

                except Exception as e:
                    logger.error(f"  Error forecasting {commodity}: {str(e)}")
                    all_forecasts[commodity] = {
                        'error': 'processing_error',
                        'message': str(e)
                    }

            # Persist all forecasts to the output directory
            forecast_dir = os.path.join(data_dir, 'forecasts')
            os.makedirs(forecast_dir, exist_ok=True)

            timestamp = datetime.now()
            forecast_filename = f'forecasts_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
            forecast_filepath = os.path.join(forecast_dir, forecast_filename)

            with open(forecast_filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp.isoformat(),
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'forecasts': all_forecasts
                }, f, indent=2)

            logger.info(f"\n  Forecasts saved: {forecast_filepath}")

        # ========== FINAL SUMMARY ==========
        logger.info("\n" + "=" * 70)
        logger.info("RUN COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Data     : {filepath}")
        logger.info(f"  Ratios   : {ratios_filepath if ratios_filepath else 'N/A'}")
        logger.info(f"  Forecasts: {forecast_filepath if forecast_filepath else 'N/A'}")

        successful = sum(
            1 for v in all_forecasts.values()
            if 'ensemble' in v and v.get('ensemble') is not None
        )
        logger.info(f"  Successful forecasts: {successful}/{len(all_forecasts)}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)