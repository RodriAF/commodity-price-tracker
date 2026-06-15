"""
Data Pipeline — Statistical Signal Computation
Handles data cleaning, metric calculation, and persistence for the commodity tracker.

The pipeline operates in four stages:
    1. clean             — type coercion, deduplication, forward-fill, rounding
    2. merge             — append new data to the historical database without duplication
    3. calculate_metrics — frequency-aware change, moving average, and z-score
    4. save              — persist the enriched DataFrame to DuckDB
"""

import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config_loader import ConfigLoader

# Import the centralized DuckDB operations module
import utils.db as db

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data Pipeline Core                                                 #
# ------------------------------------------------------------------ #

class DataPipeline:
    """
    End-to-end processing pipeline for commodity price data.

    Migrated to use DuckDB for persistence. Derived columns (change %, MA, 
    z-score, signal) are always recomputed from the raw price series to ensure 
    consistency after any backfill or data correction. Public interfaces 
    maintain the "wide" DataFrame format for backwards compatibility.
    """

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the pipeline. Parameters are kept for backward 
        compatibility with existing callers, but storage is now in DuckDB.
        """
        self.data_dir = data_dir
        self.main_file = os.path.join(data_dir, 'commodity_data.csv')
        
        logger.info("DataPipeline initialized — Storage mode: DuckDB")
        
        # Ensure the database schema is ready before any operations
        db.initialize()

    # ------------------------------------------------------------------ #
    # Processing Methods (In-Memory)                                     #
    # ------------------------------------------------------------------ #

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the raw collected DataFrame before any metric computation.

        Steps:
            - Parse the date column to pandas Timestamp for consistent indexing.
            - Sort chronologically (FRED API may return unsorted series).
            - Forward-fill NaN values produced by the outer-join merge.
            - Drop duplicate dates, keeping the most recent observation.
            - Round all float columns to 2 decimal places to avoid noise.
        """
        logger.info("Cleaning raw data...")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        df = df.ffill()
        df = df.drop_duplicates(subset=['date'], keep='last')

        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].round(2)

        logger.info(f"  Cleaned dataset: {len(df)} rows")

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute frequency-aware technical metrics for every commodity series.
        """
        logger.info("Computing frequency-aware metrics (change, MA, z-score, signal)...")

        base_commodities = [
            col for col in df.columns
            if col != 'date'
            and not col.endswith(('_change_pct', '_ma', '_zscore', '_signal'))
        ]

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        freq_map = {
            'daily':     'D',
            'weekly':    'W',
            'monthly':   'M',
            'quarterly': 'Q'
        }

        z_window_map = {
            'daily':     252,
            'weekly':    52,
            'monthly':   24,
            'quarterly': 12
        }

        for commodity in base_commodities:

            frequency     = ConfigLoader.get_commodity_frequency(commodity)
            metric_config = ConfigLoader.get_metric_config(frequency)

            change_periods = metric_config['change_periods']
            ma_window      = metric_config['ma_window']
            pandas_freq    = freq_map.get(frequency, 'M')

            ts = df.set_index('date')[commodity]
            ts_period = ts.resample(pandas_freq).last().dropna()
            ts_period.index = ts_period.index.to_period(pandas_freq)

            change = ts_period.pct_change(periods=change_periods) * 100

            # Rolling moving average as a trend baseline.
            # UPDATED: Require at least 50% of the window for a statistically valid mean,
            # with an absolute minimum of 3 periods.
            valid_min_periods = max(3, int(ma_window * 0.5))
            ma = ts_period.rolling(window=ma_window, min_periods=valid_min_periods).mean()

            z_window     = z_window_map.get(frequency, 24)
            rolling_mean = change.rolling(window=z_window, min_periods=5).mean()
            rolling_std  = change.rolling(window=z_window, min_periods=5).std()
            zscore       = (change - rolling_mean) / (rolling_std + 1e-8)

            signal = pd.Series(
                np.where(zscore.isna(), 'no_data',
                np.where(abs(zscore) > 2, 'extreme',
                np.where(abs(zscore) > 1, 'notable', 'normal'))),
                index=zscore.index
            )

            df_period_index = df['date'].dt.to_period(pandas_freq)

            df[f'{commodity}_change_pct'] = df_period_index.map(change)
            df[f'{commodity}_ma']         = df_period_index.map(ma)
            df[f'{commodity}_zscore']     = df_period_index.map(zscore)
            df[f'{commodity}_signal']     = df_period_index.map(signal)

            logger.info(
                f"  {commodity}: frequency={frequency} | periods={len(ts_period)} "
                f"| change_periods={change_periods} | MA_window={ma_window}"
            )

        return df

    # ------------------------------------------------------------------ #
    # I/O Operations (DuckDB Integration)                                #
    # ------------------------------------------------------------------ #

    def merge_with_existing(self, new_df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        """
        Append new observations to the existing historical data fetched from DuckDB.
        
        Transforms the 'long' format returned by the database into the 'wide'
        format needed by the pipeline. Derived metrics are ignored here, ensuring
        they are recomputed from scratch.
        """
        existing_long_df = db.load_prices()
        
        if existing_long_df.empty:
            return new_df

        # Pivot the database long format back into our expected wide format
        existing_df = existing_long_df.pivot(
            index='date', 
            columns='commodity', 
            values='value'
        ).reset_index()

        existing_df['date'] = pd.to_datetime(existing_df['date'])
        new_df['date'] = pd.to_datetime(new_df['date'])

        # Combine, resolve duplicates by keeping the latest, and sort
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)

        return combined

    def save(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Persist the processed DataFrame to DuckDB.
        
        Unpivots the 'wide' DataFrame into 'long' schemas expected by `db.prices`
        and `db.signals` tables, then triggers the upserts. Returns the original
        filepath parameter to maintain interface compatibility.
        """
        base_commodities = [
            col for col in df.columns
            if col != 'date'
            and not col.endswith(('_change_pct', '_ma', '_zscore', '_signal'))
        ]

        # 1. Prepare and upsert prices
        prices_list = []
        for commodity in base_commodities:
            if commodity in df.columns:
                temp_df = df[['date', commodity]].copy()
                temp_df = temp_df.rename(columns={commodity: 'value'})
                temp_df['commodity'] = commodity
                prices_list.append(temp_df)

        if prices_list:
            prices_df = pd.concat(prices_list, ignore_index=True)
            prices_df = prices_df.dropna(subset=['value'])
            prices_df['is_imputed'] = False
            prices_df['created_at'] = datetime.now()
            
            db.upsert_prices(prices_df)

        # 2. Prepare and upsert signals
        signals_list = []
        for commodity in base_commodities:
            zscore_col = f'{commodity}_zscore'
            signal_col = f'{commodity}_signal'
            
            if zscore_col in df.columns and signal_col in df.columns:
                temp_sig = df[['date', zscore_col, signal_col]].copy()
                temp_sig = temp_sig.rename(columns={
                    'date': 'run_date',
                    zscore_col: 'z_score',
                    signal_col: 'signal_type'
                })
                temp_sig['commodity'] = commodity
                temp_sig['metric'] = 'price_momentum'
                temp_sig['percentile'] = np.nan
                signals_list.append(temp_sig)

        if signals_list:
            signals_df = pd.concat(signals_list, ignore_index=True)
            signals_df = signals_df.dropna(subset=['z_score'])
            
            db.upsert_signals(signals_df)

        logger.info("  Dataset persisted to DuckDB successfully.")
        return filepath

    def process_and_save(self, df: pd.DataFrame) -> str:
        """
        Execute the full pipeline: clean -> merge -> compute metrics -> save.
        """
        logger.info("=" * 70)
        logger.info("DATA PIPELINE — PROCESSING AND SAVING (DUCKDB)")
        logger.info("=" * 70)

        df_clean   = self.clean(df)
        df_merged  = self.merge_with_existing(df_clean, self.main_file)
        df_metrics = self.calculate_metrics(df_merged)
        filepath   = self.save(df_metrics, self.main_file)

        return filepath

    def load_latest(self) -> pd.DataFrame:
        """
        Load the most recently saved processed dataset from DuckDB.

        Fetches raw prices, pivots them into a wide DataFrame, and recalculates
        the metrics to guarantee callers receive the exact expected format
        without needing to query the `signals` table separately.
        """
        existing_long_df = db.load_prices()
        
        if existing_long_df.empty:
            return pd.DataFrame()

        # Pivot prices into wide format
        existing_df = existing_long_df.pivot(
            index='date', 
            columns='commodity', 
            values='value'
        ).reset_index()
        
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        
        # Reconstruct metrics over the raw prices
        return self.calculate_metrics(existing_df)