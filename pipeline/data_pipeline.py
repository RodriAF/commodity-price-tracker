"""
Data Pipeline — Statistical Signal Computation
Handles data cleaning, metric calculation, and persistence for the commodity tracker.

The pipeline operates in four stages:
    1. clean          — type coercion, deduplication, forward-fill, rounding
    2. merge          — append new data to the historical CSV without duplication
    3. calculate_metrics — frequency-aware change, moving average, and z-score
    4. save           — persist the enriched DataFrame to disk
"""

import os
import pandas as pd
import numpy as np
import logging
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end processing pipeline for commodity price data.

    Designed around a single persistent CSV file that is updated incrementally
    on each daily run. Derived columns (change %, MA, z-score, signal) are
    always recomputed from the raw price series to ensure consistency after
    any backfill or data correction.
    """

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.main_file = os.path.join(data_dir, 'commodity_data.csv')
        logger.info(f"DataPipeline initialised — data directory: {data_dir}")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise the raw collected DataFrame before any metric computation.

        Steps:
            - Parse the date column to pandas Timestamp for consistent indexing
            - Sort chronologically (FRED API may return unsorted series)
            - Forward-fill NaN values produced by the outer-join merge in the
              collector (mixed-frequency series leave gaps on non-observation dates)
            - Drop duplicate dates, keeping the most recent observation
            - Round all float columns to 2 decimal places to avoid floating-point
              noise accumulating in derived metrics
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

        The pipeline handles four data frequencies (daily, weekly, monthly,
        quarterly) and applies different parameters to each so that metrics
        are statistically meaningful regardless of how often a series updates.

        For each commodity the following are computed at its native frequency
        and then mapped back onto the daily date index:

        1. Percentage change (change_pct)
           Measures price momentum over a frequency-appropriate lookback:
               change_t = (P_t - P_{t-n}) / P_{t-n} * 100
           where n = change_periods from the frequency config. This avoids
           comparing weekly oil prices on a 1-period basis (too noisy) or
           quarterly corn on a 12-period basis (too long).

        2. Moving average (ma)
           Simple rolling mean over ma_window periods at native frequency.
           Acts as a trend filter: price above MA implies uptrend, below implies
           downtrend. Window length is calibrated to be roughly 3-6 months of
           observations regardless of frequency.

        3. Z-score (zscore)
           Standardises the percentage change series using a rolling window:
               z_t = (change_t - mu_t) / sigma_t
           The rolling window (z_window_map) is set to approximately 2 years
           of observations at each frequency. This captures medium-term
           distributional shifts without treating structural price level changes
           as permanent anomalies.

           A small epsilon (1e-8) is added to sigma to prevent division-by-zero
           on series with very low variance (e.g. administered price indices).

        4. Signal category (signal)
           Discretises the z-score into three buckets for dashboard display:
               |z| > 2  -> 'extreme'  (outside ~95% of the rolling distribution)
               |z| > 1  -> 'notable'  (outside ~68%)
               else     -> 'normal'

        Frequency alignment:
           Raw FRED data arrives at mixed frequencies. After resampling to
           native period frequency using .resample().last(), metrics are
           computed on the period index. The result is mapped back to the
           daily DataFrame using a PeriodIndex to avoid look-ahead bias —
           each daily row receives only the metric value known at the start
           of that period.
        """
        logger.info("Computing frequency-aware metrics (change, MA, z-score, signal)...")

        base_commodities = [
            col for col in df.columns
            if col != 'date'
            and not col.endswith(('_change_pct', '_ma', '_zscore', '_signal'))
        ]

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Pandas resample frequency aliases
        freq_map = {
            'daily':     'D',
            'weekly':    'W',
            'monthly':   'M',
            'quarterly': 'Q'
        }

        # Rolling z-score window: approximately 2 years of observations
        z_window_map = {
            'daily':     252,   # ~252 trading days per year
            'weekly':    52,    # 52 weeks per year
            'monthly':   24,    # 2 years of monthly data
            'quarterly': 12     # 3 years of quarterly data (more stable)
        }

        for commodity in base_commodities:

            frequency     = ConfigLoader.get_commodity_frequency(commodity)
            metric_config = ConfigLoader.get_metric_config(frequency)

            change_periods = metric_config['change_periods']
            ma_window      = metric_config['ma_window']
            pandas_freq    = freq_map.get(frequency, 'M')

            # Isolate the price series indexed by date
            ts = df.set_index('date')[commodity]

            # Resample to native frequency, keeping only the last observation
            # per period (avoids double-counting on the daily spine)
            ts_period = ts.resample(pandas_freq).last().dropna()

            # Convert to PeriodIndex — required for correct period-based mapping
            # back to the daily DataFrame without introducing look-ahead bias
            ts_period.index = ts_period.index.to_period(pandas_freq)

            # Percentage change over the frequency-appropriate number of periods
            change = ts_period.pct_change(periods=change_periods) * 100

            # Rolling moving average as a trend baseline
            ma = ts_period.rolling(window=ma_window, min_periods=1).mean()

            # Rolling z-score on the change series
            z_window     = z_window_map.get(frequency, 24)
            rolling_mean = change.rolling(window=z_window, min_periods=5).mean()
            rolling_std  = change.rolling(window=z_window, min_periods=5).std()
            zscore       = (change - rolling_mean) / (rolling_std + 1e-8)

            # Discretise z-score into signal categories for dashboard display
            signal = pd.Series(
                np.where(zscore.isna(), 'no_data',
                np.where(abs(zscore) > 2, 'extreme',
                np.where(abs(zscore) > 1, 'notable', 'normal'))),
                index=zscore.index
            )

            # Map period-level metrics back onto the daily date spine.
            # Each daily row is assigned the metric value of the period it
            # belongs to — no future information leaks across period boundaries.
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

    def merge_with_existing(self, new_df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        """
        Append new observations to the existing historical CSV without duplication.

        Only the raw price columns (no derived metrics) are read from the existing
        file. Derived columns are always recomputed from scratch by calculate_metrics
        to ensure they reflect the full updated history — this avoids stale z-scores
        or moving averages that would result from appending pre-computed values.

        Duplicates on the date key are resolved by keeping the newer observation,
        which allows backfill corrections from FRED to propagate correctly.
        """
        if not os.path.exists(filepath):
            return new_df

        existing_df = pd.read_csv(filepath)
        existing_df['date'] = pd.to_datetime(existing_df['date'])

        # Retain only raw price columns from the existing file
        existing_base_cols = [
            col for col in existing_df.columns
            if col == 'date'
            or not any(col.endswith(suffix) for suffix in [
                '_change_pct', '_ma', '_zscore', '_signal'
            ])
        ]

        existing_df = existing_df[existing_base_cols]

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)

        return combined

    def save(self, df: pd.DataFrame, filepath: str) -> str:
        """Persist the processed DataFrame to CSV and return the file path."""
        df.to_csv(filepath, index=False)
        logger.info(f"  Dataset saved: {filepath}")
        return filepath

    def process_and_save(self, df: pd.DataFrame) -> str:
        """
        Execute the full pipeline: clean -> merge -> compute metrics -> save.

        This is the primary entry point called by run_daily.py. The pipeline
        is intentionally sequential and stateless between runs — each execution
        reads from disk, updates in memory, and writes back to the same file.
        """
        logger.info("=" * 70)
        logger.info("DATA PIPELINE — PROCESSING AND SAVING")
        logger.info("=" * 70)

        df_clean   = self.clean(df)
        df_merged  = self.merge_with_existing(df_clean, self.main_file)
        df_metrics = self.calculate_metrics(df_merged)
        filepath   = self.save(df_metrics, self.main_file)

        return filepath

    def load_latest(self) -> pd.DataFrame:
        """
        Load the most recently saved processed dataset from disk.

        Returns an empty DataFrame if no data file exists yet, which allows
        dashboard pages to handle the first-run state gracefully.
        """
        if not os.path.exists(self.main_file):
            return pd.DataFrame()

        df = pd.read_csv(self.main_file)
        df['date'] = pd.to_datetime(df['date'])
        return df