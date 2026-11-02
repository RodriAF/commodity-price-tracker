"""
Data Pipeline — Statistical Signal Computation
Handles data cleaning, metric calculation, and persistence for the commodity tracker.

The pipeline operates in four stages:
    1. clean             — type coercion, deduplication, rounding (NO gap filling)
    2. merge             — append new data to the historical database without duplication
    3. calculate_metrics — frequency-aware change, moving average, and z-score
    4. save              — persist the enriched DataFrame to DuckDB

Storage contract: the ``prices`` table holds ONLY real observations (plus the
short, explicitly flagged gap fills done by the collector inside a single
series). Series are never forward-filled onto each other's calendars here.
Consumers that need aligned series (ratios, correlations, charts) align them
in memory with ``utils.alignment.align_wide_frame``.
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

# Boolean "this value was imputed by the collector" columns travel next to the
# value columns in the wide frame, named "<commodity>__imputed".
IMPUTED_SUFFIX = '__imputed'
_DERIVED_SUFFIXES = ('_change_pct', '_ma', '_zscore', '_signal')


def _resample_alias(base: str) -> str:
    """'M'/'Q' were renamed 'ME'/'QE' in pandas 2.2; pick whichever this pandas accepts."""
    renamed = {'M': 'ME', 'Q': 'QE'}.get(base)
    if renamed is None:
        return base
    try:
        pd.tseries.frequencies.to_offset(renamed)
        return renamed
    except ValueError:
        return base


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

    @staticmethod
    def split_imputed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate the ``<commodity>__imputed`` flag columns from the value columns.

        Returns (values_df, imputed_mask) where the mask is a wide boolean frame
        indexed by date with one column per commodity. The mask is empty when
        the input carries no flags.
        """
        flag_cols = [c for c in df.columns if c.endswith(IMPUTED_SUFFIX)]
        if not flag_cols:
            return df, pd.DataFrame()

        values = df.drop(columns=flag_cols)
        mask = df[['date'] + flag_cols].copy()
        mask['date'] = pd.to_datetime(mask['date'])
        mask = mask.set_index('date')
        mask.columns = [c[: -len(IMPUTED_SUFFIX)] for c in mask.columns]
        return values, mask.astype('boolean').fillna(False).astype(bool)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the raw collected DataFrame before any metric computation.

        Steps:
            - Parse the date column to pandas Timestamp for consistent indexing.
            - Sort chronologically (FRED API may return unsorted series).
            - Drop duplicate dates, keeping the most recent observation.
            - Round all float columns to 2 decimal places to avoid noise.

        NaN values produced by the outer join of series with different
        calendars are deliberately KEPT: they mean "no observation on this
        date", not "price unchanged". Filling them would fabricate data points
        (a monthly price repeated on every business day) and distort changes,
        forecasts and statistics downstream.
        """
        logger.info("Cleaning raw data...")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

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
            and not col.endswith(_DERIVED_SUFFIXES)
            and not col.endswith(IMPUTED_SUFFIX)
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
            ts_period = ts.resample(_resample_alias(pandas_freq)).last().dropna()
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

            # Rows dated after the series' last observed period (e.g. a monthly
            # price not yet published for the current month) have no period of
            # their own: carry the latest KNOWN metrics instead of leaving gaps.
            derived = [f'{commodity}{suffix}' for suffix in _DERIVED_SUFFIXES]
            df[derived] = df[derived].ffill()

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
        Combine freshly collected observations with the history in DuckDB.

        Transforms the 'long' format returned by the database into the 'wide'
        format needed by the pipeline. Derived metrics are ignored here, ensuring
        they are recomputed from scratch.

        The merge is done CELL by cell: a new value wins over the stored one,
        but a date where the new download has no value for a given series keeps
        the stored value. (A row-level merge would blank a series on every date
        it is missing from the new download.)
        """
        existing_long_df = db.load_prices()

        if existing_long_df.empty:
            return new_df

        existing_df = existing_long_df.pivot(
            index='date',
            columns='commodity',
            values='value'
        )
        existing_df.index = pd.to_datetime(existing_df.index)
        existing_df.columns.name = None

        new_idx = new_df.copy()
        new_idx['date'] = pd.to_datetime(new_idx['date'])
        new_idx = new_idx.set_index('date')

        combined = new_idx.combine_first(existing_df)
        combined.index.name = 'date'
        return combined.sort_index().reset_index()

    def _resolve_imputed(self, df: pd.DataFrame, new_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Wide boolean frame (index = date) telling which stored values are imputed.

        Flags from the new download take precedence; dates only present in the
        existing history keep the flag already stored in DuckDB.
        """
        dates = pd.DatetimeIndex(pd.to_datetime(df['date']))
        columns = [c for c in df.columns if c != 'date']
        resolved = np.zeros((len(dates), len(columns)), dtype=bool)

        # Flags already stored in DuckDB (for dates that are not in the new download).
        existing = db.load_prices()
        if not existing.empty and 'is_imputed' in existing.columns:
            old = existing.pivot(index='date', columns='commodity', values='is_imputed')
            old.index = pd.to_datetime(old.index)
            old = old.reindex(index=dates, columns=columns).astype('boolean')
            resolved = old.fillna(False).to_numpy(dtype=bool)

        # Flags from the new download take precedence wherever they are defined.
        if not new_mask.empty:
            new = new_mask.reindex(index=dates, columns=columns).astype('boolean')
            defined = new.notna().to_numpy()
            resolved = np.where(defined, new.fillna(False).to_numpy(dtype=bool), resolved)

        resolved = pd.DataFrame(resolved, index=dates, columns=columns)
        return resolved

    def save(self, df: pd.DataFrame, filepath: str, imputed: pd.DataFrame | None = None) -> str:
        """
        Persist the processed DataFrame to DuckDB.
        
        Unpivots the 'wide' DataFrame into 'long' schemas expected by `db.prices`
        and `db.signals` tables, then triggers the upserts. Returns the original
        filepath parameter to maintain interface compatibility.

        Only real observations are written: rows where a series has no value
        are skipped (never filled). ``imputed`` is an optional wide boolean
        frame (index = date) flagging values the collector gap-filled.
        """
        base_commodities = [
            col for col in df.columns
            if col != 'date'
            and not col.endswith(_DERIVED_SUFFIXES)
            and not col.endswith(IMPUTED_SUFFIX)
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
            prices_df = prices_df.dropna(subset=['value']).copy()

            if imputed is not None and not imputed.empty:
                flags = imputed.stack()
                flags.index.names = ['date', 'commodity']
                flags = flags.rename('is_imputed').reset_index()
                prices_df['date'] = pd.to_datetime(prices_df['date'])
                prices_df = prices_df.merge(flags, on=['date', 'commodity'], how='left')
                prices_df['is_imputed'] = prices_df['is_imputed'].fillna(False).astype(bool)
            else:
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

        On the first run after upgrading from the forward-filled storage format
        (data version < 2) the legacy price history is purged *after* the fresh
        download is already in memory, so a failed download can never leave the
        database empty.
        """
        logger.info("=" * 70)
        logger.info("DATA PIPELINE — PROCESSING AND SAVING (DUCKDB)")
        logger.info("=" * 70)

        values_df, new_imputed = self.split_imputed(df)
        df_clean = self.clean(values_df)

        if db.get_data_version() < db.CURRENT_DATA_VERSION:
            db.purge_forward_filled_history()

        df_merged  = self.merge_with_existing(df_clean, self.main_file)
        imputed    = self._resolve_imputed(df_merged, new_imputed)
        df_metrics = self.calculate_metrics(df_merged)
        filepath   = self.save(df_metrics, self.main_file, imputed=imputed)

        db.set_data_version(db.CURRENT_DATA_VERSION)
        return filepath

    def load_latest(self, aligned: bool = False) -> pd.DataFrame:
        """
        Load the most recently saved processed dataset from DuckDB.

        By default the frame is SPARSE: each series only has values on its own
        observation dates (NaN elsewhere). Pass ``aligned=True`` to get the
        price columns forward-filled onto the common calendar.

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
        existing_df.columns.name = None

        # Reconstruct metrics over the raw prices
        result = self.calculate_metrics(existing_df)
        if aligned:
            from utils.alignment import align_wide_frame
            base = [c for c in result.columns if c != 'date' and not c.endswith(_DERIVED_SUFFIXES)]
            result[base] = align_wide_frame(result[['date'] + base])[base]
        return result