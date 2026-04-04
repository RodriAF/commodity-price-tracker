"""
tests/test_pipeline.py
--------------------
DataPipeline.calculate_metrics() is the most dangerous function in the project.
It runs silently — if the frequency-aware period mapping produces NaN for a
commodity, the rest of the pipeline (analytics, forecasting, dashboard) consumes
corrupt data without any error. These tests pin down the contracts that must hold
for the pipeline to be trustworthy.

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.data_pipeline import DataPipeline


# Fixtures — reusable synthetic DataFrames that mimic real FRED output

def make_monthly_df(n=36, commodity='corn', start='2021-01-01', seed=42):
    """
    36 months of synthetic daily data for a monthly commodity (like corn).
    Using daily index because that's what the pipeline receives before resampling.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n * 30, freq='D')
    prices = 400 + np.cumsum(rng.normal(0, 2, len(dates)))
    prices = np.clip(prices, 1, None)  # prices must be positive
    return pd.DataFrame({'date': dates, commodity: prices})


def make_daily_df(n=252, commodity='crude_oil', start='2022-01-01', seed=7):
    """252 trading days of synthetic data for a daily commodity (like crude oil)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq='D')
    prices = 80 + np.cumsum(rng.normal(0, 1.5, n))
    prices = np.clip(prices, 1, None)
    return pd.DataFrame({'date': dates, commodity: prices})


def make_multi_commodity_df(n_months=36, start='2021-01-01'):
    """Multiple commodities at different frequencies merged into one DataFrame."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start=start, periods=n_months * 30, freq='D')
    return pd.DataFrame({
        'date':        dates,
        'corn':        np.clip(400 + np.cumsum(rng.normal(0, 2,   len(dates))), 1, None),
        'crude_oil':   np.clip(80  + np.cumsum(rng.normal(0, 1.5, len(dates))), 1, None),
        'urea':        np.clip(300 + np.cumsum(rng.normal(0, 3,   len(dates))), 1, None),
    })


# 1. clean()

class TestClean:
    """
    clean() is the entry point. If it silently drops rows, mis-casts dates,
    or fails to deduplicate, every downstream calculation is wrong. These tests
    establish the invariants that must always hold after cleaning.
    """

    def test_date_column_is_datetime(self):
        """
        The pipeline does date arithmetic everywhere. If 'date' stays as
        a string (common when loading from CSV), pd.Timedelta comparisons silently
        return NaN instead of raising — a classic silent corruption bug.
        """
        df = make_monthly_df()
        df['date'] = df['date'].astype(str)  # simulate CSV load
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.clean(df)
        assert result['date'].dtype == 'datetime64[ns]', (
            "date column must be datetime64 after clean()"
        )

    def test_duplicates_are_removed(self):
        """
        FRED sometimes returns overlapping date ranges when you extend history.
        The merge_with_existing() step can also produce duplicates. Duplicates in
        the time series distort rolling statistics (z-scores, MA) by double-weighting
        specific dates.
        """
        df = make_monthly_df(n=12)
        df_duped = pd.concat([df, df.iloc[:5]]).reset_index(drop=True)
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.clean(df_duped)
        assert result['date'].nunique() == len(result), (
            "No duplicate dates should remain after clean()"
        )

    def test_sorted_ascending_by_date(self):
        """
        Rolling window functions (MA, z-score) are order-dependent. If the
        DataFrame is not sorted ascending, the 'last' value in a window is actually
        an earlier date — the rolling mean looks into the future.
        """
        df = make_monthly_df(n=24)
        df = df.sample(frac=1, random_state=1)  # shuffle
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.clean(df)
        assert result['date'].is_monotonic_increasing, (
            "DataFrame must be sorted ascending by date after clean()"
        )

    def test_no_negative_prices_after_ffill(self):
        """
        FRED occasionally has gaps (NaN) for commodities with irregular
        reporting. ffill() propagates the last known value, which is always
        non-negative for prices. If a negative price slips through, log-return
        calculations in the forecaster produce NaN or complex numbers.
        """
        df = make_monthly_df(n=24)
        df.loc[5:10, 'corn'] = np.nan  # simulate gap
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.clean(df)
        numeric = result.select_dtypes(include='number')
        assert (numeric >= 0).all().all(), (
            "No negative values should remain after ffill"
        )


# 2. calculate_metrics()

class TestCalculateMetrics:
    """
    This is the highest-risk function in the project. It does frequency
    resampling, period mapping, and column expansion all in one pass. A bug here
    produces NaN columns that flow into z-score signals, analytics ratios, and
    forecasting features — all silently. These tests establish the structural
    guarantees that must hold regardless of input commodity or frequency.
    """

    @patch('pipeline.data_pipeline.ConfigLoader.get_commodity_frequency')
    @patch('pipeline.data_pipeline.ConfigLoader.get_metric_config')
    def test_expected_columns_created_monthly(self, mock_metric, mock_freq):
        """
        If the PeriodIndex mapping silently fails, the derived columns
        (_change_pct, _ma, _zscore, _signal) are simply absent. The dashboard
        then shows '—' everywhere and no signals fire — a silent failure that
        looks like 'no interesting data' rather than a bug.
        """
        mock_freq.return_value = 'monthly'
        mock_metric.return_value = {
            'change_periods': 1, 'ma_window': 3, 'z_window': 12
        }

        df = make_monthly_df(n=36)
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.calculate_metrics(df)

        for suffix in ['_change_pct', '_ma', '_zscore', '_signal']:
            col = f'corn{suffix}'
            assert col in result.columns, (
                f"Expected column '{col}' to be created by calculate_metrics()"
            )

    @patch('pipeline.data_pipeline.ConfigLoader.get_commodity_frequency')
    @patch('pipeline.data_pipeline.ConfigLoader.get_metric_config')
    def test_zscore_column_not_all_nan(self, mock_metric, mock_freq):
        """
        The z-score is the core signal of the whole system. If it's all NaN
        (e.g., because rolling std is 0 or the period mapping produced no matches),
        the analytics page shows nothing and signals.json is always empty. This
        test is the most important in the suite — it validates the end-to-end
        frequency pipeline produces usable output.
        """
        mock_freq.return_value = 'monthly'
        mock_metric.return_value = {
            'change_periods': 1, 'ma_window': 3, 'z_window': 12
        }

        df = make_monthly_df(n=36)
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.calculate_metrics(df)

        z_col = result['corn_zscore']
        non_nan_ratio = z_col.notna().mean()
        assert non_nan_ratio > 0.5, (
            f"corn_zscore is {non_nan_ratio:.1%} non-null — too many NaNs. "
            "Likely a frequency mapping failure."
        )

    @patch('pipeline.data_pipeline.ConfigLoader.get_commodity_frequency')
    @patch('pipeline.data_pipeline.ConfigLoader.get_metric_config')
    def test_signal_column_contains_valid_values(self, mock_metric, mock_freq):
        """
        The _signal column is consumed directly by the Overview dashboard
        as categorical text. If it contains unexpected values (e.g., NaN cast
        to the string 'nan'), the color-coding logic silently falls through to
        the default case, making all signals appear 'normal'.
        """
        mock_freq.return_value = 'monthly'
        mock_metric.return_value = {
            'change_periods': 1, 'ma_window': 3, 'z_window': 12
        }

        df = make_monthly_df(n=36)
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.calculate_metrics(df)

        valid_signals = {'normal', 'notable', 'extreme', 'no_data'}
        actual = set(result['corn_signal'].dropna().unique())
        unexpected = actual - valid_signals
        assert not unexpected, (
            f"Signal column contains unexpected values: {unexpected}"
        )

    @patch('pipeline.data_pipeline.ConfigLoader.get_commodity_frequency')
    @patch('pipeline.data_pipeline.ConfigLoader.get_metric_config')
    def test_row_count_preserved(self, mock_metric, mock_freq):
        """
        The period mapping expands resampled metrics back to the original
        daily index. If this expansion drops rows (e.g., off-by-one in date
        alignment), the DataFrame shape changes and pd.concat() in run_daily.py
        produces misaligned columns between the main data and the ratios CSV.
        """
        mock_freq.return_value = 'monthly'
        mock_metric.return_value = {
            'change_periods': 1, 'ma_window': 3, 'z_window': 12
        }

        df = make_monthly_df(n=24)
        original_len = len(df)
        pipeline = DataPipeline.__new__(DataPipeline)
        result = pipeline.calculate_metrics(df)

        assert len(result) == original_len, (
            f"calculate_metrics() changed row count: {original_len} → {len(result)}"
        )
