"""
tests/test_calculations.py
--------------------
CommoditiesAnalytics produces the ratios, cost indices, z-scores, signals, and
regime that flow into signals.json and commodity_ratios.csv. Both files are
consumed by the dashboard on every page load. If calculate_all() silently
produces an empty or malformed result, the Analytics page shows nothing and
the Overview signals list is always empty — no exception, no log warning.
These tests pin the structural contracts.

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.calculations import CommoditiesAnalytics, CorrelationAnalysis


# Fixtures

MOCK_CATEGORIES = {
    'crop':         ['corn', 'wheat'],
    'energy_input': ['crude_oil', 'natural_gas'],
    'fertilizer':   ['urea'],
}

def make_analytics_df(n=60, seed=0):
    """
    Synthetic DataFrame with all commodity categories represented.
    n=60 rows ensures z-score rolling windows have enough data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start='2021-01-01', periods=n, freq='MS')
    return pd.DataFrame({
        'date':        dates,
        'corn':        np.clip(400 + np.cumsum(rng.normal(0, 5, n)), 50, None),
        'wheat':       np.clip(600 + np.cumsum(rng.normal(0, 4, n)), 50, None),
        'crude_oil':   np.clip(80  + np.cumsum(rng.normal(0, 2, n)), 5,  None),
        'natural_gas': np.clip(4   + np.cumsum(rng.normal(0, .1,n)), 0.5,None),
        'urea':        np.clip(300 + np.cumsum(rng.normal(0, 6, n)), 50, None),
    })


# 1. crop_profitability_ratios()

class TestCropProfitabilityRatios:
    """
    The profitability ratios are the core of the Analytics page. If this
    method returns an empty DataFrame, the entire 'Crop Profitability' section
    is silently hidden behind a conditional block — the page renders but looks
    broken.
    """

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_returns_nonempty_dataframe(self, _):
        df = make_analytics_df()
        analytics = CommoditiesAnalytics(df)
        result = analytics.crop_profitability_ratios()
        assert not result.empty, "crop_profitability_ratios() must not return an empty DataFrame"

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_column_naming_convention(self, _):
        """
        The dashboard uses '{crop}_to_{input}' column names to build the
        display labels. If the naming convention changes here, every label on the
        Analytics page shows the raw column name instead of the formatted version.
        """
        df = make_analytics_df()
        analytics = CommoditiesAnalytics(df)
        result = analytics.crop_profitability_ratios()

        for col in result.columns:
            assert '_to_' in col, (
                f"Ratio column '{col}' does not follow the '{{crop}}_to_{{input}}' convention"
            )

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_no_infinite_values(self, _):
        """
        Ratios are divisions. If an input commodity has a zero price in the
        data (e.g., due to a bad FRED observation), the ratio becomes inf. Inf
        propagates into the z-score, making that signal permanently 'extreme'
        and masking real anomalies.
        """
        df = make_analytics_df()
        analytics = CommoditiesAnalytics(df)
        result = analytics.crop_profitability_ratios()

        has_inf = np.isinf(result.values).any()
        assert not has_inf, "Profitability ratios must not contain infinite values"


# 2. cost_indices()

class TestCostIndices:
    """
    The cost index is normalized to 100 = historical average and is the
    only number on the Analytics page that is explicitly described as 'meaningful
    in absolute terms'. If the normalization is wrong, the regime classification
    (high/normal/low) fires on wrong thresholds.
    """

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_index_mean_is_approximately_100(self, _):
        """
        The normalization formula is (value / mean) * 100. Over the full
        history, the mean of the normalized series should be ≈100 by construction.
        If it's significantly off, the normalization has a bug (e.g., using
        the wrong base, or computing mean after normalization instead of before).
        """
        df = make_analytics_df(n=60)
        analytics = CommoditiesAnalytics(df)
        result = analytics.cost_indices()

        for col in result.columns:
            mean_val = result[col].dropna().mean()
            assert abs(mean_val - 100) < 5, (
                f"Cost index '{col}' has mean {mean_val:.1f}, expected ≈100. "
                "Normalization may be incorrect."
            )

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_expected_index_columns_present(self, _):
        """
        The regime classifier looks for 'energy_input_cost_index' and
        'fertilizer_cost_index' by exact name. If the column names change,
        regime always returns {} and the dashboard regime cards show nothing.
        """
        df = make_analytics_df()
        analytics = CommoditiesAnalytics(df)
        result = analytics.cost_indices()

        assert 'energy_input_cost_index' in result.columns, (
            "Missing 'energy_input_cost_index' column"
        )
        assert 'fertilizer_cost_index' in result.columns, (
            "Missing 'fertilizer_cost_index' column"
        )


# 3. calculate_zscores()

class TestCalculateZscores:
    """
    Z-scores are the central output of the analytics module — they drive
    signals, the ratio explorer, and the dashboard color coding. Wrong z-scores
    produce wrong signals. These tests verify the statistical properties.
    """

    def setup_method(self):
        df = make_analytics_df(n=80)
        with patch('pipeline.calculations.ConfigLoader.get_categories',
                   return_value=MOCK_CATEGORIES):
            self.analytics = CommoditiesAnalytics(df)
            ratios = self.analytics.crop_profitability_ratios()
            self.zscores = self.analytics.calculate_zscores(ratios, window=20)

    def test_zscore_columns_have_suffix(self):
        """Dashboard expects '{col}_zscore' naming to look up z-score columns."""
        for col in self.zscores.columns:
            assert col.endswith('_zscore'), (
                f"Z-score column '{col}' does not end with '_zscore'"
            )

    def test_zscores_are_roughly_standardized(self):
        """
        A rolling z-score should have std ≈ 1 in steady state. If std is
        much larger, the threshold logic (|z| > 2 = extreme) fires too often.
        If std is much smaller, it never fires. This test uses a loose bound
        because rolling windows at the start of the series have fewer observations.
        """
        for col in self.zscores.columns:
            std = self.zscores[col].dropna().std()
            assert 0.5 < std < 3.0, (
                f"Z-score column '{col}' has std={std:.2f}, expected roughly 1.0. "
                "Rolling window may be too small or the formula is incorrect."
            )


# 4. detect_market_regime()

class TestDetectMarketRegime:
    """
    The regime dict is serialized to signals.json and read by both the
    Overview and Analytics pages. If it returns unexpected keys or values, the
    CSS class lookup in the dashboard fails silently (wrong color, no accent bar).
    """

    def setup_method(self):
        df = make_analytics_df(n=60)
        with patch('pipeline.calculations.ConfigLoader.get_categories',
                   return_value=MOCK_CATEGORIES):
            self.analytics = CommoditiesAnalytics(df)
            self.indices = self.analytics.cost_indices()

    def test_regime_keys_are_expected(self):
        """Only 'energy' and 'fertilizer' keys should ever appear."""
        regime = self.analytics.detect_market_regime(self.indices)
        valid_keys = {'energy', 'fertilizer'}
        unexpected = set(regime.keys()) - valid_keys
        assert not unexpected, f"Unexpected regime keys: {unexpected}"

    def test_regime_values_are_valid_strings(self):
        """
        The Analytics page uses regime values as CSS class names. An
        unexpected value means the card renders without its color accent.
        """
        valid_values = {'high_cost', 'low_cost', 'normal', 'expensive', 'cheap'}
        regime = self.analytics.detect_market_regime(self.indices)
        for key, val in regime.items():
            assert val in valid_values, (
                f"regime['{key}'] = '{val}' is not a valid regime value. "
                f"Expected one of: {valid_values}"
            )

    def test_empty_indices_returns_empty_dict(self):
        """
        run_daily.py calls detect_market_regime() unconditionally. If it
        raises on an empty DataFrame instead of returning {}, the entire analytics
        step crashes and signals.json is never written.
        """
        regime = self.analytics.detect_market_regime(pd.DataFrame())
        assert regime == {}, "Empty input must return empty dict, not raise"


# 5. CorrelationAnalysis

class TestCorrelationAnalysis:
    """
    key_correlations() output is logged in run_daily.py and used for
    interpretability context. If it crashes or returns malformed data, the
    whole daily run fails at the logging step — after analytics has already
    completed successfully.
    """

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_returns_dataframe_with_expected_columns(self, _):
        df = make_analytics_df()
        corr = CorrelationAnalysis(df)
        result = corr.key_correlations()

        for col in ['crop', 'input', 'correlation']:
            assert col in result.columns, (
                f"key_correlations() missing expected column: '{col}'"
            )

    @patch('pipeline.calculations.ConfigLoader.get_categories', return_value=MOCK_CATEGORIES)
    def test_correlations_bounded(self, _):
        """
        Pearson r is mathematically bounded to [-1, 1]. A value outside
        this range indicates a numerical error (e.g., division by zero std,
        or ffill producing a constant series that breaks the formula).
        """
        df = make_analytics_df()
        corr = CorrelationAnalysis(df)
        result = corr.key_correlations()

        if not result.empty:
            assert (result['correlation'] >= -1.01).all(), "Correlation below -1"
            assert (result['correlation'] <=  1.01).all(), "Correlation above +1"
