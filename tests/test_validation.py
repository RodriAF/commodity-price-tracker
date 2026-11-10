"""
tests/test_validation.py
------------------------
pipeline.validation.validate(): which series pass, which are rejected, and
which only raise a warning.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import validation
from pipeline.validation import validate


def _series(dates, values=None, **extra):
    """DataFrame in the shape validate() expects: DatetimeIndex + 'value'."""
    values = values if values is not None else [100.0 + i for i in range(len(dates))]
    return pd.DataFrame({'value': values, **extra}, index=pd.DatetimeIndex(dates))


MONTHLY = {'corn': {'frequency': 'monthly'}}


class TestValidate:

    def test_clean_series_passes_without_warnings(self):
        df = _series(pd.date_range('2024-01-01', periods=12, freq='MS'))
        valid, report = validate({'corn': df}, MONTHLY)
        assert 'corn' in valid
        assert report.valid_series == ['corn']
        assert not report.rejected_series
        assert not report.warnings

    def test_empty_dataframe_is_rejected(self):
        empty = pd.DataFrame({'value': []}, index=pd.DatetimeIndex([]))
        valid, report = validate({'corn': empty}, MONTHLY)
        assert valid == {}
        assert 'corn' in report.rejected_series

    def test_missing_value_column_is_rejected(self):
        df = pd.DataFrame({'price': [1.0, 2.0]}, index=pd.date_range('2024-01-01', periods=2, freq='MS'))
        valid, report = validate({'corn': df}, MONTHLY)
        assert valid == {}
        assert 'corn' in report.rejected_series

    def test_one_bad_series_does_not_block_the_others(self):
        good = _series(pd.date_range('2024-01-01', periods=6, freq='MS'))
        bad = pd.DataFrame({'value': []}, index=pd.DatetimeIndex([]))
        valid, report = validate({'corn': good, 'wheat': bad}, MONTHLY)
        assert list(valid) == ['corn']
        assert list(report.rejected_series) == ['wheat']

    def test_large_gap_is_a_warning_not_a_rejection(self):
        # Jan, Feb, then June: a ~120-day hole in a monthly series
        df = _series(['2024-01-01', '2024-02-01', '2024-06-01'])
        valid, report = validate({'corn': df}, MONTHLY)
        assert 'corn' in valid
        assert any('gap' in w.lower() for w in report.warnings['corn'])

    def test_high_imputation_is_a_warning_not_a_rejection(self, monkeypatch):
        # The shipped threshold is 100%, so lower it to exercise the branch.
        monkeypatch.setattr(validation, 'HIGH_IMPUTATION_THRESHOLD_PCT', 50)
        dates = pd.date_range('2024-01-01', periods=10, freq='MS')
        df = _series(dates, is_imputed=[True] * 8 + [False] * 2)
        valid, report = validate({'corn': df}, MONTHLY)
        assert 'corn' in valid
        assert any('imputation' in w.lower() for w in report.warnings['corn'])
