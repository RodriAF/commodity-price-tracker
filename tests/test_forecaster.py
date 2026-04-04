"""
tests/test_forecaster.py
--------------------
The forecaster is the most complex module and the one most likely to produce
plausible-looking but wrong output. A model that overfits, produces infinite
values, or passes metrics computed on training data instead of test data will
still "work" — it will write JSON, the dashboard will render it, and no error
will be raised. These tests pin the contracts that protect against silent
metric corruption and ensemble failure.

Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.predictions import CommoditiesForecaster


# Fixtures

def make_forecaster(n=48, frequency='monthly', commodity='corn', seed=42):
    """
    Minimal CommoditiesForecaster with synthetic monthly data.
    48 months > the minimum required (24 + test_size=6 + horizon=3).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start='2020-01-01', periods=n, freq='MS')
    prices = np.clip(400 + np.cumsum(rng.normal(0, 5, n)), 50, None)
    df = pd.DataFrame({'date': dates, commodity: prices})
    return CommoditiesForecaster(
        data=df, commodity=commodity, frequency=frequency, horizon=3
    )


# 1. calculate_metrics()

class TestCalculateMetrics:
    """
    calculate_metrics() is used to rank models and build the ensemble.
    If it silently accepts wrong-length arrays, computes MAPE on training data,
    or produces NaN, the ensemble weights are meaningless — you could end up
    selecting the worst model as the best one.
    """

    def setup_method(self):
        self.fc = make_forecaster()

    def test_all_metric_keys_present(self):
        """
        The ensemble uses 'mape' and 'mase' for ranking. The dashboard
        displays 'mae'. If any key is missing, the ensemble falls back to wrong
        defaults and the dashboard raises a KeyError that only appears at render
        time, not during the pipeline run.
        """
        actual   = np.array([100, 110, 105, 108, 103, 107])
        predicted = np.array([102, 108, 106, 110, 101, 105])
        metrics = self.fc.calculate_metrics(actual, predicted)

        required_keys = {'mae', 'mape', 'rmse', 'smape', 'mase', 'mbe', 'directional_accuracy'}
        missing = required_keys - set(metrics.keys())
        assert not missing, f"calculate_metrics() is missing keys: {missing}"

    def test_mape_is_non_negative(self):
        """
        MAPE is used as an exponential weight in the ensemble (np.exp(-mape)).
        A negative MAPE would make that model's weight explode to infinity, making
        the ensemble a single-model prediction dressed up as an ensemble.
        """
        actual    = np.array([100.0, 200.0, 150.0, 180.0])
        predicted = np.array([110.0, 190.0, 160.0, 170.0])
        metrics = self.fc.calculate_metrics(actual, predicted)
        assert metrics['mape'] >= 0, "MAPE must be non-negative"

    def test_perfect_forecast_gives_zero_mae(self):
        """
        Sanity check that the metric formula is correct. If MAE is non-zero
        for a perfect forecast, the function has a systematic bias (e.g., computing
        on the wrong arrays, off-by-one indexing, or wrong axis).
        """
        values = np.array([100.0, 120.0, 115.0, 130.0])
        metrics = self.fc.calculate_metrics(values, values)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-8), (
            "Perfect forecast should produce MAE = 0"
        )

    def test_mismatched_lengths_raises(self):
        """
        If actual and predicted have different lengths (e.g., because a model
        returned fewer periods than the test set), numpy broadcasts silently in
        some configurations. We want a hard failure here, not a wrong number.
        """
        actual    = np.array([100.0, 110.0, 105.0])
        predicted = np.array([102.0, 108.0])  # one less
        with pytest.raises((ValueError, IndexError)):
            self.fc.calculate_metrics(actual, predicted)

    def test_mase_is_finite_for_non_constant_series(self):
        """
        MASE divides by the mean absolute difference of the actual series.
        For a non-constant series this should always be finite. If it's NaN or
        inf, assign_confidence() falls back to MAPE-based confidence, but the
        NaN propagates into the ensemble ranking key function — sorted() raises
        a TypeError when comparing NaN to float in Python 3.
        """
        actual    = np.array([100.0, 110.0, 108.0, 115.0, 112.0, 120.0])
        predicted = np.array([102.0, 109.0, 110.0, 113.0, 114.0, 118.0])
        metrics = self.fc.calculate_metrics(actual, predicted)
        if metrics['mase'] is not None:
            assert np.isfinite(metrics['mase']), "MASE must be finite for non-constant series"


# 2. validate_data()

class TestValidateData:
    """
    validate_data() is the gate that prevents models from running on
    insufficient data. If it passes data that's too short, auto_arima and
    GARCH crash with cryptic internal errors that are hard to debug from logs.
    If it's too strict, you get no forecasts for valid commodities.
    """

    def test_sufficient_data_passes(self):
        """48 months is clearly enough for monthly frequency."""
        fc = make_forecaster(n=48)
        is_valid, msg = fc.validate_data()
        assert is_valid, f"48-month series should be valid, got: {msg}"

    def test_insufficient_data_fails(self):
        """
        The minimum for monthly is 24 + test_size(6) + horizon(3) = 33.
        A 20-row series must be rejected. Without this gate, auto_arima raises
        a MLE convergence error that gets caught by the broad except in
        run_daily.py, logged as a generic 'processing_error', and silently skipped.
        """
        fc = make_forecaster(n=20)
        is_valid, msg = fc.validate_data()
        assert not is_valid, "20-month series should fail validation"
        assert len(msg) > 0, "Failure message must not be empty"


# 3. create_ensemble()

class TestCreateEnsemble:
    """
    create_ensemble() is the final output of the forecasting system.
    Its output is written to JSON and consumed by the dashboard without further
    validation. These tests ensure the ensemble's structural guarantees hold —
    correct length, finite values, and graceful failure when no models succeed.
    """

    def setup_method(self):
        self.fc = make_forecaster(n=48, seed=0)

    def _make_valid_result(self, predictions, mape=5.0):
        """Helper: minimal valid model result dict."""
        return {
            'predictions': predictions,
            'metrics': {'mape': mape, 'mae': 1.0, 'mase': 0.5},
            'confidence': 'medium',
            'method': 'TestModel'
        }

    def test_ensemble_prediction_length_matches_horizon(self):
        """
        The dashboard iterates over ensemble['predictions'] with a fixed
        range(horizon). If the list is shorter than horizon, it raises an IndexError
        at render time — after the pipeline has already exited successfully.
        """
        results = {
            'model_a': self._make_valid_result([410.0, 415.0, 420.0], mape=4.0),
            'model_b': self._make_valid_result([408.0, 412.0, 418.0], mape=6.0),
            'model_c': self._make_valid_result([412.0, 416.0, 422.0], mape=5.5),
        }
        ensemble = self.fc.create_ensemble(results)
        assert ensemble is not None
        assert len(ensemble['predictions']) == self.fc.horizon, (
            f"Ensemble predictions length {len(ensemble['predictions'])} "
            f"!= horizon {self.fc.horizon}"
        )

    def test_ensemble_predictions_are_finite(self):
        """
        The ensemble uses np.exp(-errors) for weighting. If any model has
        MAPE=0 exactly (overfit), exp(0)=1 and the weight is fine. But if MAPE
        is negative or NaN (corrupted metrics), exp() produces inf or NaN, which
        propagates into the weighted sum. The dashboard would render 'NaN' or
        'Infinity' as the forecast price.
        """
        results = {
            'model_a': self._make_valid_result([410.0, 415.0, 420.0], mape=4.0),
            'model_b': self._make_valid_result([408.0, 412.0, 418.0], mape=6.0),
        }
        ensemble = self.fc.create_ensemble(results)
        assert ensemble is not None
        for i, p in enumerate(ensemble['predictions']):
            assert np.isfinite(p), f"Ensemble prediction[{i}] = {p} is not finite"

    def test_ensemble_returns_none_when_all_models_fail(self):
        """
        If every model errors out (e.g., insufficient variance, convergence
        failure), create_ensemble() must return None explicitly. The caller in
        run_daily.py checks `if ensemble` — if it returns an empty dict instead
        of None, that check passes and the code tries to access
        ensemble['predictions'], raising a KeyError.
        """
        results = {
            'model_a': {'error': 'convergence failure', 'method': 'ARIMA'},
            'model_b': {'error': 'insufficient data',   'method': 'SARIMA'},
        }
        ensemble = self.fc.create_ensemble(results)
        assert ensemble is None, (
            "create_ensemble() must return None when no models have valid predictions"
        )

    def test_ensemble_rejects_extreme_predictions(self):
        """
        The sanity filter in create_ensemble() rejects models whose first
        prediction deviates > 40% from the last known price. This test verifies
        the filter actually works — without it, a diverged ML model (e.g., Ridge
        on a short series) can dominate the ensemble with a prediction of $0 or
        $10,000 for a commodity normally priced at $400.
        """
        last_price = float(self.fc.series.iloc[-1])
        extreme    = last_price * 3.0  # 200% above — must be filtered

        results = {
            'model_good':    self._make_valid_result(
                [last_price * 1.01] * 3, mape=3.0
            ),
            'model_extreme': self._make_valid_result(
                [extreme] * 3, mape=0.1  # best MAPE but extreme prediction
            ),
        }
        ensemble = self.fc.create_ensemble(results)
        assert ensemble is not None
        # The extreme model must not dominate — its prediction is 3x last price
        for pred in ensemble['predictions']:
            assert pred < last_price * 2.0, (
                f"Extreme model leaked into ensemble: predicted {pred:.2f} "
                f"vs last known {last_price:.2f}"
            )


# 4. assign_confidence()

class TestAssignConfidence:
    """
    assign_confidence() is displayed prominently on the Forecasting page
    as HIGH / MEDIUM / LOW. If the thresholds are wrong or the fallback path
    (when mase is None) produces unexpected output, hiring managers running the
    dashboard will see inconsistent confidence labels.
    """

    def setup_method(self):
        self.fc = make_forecaster(n=48)

    def test_valid_confidence_levels_only(self):
        """Output must always be one of the three expected strings."""
        valid = {'high', 'medium', 'low'}
        for mase in [0.3, 0.8, 0.9, 1.2, 2.0]:
            result = self.fc.assign_confidence({'mase': mase, 'mape': 10.0})
            assert result in valid, (
                f"assign_confidence() returned '{result}' for mase={mase}, "
                f"expected one of {valid}"
            )

    def test_mase_thresholds_correct(self):
        """
        The docstring says HIGH < 0.8, MEDIUM < 1.2. These are the
        thresholds communicated in the dashboard methodology section. If the
        implementation drifts from the documentation, the displayed confidence
        is misleading.
        """
        assert self.fc.assign_confidence({'mase': 0.5,  'mape': 5.0})  == 'high'
        assert self.fc.assign_confidence({'mase': 0.79, 'mape': 5.0})  == 'high'
        assert self.fc.assign_confidence({'mase': 0.8,  'mape': 5.0})  == 'medium'
        assert self.fc.assign_confidence({'mase': 1.19, 'mape': 5.0})  == 'medium'
        assert self.fc.assign_confidence({'mase': 1.2,  'mape': 5.0})  == 'low'
        assert self.fc.assign_confidence({'mase': 2.0,  'mape': 5.0})  == 'low'

    def test_none_mase_falls_back_gracefully(self):
        """
        MASE is None when naive_error ≈ 0 (constant series) or when the
        denominator is too small. The fallback must still return a valid string —
        not raise AttributeError or return None, which would crash the JSON
        serialization in run_daily.py.
        """
        result = self.fc.assign_confidence({'mase': None, 'mape': 15.0})
        assert result in {'high', 'medium', 'low'}, (
            f"Fallback with mase=None returned unexpected value: '{result}'"
        )
