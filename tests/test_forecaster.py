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
    48 months > the minimum required (24 min points + horizon 3 + 2 fold steps).
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
        The minimum for monthly with horizon 3 is 24 + 3 + 2*1 = 29 (three
        rolling-origin folds must fit).
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



# =============================================================================
# Native series, rolling-origin backtest, naive baseline, intervals
# =============================================================================

def make_series_df(frequency='monthly', n=96, seed=1, vol=0.04, drift=0.001, commodity='x'):
    """Random-walk prices on the series' own calendar."""
    rng = np.random.default_rng(seed)
    grid = {'daily': 'B', 'weekly': 'W-MON', 'monthly': 'MS', 'quarterly': 'QS'}[frequency]
    idx = pd.date_range('2015-01-01', periods=n, freq=grid)
    prices = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.DataFrame({'date': idx, commodity: prices})


class TestNativeSeries:
    """
    Each commodity is forecast on its OWN calendar. If the forecaster is handed a
    sparse wide frame (NaN where the series has no observation), the NaNs must
    be dropped — never forward-filled into fake "unchanged" points.
    """

    def test_sparse_wide_frame_yields_only_real_observations(self):
        monthly = make_series_df('monthly', 48)
        busy = pd.DataFrame({'date': pd.bdate_range('2015-01-01', monthly['date'].iloc[-1])})
        wide = busy.merge(monthly, on='date', how='outer').sort_values('date')   # monthly NaN on most days
        fc = CommoditiesForecaster(wide, 'x', 'monthly', horizon=3)
        assert len(fc.native) == 48
        assert len(fc.series) == 48, "no forward-filled points may be added"
        assert fc.n_gap_filled == 0

    def test_last_observation_date_is_the_last_real_one(self):
        monthly = make_series_df('monthly', 48)
        extra = pd.DataFrame({'date': pd.bdate_range(monthly['date'].iloc[-1] + pd.Timedelta(days=1), periods=20)})
        wide = monthly.merge(extra, on='date', how='outer').sort_values('date')
        fc = CommoditiesForecaster(wide, 'x', 'monthly', horizon=3)
        assert fc.last_observation_date == monthly['date'].iloc[-1]

    def test_future_dates_follow_the_series_own_calendar(self):
        fc = CommoditiesForecaster(make_series_df('monthly', 48), 'x', 'monthly', horizon=3)
        future = pd.to_datetime(fc.future_dates())
        assert len(future) == 3
        assert future[0] == fc.series.index[-1] + pd.offsets.MonthBegin(1)
        assert all(d.day == 1 for d in future)

    def test_interior_holes_are_interpolated_and_counted(self):
        df = make_series_df('daily', 200)
        df = df.drop(df.index[[50, 51, 120]])          # market holidays
        fc = CommoditiesForecaster(df, 'x', 'daily', horizon=5)
        assert fc.n_gap_filled == 3
        assert fc.series.notna().all() and len(fc.series) == 200

    def test_never_extrapolates_beyond_last_observation(self):
        fc = CommoditiesForecaster(make_series_df('monthly', 48), 'x', 'monthly', horizon=3)
        assert fc.series.index[-1] == fc.native.index[-1]


class TestHorizonPerFrequency:

    @pytest.mark.parametrize('frequency,expected', [
        ('daily', 20), ('weekly', 12), ('monthly', 6), ('quarterly', 4),
    ])
    def test_default_horizon_comes_from_config(self, frequency, expected):
        fc = CommoditiesForecaster(make_series_df(frequency, 300), 'x', frequency)
        assert fc.horizon == expected

    def test_explicit_horizon_wins(self):
        fc = CommoditiesForecaster(make_series_df('monthly', 60), 'x', 'monthly', horizon=2)
        assert fc.horizon == 2


class TestRollingOrigin:

    def test_origins_are_increasing_and_last_fold_ends_at_series_end(self):
        fc = make_forecaster(n=60)
        origins = fc._origins()
        assert origins == sorted(origins) and len(origins) >= fc.MIN_FOLDS
        assert origins[-1] + fc.horizon == len(fc.series), "last fold must end on the last observation"

    def test_every_fold_has_enough_training_data(self):
        fc = make_forecaster(n=60)
        assert min(fc._origins()) >= fc.config['min_points']

    def test_folds_never_exceed_configured_maximum(self):
        fc = CommoditiesForecaster(make_series_df('monthly', 120), 'x', 'monthly', horizon=3)
        assert len(fc._origins()) <= fc.config['max_folds']

    def test_validate_requires_room_for_three_folds(self):
        fc = CommoditiesForecaster(make_series_df('monthly', 28), 'x', 'monthly', horizon=3)
        ok, msg = fc.validate_data()
        assert not ok and 'minimum required' in msg


class TestNaiveBaseline:
    """The baseline every model is judged against must be exactly 'the last value'."""

    def setup_method(self):
        self.fc = CommoditiesForecaster(make_series_df('monthly', 72), 'x', 'monthly', horizon=3)
        self.results = self.fc.forecast_all_models()

    def test_naive_is_always_in_the_pool(self):
        assert 'naive' in self.results and 'predictions' in self.results['naive']

    def test_naive_forecast_equals_last_value(self):
        last = float(self.fc.series.iloc[-1])
        assert np.allclose(self.results['naive']['predictions'], last)

    def test_naive_has_rmae_of_exactly_one(self):
        assert self.results['naive']['metrics']['rmae'] == pytest.approx(1.0)

    def test_every_scored_model_has_relative_error(self):
        for key, res in self.results.items():
            if 'metrics' in res and res['metrics'].get('mae') is not None:
                assert 'rmae' in res['metrics'], f"{key} is missing rmae"

    def test_pool_is_chosen_per_series_type(self):
        monthly = set(self.results)
        assert {'naive', 'drift', 'ets', 'theta', 'arima', 'ets_seasonal', 'sarima', 'seasonal_naive'} <= monthly
        assert 'garch' not in monthly, "GARCH needs daily/weekly data"
        assert 'random_forest' not in monthly, "tree models need long histories"

    def test_garch_only_for_long_high_frequency_series(self):
        weekly = CommoditiesForecaster(make_series_df('weekly', 300), 'x', 'weekly', horizon=4)
        assert 'garch' in weekly._get_model_pool()
        daily_short = CommoditiesForecaster(make_series_df('daily', 150), 'x', 'daily', horizon=5)
        assert 'garch' not in daily_short._get_model_pool()


class TestAssignConfidenceRelativeToNaive:

    def setup_method(self):
        self.fc = make_forecaster()

    @pytest.mark.parametrize('rmae,label', [(0.60, 'high'), (0.84, 'high'), (0.90, 'medium'),
                                            (0.99, 'medium'), (1.00, 'low'), (1.40, 'low')])
    def test_rmae_thresholds(self, rmae, label):
        assert self.fc.assign_confidence({'rmae': rmae, 'mase': 0.1}) == label

    def test_rmae_takes_precedence_over_mase(self):
        """A model with a great MASE that still loses to naive must not be 'high' confidence."""
        assert self.fc.assign_confidence({'rmae': 1.3, 'mase': 0.2}) == 'low'


class TestNoTargetLeakage:
    """
    The ML models predict the change from t to t+1. A feature that already
    contains that change makes training errors look superb and inference flat —
    the earlier implementation had exactly this bug (diff_1 == target).
    """

    def test_features_at_t_do_not_depend_on_later_values(self):
        fc = CommoditiesForecaster(make_series_df('daily', 200), 'x', 'daily', horizon=5)
        y = np.log(fc.series)
        t = 120
        base = fc._features(y.iloc[: t + 1]).iloc[-1]

        tampered = y.copy()
        tampered.iloc[t + 1:] += 5.0                      # change the future
        after = fc._features(tampered.iloc[: t + 1]).iloc[-1]
        # features at t are computed from y[:t+1] only; also compare against the full-series row
        full_row = fc._features(tampered).iloc[t]
        pd.testing.assert_series_equal(base, after)
        pd.testing.assert_series_equal(base, full_row, check_names=False)

    def test_no_feature_equals_the_target(self):
        fc = CommoditiesForecaster(make_series_df('daily', 200), 'x', 'daily', horizon=5)
        y = np.log(fc.series)
        feats = fc._features(y)
        target = y.diff().shift(-1)
        for col in feats.columns:
            aligned = pd.concat([feats[col], target], axis=1).dropna()
            assert not np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1]), f"{col} leaks the target"

    def test_ml_forecast_is_not_flat(self):
        """Recursive ML forecasts must move (calendar and lags advance), not stay on the last value."""
        fc = CommoditiesForecaster(make_series_df('weekly', 300, drift=0.004), 'x', 'weekly', horizon=6)
        train = fc._fwd(fc.series)
        out = fc._m_gradient_boosting(train, 6)
        assert np.ptp(out) > 0


class TestPredictionIntervals:

    def setup_method(self):
        self.fc = CommoditiesForecaster(make_series_df('monthly', 96), 'x', 'monthly', horizon=6)

    def test_cone_never_narrows_with_horizon(self):
        sigma, dof = self.fc._cone(self.fc.series, 6)
        assert np.all(np.diff(sigma) >= -1e-12) and np.all(sigma > 0)

    def test_intervals_nest_and_bracket_the_forecast(self):
        pred = np.full(6, 100.0)
        cone = self.fc._cone(self.fc.series, 6)
        lo80, hi80 = self.fc._interval(pred, cone, 0.80)
        lo95, hi95 = self.fc._interval(pred, cone, 0.95)
        assert np.all(lo95 < lo80) and np.all(lo80 < pred) and np.all(pred < hi80) and np.all(hi80 < hi95)

    def test_log_space_intervals_are_positive(self):
        pred = np.full(6, 5.0)
        lo, hi = self.fc._interval(pred, self.fc._cone(self.fc.series, 6), 0.95)
        assert np.all(lo > 0)

    def test_short_history_gives_wider_intervals_than_long(self):
        """Student-t degrees of freedom: less data -> more honest (wider) intervals."""
        long_fc = CommoditiesForecaster(make_series_df('monthly', 240, seed=3), 'x', 'monthly', horizon=6)
        short_fc = CommoditiesForecaster(make_series_df('monthly', 30, seed=3), 'x', 'monthly', horizon=6)
        sig_l, dof_l = long_fc._cone(long_fc.series, 6)
        sig_s, dof_s = short_fc._cone(short_fc.series, 6)
        assert dof_s[-1] < dof_l[-1]

    def test_empirical_coverage_is_close_to_nominal_on_random_walks(self):
        """
        Over many random walks and origins, the 80% interval should contain
        roughly 80% of outcomes. (Loose bounds: this is a calibration guard, not
        a statistical test.)
        """
        hits = total = 0
        for seed in range(25):
            fc = CommoditiesForecaster(make_series_df('monthly', 110, seed=seed, drift=0.0), 'x', 'monthly', horizon=4)
            s = fc.series
            for o in range(60, len(s) - 4, 4):
                cone = fc._cone(s.iloc[:o], 4)
                lo, hi = fc._interval(np.repeat(s.iloc[o - 1], 4), cone, 0.80)
                act = s.values[o:o + 4]
                hits += int(((act >= lo) & (act <= hi)).sum())
                total += 4
        assert 0.68 <= hits / total <= 0.90, f"80% interval covered {hits / total:.2f}"


class TestEnsembleEvaluation:

    def setup_method(self):
        self.fc = CommoditiesForecaster(make_series_df('monthly', 96, seed=5), 'x', 'monthly', horizon=3)
        self.results = self.fc.forecast_all_models()
        self.ensemble = self.fc.create_ensemble(self.results)

    def test_ensemble_has_intervals_for_every_step(self):
        for level in ('80', '95'):
            band = self.ensemble['intervals'][level]
            assert len(band['lower']) == len(band['upper']) == 3

    def test_ensemble_forecast_lies_inside_its_intervals(self):
        preds = np.array(self.ensemble['predictions'])
        iv = self.ensemble['intervals']['80']
        assert np.all(np.array(iv['lower']) <= preds) and np.all(preds <= np.array(iv['upper']))

    def test_ensemble_reports_skill_and_coverage(self):
        m = self.ensemble['metrics']
        for key in ('mae', 'rmae', 'skill_vs_naive', 'coverage_80', 'coverage_95', 'n_folds'):
            assert key in m, f"ensemble metrics missing {key}"
        assert m['skill_vs_naive'] == pytest.approx(1 - m['rmae'])
        assert 0 <= m['coverage_80'] <= 1

    def test_ensemble_carries_the_naive_baseline(self):
        assert self.ensemble['baseline']['method'].startswith('Naive')
        assert np.allclose(self.ensemble['baseline']['predictions'], float(self.fc.series.iloc[-1]))

    def test_weights_sum_to_one(self):
        assert sum(self.ensemble['weights'].values()) == pytest.approx(1.0)

    def test_naive_can_be_an_ensemble_member(self):
        """When nothing beats naive the ensemble must be able to shrink toward 'no change'."""
        results = {
            'naive': {'method': 'Naive', 'predictions': [100.0] * 3,
                      'metrics': {'mape': 5.0, 'mae': 5.0, 'rmae': 1.0}},
            'bad_model': {'method': 'Bad', 'predictions': [101.0] * 3,
                          'metrics': {'mape': 9.0, 'mae': 9.0, 'rmae': 1.8}},
        }
        fc = make_forecaster()
        fc.series = pd.Series([100.0] * 40, index=pd.date_range('2020-01-01', periods=40, freq='MS'))
        ens = fc.create_ensemble(results)
        assert ens['top_models'][0] == 'naive'
        assert ens['weights']['naive'] > ens['weights']['bad_model']

    def test_garch_is_never_an_ensemble_member(self):
        results = dict(self.results)
        results['garch'] = {'method': 'GARCH', 'predictions': [float(self.fc.series.iloc[-1])] * 3,
                            'metrics': {'mape': 0.0, 'mae': 0.0, 'rmae': 0.0}}
        ens = self.fc.create_ensemble(results)
        assert 'garch' not in ens['top_models']
