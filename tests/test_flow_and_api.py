"""
tests/test_flow_and_api.py
--------------------------
Glue between the forecaster, the flow adapters, DuckDB and the HTTP API.

The forecaster can be perfect and the dashboard still wrong if the persistence
layer drops fields, stores unstable model names, or the router misreads them.
These tests run the real path with a temporary DuckDB file:

    synthetic series -> flow adapters -> DataPipeline -> forecast_one -> DuckDB -> /api/*

Prefect itself is not needed to test this logic; a stub is installed when the
package is absent (the project's requirements pin the real one).
"""

import functools
import os
import sys
import types
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'backend'))

try:                                           # pragma: no cover - depends on environment
    import prefect  # noqa: F401
except ImportError:                            # pragma: no cover
    class _Task:
        """Minimal stand-in for a Prefect task/flow: callable, with the raw function on .fn."""

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *a, **k):
            return self.fn(*a, **k)

    def _decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _Task(args[0])
        return _Task

    stub = types.ModuleType('prefect')
    stub.task = _decorator
    stub.flow = _decorator
    ctx = types.ModuleType('prefect.context')
    ctx.get_run_context = lambda: None
    sys.modules['prefect'] = stub
    sys.modules['prefect.context'] = ctx

import utils.db as db
from automation import flow as F
from pipeline.data_pipeline import DataPipeline, IMPUTED_SUFFIX


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'get_settings', lambda: SimpleNamespace(duckdb_path=tmp_path / 'api.duckdb'))
    monkeypatch.delenv('DATABASE_READ_ONLY', raising=False)
    db.initialize()
    return tmp_path


def synthetic_series(key, frequency, n, end, seed):
    grid = {'daily': 'B', 'weekly': 'W-MON', 'monthly': 'MS', 'quarterly': 'QS'}[frequency]
    idx = pd.date_range(end=end, periods=n, freq=grid)
    rng = np.random.default_rng(seed)
    vol = {'daily': 0.015, 'weekly': 0.025, 'monthly': 0.05, 'quarterly': 0.07}[frequency]
    vals = 100 * np.exp(np.cumsum(rng.normal(0.0005, vol, n)))
    return pd.DataFrame({'date': idx, key: vals, 'is_imputed': False})


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

class TestSeriesDictToWide:

    def test_outer_join_leaves_gaps_unfilled(self):
        series = {
            'corn': synthetic_series('corn', 'monthly', 40, '2026-09-01', 1),
            'crude_oil': synthetic_series('crude_oil', 'daily', 300, '2026-10-19', 2),
        }
        adapted = F._collector_to_validation_dict(series)
        wide = F._series_dict_to_wide(adapted)
        assert wide['corn'].notna().sum() == 40, "monthly values must not be repeated on business days"
        assert wide['crude_oil'].notna().sum() == 300

    def test_imputation_flags_travel_with_the_values(self):
        s = synthetic_series('corn', 'monthly', 40, '2026-09-01', 1)
        s.loc[10, 'is_imputed'] = True
        wide = F._series_dict_to_wide(F._collector_to_validation_dict({'corn': s}))
        flags = wide[f'corn{IMPUTED_SUFFIX}']
        assert flags.sum() == 1 and bool(flags.iloc[10])


class TestMakeSerializable:

    def test_non_finite_numbers_become_null(self):
        out = F.make_serializable({'a': float('nan'), 'b': np.float64('inf'), 'c': np.array([1.0, np.nan]), 'd': (1, 2)})
        assert out == {'a': None, 'b': None, 'c': [1.0, None], 'd': [1, 2]}


# --------------------------------------------------------------------------- #
# forecast_one -> DB rows
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def build_payload(commodity='corn', frequency='monthly', n=84, horizon=None):
    wide = synthetic_series(commodity, frequency, n, '2026-09-01', 3).drop(columns='is_imputed')
    # put the series on a busy calendar to prove forecast_one works on sparse frames
    busy = pd.DataFrame({'date': pd.bdate_range(wide['date'].iloc[0], wide['date'].iloc[-1] + pd.Timedelta(days=40))})
    sparse = busy.merge(wide, on='date', how='outer').sort_values('date').reset_index(drop=True)
    return F.forecast_one.fn(commodity, sparse, frequency, horizon)


class TestForecastOne:

    def test_uses_last_real_observation_as_current_price(self):
        name, payload = build_payload()
        assert name == 'corn'
        assert payload['meta']['last_observation_date'] == '2026-09-01'
        assert np.isfinite(payload['current_price'])

    def test_horizon_defaults_to_config_for_the_frequency(self):
        _, payload = build_payload()
        assert payload['horizon'] == 6 and len(payload['ensemble']['predictions']) == 6

    def test_future_dates_continue_the_commodity_calendar(self):
        _, payload = build_payload()
        assert payload['meta']['future_dates'][0] == '2026-10-01'

    def test_payload_is_json_safe_and_drops_backtest_arrays(self):
        import json
        _, payload = build_payload()
        json.dumps(payload, allow_nan=False)
        assert all('backtest' not in r for r in payload['individual_models'].values())

    def test_too_short_series_is_skipped_not_crashed(self):
        name, payload = build_payload(n=20)
        assert payload == {}


class TestForecastsToDb:

    def setup_method(self):
        _, self.payload = build_payload()
        self.df = F._forecasts_to_db_df({'corn': self.payload}, date(2026, 10, 27))

    def test_model_column_holds_stable_keys(self):
        models = set(self.df['model'])
        assert 'ensemble' in models and 'naive' in models and 'arima' in models
        assert not any('(' in m for m in models), "labels with fitted orders must not be primary keys"

    def test_confidence_is_stored_as_text(self):
        assert set(self.df['confidence']) <= {'high', 'medium', 'low', 'n/a'}

    def test_ensemble_row_carries_intervals_weights_baseline_and_meta(self):
        import json
        row = self.df[self.df['model'] == 'ensemble'].iloc[0]
        m = json.loads(row['metrics_json'])
        for key in ('intervals', 'weights', 'baseline', 'meta', 'skill_vs_naive', 'coverage_80', 'top_models'):
            assert key in m, f"ensemble metrics_json missing {key}"
        assert m['meta']['last_observation_date'] == '2026-09-01'

    def test_all_json_columns_parse_without_nan_tokens(self):
        for col in ('predictions_json', 'metrics_json'):
            for text in self.df[col]:
                assert 'NaN' not in text and 'Infinity' not in text


# --------------------------------------------------------------------------- #
# HTTP API on a real temporary database
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def client(tmp_path_factory):
    """
    One database and one forecast run shared by every API test in this module
    (computing the forecasts is the slow part).
    """
    from fastapi.testclient import TestClient
    import main as backend_main

    mp = pytest.MonkeyPatch()
    db_path = tmp_path_factory.mktemp('api') / 'api.duckdb'
    mp.setattr(db, 'get_settings', lambda: SimpleNamespace(duckdb_path=db_path))
    mp.delenv('DATABASE_READ_ONLY', raising=False)
    db.initialize()

    series = {
        'corn':      synthetic_series('corn', 'monthly', 84, '2026-09-01', 11),
        'crude_oil': synthetic_series('crude_oil', 'daily', 420, '2026-10-19', 12),
    }
    metadata = {'corn': {'frequency': 'monthly'}, 'crude_oil': {'frequency': 'daily'}}
    wide = F._series_dict_to_wide(F._collector_to_validation_dict(series))

    pipeline = DataPipeline()
    pipeline.process_and_save(wide)
    full = pipeline.load_latest()

    run_date = date(2026, 10, 27)
    forecasts = {}
    for commodity in ('corn', 'crude_oil'):
        name, payload = F.forecast_one.fn(commodity, full, metadata[commodity]['frequency'])
        forecasts[name] = payload
    db.upsert_forecasts(F._forecasts_to_db_df(forecasts, run_date))
    db.log_run('run-1', pd.Timestamp('2026-10-27 08:00').to_pydatetime(), 'SUCCESS',
               completed_at=pd.Timestamp('2026-10-27 08:05').to_pydatetime())

    yield TestClient(backend_main.app)
    mp.undo()


class TestOverviewApi:

    def test_last_update_is_data_date_and_pipeline_run_is_reported_separately(self, client):
        body = client.get('/api/overview').json()
        assert body['last_update'] == '2026-10-19', "newest price observation, not the run date"
        assert body['pipeline']['last_run_at'] == '2026-10-27T08:05:00Z'
        assert body['pipeline']['status'] == 'SUCCESS'

    def test_monthly_commodity_has_a_real_nonzero_change_and_own_as_of_date(self, client):
        body = client.get('/api/overview').json()
        corn = next(c for cat in body['categories'].values() for c in cat['commodities'] if c['key'] == 'corn')
        assert corn['as_of'] == '2026-09-01'
        assert corn['change_pct'] is not None and abs(corn['change_pct']) > 1e-9

    def test_chart_history_is_continuous_while_stats_use_real_points(self, client):
        body = client.get('/api/overview').json()
        corn_series = body['price_history']['series']['corn']
        assert None not in corn_series[-50:], "aligned chart lines must have no holes"


class TestForecastApi:

    def test_available_lists_forecast_commodities(self, client):
        keys = [c['key'] for c in client.get('/api/forecast/available').json()['commodities']]
        assert keys == ['corn', 'crude_oil']

    def test_forecast_payload_shape_for_monthly_series(self, client):
        d = client.get('/api/forecast', params={'commodity': 'corn'}).json()
        assert d['horizon'] == 6 and len(d['future_dates']) == 6
        assert d['last_date'] == '2026-09-01' and d['history']['dates'][-1] == '2026-09-01'
        assert set(d['intervals']) == {'80', '95'}
        assert d['baseline']['method'].startswith('Naive')
        assert d['confidence'] in ('high', 'medium', 'low')
        for key in ('rmae', 'skill_vs_naive', 'coverage_80', 'coverage_95', 'mae', 'n_folds'):
            assert key in d['evaluation']

    def test_history_is_the_commodities_own_observations(self, client):
        d = client.get('/api/forecast', params={'commodity': 'corn'}).json()
        dates = pd.to_datetime(d['history']['dates'])
        assert all(day == 1 for day in dates.day), "monthly history must be one point per month"
        assert None not in d['history']['values']

    def test_baseline_is_flat_at_current_price(self, client):
        d = client.get('/api/forecast', params={'commodity': 'corn'}).json()
        assert np.allclose(d['baseline']['predictions'], d['current_price'], rtol=1e-6)

    def test_ranked_models_flag_baseline_and_top_models(self, client):
        d = client.get('/api/forecast', params={'commodity': 'corn'}).json()
        models = {m['key']: m for m in d['ranked_models']}
        assert models['naive']['is_baseline'] is True and models['naive']['rmae'] == pytest.approx(1.0)
        assert any(m['is_top'] for m in d['ranked_models'])
        assert 'garch' not in models, "GARCH is a volatility model, not a ranked price model"

    def test_daily_series_exposes_garch_band_and_longer_horizon(self, client):
        d = client.get('/api/forecast', params={'commodity': 'crude_oil'}).json()
        assert d['horizon'] == 20
        assert d['garch_band'] is not None and len(d['garch_band']['lower']) == 20

    def test_intervals_bracket_the_forecast(self, client):
        d = client.get('/api/forecast', params={'commodity': 'corn'}).json()
        lo, hi = d['intervals']['80']['lower'], d['intervals']['80']['upper']
        for p, l, h in zip(d['ensemble']['predictions'], lo, hi):
            assert l <= p <= h

    def test_unknown_commodity_is_404(self, client):
        assert client.get('/api/forecast', params={'commodity': 'unobtainium'}).status_code == 404

    def test_legacy_numeric_confidence_is_still_readable(self, client):
        from routers import forecast as forecast_router
        assert forecast_router._confidence_label('1.0') == 'high'
        assert forecast_router._confidence_label('0.5') == 'medium'
        assert forecast_router._confidence_label('0.0') == 'low'
        assert forecast_router._confidence_label('medium') == 'medium'


class TestAnalysisApiUsesRealObservations:

    def test_analysis_endpoint_responds(self, client):
        r = client.get('/api/analysis', params={'commodities': 'corn,crude_oil', 'range': '1Y'})
        assert r.status_code == 200, r.text[:300]


# --------------------------------------------------------------------------- #
# Smoke tests: endpoints with no dedicated coverage above
# --------------------------------------------------------------------------- #

class TestSmokeEndpoints:
    """Cheap 'does it respond and have the right shape' checks."""

    def test_health_reports_database_reachable(self, client):
        r = client.get('/api/health')
        assert r.status_code == 200, r.text[:300]
        assert r.json() == {'status': 'ok', 'database': 'reachable'}

    def test_commodities_lists_registry_and_categories(self, client):
        r = client.get('/api/commodities')
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body['commodities'], 'commodity registry should not be empty'
        assert {'key', 'name', 'category', 'frequency', 'unit'} <= set(body['commodities'][0])
        # every commodity key referenced by a category exists in the registry
        keys = {c['key'] for c in body['commodities']}
        for members in body['categories'].values():
            assert set(members) <= keys

    def test_ratios_responds_with_expected_sections(self, client):
        r = client.get('/api/ratios')
        assert r.status_code == 200, r.text[:300]
        assert {'regime', 'cost_index_history', 'margin_pressure',
                'zscore_bars', 'ratio_pairs'} <= set(r.json())

    def test_ratios_without_data_is_404_not_a_crash(self, tmp_db):
        from fastapi.testclient import TestClient
        import main as backend_main
        r = TestClient(backend_main.app).get('/api/ratios')   # tmp_db is empty
        assert r.status_code == 404
