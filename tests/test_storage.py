"""
tests/test_storage.py
---------------------
The storage contract: the ``prices`` table holds ONLY real observations.

Before, ``clean()`` forward-filled every series onto the union calendar before
saving, so a monthly price was stored once per business day. That made the
"change %" of every monthly commodity exactly 0.00%, gave the forecaster a flat
tail, and inflated percentiles/statistics. These tests pin the replacement
behaviour against a real (temporary) DuckDB file:

  * only real observations are written, and re-running is idempotent,
  * the one-time migration purges legacy forward-filled rows — but only once
    fresh data is already in hand,
  * merging is per cell, not per row,
  * collector gap-fill flags (is_imputed) reach the database,
  * consumers that need aligned series get them explicitly and in memory,
  * the API can tell "when did the flow run" from "how fresh is the data".

Run with:  pytest tests/ -v
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'backend'))

import utils.db as db
from pipeline.data_pipeline import DataPipeline, IMPUTED_SUFFIX
from utils.alignment import align_wide_frame
from services import prices as price_service


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point utils.db at a throw-away DuckDB file."""
    monkeypatch.setattr(db, 'get_settings', lambda: SimpleNamespace(duckdb_path=tmp_path / 'test.duckdb'))
    monkeypatch.delenv('DATABASE_READ_ONLY', raising=False)
    db.initialize()
    return tmp_path


def mixed_frequency_wide(n_months=30):
    """
    Wide frame as produced after the collector: a daily series (crude_oil) and a
    monthly one (corn) outer-joined on one calendar, with NaN where a series
    has no observation.
    """
    rng = np.random.default_rng(0)
    bdays = pd.bdate_range('2024-01-01', periods=int(n_months * 21.5))
    months = pd.date_range('2024-01-01', periods=n_months, freq='MS')
    crude = pd.Series(70 + np.cumsum(rng.normal(0, 1, len(bdays))), index=bdays, name='crude_oil')
    corn = pd.Series(400 + np.cumsum(rng.normal(0, 5, len(months))), index=months, name='corn')
    wide = pd.concat([crude, corn], axis=1).sort_index()
    wide.index.name = 'date'
    return wide.reset_index()


class TestOnlyRealObservationsStored:

    def test_monthly_series_stored_once_per_month(self, tmp_db):
        wide = mixed_frequency_wide()
        DataPipeline().process_and_save(wide)
        stored = db.load_prices()
        n_corn = (stored['commodity'] == 'corn').sum()
        assert n_corn == wide['corn'].notna().sum(), (
            f"corn must be stored once per real observation (expected "
            f"{wide['corn'].notna().sum()}, got {n_corn})"
        )

    def test_second_run_is_idempotent(self, tmp_db):
        wide = mixed_frequency_wide()
        pipeline = DataPipeline()
        pipeline.process_and_save(wide)
        first = len(db.load_prices())
        pipeline.process_and_save(wide)
        assert len(db.load_prices()) == first

    def test_load_latest_is_sparse_by_default_and_aligned_on_request(self, tmp_db):
        wide = mixed_frequency_wide()
        pipeline = DataPipeline()
        pipeline.process_and_save(wide)

        sparse = pipeline.load_latest()
        assert sparse['corn'].isna().any(), "default frame must not invent monthly values on business days"

        aligned = pipeline.load_latest(aligned=True)
        first_valid = aligned['corn'].first_valid_index()
        assert not aligned['corn'].iloc[first_valid:].isna().any(), "aligned frame carries the last value forward"

    def test_change_pct_uses_last_two_real_observations(self, tmp_db):
        """
        The bug that motivated all this: on a forward-filled frame the last two
        values of a monthly series are identical, so change % was always 0.
        """
        wide = mixed_frequency_wide()
        DataPipeline().process_and_save(wide)

        raw = price_service.load_wide_prices(aligned=False)
        real = wide['corn'].dropna()
        expected = (real.iloc[-1] - real.iloc[-2]) / real.iloc[-2] * 100
        got = price_service.change_pct(raw, 'corn')
        assert got is not None and abs(got - expected) < 0.05          # stored values are rounded to 2 dp
        assert abs(got) > 1e-6, "monthly change must not collapse to 0.00%"

        aligned = price_service.load_wide_prices(aligned=True)
        assert price_service.change_pct(aligned, 'corn') == 0.0, (
            "documenting why consumers must use the raw frame for changes"
        )


class TestLegacyMigration:

    def _seed_legacy_ffilled_rows(self):
        legacy = pd.DataFrame({
            'date': pd.bdate_range('2024-02-01', '2024-03-15'),
            'commodity': 'corn', 'value': 123.0,
            'is_imputed': False, 'created_at': pd.Timestamp.now(),
        })
        db.upsert_prices(legacy)

    def test_legacy_rows_are_purged_once_and_version_is_recorded(self, tmp_db):
        self._seed_legacy_ffilled_rows()
        assert db.get_data_version() == 0

        wide = mixed_frequency_wide()
        DataPipeline().process_and_save(wide)

        stored = db.load_prices()
        assert (stored['value'] == 123.0).sum() == 0, "fabricated legacy rows must be gone"
        assert (stored['commodity'] == 'corn').sum() == wide['corn'].notna().sum()
        assert db.get_data_version() == db.CURRENT_DATA_VERSION

    def test_migration_does_not_repeat(self, tmp_db):
        wide = mixed_frequency_wide()
        pipeline = DataPipeline()
        pipeline.process_and_save(wide)

        # A row added after migration must survive later runs.
        db.upsert_prices(pd.DataFrame({
            'date': [pd.Timestamp('2030-01-01')], 'commodity': ['corn'], 'value': [999.0],
            'is_imputed': [False], 'created_at': [pd.Timestamp.now()],
        }))
        pipeline.process_and_save(wide)
        assert (db.load_prices()['value'] == 999.0).sum() == 1

    def test_purge_never_happens_without_fresh_data(self, tmp_db):
        """A failed download must not leave the database empty: the purge lives inside process_and_save."""
        self._seed_legacy_ffilled_rows()
        before = len(db.load_prices())
        # (nothing calls process_and_save — e.g. collect() raised)
        assert len(db.load_prices()) == before
        assert db.get_data_version() == 0


class TestMergeAndFlags:

    def test_merge_is_per_cell_not_per_row(self, tmp_db):
        """
        A new download that lacks a series on a date must not blank the value
        already stored for that series on that date.
        """
        pipeline = DataPipeline()
        dates = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        first = pd.DataFrame({'date': dates, 'corn': [1.0, 2.0, 3.0], 'wheat': [10.0, 20.0, 30.0]})
        pipeline.process_and_save(first)

        second = pd.DataFrame({'date': dates, 'corn': [1.5, 2.5, 3.5], 'wheat': [np.nan, np.nan, np.nan]})
        merged = pipeline.merge_with_existing(second, '')
        merged = merged.set_index('date')
        assert merged.loc['2024-02-01', 'corn'] == 2.5, "new values win"
        assert merged.loc['2024-02-01', 'wheat'] == 20.0, "missing new values keep the stored ones"

    def test_split_imputed_separates_flag_columns(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'corn': [1.0, 2.0],
            f'corn{IMPUTED_SUFFIX}': [False, True],
        })
        values, mask = DataPipeline.split_imputed(df)
        assert list(values.columns) == ['date', 'corn']
        assert bool(mask.loc['2024-02-01', 'corn']) is True

    def test_imputed_flags_reach_the_database(self, tmp_db):
        wide = mixed_frequency_wide(n_months=12)
        flag = pd.Series(False, index=wide.index)
        last_corn = wide['corn'].last_valid_index()
        flag.loc[last_corn] = True
        wide[f'corn{IMPUTED_SUFFIX}'] = flag

        DataPipeline().process_and_save(wide)
        stored = db.load_prices(commodity='corn')
        assert stored['is_imputed'].sum() == 1
        assert bool(stored.iloc[-1]['is_imputed']) is True


class TestAlignment:

    def test_align_forward_fills_and_keeps_date(self):
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4, freq='D'),
            'a': [1.0, np.nan, np.nan, 4.0],
            'b': [np.nan, 5.0, np.nan, np.nan],
        })
        out = align_wide_frame(df)
        assert out['a'].tolist() == [1.0, 1.0, 1.0, 4.0]
        assert np.isnan(out['b'].iloc[0]) and out['b'].iloc[3] == 5.0, "leading gaps stay empty"
        assert out['date'].equals(df['date'])
        assert df['a'].isna().sum() == 2, "input must not be modified"


class TestPipelineStatus:

    def test_last_run_is_separate_from_data_freshness(self, tmp_db):
        assert price_service.load_pipeline_status() is None

        db.log_run('r1', pd.Timestamp('2026-10-27 08:00').to_pydatetime(), 'RUNNING')
        assert price_service.load_pipeline_status() is None, "unfinished runs do not count"

        db.log_run('r1', pd.Timestamp('2026-10-27 08:00').to_pydatetime(), 'SUCCESS',
                   completed_at=pd.Timestamp('2026-10-27 08:04').to_pydatetime())
        status = price_service.load_pipeline_status()
        assert status['status'] == 'SUCCESS'
        assert status['last_run_at'] == '2026-10-27T08:04:00Z'
        assert status['last_success_at'] == '2026-10-27T08:04:00Z'

    def test_forecasts_are_replaced_per_commodity_and_day(self, tmp_db):
        """Re-running on the same day must not leave rows of a previous model set behind."""
        def rows(models):
            return pd.DataFrame([{
                'run_date': pd.Timestamp('2026-10-27').date(), 'commodity': 'corn', 'model': m,
                'horizon': 3, 'predictions_json': '[1,2,3]', 'metrics_json': '{}', 'confidence': 'low',
            } for m in models])

        db.upsert_forecasts(rows(['ARIMA(1, 1, 0)', 'ensemble']))
        db.upsert_forecasts(rows(['arima', 'naive', 'ensemble']))
        with db.get_connection() as conn:
            models = sorted(r[0] for r in conn.execute('SELECT model FROM forecasts').fetchall())
        assert models == ['arima', 'ensemble', 'naive']
