"""
Multi-Horizon Forecasting System

Generates price forecasts for the key commodity series using a pool of
statistical, econometric and machine-learning models, chosen per series type
(frequency and length), and combines the best ones into a weighted ensemble.

Design principles
-----------------
1. Native series.
   Every commodity is forecast on ITS OWN observation calendar (business days,
   weeks, months or quarters). Nothing is forward-filled onto another series'
   calendar, so the models never see a "price unchanged" that never happened.

2. Honest evaluation.
   Models are scored with a ROLLING-ORIGIN backtest (several train/test splits
   moving through time) instead of a single small hold-out. Every metric is
   also expressed relative to a NAIVE "last value" forecast evaluated on exactly
   the same folds (rMAE). If nothing beats the naive baseline the dashboard says
   so (confidence = low) instead of dressing a flat line up as a prediction.

3. Intervals, not just a line.
   The ensemble ships 80% and 95% prediction intervals built from the
   historical k-step-ahead changes of the series, rescaled by the recent
   volatility regime. Their empirical coverage is measured in the same
   backtest, so calibration is reported, not assumed.

Model pool (selected per series type)
-------------------------------------
    Baselines / benchmarks : Naive (last value), Drift, Seasonal naive
    Exponential smoothing  : ETS damped trend, ETS damped + seasonal
    Decomposition          : Theta
    Box-Jenkins            : ARIMA, SARIMA (order picked once, parameters refit per fold)
    ML on returns          : Ridge, LASSO, Gradient Boosting, Random Forest, XGBoost
    Volatility             : GARCH(1,1) (daily/weekly only; feeds an alternative 90% band)

Prices are modelled in log space (when all values are positive) so that
percentage moves, not absolute ones, are what the models learn, and forecasts
can never go negative.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from pmdarima import auto_arima
from arch import arch_model

# Configuration loader (forecast horizon per frequency, per-commodity flags)
try:
    from utils.config_loader import ConfigLoader
except ImportError:  # pragma: no cover - only when run outside the project
    ConfigLoader = None


# Interval levels published with every ensemble forecast.
INTERVAL_LEVELS = (0.80, 0.95)
_Z_GARCH_90 = 1.6449


class CommoditiesForecaster:
    """
    Forecasting engine for ONE commodity on its native observation calendar.
    """

    # ------------------------------------------------------------------ #
    # Per-frequency settings                                               #
    # ------------------------------------------------------------------ #
    #   pd_freq      pandas grid the native observations are snapped to
    #   min_points   minimum history needed before backtesting at all
    #   lags         momentum look-backs used by the ML models
    #   season       natural seasonal period in observations (None = none)
    #   max_folds    maximum number of rolling-origin folds
    #   vol_window   look-back (observations) for the "recent volatility" regime
    #   train_window cap on training observations (keeps daily series fast)
    #   horizon      default number of steps ahead (overridden by commodities.json)
    FREQ_SETTINGS = {
        'daily':     dict(pd_freq='B',     min_points=90, lags=[1, 5, 21],     season=None,
                          max_folds=6, vol_window=60, train_window=1000, horizon=20),
        'weekly':    dict(pd_freq='W-MON', min_points=52, lags=[1, 4, 12],     season=52,
                          max_folds=6, vol_window=26, train_window=400,  horizon=12),
        'monthly':   dict(pd_freq='MS',    min_points=24, lags=[1, 3, 6, 12],  season=12,
                          max_folds=8, vol_window=12, train_window=None, horizon=6),
        'quarterly': dict(pd_freq='QS',    min_points=12, lags=[1, 2, 4],      season=4,
                          max_folds=6, vol_window=8,  train_window=None, horizon=4),
    }

    # Minimum number of rolling-origin folds for an evaluation to be meaningful.
    MIN_FOLDS = 3

    # Supervised ML models need a training set materially larger than their
    # feature space; below this they overfit and poison the ranking.
    _ML_MIN_OBSERVATIONS = 150
    _RIDGE_MIN_OBSERVATIONS = 60

    # GARCH needs a long, high-frequency sample to be meaningful.
    _GARCH_MIN_OBSERVATIONS = 250

    BASELINE_KEY = 'naive'
    BENCHMARK_KEYS = ('drift', 'seasonal_naive')

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, data: pd.DataFrame, commodity: str, frequency: str = 'monthly',
                 horizon: int | None = None):
        self.data      = data
        self.commodity = commodity
        self.frequency = frequency if frequency in self.FREQ_SETTINGS else 'monthly'

        self.config  = dict(self.FREQ_SETTINGS[self.frequency])
        self.pd_freq = self.config['pd_freq']
        self.horizon = int(horizon) if horizon is not None else self._default_horizon()

        # Commodity-specific flags from commodities.json
        self.allow_negative_prices = False
        if ConfigLoader is not None:
            try:
                info = ConfigLoader.get_commodity_info(self.commodity)
                self.allow_negative_prices = bool(info.get('allow_negative_prices', False))
            except Exception:
                pass

        self.native, self.series, self.n_gap_filled = self._build_series(data, commodity)
        self.use_log = bool(len(self.series) and (self.series > 0).all()) and not self.allow_negative_prices

        self._order_cache: dict[str, tuple] = {}
        self._labels: dict[str, str] = {}

    def _default_horizon(self) -> int:
        """Forecast steps for this frequency: commodities.json first, built-in default second."""
        default = self.config['horizon']
        if ConfigLoader is None:
            return default
        try:
            freq_cfg = ConfigLoader.get_frequency_config().get(self.frequency, {})
            steps = int(freq_cfg.get('forecast_steps', default))
            return steps if steps > 0 else default
        except Exception:
            return default

    # ------------------------------------------------------------------ #
    # Series construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_series(self, data: pd.DataFrame, commodity: str):
        """
        Extract the commodity's NATIVE observations and put them on a regular
        grid of its own frequency.

        Returns (native, regular, n_gap_filled):
            native         real observations exactly as stored
            regular        the same values on a regular pandas grid
            n_gap_filled   grid points that had no observation and were
                           linearly interpolated (e.g. a market holiday)
        """
        if 'date' in data.columns:
            native = data.set_index('date')[commodity]
        else:
            native = data[commodity]

        native = native.dropna().astype(float)

        if not isinstance(native.index, pd.DatetimeIndex):
            try:
                native.index = pd.to_datetime(native.index)
            except Exception:
                native.index = pd.date_range('2000-01-01', periods=len(native), freq=self.pd_freq)
            if native.index.min() < pd.Timestamp('1990-01-01'):   # integer index misparsed as epoch
                native.index = pd.date_range('2000-01-01', periods=len(native), freq=self.pd_freq)

        native = native[~native.index.duplicated(keep='last')].sort_index()

        if native.empty:
            return native, native, 0

        grid = native.resample(self.pd_freq).last()
        n_gaps = int(grid.isna().sum())
        # Interior holes only: never extrapolate past the last real observation.
        grid = grid.interpolate(limit_area='inside').dropna()
        return native, grid, n_gaps

    @property
    def last_observation_date(self) -> pd.Timestamp:
        """Date of the last REAL observation of this commodity."""
        return self.native.index[-1]

    def future_dates(self) -> list[str]:
        """ISO dates of the forecast steps, following the series' own calendar."""
        idx = pd.date_range(self.series.index[-1], periods=self.horizon + 1, freq=self.pd_freq)[1:]
        return [d.strftime('%Y-%m-%d') for d in idx]

    # Model-space transforms (log for strictly positive series, identity otherwise)
    def _fwd(self, s):
        return np.log(s) if self.use_log else s

    def _inv(self, a):
        return np.exp(a) if self.use_log else a

    # ------------------------------------------------------------------ #
    # Validation & rolling-origin design                                   #
    # ------------------------------------------------------------------ #

    def _fold_step(self) -> int:
        """Distance between consecutive fold origins (half a horizon, overlapping)."""
        return max(1, self.horizon // 2)

    def validate_data(self) -> tuple[bool, str]:
        """
        Check that there is enough history to train AND run a rolling-origin
        backtest with at least MIN_FOLDS folds.
        """
        min_required = (
            self.config['min_points'] + self.horizon + (self.MIN_FOLDS - 1) * self._fold_step()
        )
        if len(self.series) < min_required:
            return False, f"Insufficient data: {len(self.series)} points (minimum required: {min_required})"
        return True, "OK"

    def _origins(self) -> list[int]:
        """
        Fold origins: position `o` means train on series[:o], test on series[o:o+horizon].
        The last fold always ends exactly at the end of the series.
        """
        n, h, step = len(self.series), self.horizon, self._fold_step()
        available = (n - self.config['min_points'] - h) // step + 1
        n_folds = max(0, min(self.config['max_folds'], available))
        last = n - h
        return [last - i * step for i in range(n_folds)][::-1]

    def _train_slice(self, end: int) -> pd.Series:
        window = self.config['train_window']
        start = 0 if window is None else max(0, end - window)
        return self.series.iloc[start:end]

    def _mase_scale(self, upto: int) -> float:
        """Mean absolute one-step change of the history before the last fold (MASE denominator)."""
        d = np.abs(np.diff(self.series.values[:upto]))
        return float(d.mean()) if len(d) else float('nan')

    # ------------------------------------------------------------------ #
    # Metrics & confidence                                                 #
    # ------------------------------------------------------------------ #

    def calculate_metrics(self, actual, predicted, scale: float | None = None, base=None) -> dict:
        """
        Forecast-error metrics.

        Args:
            actual, predicted: arrays of the same shape (any number of dims).
            scale: MASE denominator (mean absolute in-sample one-step change).
                   When omitted, the mean absolute difference of ``actual`` is used.
            base:  the last observed value each forecast was issued from, same
                   shape as ``actual``. With it, directional accuracy asks "did
                   the forecast get the direction of the move from the last
                   known price right?". Without it, it compares successive diffs.
        """
        actual    = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        err  = actual - predicted        # raises ValueError on shape mismatch
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        mape = float(np.mean(np.abs(err / np.where(actual == 0, 1, actual))) * 100)

        smape = float(np.mean(
            2 * np.abs(err) / (np.abs(actual) + np.abs(predicted) + 1e-8)
        ) * 100)

        if scale is None:
            flat = actual.ravel()
            scale = float(np.mean(np.abs(np.diff(flat)))) if flat.size > 1 else float('nan')
        mase = mae / scale if (scale is not None and np.isfinite(scale) and scale > 1e-8) else np.nan

        mbe = float(np.mean(predicted - actual))

        if base is not None:
            base = np.asarray(base, dtype=float)
            directional_accuracy = float(
                np.mean(np.sign(predicted - base) == np.sign(actual - base)) * 100
            )
        elif actual.size > 1:
            a, p = actual.ravel(), predicted.ravel()
            directional_accuracy = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100)
        else:
            directional_accuracy = float('nan')

        return {
            'mae':                  mae,
            'mape':                 mape,
            'rmse':                 rmse,
            'smape':                smape,
            'mase':                 float(mase) if not np.isnan(mase) else None,
            'mbe':                  mbe,
            'directional_accuracy': directional_accuracy,
        }

    def assign_confidence(self, metrics_or_mape) -> str:
        """
        Qualitative confidence label from out-of-sample error.

        Preferred signal: ``rmae`` = MAE of the model / MAE of the naive
        "last value" forecast on the same folds. A model only deserves
        confidence if it clearly beats doing nothing:

            rmae < 0.85  -> high    (>15% better than naive)
            rmae < 1.00  -> medium  (better than naive)
            otherwise    -> low     (no better than "the price stays where it is")

        Fallbacks when rmae is unavailable: MASE thresholds, then MAPE relative
        to the series' historical volatility.
        """
        if isinstance(metrics_or_mape, dict):
            rmae = metrics_or_mape.get('rmae')
            if rmae is not None and np.isfinite(rmae):
                if rmae < 0.85:
                    return 'high'
                elif rmae < 1.0:
                    return 'medium'
                return 'low'

            mase = metrics_or_mape.get('mase')
            if mase is not None and not np.isnan(mase):
                if mase < 0.8:
                    return 'high'
                elif mase < 1.2:
                    return 'medium'
                else:
                    return 'low'
            mape = metrics_or_mape.get('mape', 100)
        else:
            mape = metrics_or_mape

        volatility = self.series.pct_change().std() * 100
        if not np.isfinite(volatility) or volatility == 0:
            return 'low'

        score = mape / volatility
        if score < 0.5:
            return 'high'
        elif score < 1.0:
            return 'medium'
        return 'low'

    # ------------------------------------------------------------------ #
    # Prediction intervals                                                 #
    # ------------------------------------------------------------------ #

    def _cone(self, series: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Spread of the k-step-ahead change (model space) for k = 1..h.

        Returns ``(sigma, df)``:
            sigma  standard deviation of the k-step change
            df     effective degrees of freedom of that estimate

        sigma comes from the OVERLAPPING k-step changes of the history (drift
        removed), which reflects real autocorrelation better than sqrt(k)
        scaling; it falls back to sqrt(k) scaling when there are too few
        windows. The whole cone is rescaled by the ratio of recent to
        full-history one-step volatility (clipped to [0.7, 1.5]) so calm and
        turbulent regimes narrow or widen the intervals.

        ``df`` ~ number of INDEPENDENT k-step increments in the sample. Using
        Student-t quantiles with these degrees of freedom keeps the intervals
        honest when history is short (a normal quantile would under-cover).
        """
        x = np.asarray(self._fwd(series.values.astype(float)))
        if len(x) < 6:
            return np.zeros(h), np.full(h, 4.0)

        d1 = np.diff(x)
        full_sd = float(np.std(d1, ddof=1))
        if not np.isfinite(full_sd) or full_sd <= 0:
            return np.zeros(h), np.full(h, 4.0)

        w = min(self.config['vol_window'], len(d1))
        recent_sd = float(np.std(d1[-w:], ddof=1)) if w > 2 else full_sd
        regime = float(np.clip(recent_sd / full_sd, 0.7, 1.5)) if np.isfinite(recent_sd) else 1.0

        sig = []
        for k in range(1, h + 1):
            if len(x) - k >= 12:
                dk = x[k:] - x[:-k]
                dk = dk - dk.mean()
                sig.append(float(np.sqrt(np.mean(dk ** 2))))
            else:
                sig.append(full_sd * np.sqrt(k))

        # A forecast cone can only widen with the horizon.
        sigma = np.maximum.accumulate(np.array(sig)) * regime
        dof = np.array([max(4.0, (len(x) - 1) / k) for k in range(1, h + 1)])
        return sigma, dof

    def _interval(self, pred: np.ndarray, cone: tuple[np.ndarray, np.ndarray], level: float):
        """Lower/upper bounds around ``pred`` for a level (log-symmetric when modelled in logs)."""
        sigma, dof = cone
        z = stats.t.ppf(0.5 + level / 2.0, dof)
        pred = np.asarray(pred, dtype=float)
        if self.use_log:
            return pred * np.exp(-z * sigma), pred * np.exp(z * sigma)
        return pred - z * sigma, pred + z * sigma

    # ------------------------------------------------------------------ #
    # Individual models — all share the signature fn(train, h) -> array    #
    # `train` is a pandas Series in MODEL space (log or level);            #
    # the return value is an array of h forecasts in the same space.       #
    # ------------------------------------------------------------------ #

    def _m_naive(self, train: pd.Series, h: int) -> np.ndarray:
        return np.repeat(float(train.iloc[-1]), h)

    def _m_drift(self, train: pd.Series, h: int) -> np.ndarray:
        slope = (float(train.iloc[-1]) - float(train.iloc[0])) / max(len(train) - 1, 1)
        return float(train.iloc[-1]) + slope * np.arange(1, h + 1)

    def _m_seasonal_naive(self, train: pd.Series, h: int) -> np.ndarray:
        m = self.config['season']
        if m is None or len(train) < m:
            raise ValueError('not enough history for a seasonal naive forecast')
        idx = len(train) - m + (np.arange(h) % m)
        return train.iloc[idx].values.astype(float)

    def _m_ets(self, train: pd.Series, h: int, seasonal: bool = False) -> np.ndarray:
        """Damped-trend exponential smoothing (optionally with additive seasonality)."""
        kw = dict(trend='add', damped_trend=True, initialization_method='estimated')
        if seasonal:
            kw.update(seasonal='add', seasonal_periods=self.config['season'])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fc = np.asarray(ExponentialSmoothing(train, **kw).fit(optimized=True).forecast(h), dtype=float)
                if np.all(np.isfinite(fc)):
                    return fc
            except Exception:
                if seasonal:
                    raise
            if seasonal:
                raise ValueError('seasonal ETS produced non-finite forecasts')
            # Non-seasonal fallback: simple exponential smoothing (flat forecast)
            fit = SimpleExpSmoothing(train, initialization_method='estimated').fit()
            return np.asarray(fit.forecast(h), dtype=float)

    def _m_ets_seasonal(self, train: pd.Series, h: int) -> np.ndarray:
        return self._m_ets(train, h, seasonal=True)

    def _m_theta(self, train: pd.Series, h: int) -> np.ndarray:
        season = self.config['season']
        deseason = bool(season and season <= 12 and len(train) >= 2 * season)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = ThetaModel(
                train, period=season if deseason else 1, deseasonalize=deseason
            ).fit()
            return np.asarray(fit.forecast(h), dtype=float)

    def _select_arima_order(self, key: str, train: pd.Series, seasonal: bool) -> None:
        """
        Pick (p,d,q)(P,D,Q,m) ONCE, on the training data of the first (earliest)
        fold, so the order never sees the test periods. Later folds and the final
        forecast only re-estimate the parameters for that fixed order.
        """
        if key in self._order_cache:
            return
        sel = train.iloc[-750:]
        m = self.config['season'] if seasonal else 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = auto_arima(
                sel.values,
                start_p=0, max_p=3, start_q=0, max_q=3, d=None,
                seasonal=seasonal, m=m,
                start_P=0, max_P=1, start_Q=0, max_Q=1, D=None,
                information_criterion='aic', stepwise=True,
                error_action='ignore', suppress_warnings=True,
            )
        order = tuple(int(v) for v in model.order)
        sorder = tuple(int(v) for v in model.seasonal_order) if seasonal else (0, 0, 0, 0)
        self._order_cache[key] = (order, sorder)
        self._labels[key] = f'SARIMA{order}x{sorder}' if seasonal else f'ARIMA{order}'

    def _m_arima(self, train: pd.Series, h: int, seasonal: bool = False) -> np.ndarray:
        key = 'sarima' if seasonal else 'arima'
        self._select_arima_order(key, train, seasonal)
        order, sorder = self._order_cache[key]

        if seasonal and not any(sorder[:3]):
            raise ValueError('no seasonal structure selected — SARIMA would duplicate ARIMA')

        integrated = order[1] > 0 or sorder[1] > 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = ARIMA(train, order=order, seasonal_order=sorder,
                        trend='n' if integrated else 'c').fit()
            return np.asarray(res.forecast(h), dtype=float)

    def _m_sarima(self, train: pd.Series, h: int) -> np.ndarray:
        return self._m_arima(train, h, seasonal=True)

    # ------------------------------------------------------------------ #
    # Machine-learning models (on returns)                                 #
    # ------------------------------------------------------------------ #

    def _feature_window(self) -> int:
        return max(self.config['lags']) + 6

    def _features(self, y: pd.Series) -> pd.DataFrame:
        """
        Features available AT TIME t (they use observations up to and including t):
        the last three changes, momentum over each configured look-back, the
        distance from a rolling mean, rolling volatility and calendar position.

        The supervised target is the change from t to t+1, so no feature can
        contain the answer. (An earlier version fed the 1-step change of the
        row into its own target, which made training errors look excellent and
        inference flat.)
        """
        lags = self.config['lags']
        w = max(lags)
        d1 = y.diff()

        f = pd.DataFrame(index=y.index)
        for j in (1, 2, 3):
            f[f'chg_lag_{j}'] = d1.shift(j - 1)
        for k in lags:
            if k > 1:
                f[f'mom_{k}'] = y.diff(k)
        f['dev_from_mean'] = y - y.rolling(w).mean()
        f['vol_w'] = d1.rolling(w).std()
        f['vol_3'] = d1.rolling(3).std()

        f['month'] = y.index.month
        f['quarter'] = y.index.quarter
        if self.frequency == 'daily':
            f['dayofweek'] = y.index.dayofweek
        return f

    def _ml_fit_predict(self, make_model, use_scaler: bool, train: pd.Series, h: int) -> np.ndarray:
        feats = self._features(train)
        target = train.diff().shift(-1)           # change from t to t+1

        frame = feats.assign(__target=target).dropna()
        if len(frame) < 30:
            raise ValueError(f'only {len(frame)} usable training rows')

        X = frame.drop(columns='__target').astype(float)
        y = frame['__target'].astype(float)
        cols = list(X.columns)

        scaler = StandardScaler().fit(X) if use_scaler else None
        X_fit = pd.DataFrame(scaler.transform(X), columns=cols, index=X.index) if scaler else X

        model = make_model()
        model.fit(X_fit, y)

        # Recursive multi-step forecast; the calendar advances with each step.
        cap = 5.0 * float(train.diff().std())
        future_idx = pd.date_range(train.index[-1], periods=h + 1, freq=self.pd_freq)[1:]
        ext = train.copy()
        window = self._feature_window()
        out = []

        for i in range(h):
            feat = self._features(ext.iloc[-window:]).iloc[[-1]][cols].astype(float)
            feat = feat.fillna(0.0)
            if scaler is not None:
                feat = pd.DataFrame(scaler.transform(feat), columns=cols)
            step = float(model.predict(feat)[0])
            if np.isfinite(cap) and cap > 0:
                step = float(np.clip(step, -cap, cap))
            new_val = float(ext.iloc[-1]) + step
            out.append(new_val)
            ext = pd.concat([ext, pd.Series([new_val], index=[future_idx[i]])])

        return np.asarray(out, dtype=float)

    def _m_ridge(self, train, h):
        return self._ml_fit_predict(lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]), True, train, h)

    def _m_lasso(self, train, h):
        return self._ml_fit_predict(
            lambda: LassoCV(alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], max_iter=5000), True, train, h)

    def _m_gradient_boosting(self, train, h):
        return self._ml_fit_predict(
            lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, random_state=42), False, train, h)

    def _m_random_forest(self, train, h):
        return self._ml_fit_predict(
            lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=2),
            False, train, h)

    def _m_xgboost(self, train, h):
        return self._ml_fit_predict(
            lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42, n_jobs=2), False, train, h)

    # ------------------------------------------------------------------ #
    # GARCH — volatility only (not an ensemble price candidate)            #
    # ------------------------------------------------------------------ #

    def forecast_garch(self) -> dict:
        """
        GARCH(1,1) with Student-t innovations on log returns. Used for an
        ALTERNATIVE 90% volatility band; it is not ranked or ensembled as a
        price model (its mean equation is close to a random walk by design).
        """
        method = 'GARCH(1,1)'
        try:
            n = len(self.series)
            if self.frequency not in ('daily', 'weekly') or n < self._GARCH_MIN_OBSERVATIONS:
                return {'error': f'GARCH needs daily/weekly data with >= {self._GARCH_MIN_OBSERVATIONS} '
                                 f'observations (have {n}, {self.frequency})', 'method': method}

            if self.use_log:
                returns = np.log(self.series).diff()
            else:
                returns = self.series.pct_change()
            returns = returns.dropna() * 100
            returns = returns.iloc[-2000:]

            if returns.std() < 1e-8:
                return {'error': 'Return series has effectively zero variance', 'method': method}

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = arch_model(returns, vol='Garch', p=1, q=1, dist='t', mean='AR', lags=1).fit(
                    disp='off', show_warning=False)
                fc = fit.forecast(horizon=self.horizon, reindex=False)

            mean_rets = fc.mean.iloc[-1].values / 100
            vols = np.sqrt(fc.variance.iloc[-1].values) / 100
            cum = np.sqrt(np.cumsum(vols ** 2))

            price = float(self.series.iloc[-1])
            preds = []
            for r in mean_rets:
                price = price * float(np.exp(r))
                preds.append(price)
            preds = np.asarray(preds)

            if not np.all(np.isfinite(preds)):
                raise ValueError('GARCH produced non-finite forecast values')

            return {
                'method':              method,
                'role':                'volatility',
                'predictions':         [float(p) for p in preds],
                'metrics':             {},
                'confidence':          'n/a',
                'volatility_forecast': [float(v) for v in vols],
                'band': {
                    'level': 0.90,
                    'lower': [float(p * np.exp(-_Z_GARCH_90 * c)) for p, c in zip(preds, cum)],
                    'upper': [float(p * np.exp(_Z_GARCH_90 * c)) for p, c in zip(preds, cum)],
                },
            }
        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': method}

    # ------------------------------------------------------------------ #
    # Backtest harness                                                     #
    # ------------------------------------------------------------------ #

    def _run_model(self, key: str, default_label: str, fn, role: str = 'model') -> dict:
        """
        Evaluate one model with the rolling-origin backtest, then fit it on the
        full history for the real forecast.
        """
        try:
            h = self.horizon
            origins = self._origins()
            if len(origins) < self.MIN_FOLDS:
                raise ValueError(f'only {len(origins)} backtest folds available')

            actual, pred, base = [], [], []
            for o in origins:
                p = self._inv(np.asarray(fn(self._fwd(self._train_slice(o)), h), dtype=float))
                if p.shape != (h,) or not np.all(np.isfinite(p)):
                    raise ValueError('non-finite or mis-shaped backtest forecast')
                pred.append(p)
                actual.append(self.series.values[o:o + h])
                base.append(np.repeat(float(self.series.iloc[o - 1]), h))

            actual, pred, base = np.array(actual), np.array(pred), np.array(base)
            metrics = self.calculate_metrics(actual, pred, scale=self._mase_scale(origins[-1]), base=base)
            if key == self.BASELINE_KEY:
                metrics['directional_accuracy'] = None       # a flat forecast has no direction

            final = self._inv(np.asarray(fn(self._fwd(self._train_slice(len(self.series))), h), dtype=float))
            if not self.allow_negative_prices:
                final = np.maximum(final, 0.0)
            if final.shape != (h,) or not np.all(np.isfinite(final)):
                raise ValueError('non-finite forecast values')

            return {
                'method':      self._labels.get(key, default_label),
                'role':        role,
                'predictions': [float(x) for x in final],
                'metrics':     metrics,
                'confidence':  self.assign_confidence(metrics),
                'n_folds':     len(origins),
                'backtest': {
                    'origins': [int(o) for o in origins],
                    'actual':  actual.tolist(),
                    'pred':    pred.tolist(),
                    'base':    [float(b[0]) for b in base],
                },
            }
        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(),
                    'method': self._labels.get(key, default_label)}

    # ------------------------------------------------------------------ #
    # Model pool & orchestration                                           #
    # ------------------------------------------------------------------ #

    def _get_model_pool(self) -> dict:
        """
        Select the models that make sense for THIS series type.

        Always            Naive (the baseline everything is judged against),
                          Drift, damped-trend ETS, Theta, ARIMA.
        Seasonal series   Seasonal naive (>= 2 seasons of data); seasonal ETS and
                          SARIMA for monthly/quarterly data (>= 3 seasons).
                          Weekly data has a 52-period season, too long for
                          SARIMA/ETS to be reliable, so only the seasonal naive
                          benchmark is used; daily prices have no useful season.
        Supervised ML     Ridge from 60 observations; LASSO and the tree models
                          only with long histories (daily/weekly), where they
                          have enough rows not to overfit.
        GARCH             Daily/weekly with long history; volatility band only.

        Returns {key: zero-argument callable} ready for concurrent execution.
        """
        n = len(self.series)
        m = self.config['season']

        spec: list[tuple[str, str, object, str]] = [
            ('naive', 'Naive (last value)',         self._m_naive,  'baseline'),
            ('drift', 'Drift',                      self._m_drift,  'benchmark'),
            ('ets',   'ETS (damped trend)',         self._m_ets,    'model'),
            ('theta', 'Theta',                      self._m_theta,  'model'),
            ('arima', 'ARIMA',                      self._m_arima,  'model'),
        ]

        if m and n >= 2 * m:
            spec.append(('seasonal_naive', 'Seasonal naive', self._m_seasonal_naive, 'benchmark'))
        if m and m <= 12 and n >= 3 * m:
            spec.append(('ets_seasonal', 'ETS (damped + seasonal)', self._m_ets_seasonal, 'model'))
            spec.append(('sarima',       'SARIMA',                  self._m_sarima,       'model'))

        if n >= self._RIDGE_MIN_OBSERVATIONS:
            spec.append(('ridge', 'Ridge Regression', self._m_ridge, 'model'))
        if n >= self._ML_MIN_OBSERVATIONS:
            spec += [
                ('lasso',             'LASSO Regression',  self._m_lasso,             'model'),
                ('gradient_boosting', 'Gradient Boosting', self._m_gradient_boosting, 'model'),
                ('random_forest',     'Random Forest',     self._m_random_forest,     'model'),
                ('xgboost',           'XGBoost',           self._m_xgboost,           'model'),
            ]

        pool = {
            key: (lambda k=key, lbl=label, f=fn, r=role: self._run_model(k, lbl, f, r))
            for key, label, fn, role in spec
        }

        if self.frequency in ('daily', 'weekly') and n >= self._GARCH_MIN_OBSERVATIONS:
            pool['garch'] = self.forecast_garch

        return pool

    def forecast_all_models(self) -> dict:
        """
        Run every applicable model concurrently and return {key: result}.

        After all models finish, each result gets its error RELATIVE to the
        naive baseline (``rmae``) and a confidence label derived from it.
        """
        is_valid, message = self.validate_data()
        if not is_valid:
            return {'error': message}

        pool = self._get_model_pool()
        results: dict = {}

        with ThreadPoolExecutor(max_workers=min(len(pool), 6)) as executor:
            future_to_name = {executor.submit(func): name for name, func in pool.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {'error': str(e), 'trace': traceback.format_exc(), 'method': name}

        self._attach_relative_metrics(results)
        return results

    def _attach_relative_metrics(self, results: dict) -> None:
        """Add rMAE / skill vs naive to every scored model and refresh its confidence."""
        naive = results.get(self.BASELINE_KEY, {})
        naive_mae = (naive.get('metrics') or {}).get('mae')

        for res in results.values():
            metrics = res.get('metrics')
            if not metrics or 'mae' not in metrics:
                continue
            if naive_mae and np.isfinite(naive_mae) and naive_mae > 0:
                metrics['rmae'] = float(metrics['mae'] / naive_mae)
                metrics['skill_vs_naive'] = float(1.0 - metrics['rmae'])
                res['confidence'] = self.assign_confidence(metrics)

    # ------------------------------------------------------------------ #
    # Ensemble                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rank_value(result: dict) -> float:
        """Lower is better: rMAE, else MASE, else MAPE as a fraction."""
        m = result['metrics']
        for k in ('rmae', 'mase'):
            v = m.get(k)
            if v is not None and np.isfinite(v):
                return float(v)
        return float(m['mape']) / 100

    def create_ensemble(self, results: dict) -> dict | None:
        """
        Combine the three best models into a weighted forecast, with prediction
        intervals and its own backtest metrics.

        * Candidates are ranked by rMAE (error relative to naive). The naive
          baseline is itself a candidate: when nothing beats it, the ensemble
          shrinks toward "the last value", which is the honest answer.
        * Weights are proportional to 1 / rank_value^2.
        * A volatility-aware sanity filter drops models whose forecast jumps
          implausibly far from the last known price.
        * The ensemble's backtest is rebuilt from the members' fold-by-fold
          forecasts with the same weights, so it carries real, comparable
          metrics (MAE, rMAE, skill vs naive, interval coverage).
        """
        h = self.horizon
        last_known = float(self.series.iloc[-1])

        hist_std = self.series.pct_change().std()
        adaptive_threshold = 2 * hist_std if pd.notna(hist_std) and hist_std > 0 else 0.40

        def is_valid(v):
            if 'predictions' not in v or 'metrics' not in v:
                return False
            preds = v['predictions']
            if not preds or not all(np.isfinite(p) for p in preds):
                return False
            if not np.isfinite(v['metrics'].get('mape', np.inf)):
                return False
            for k, p in enumerate(preds, start=1):
                change = abs(p - last_known) / last_known if last_known else 1
                limit = adaptive_threshold if k == 1 else adaptive_threshold * np.sqrt(k) * 2
                if change > limit:
                    return False
            return True

        # GARCH is a volatility model; it never competes as a price forecast.
        candidates = {k: v for k, v in results.items() if k != 'garch'}
        successful = {k: v for k, v in candidates.items() if is_valid(v)}
        if not successful:
            return None

        ranked = sorted(successful.items(), key=lambda kv: self._rank_value(kv[1]))
        top = ranked[:3]

        scores = np.array([max(self._rank_value(r), 0.05) for _, r in top])
        weights = (1.0 / scores ** 2)
        weights = weights / weights.sum()

        preds = np.zeros(h)
        for (_, res), w in zip(top, weights):
            preds += np.array(res['predictions'][:h]) * w

        avg_mape = float(np.mean([r['metrics']['mape'] for _, r in top]))

        # ----- intervals (final forecast) -----
        cone = self._cone(self.series, h)
        intervals = {}
        for level in INTERVAL_LEVELS:
            lo, hi = self._interval(preds, cone, level)
            if not self.allow_negative_prices:
                lo = np.maximum(lo, 0.0)
            intervals[str(int(level * 100))] = {'lower': lo.tolist(), 'upper': hi.tolist()}

        # ----- ensemble backtest (needs fold-by-fold forecasts from every member) -----
        ens_metrics = self._ensemble_backtest(top, weights, results)

        if ens_metrics is None:
            # No fold-level data (e.g. results injected by hand): summarise from members.
            ens_metrics = {'mape': avg_mape}
            confidence = self.assign_confidence({'mape': avg_mape})
        else:
            confidence = self.assign_confidence(ens_metrics)

        out = {
            'method':      'Weighted Ensemble',
            'predictions': preds.tolist(),
            'top_models':  [k for k, _ in top],
            'weights':     {k: float(w) for (k, _), w in zip(top, weights)},
            'confidence':  confidence,
            'avg_mape':    avg_mape,
            'metrics':     ens_metrics,
            'intervals':   intervals,
        }

        naive = results.get(self.BASELINE_KEY)
        if naive and 'predictions' in naive:
            out['baseline'] = {
                'method':      naive['method'],
                'predictions': naive['predictions'],
                'metrics':     naive.get('metrics', {}),
            }
        return out

    def _ensemble_backtest(self, top, weights, results) -> dict | None:
        """Backtest metrics for the weighted ensemble, or None when fold data is missing."""
        bts = [res.get('backtest') for _, res in top]
        if not bts or any(b is None for b in bts):
            return None
        origins = bts[0]['origins']
        if any(b['origins'] != origins for b in bts):
            return None

        actual = np.array(bts[0]['actual'], dtype=float)
        base = np.repeat(np.array(bts[0]['base'], dtype=float)[:, None], self.horizon, axis=1)
        pred = sum(np.array(b['pred'], dtype=float) * w for b, w in zip(bts, weights))

        metrics = self.calculate_metrics(actual, pred, scale=self._mase_scale(origins[-1]), base=base)
        metrics['n_folds'] = len(origins)

        naive_mae = ((results.get(self.BASELINE_KEY) or {}).get('metrics') or {}).get('mae')
        if naive_mae and np.isfinite(naive_mae) and naive_mae > 0:
            metrics['rmae'] = float(metrics['mae'] / naive_mae)
            metrics['skill_vs_naive'] = float(1.0 - metrics['rmae'])

        # Empirical coverage of the intervals: were the real values inside them?
        for level in INTERVAL_LEVELS:
            hits, total = 0, 0
            for f, o in enumerate(origins):
                cone = self._cone(self.series.iloc[:o], self.horizon)
                lo, hi = self._interval(pred[f], cone, level)
                inside = (actual[f] >= lo) & (actual[f] <= hi)
                hits += int(inside.sum())
                total += inside.size
            metrics[f'coverage_{int(level * 100)}'] = float(hits / total) if total else None

        return metrics
