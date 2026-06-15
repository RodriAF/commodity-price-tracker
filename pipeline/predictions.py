"""
Multi-Horizon Forecasting System

Generates price forecasts for key agricultural commodity series using an ensemble
of statistical, econometric, and machine learning models.

Model suite:
    Statistical / econometric:
        ARIMA         — AutoRegressive Integrated Moving Average
        SARIMA        — Seasonal ARIMA (handles periodic cycles)
        Exponential Smoothing — Holt's trend-adjusted exponential smoothing
        GARCH(1,1)    — Generalised AutoRegressive Conditional Heteroskedasticity

    Machine learning (lag-feature based):
        Gradient Boosting, Random Forest, XGBoost — tree-based ensemble methods
        Ridge Regression, LASSO                   — regularised linear models

    Combination:
        Weighted Ensemble — exponential-error-weighted average of the top 3 models
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from pmdarima import auto_arima
from arch import arch_model

# Import configuration loader to extract JSON parameters
try:
    from config.config_loader import ConfigLoader
except ImportError:
    ConfigLoader = None


class CommoditiesForecaster:
    """
    Forecasting engine to predict future price actions based on historical data.
    """

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, data: pd.DataFrame, commodity: str, frequency: str = 'monthly', horizon: int | None = None):
        self.data      = data
        self.commodity = commodity
        self.frequency = frequency

        # Pandas date offset aliases used for resampling and date generation
        self.freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'quarterly': 'QS'}
        self.pd_freq  = self.freq_map.get(frequency, 'MS')

        # Default forecast horizon if not explicitly set: calibrated per frequency
        self.horizon_map = {
            'daily':     10,   # ~2 trading weeks
            'weekly':    10,   # ~2.5 months
            'monthly':   5,    # ~5 months
            'quarterly': 4     # ~1 year
        }
        self.horizon = horizon if horizon is not None else self.horizon_map.get(frequency, 5)

        # Extract and regularise the target series
        if 'date' in data.columns:
            self.series = data.set_index('date')[commodity].dropna()
            self.series.index = pd.to_datetime(self.series.index)
            if getattr(self.series.index, 'freq', None) is None:
                self.series = self.series.asfreq(self.pd_freq).ffill()
        else:
            self.series = data[commodity].dropna()

        # Frequency-specific configuration for validation, test sizing, and lag selection
        self.freq_configs = {
            'daily':     {'min_points': 90,  'test_size': 14, 'lags': [1, 7, 30]},
            'weekly':    {'min_points': 52,  'test_size': 10, 'lags': [1, 4, 12]},
            'monthly':   {'min_points': 24,  'test_size': 6,  'lags': [1, 3, 6, 12]},
            'quarterly': {'min_points': 12,  'test_size': 4,  'lags': [1, 2, 4]}
        }
        self.config = self.freq_configs.get(frequency, self.freq_configs['monthly'])

        # Resolve commodity-specific settings from JSON config
        self.allow_negative_prices = False
        if ConfigLoader is not None:
            try:
                info = ConfigLoader.get_commodity_info(self.commodity)
                self.allow_negative_prices = info.get('allow_negative_prices', False)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Data Validation & Processing                                         #
    # ------------------------------------------------------------------ #

    def validate_data(self) -> tuple[bool, str]:
        """
        Check that sufficient observations exist to train models and hold out a test set.
        """
        min_required = self.config['min_points'] + max(self.horizon, self.config['test_size'])
        if len(self.series) < min_required:
            return False, f"Insufficient data: {len(self.series)} points (minimum required: {min_required})"
        return True, "OK"

    def train_test_split(self) -> tuple[pd.Series, pd.Series]:
        """
        Split the series into a training set and a held-out test set via walk-forward split.
        """
        test_size = self.config['test_size']
        return self.series[:-test_size], self.series[-test_size:]

    # ------------------------------------------------------------------ #
    # Metrics & Evaluation                                                 #
    # ------------------------------------------------------------------ #

    def calculate_metrics(self, actual, predicted) -> dict:
        """
        Compute a comprehensive set of forecast error metrics on the test set.
        """
        actual    = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)

        mae  = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        mape = float(np.mean(
            np.abs((actual - predicted) / np.where(actual == 0, 1, actual))
        ) * 100)

        smape = float(np.mean(
            2 * np.abs(actual - predicted) /
            (np.abs(actual) + np.abs(predicted) + 1e-8)
        ) * 100)

        naive_error = float(np.mean(np.abs(np.diff(actual))))
        mase        = mae / naive_error if naive_error > 1e-8 else np.nan

        mbe = float(np.mean(predicted - actual))

        if len(actual) > 1:
            actual_dir           = np.sign(np.diff(actual))
            pred_dir             = np.sign(np.diff(predicted))
            directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
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
        Assign a qualitative confidence label based on out-of-sample error.
        """
        if isinstance(metrics_or_mape, dict):
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
        if volatility == 0:
            return 'low'
        
        score = mape / volatility
        if score < 0.5:
            return 'high'
        elif score < 1.0:
            return 'medium'
        else:
            return 'low'

    # ------------------------------------------------------------------ #
    # Feature Engineering                                                  #
    # ------------------------------------------------------------------ #

    def create_lag_features(self, series: pd.Series, lags: list) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a supervised learning design matrix from the price series.
        Excludes dayofweek for monthly and quarterly frequencies.
        """
        df = pd.DataFrame({'y': series})
        df['target'] = df['y'].diff()

        for lag in lags:
            df[f'lag_{lag}']  = df['y'].shift(lag)
            df[f'diff_{lag}'] = df['y'].diff(lag)

        df['rolling_mean_3'] = df['y'].rolling(3).mean()
        df['rolling_std_3']  = df['y'].rolling(3).std()

        if isinstance(series.index, pd.DatetimeIndex):
            df['month']   = series.index.month
            df['quarter'] = series.index.quarter
            # Adaptive feature exclusion: discard dayofweek for low frequencies
            if self.frequency not in ['monthly', 'quarterly']:
                df['dayofweek'] = series.index.dayofweek

        df = df.dropna()

        X = df.drop(['y', 'target'], axis=1)
        y = df['target']

        return X, y

    def _get_last_features(self, values: list, lags: list, feature_columns: list) -> pd.DataFrame:
        """
        Construct the feature vector for the next forecast step using recent history.
        """
        v = np.array(values)

        feat = {}
        for lag in lags:
            feat[f'lag_{lag}']  = v[-lag]
            feat[f'diff_{lag}'] = v[-1] - v[-lag]

        feat['rolling_mean_3'] = np.mean(v[-3:])
        feat['rolling_std_3']  = np.std(v[-3:])

        last_date = (
            self.series.index[-1]
            if isinstance(self.series.index, pd.DatetimeIndex)
            else pd.Timestamp.today()
        )
        feat['month']   = last_date.month
        feat['quarter'] = last_date.quarter
        
        if self.frequency not in ['monthly', 'quarterly']:
            feat['dayofweek'] = last_date.dayofweek

        return pd.DataFrame([feat]).reindex(columns=feature_columns, fill_value=0)

    # ------------------------------------------------------------------ #
    # Statistical & Econometric Models                                     #
    # ------------------------------------------------------------------ #

    def forecast_arima(self) -> dict:
        """
        AutoRegressive Integrated Moving Average (ARIMA) forecast.
        """
        try:
            train, test = self.train_test_split()

            model = auto_arima(
                train,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None,
                seasonal=False,
                information_criterion='aic',
                stepwise=True,
                error_action='ignore',
                suppress_warnings=True,
            )

            test_pred = model.predict(len(test))
            metrics   = self.calculate_metrics(test.values, test_pred)
            forecast  = model.predict(self.horizon)

            return {
                'method':      f'ARIMA{model.order}',
                'predictions': [float(x) for x in forecast],
                'metrics':     metrics,
                'confidence':  self.assign_confidence(metrics),
                'order':       model.order
            }

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': 'ARIMA'}

    def forecast_sarima(self) -> dict:
        """
        Seasonal ARIMA (SARIMA) forecast. Falls back transparency to ARIMA 
        if seasonal observations are insufficient.
        """
        try:
            train, test = self.train_test_split()
            s = {'daily': 7, 'weekly': 52, 'monthly': 12, 'quarterly': 4}.get(self.frequency, 12)

            if len(train) < 2 * s:
                # Transparent fallback handler
                fallback_res = self.forecast_arima()
                if 'method' in fallback_res:
                    fallback_res['method'] = fallback_res['method'].replace('ARIMA', 'SARIMA (fallback to ARIMA') + ')'
                return fallback_res

            model = auto_arima(
                train,
                start_p=0, max_p=2,
                start_q=0, max_q=2,
                d=None,
                seasonal=True,
                m=s,
                start_P=0, max_P=1,
                start_Q=0, max_Q=1,
                D=None,
                information_criterion='aic',
                stepwise=True,
                error_action='ignore',
                suppress_warnings=True,
            )

            test_pred = model.predict(len(test))
            metrics   = self.calculate_metrics(test.values, test_pred)
            forecast  = model.predict(self.horizon)

            return {
                'method':         f'SARIMA{model.order}x{model.seasonal_order}',
                'predictions':    [float(x) for x in forecast],
                'metrics':        metrics,
                'confidence':     self.assign_confidence(metrics),
                'order':          model.order,
                'seasonal_order': model.seasonal_order
            }

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': 'SARIMA'}

    def forecast_garch(self) -> dict:
        """
        GARCH(1,1) volatility and price forecast. Handles configurable price baselines.
        """
        try:
            train, test = self.train_test_split()
            min_garch_points = 30
            
            if len(train) < min_garch_points:
                return {
                    'error': f'Insufficient data for GARCH: {len(train)} observations (minimum: {min_garch_points})',
                    'method': 'GARCH(1,1)'
                }

            log_returns_full = np.log(self.series / self.series.shift(1)).dropna() * 100

            if log_returns_full.std() < 1e-8:
                return {
                    'error': 'Log-return series has effectively zero variance — series appears constant',
                    'method': 'GARCH(1,1)'
                }

            n_test           = len(test)
            n_full           = len(log_returns_full)
            test_preds_price = []

            for i in range(n_test):
                cutoff  = n_full - n_test + i
                history = log_returns_full.iloc[:cutoff]

                if len(history) < min_garch_points:
                    test_preds_price.append(float(train.iloc[-1]))
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    g  = arch_model(history, vol='Garch', p=1, q=1, dist='t', mean='AR', lags=1)
                    f  = g.fit(disp='off', show_warning=False)
                    fc = f.forecast(horizon=1, reindex=False)

                mean_ret   = fc.mean.iloc[-1, 0] / 100
                base_price = float(self.series.iloc[-(n_test - i + 1)])
                test_preds_price.append(float(base_price * np.exp(mean_ret)))

            metrics = self.calculate_metrics(test.values, np.array(test_preds_price))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                garch_final  = arch_model(log_returns_full, vol='Garch', p=1, q=1, dist='t', mean='AR', lags=1)
                fitted_final = garch_final.fit(disp='off', show_warning=False)
                fc_future    = fitted_final.forecast(horizon=self.horizon, reindex=False)

            mean_rets = fc_future.mean.iloc[-1].values / 100
            vol_pcts  = np.sqrt(fc_future.variance.iloc[-1].values) / 100

            price       = float(self.series.iloc[-1])
            predictions = []
            vols        = []
            
            for r, v in zip(mean_rets, vol_pcts):
                price = price * np.exp(r)
                pred_price = price if self.allow_negative_prices else max(0, price)
                predictions.append(float(pred_price))
                vols.append(float(v))

            if not all(np.isfinite(p) for p in predictions):
                raise ValueError("GARCH produced non-finite forecast values")

            return {
                'method':              'GARCH(1,1)',
                'predictions':         predictions,
                'metrics':             metrics,
                'confidence':          self.assign_confidence(metrics),
                'volatility_forecast': vols
            }

        except Exception as e:
            return {
                'error':  str(e),
                'trace':  traceback.format_exc(),
                'method': 'GARCH(1,1)'
            }

    def forecast_exponential_smoothing(self) -> dict:
        """
        Holt's exponential smoothing with additive trend.
        """
        try:
            train, test = self.train_test_split()
            model = None
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                try:
                    fit = ExponentialSmoothing(
                        train, trend='add',
                        initialization_method='estimated'
                    ).fit(optimized=True, remove_bias=True,
                          minimize_kwargs={'options': {'maxiter': 200}})
                    if all(np.isfinite(f) for f in fit.forecast(self.horizon)):
                        model = fit
                except Exception:
                    pass

            if model is None:
                model = SimpleExpSmoothing(train).fit(
                    smoothing_level=0.3,
                    optimized=False
                )

            test_pred = model.forecast(len(test))
            forecast  = model.forecast(self.horizon)

            if not all(np.isfinite(f) for f in forecast):
                raise ValueError("Forecast contains non-finite values after all fallback attempts")

            metrics = self.calculate_metrics(test.values, test_pred)
            return {
                'method':      'Exponential Smoothing',
                'predictions': [float(x) for x in forecast],
                'metrics':     metrics,
                'confidence':  self.assign_confidence(metrics)
            }

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': 'Exponential Smoothing'}

    # ------------------------------------------------------------------ #
    # Machine Learning Models                                              #
    # ------------------------------------------------------------------ #

    def _forecast_ml_model(self, model, name: str, use_scaler: bool = False) -> dict:
        """
        Shared forecasting loop for all supervised ML models.
        """
        try:
            lags  = self.config['lags']
            train, test = self.train_test_split()

            X_train, y_train = self.create_lag_features(train, lags)
            feature_columns  = list(X_train.columns)

            combined       = pd.concat([train.iloc[-max(lags):], test])
            X_test_full, _ = self.create_lag_features(combined, lags)
            X_test         = X_test_full[-len(test):].reindex(columns=feature_columns, fill_value=0)

            scaler = StandardScaler()

            if use_scaler:
                X_train_fit = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
                X_test_fit  = pd.DataFrame(scaler.transform(X_test),      columns=feature_columns)
            else:
                X_train_fit = X_train
                X_test_fit  = X_test

            model.fit(X_train_fit, y_train)

            last_train_price = train.iloc[-1]
            preds_diff       = model.predict(X_test_fit)
            preds            = last_train_price + np.cumsum(preds_diff)

            metrics = self.calculate_metrics(test.values, preds)

            predictions = []
            temp_series = list(self.series.values)
            current     = temp_series[-1]

            for _ in range(self.horizon):
                feat = self._get_last_features(temp_series, lags, feature_columns)

                if use_scaler:
                    feat = pd.DataFrame(scaler.transform(feat), columns=feature_columns)

                diff    = model.predict(feat)[0]
                new_val = current + diff
                
                # Configurable negative price ceiling
                new_val = new_val if self.allow_negative_prices else max(0, new_val)

                predictions.append(float(new_val))
                temp_series.append(new_val)
                current = new_val

            return {
                'method':      name,
                'predictions': predictions,
                'metrics':     metrics,
                'confidence':  self.assign_confidence(metrics)
            }

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': name}

    def forecast_gradient_boosting(self) -> dict:
        return self._forecast_ml_model(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting"
        )

    def forecast_random_forest(self) -> dict:
        return self._forecast_ml_model(
            RandomForestRegressor(n_estimators=100, random_state=42),
            "Random Forest"
        )

    def forecast_ridge(self) -> dict:
        return self._forecast_ml_model(
            RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
            "Ridge Regression", use_scaler=True
        )

    def forecast_lasso(self) -> dict:
        return self._forecast_ml_model(
            LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], max_iter=5000),
            "LASSO Regression", use_scaler=True
        )

    def forecast_xgboost(self) -> dict:
        return self._forecast_ml_model(
            XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            "XGBoost"
        )

    # ------------------------------------------------------------------ #
    # Orchestration & Ensemble Construction                                #
    # ------------------------------------------------------------------ #

    # Minimum training observations required to include supervised ML models.
    # The rule of thumb is: train set must have at least 10x the number of lag
    # features (~13 features) after the test set is removed. Below this threshold
    # tree-based models will overfit the training set and produce unreliable MAPE
    # estimates on the small hold-out test, which in turn contaminates ensemble weights.
    _ML_MIN_OBSERVATIONS = 150

    def _get_model_pool(self) -> dict:
        """
        Select the set of models to run based on data frequency and series length.

        Rationale:
            Supervised ML models (tree ensembles, LASSO) require a training set
            that is materially larger than the feature space to avoid overfitting.
            With monthly or quarterly data, the available observations rarely meet
            this threshold, so only the statistical models (ARIMA, SARIMA,
            Exponential Smoothing) and Ridge (linear, strongly regularised) are
            included. With daily or weekly data and sufficient history, the full
            suite is appropriate.

            GARCH is included in the pool for all frequencies because its output
            is used for the volatility confidence band on the dashboard, not for
            the ensemble price prediction. It is excluded from ensemble candidates
            in create_ensemble() regardless of its MAPE.

        Returns:
            dict mapping model key -> bound method, ready for concurrent execution.
        """
        n = len(self.series)

        # Statistical core: valid for all frequencies and sample sizes
        pool = {
            'arima':         self.forecast_arima,
            'sarima':        self.forecast_sarima,
            'exp_smoothing': self.forecast_exponential_smoothing,
        }

        # GARCH: always included for volatility band output; excluded from
        # ensemble price candidates in create_ensemble()
        if n >= 30:
            pool['garch'] = self.forecast_garch

        # Ridge: linear and strongly regularised — works reliably with moderate
        # sample sizes; included once there is enough data to estimate a test set
        if n >= 60:
            pool['ridge'] = self.forecast_ridge

        # Supervised ML models: only when the training set is large enough to
        # avoid overfitting across ~13 lag and calendar features
        if n >= self._ML_MIN_OBSERVATIONS:
            pool['gradient_boosting'] = self.forecast_gradient_boosting
            pool['random_forest']     = self.forecast_random_forest
            pool['xgboost']           = self.forecast_xgboost
            pool['lasso']             = self.forecast_lasso

        return pool

    def forecast_all_models(self) -> dict:
        """
        Run every applicable model concurrently and return their results as a dict.

        The model pool is determined dynamically by _get_model_pool() based on
        data frequency and series length. Parallel execution uses ThreadPoolExecutor
        with a bounded worker count to avoid saturating CPU on multi-commodity runs.
        """
        is_valid, message = self.validate_data()

        if not is_valid:
            return {'error': message}

        models_to_run = self._get_model_pool()

        results = {}

        # Bound the worker count: prevents CPU saturation when forecasting
        # multiple commodities concurrently from the pipeline orchestrator
        max_workers = min(len(models_to_run), 6)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(func): name for name, func in models_to_run.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {'error': str(e), 'trace': traceback.format_exc(), 'method': name}

        return results

    def create_ensemble(self, results: dict) -> dict | None:
        """
        Combine the top-performing models into a weighted ensemble forecast using 
        an adaptive historical volatility filter.
        """
        last_known = float(self.series.iloc[-1])
        
        # Adaptive Threshold Generation: 2x Historical Standard Deviation
        hist_std = self.series.pct_change().std()
        adaptive_threshold = 2 * hist_std if pd.notna(hist_std) else 0.40

        def is_valid(v):
            if 'predictions' not in v:
                return False
            preds = v['predictions']
            if not preds or not all(np.isfinite(p) for p in preds):
                return False
            if not np.isfinite(v['metrics'].get('mape', np.inf)):
                return False
                
            first_change = abs(preds[0] - last_known) / last_known if last_known else 1
            if first_change > adaptive_threshold:
                return False
            return True

        # GARCH is excluded from ensemble price candidates: its mean equation
        # produces near-zero return forecasts by design (modelling volatility,
        # not price direction). Including it biases ensemble predictions toward
        # the last known price and dilutes the signal from the other models.
        # GARCH output is still available in results for the volatility band on
        # the dashboard.
        price_candidates = {k: v for k, v in results.items() if k != 'garch'}
        successful = {k: v for k, v in price_candidates.items() if is_valid(v)}

        if not successful:
            return None

        def rank_key(item):
            m    = item[1]['metrics']
            mase = m.get('mase')
            return mase if (mase is not None and not np.isnan(mase)) else m['mape'] / 100

        ranked     = sorted(successful.items(), key=rank_key)
        top_models = ranked[:3]

        errors  = np.array([res['metrics']['mape'] for _, res in top_models])
        weights = np.exp(-errors) / np.sum(np.exp(-errors))

        ensemble_preds = np.zeros(self.horizon)
        for (name, res), w in zip(top_models, weights):
            ensemble_preds += np.array(res['predictions'][:self.horizon]) * w

        avg_mape = float(np.mean(errors))

        return {
            'method':      'Weighted Ensemble',
            'predictions': ensemble_preds.tolist(),
            'top_models':  [x[0] for x in top_models],
            'confidence':  self.assign_confidence({'mape': avg_mape}),
            'avg_mape':    avg_mape
        }