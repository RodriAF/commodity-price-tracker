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

All models are evaluated on a held-out test set sized to their data frequency.
The ensemble selects and weights models by their out-of-sample MASE or MAPE.
"""

import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import traceback

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


class CommoditiesForecaster:

    def __init__(self, data, commodity, frequency='monthly', horizon=None):
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
            if self.series.index.freq is None:
                # Assign explicit frequency and forward-fill any remaining gaps
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

    # ── Data validation ───────────────────────────────────────────────────────

    def validate_data(self):
        """
        Check that sufficient observations exist to train models and hold out a test set.

        The minimum required is min_points + max(horizon, test_size) to guarantee
        that both the training window and the evaluation window are fully populated.
        Attempting to fit ARIMA or tree models on very short series produces
        unreliable parameter estimates and artificially low error metrics.
        """
        min_required = self.config['min_points'] + max(self.horizon, self.config['test_size'])
        if len(self.series) < min_required:
            return False, f"Insufficient data: {len(self.series)} points (minimum required: {min_required})"
        return True, "OK"

    def train_test_split(self):
        """
        Split the series into a training set and a held-out test set.

        The test set size is fixed per frequency (see freq_configs). This
        implements a simple walk-forward split — the test set is always the
        most recent observations, mimicking the real forecasting scenario where
        only past data is available at prediction time.
        """
        test_size = self.config['test_size']
        return self.series[:-test_size], self.series[-test_size:]

    # ── Error metrics ─────────────────────────────────────────────────────────

    def calculate_metrics(self, actual, predicted):
        """
        Compute a comprehensive set of forecast error metrics on the test set.

        Metrics:
            MAE   — Mean Absolute Error: average magnitude of errors, same units as price.
            MAPE  — Mean Absolute Percentage Error: scale-free, but undefined at zero.
            RMSE  — Root Mean Squared Error: penalises large errors more than MAE.
            sMAPE — Symmetric MAPE: bounded [0, 200%], more robust than MAPE near zero.
            MASE  — Mean Absolute Scaled Error: compares model to a naive random-walk
                    baseline (mean of first differences). MASE < 1 means the model
                    outperforms the naive forecast. This is the primary ranking metric
                    for ensemble selection because it is scale-free and defined at zero.
            MBE   — Mean Bias Error: signed average error, indicates systematic over-
                    or under-prediction.
            Directional Accuracy — percentage of periods where the predicted direction
                    of change matches the actual direction. Relevant for hedging decisions.
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

        # MASE denominator: mean absolute first difference of the actual series
        # (the expected error of a naive one-step-ahead forecast)
        naive_error = float(np.mean(np.abs(np.diff(actual))))
        mase        = mae / naive_error if naive_error > 1e-8 else np.nan

        mbe = float(np.mean(predicted - actual))

        if len(actual) > 1:
            actual_dir            = np.sign(np.diff(actual))
            pred_dir              = np.sign(np.diff(predicted))
            directional_accuracy  = float(np.mean(actual_dir == pred_dir) * 100)
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

    def assign_confidence(self, metrics_or_mape):
        """
        Assign a qualitative confidence label based on out-of-sample error.

        Primary criterion — MASE (preferred because it is scale-free):
            MASE < 0.8  -> HIGH    (model substantially beats naive baseline)
            MASE < 1.2  -> MEDIUM  (model roughly comparable to naive baseline)
            MASE >= 1.2 -> LOW     (model does not outperform naive forecast)

        Fallback criterion — MAPE normalised by series volatility:
            Used when MASE is unavailable (e.g. constant actuals).
            A score of MAPE / volatility < 0.5 indicates the model error is
            small relative to the natural variability of the series.
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

    # ── Feature engineering ───────────────────────────────────────────────────

    def create_lag_features(self, series, lags):
        """
        Build a supervised learning design matrix from the price series.

        ML models cannot directly consume time series; instead, past values
        are used as input features to predict the next period's change.

        Features created:
            lag_k     — price k periods ago (level feature, captures autocorrelation)
            diff_k    — price change over the last k periods (momentum feature)
            rolling_mean_3 / rolling_std_3 — short-term trend and volatility context
            month, quarter, dayofweek — calendar seasonality features (if DatetimeIndex)

        The target is the first difference of the price series (delta, not level).
        Forecasting differences rather than levels improves stationarity and makes
        the model less sensitive to the absolute price regime.
        """
        df = pd.DataFrame({'y': series})
        df['target'] = df['y'].diff()

        for lag in lags:
            df[f'lag_{lag}']  = df['y'].shift(lag)
            df[f'diff_{lag}'] = df['y'].diff(lag)

        df['rolling_mean_3'] = df['y'].rolling(3).mean()
        df['rolling_std_3']  = df['y'].rolling(3).std()

        if isinstance(series.index, pd.DatetimeIndex):
            df['month']     = series.index.month
            df['quarter']   = series.index.quarter
            df['dayofweek'] = series.index.dayofweek

        df = df.dropna()

        X = df.drop(['y', 'target'], axis=1)
        y = df['target']

        return X, y

    # ── ARIMA ─────────────────────────────────────────────────────────────────

    def forecast_arima(self):
        """
        AutoRegressive Integrated Moving Average (ARIMA) forecast.

        ARIMA(p, d, q) models a time series as a linear function of its own
        past values (AR component, order p), past forecast errors (MA component,
        order q), and accounts for non-stationarity via differencing (order d).

        Order selection is delegated to pmdarima's auto_arima, which searches
        over a grid of (p, d, q) combinations and selects the specification
        with the lowest AIC (Akaike Information Criterion). AIC penalises model
        complexity, preventing overfitting to the training set.

        ARIMA is most reliable for series without strong seasonal patterns and
        with a moderate number of observations. It serves as the baseline
        statistical model in the ensemble.
        """
        try:
            train, test = self.train_test_split()

            model = auto_arima(
                train,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None,                        # let auto_arima determine integration order
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
                'method':     f'ARIMA{model.order}',
                'predictions': [float(x) for x in forecast],
                'metrics':     metrics,
                'confidence':  self.assign_confidence(metrics),
                'order':       model.order
            }

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc(), 'method': 'ARIMA'}

    # ── SARIMA ────────────────────────────────────────────────────────────────

    def forecast_sarima(self):
        """
        Seasonal ARIMA (SARIMA) forecast.

        Extends ARIMA with seasonal AR and MA terms at lag s, where s is the
        number of periods in one seasonal cycle (12 for monthly, 4 for quarterly,
        7 for daily data with a weekly cycle, 52 for weekly data).

        SARIMA(p,d,q)(P,D,Q)[s] adds:
            P — seasonal AR order
            D — seasonal differencing
            Q — seasonal MA order

        Agricultural commodity prices frequently exhibit calendar seasonality
        (harvest cycles, heating season demand for natural gas, etc.). SARIMA
        can capture these periodic patterns while ARIMA cannot.

        Falls back to ARIMA if the training set is shorter than 2 full seasonal
        cycles, since SARIMA parameter estimates become unreliable without
        sufficient seasonal repetitions.
        """
        try:
            train, test = self.train_test_split()

            s = {'daily': 7, 'weekly': 52, 'monthly': 12, 'quarterly': 4}.get(self.frequency, 12)

            if len(train) < 2 * s:
                # Insufficient seasonal history — defer to non-seasonal ARIMA
                return self.forecast_arima()

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

    # ── GARCH ─────────────────────────────────────────────────────────────────

    def forecast_garch(self):
        """
        GARCH(1,1) volatility and price forecast.

        GARCH(1,1) — Generalised AutoRegressive Conditional Heteroskedasticity —
        is the industry standard model for commodity price series where variance
        is not constant over time (a property called heteroskedasticity).

        It captures two empirically well-documented phenomena that ARIMA ignores:

        1. Volatility clustering:
           Periods of high volatility tend to cluster together. A large price
           swing in crude oil (e.g. a supply shock) is typically followed by
           further large movements before markets stabilise. GARCH models this
           by making today's variance a function of yesterday's squared return
           and yesterday's variance:

               sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

           where omega is the long-run variance, alpha captures the sensitivity
           to recent shocks (ARCH effect), and beta controls persistence.

        2. Heavy tails:
           Commodity returns have fatter tails than a normal distribution —
           extreme moves occur more often than Gaussian models predict. Using
           dist='t' (Student-t) in arch_model accounts for this, reducing the
           systematic underestimation of tail risk.

        Implementation:
           - Works on log-returns scaled to percentage (x100), the standard
             input for the arch library.
           - Prices are reconstructed by cumulating predicted log-returns:
               P_t = P_{t-1} * exp(r_t / 100)
           - Test set evaluation uses rolling one-step-ahead re-estimation to
             prevent look-ahead bias: each point in the test window is predicted
             using only data available up to that point.
           - The volatility_forecast output (sigma per period) is the most
             distinctive output of GARCH. The Forecasting page uses it to draw
             90% confidence bands (±1.645 sigma, assuming normally distributed
             returns) around the central price forecast.

        Minimum data requirement: 30 observations of log-returns, below which
        GARCH parameter estimation is unreliable.
        """
        try:
            train, test = self.train_test_split()

            min_garch_points = 30
            if len(train) < min_garch_points:
                return {
                    'error': f'Insufficient data for GARCH: {len(train)} observations (minimum: {min_garch_points})',
                    'method': 'GARCH(1,1)'
                }

            # Log-returns scaled to percentage — arch library works best in this range
            log_returns_full = np.log(self.series / self.series.shift(1)).dropna() * 100

            if log_returns_full.std() < 1e-8:
                return {
                    'error': 'Log-return series has effectively zero variance — series appears constant',
                    'method': 'GARCH(1,1)'
                }

            # Rolling one-step-ahead evaluation on the test set
            n_test              = len(test)
            n_full              = len(log_returns_full)
            test_preds_price    = []

            for i in range(n_test):
                # Use only log-returns observed strictly before this test point
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

            # Final forecast: fit on the full series, project h periods ahead
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                garch_final  = arch_model(log_returns_full, vol='Garch', p=1, q=1, dist='t', mean='AR', lags=1)
                fitted_final = garch_final.fit(disp='off', show_warning=False)
                fc_future    = fitted_final.forecast(horizon=self.horizon, reindex=False)

            mean_rets = fc_future.mean.iloc[-1].values / 100
            vol_pcts  = np.sqrt(fc_future.variance.iloc[-1].values) / 100

            # Reconstruct price path by compounding log-returns
            price       = float(self.series.iloc[-1])
            predictions = []
            vols        = []
            for r, v in zip(mean_rets, vol_pcts):
                price = price * np.exp(r)
                predictions.append(float(max(0, price)))
                vols.append(float(v))

            if not all(np.isfinite(p) for p in predictions):
                raise ValueError("GARCH produced non-finite forecast values")

            return {
                'method':             'GARCH(1,1)',
                'predictions':        predictions,
                'metrics':            metrics,
                'confidence':         self.assign_confidence(metrics),
                # Per-period volatility (sigma): use for confidence bands
                # and risk-adjusted position sizing (higher vol -> smaller position)
                'volatility_forecast': vols
            }

        except Exception as e:
            return {
                'error':  str(e),
                'trace':  traceback.format_exc(),
                'method': 'GARCH(1,1)'
            }

    # ── ML model shared infrastructure ────────────────────────────────────────

    def _forecast_ml_model(self, model, name, use_scaler=False):
        """
        Shared forecasting loop for all supervised ML models.

        All ML models in this suite follow an identical structure:
            1. Create lag features from the training series
            2. Fit the model on training features
            3. Predict on test features (aligned to avoid look-ahead bias)
            4. Evaluate on the held-out test set
            5. Generate the multi-step forecast by iterative one-step-ahead
               prediction — each step appends its prediction to the feature
               window so that the next step can use it as a lag input

        The target variable is the first difference of the price series
        (see create_lag_features). Predictions are converted back to price
        levels by cumulatively summing the predicted differences from the
        last observed price.

        Scaling (use_scaler=True):
            Linear models (Ridge, LASSO) are sensitive to feature scale.
            StandardScaler is fit on the training set and applied to both
            training and test/forecast features to prevent data leakage.
            Tree-based models (GBM, RF, XGBoost) are scale-invariant and
            do not require this transformation.
        """
        try:
            lags  = self.config['lags']
            train, test = self.train_test_split()

            X_train, y_train   = self.create_lag_features(train, lags)
            feature_columns    = list(X_train.columns)

            # Build test features by prepending the tail of the training series
            # to give the lagged features enough context
            combined           = pd.concat([train.iloc[-max(lags):], test])
            X_test_full, _     = self.create_lag_features(combined, lags)
            X_test             = X_test_full[-len(test):].reindex(columns=feature_columns, fill_value=0)

            scaler = StandardScaler()

            if use_scaler:
                X_train_fit = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
                X_test_fit  = pd.DataFrame(scaler.transform(X_test),      columns=feature_columns)
            else:
                X_train_fit = X_train
                X_test_fit  = X_test

            model.fit(X_train_fit, y_train)

            # Test set evaluation: cumulative sum of predicted differences
            last_train_price = train.iloc[-1]
            preds_diff       = model.predict(X_test_fit)
            preds            = last_train_price + np.cumsum(preds_diff)

            metrics = self.calculate_metrics(test.values, preds)

            # Multi-step forecast: iterative prediction using a rolling window
            predictions = []
            temp_series = list(self.series.values)
            current     = temp_series[-1]

            for _ in range(self.horizon):
                feat = self._get_last_features(temp_series, lags, feature_columns)

                if use_scaler:
                    feat = pd.DataFrame(scaler.transform(feat), columns=feature_columns)

                diff    = model.predict(feat)[0]
                new_val = max(0, current + diff)

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

    def _get_last_features(self, values, lags, feature_columns):
        """
        Construct the feature vector for the next forecast step.

        Mirrors the feature engineering in create_lag_features exactly,
        applied to the rolling values list (which grows by one entry per
        forecast step as new predicted prices are appended).
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
        feat['month']     = last_date.month
        feat['quarter']   = last_date.quarter
        feat['dayofweek'] = last_date.dayofweek

        return pd.DataFrame([feat]).reindex(columns=feature_columns, fill_value=0)

    # ── Individual ML models ──────────────────────────────────────────────────

    def forecast_gradient_boosting(self):
        """
        Gradient Boosting Regressor.

        Builds an additive ensemble of shallow decision trees where each
        successive tree fits the residuals of the previous ensemble (gradient
        descent in function space). Well-suited to tabular lag-feature data
        with non-linear interactions between lags.
        """
        return self._forecast_ml_model(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting"
        )

    def forecast_random_forest(self):
        """
        Random Forest Regressor.

        Averages predictions from many independently grown decision trees,
        each trained on a bootstrap sample with a random feature subset.
        High bias reduction through averaging makes it robust to outliers
        in the training set — useful for commodity series with price spikes.
        """
        return self._forecast_ml_model(
            RandomForestRegressor(n_estimators=100, random_state=42),
            "Random Forest"
        )

    def forecast_ridge(self):
        """
        Ridge Regression (L2 regularisation).

        Linear model that adds an L2 penalty on the coefficient vector:
            min ||y - Xw||^2 + alpha * ||w||^2

        The penalty shrinks all coefficients toward zero, reducing overfitting
        when lag features are highly correlated (common in price series).
        Alpha is selected via cross-validation from a predefined grid.
        Feature scaling is required and applied via StandardScaler.
        """
        return self._forecast_ml_model(
            RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
            "Ridge Regression", use_scaler=True
        )

    def forecast_lasso(self):
        """
        LASSO Regression (L1 regularisation).

        Similar to Ridge but uses an L1 penalty, which produces sparse solutions
        by driving some coefficients exactly to zero:
            min ||y - Xw||^2 + alpha * ||w||_1

        Effective feature selection in disguise — LASSO effectively discards
        lag features that carry little predictive signal for a given commodity,
        which is useful given the varying importance of different lag lengths
        across series with different autocorrelation structures.
        """
        return self._forecast_ml_model(
            LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], max_iter=5000),
            "LASSO Regression", use_scaler=True
        )

    def forecast_xgboost(self):
        """
        XGBoost Regressor.

        An optimised implementation of gradient boosted trees with additional
        regularisation (L1 and L2 on leaf weights), column subsampling per
        tree (colsample_bytree), and row subsampling (subsample). These
        regularisation techniques reduce variance compared to standard GBM,
        making XGBoost generally more robust on smaller datasets.
        """
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

    # ── Exponential Smoothing ─────────────────────────────────────────────────

    def forecast_exponential_smoothing(self):
        """
        Holt's exponential smoothing with additive trend.

        Exponential smoothing assigns exponentially decreasing weights to past
        observations — recent data matters more than distant history. Holt's
        extension adds a trend component:

            Level:  L_t = alpha * y_t + (1 - alpha) * (L_{t-1} + T_{t-1})
            Trend:  T_t = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
            Forecast: y_{t+h} = L_t + h * T_t

        Alpha (level smoothing) and beta (trend smoothing) are optimised by
        minimising the sum of squared in-sample errors.

        Fallback logic:
            Holt's ExponentialSmoothing with 'estimated' initialisation can
            fail to converge on short or irregular commodity series. When this
            occurs the model falls back to SimpleExpSmoothing (alpha only, no
            trend component) with a fixed smoothing level of 0.3, which is
            always stable but less accurate for trending series.
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
                # Primary model did not converge — use simple level-only smoothing
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
            return {'error': str(e), 'trace': traceback.format_exc(),
                    'method': 'Exponential Smoothing'}

    # ── Model orchestration ───────────────────────────────────────────────────

    def forecast_all_models(self):
        """
        Run every model in the suite and return their results as a dict.

        Each model is run independently so that a failure in one does not
        prevent the others from completing. Results keyed by model identifier
        are passed to create_ensemble for ranking and combination.
        """
        is_valid, message = self.validate_data()

        if not is_valid:
            return {'error': message}

        return {
            'arima':             self.forecast_arima(),
            'sarima':            self.forecast_sarima(),
            'exp_smoothing':     self.forecast_exponential_smoothing(),
            'garch':             self.forecast_garch(),
            'gradient_boosting': self.forecast_gradient_boosting(),
            'random_forest':     self.forecast_random_forest(),
            'ridge':             self.forecast_ridge(),
            'lasso':             self.forecast_lasso(),
            'xgboost':           self.forecast_xgboost(),
        }

    # ── Ensemble construction ─────────────────────────────────────────────────

    def create_ensemble(self, results):
        """
        Combine the top-performing models into a weighted ensemble forecast.

        Model selection:
            A model is included in the candidate pool only if it:
            - produced a complete prediction vector (no errors)
            - all predictions are finite
            - MAPE is finite and defined
            - the first-period prediction does not deviate more than 40% from
              the last observed price (guards against degenerate forecasts
              caused by integration order misspecification or data artifacts)

        Ranking:
            Candidates are ranked by MASE if available (preferred, scale-free),
            falling back to MAPE/100 otherwise. The top 3 models are selected.

        Weighting:
            Ensemble weights are derived from the exponential of the negative
            MAPE values, then normalised to sum to 1:

                w_i = exp(-MAPE_i) / sum_j exp(-MAPE_j)

            This gives disproportionately higher weight to lower-error models
            (softmax-style) while still allowing weaker models a small
            contribution, which has been shown empirically to reduce variance
            in forecast combinations (the "forecast combination puzzle" result
            in forecasting literature).

        The ensemble confidence label is derived from the average MAPE of
        the top 3 models using the MAPE-based fallback in assign_confidence.
        """
        last_known = float(self.series.iloc[-1])

        def is_valid(v):
            if 'predictions' not in v:
                return False
            preds = v['predictions']
            if not preds or not all(np.isfinite(p) for p in preds):
                return False
            if not np.isfinite(v['metrics'].get('mape', np.inf)):
                return False
            # Reject models whose first prediction is implausibly far from current price
            first_change = abs(preds[0] - last_known) / last_known if last_known else 1
            if first_change > 0.40:
                return False
            return True

        successful = {k: v for k, v in results.items() if is_valid(v)}

        if not successful:
            return None

        def rank_key(item):
            m    = item[1]['metrics']
            mase = m.get('mase')
            return mase if (mase is not None and not np.isnan(mase)) else m['mape'] / 100

        ranked     = sorted(successful.items(), key=rank_key)
        top_models = ranked[:3]

        # Exponential inverse-error weighting (softmax on negative MAPE)
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