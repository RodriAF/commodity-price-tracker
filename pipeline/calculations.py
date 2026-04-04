"""
Agricultural Commodities Calculations
Focus: meaningful trend indicators, z-score signals, and market regime detection.

What was removed and why:
- crop_spreads (nominal): corn minus wheat in USD is meaningless across different units
- crop_spreads (ratio): corn/wheat retained only as part of profitability context
- categorize_ratios: fragile string-matching replaced with explicit mapping logic

What remains and why it matters:
- crop_profitability_ratios: directional signal of crop price vs input cost pressure
- cost_indices: normalised input cost environment, actionable across commodities
- z-score signals: statistical anomaly detection — the core value of this module
- market regime: classifies the current cost environment for dashboard context
- CorrelationAnalysis: crop-input price transmission, useful for forecasting context
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class CommoditiesAnalytics:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.categories = ConfigLoader.get_categories()
        logger.info("CommoditiesAnalytics initialised")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _align(self, c1, c2):
        """
        Forward-fill both series before operating on them.

        Mixed-frequency data (e.g. monthly fertilizer vs weekly crude oil) creates
        NaN gaps after the outer join in the collector. Forward-filling propagates
        the last known value across missing periods, ensuring ratio calculations
        do not silently drop rows or introduce NaN artifacts.
        """
        return self.df[c1].ffill(), self.df[c2].ffill()

    # ── Ratio calculations ────────────────────────────────────────────────────

    def crop_profitability_ratios(self):
        """
        Compute crop price divided by input cost for every crop-input pair.

        Interpretation note:
            The absolute value is NOT comparable across pairs because the
            numerator and denominator carry different units (e.g. USD/bushel
            vs USD/MMBtu vs USD/ton). A ratio of 5.2 for corn-to-crude and
            0.8 for wheat-to-urea convey nothing by themselves.

            What IS meaningful:
            - The trend of each ratio over time (rising = improving margins)
            - Its z-score relative to its own history (deviation from norm)

        Economic interpretation:
            High ratio -> crop price elevated relative to input cost
                       -> favourable margin environment for producers
            Low ratio  -> input cost elevated relative to crop price
                       -> margin pressure; producers may reduce planted area

        All pairs (crop x energy) and (crop x fertilizer) are computed.
        Columns with all-NaN values are dropped before returning.
        """
        ratios = pd.DataFrame(index=self.df.index)

        crops       = self.categories.get('crop', [])
        energies    = self.categories.get('energy_input', [])
        fertilizers = self.categories.get('fertilizer', [])

        for crop in crops:
            if crop not in self.df.columns:
                continue

            for energy in energies:
                if energy in self.df.columns:
                    s1, s2 = self._align(crop, energy)
                    ratios[f'{crop}_to_{energy}'] = s1 / s2

            for fert in fertilizers:
                if fert in self.df.columns:
                    s1, s2 = self._align(crop, fert)
                    ratios[f'{crop}_to_{fert}'] = s1 / s2

        return ratios.dropna(axis=1, how='all')

    def cost_indices(self):
        """
        Build a normalised cost index for energy inputs and fertilizers.

        Method:
            Each commodity in the category is divided by its own full-period
            mean and multiplied by 100. This transforms all series to the same
            scale regardless of original units (USD/gallon, USD/ton, etc.).
            The index is the simple average of all normalised series in the
            category.

                Index_t = mean_i [ (P_i_t / mean(P_i)) * 100 ]

        Interpretation:
            Index = 100  -> current costs at their historical average
            Index > 110  -> above-average cost environment (~1 normalised std)
            Index < 90   -> below-average cost environment

        Unlike the raw profitability ratios above, the index value IS
        meaningful as an absolute number because all inputs share the same
        baseline (100 = own historical mean). This makes cross-time comparison
        valid and powers the regime classification downstream.
        """
        indices = pd.DataFrame(index=self.df.index)

        for cat in ['energy_input', 'fertilizer']:
            commodities = [c for c in self.categories.get(cat, []) if c in self.df.columns]

            if not commodities:
                continue

            normalized = pd.DataFrame(index=self.df.index)

            for c in commodities:
                mean_val = self.df[c].mean()
                if mean_val > 0:
                    normalized[c] = (self.df[c] / mean_val) * 100

            if not normalized.empty:
                indices[f'{cat}_cost_index'] = normalized.mean(axis=1)
                logger.info(f"  {cat} cost index computed from {len(commodities)} series: {commodities}")

        return indices

    # ── Z-score signals ───────────────────────────────────────────────────────

    def calculate_zscores(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Compute rolling z-scores for every column in the input DataFrame.

        A z-score measures how many standard deviations the current value
        lies from the rolling mean:

            z_t = (x_t - mu_t) / sigma_t

        where mu_t and sigma_t are the rolling mean and standard deviation
        over the specified window.

        Window choice (default = 60):
            A window of 60 observations balances two competing goals:
            - Long enough to estimate a stable mean and std (reduces noise)
            - Short enough to adapt to structural shifts in market levels
              (avoids flagging genuinely new price regimes as anomalies)

            For commodity ratios that update at monthly or quarterly
            frequency, 60 observations corresponds to roughly 5 years of
            monthly data — a reasonable lookback for agricultural markets.

        A small epsilon (1e-8) is added to the denominator to prevent
        division-by-zero on constant or near-constant sub-series.
        """
        zscores = pd.DataFrame(index=df.index)

        for col in df.columns:
            mean = df[col].rolling(window).mean()
            std  = df[col].rolling(window).std()
            zscores[col + '_zscore'] = (df[col] - mean) / (std + 1e-8)

        return zscores

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(self, zscores: pd.DataFrame, threshold: float = 2.0) -> list:
        """
        Identify statistically anomalous values in the most recent observation.

        A signal is raised when |z| > threshold. The default threshold of 2.0
        corresponds to the outermost ~5% of a normal distribution (top and
        bottom 2.5% combined), making it a meaningful but not overly sensitive
        trigger for commodity markets.

        Setting threshold = 2.0 rather than 1.5 deliberately filters out
        moderate deviations that are common in volatile commodity series,
        keeping only those that warrant attention in a dashboard context.

        Returns a list of signal dicts sorted by absolute z-score descending,
        so the most anomalous metrics surface first.
        """
        signals = []

        for col in zscores.columns:
            val = zscores[col].iloc[-1]

            if pd.isna(val):
                continue

            if abs(val) > threshold:
                signals.append({
                    'metric':   col.replace('_zscore', ''),
                    'type':     'overvalued' if val > 0 else 'undervalued',
                    'z_score':  float(val),
                    'strength': abs(float(val))
                })

        return sorted(signals, key=lambda x: x['strength'], reverse=True)

    # ── Regime detection ──────────────────────────────────────────────────────

    def detect_market_regime(self, indices: pd.DataFrame) -> dict:
        """
        Classify the current input cost environment using index levels.

        Thresholds:
            > 110 -> HIGH  (cost above historical average by ~10%)
            < 90  -> LOW   (cost below historical average by ~10%)
            else  -> NORMAL

        The symmetric ±10 band around 100 is intentionally conservative.
        Commodity input costs are structurally volatile; labelling anything
        outside a ±5 band as extreme would generate excessive regime switches.
        A ±10 band corresponds roughly to one normalised standard deviation
        across the energy and fertilizer series in this dataset.

        The result is a plain dict (JSON-serialisable) consumed by the
        Analytics dashboard page and the daily automation summary log.
        """
        if indices.empty:
            return {}

        latest = indices.iloc[-1]
        regime = {}

        threshold_high = 110
        threshold_low  = 90

        mapping = {
            'energy_input_cost_index': {
                'key': 'energy',
                'high': 'high_cost',
                'low':  'low_cost',
                'mid':  'normal'
            },
            'fertilizer_cost_index': {
                'key': 'fertilizer',
                'high': 'expensive',
                'low':  'cheap',
                'mid':  'normal'
            }
        }

        for index_col, labels in mapping.items():
            if index_col not in latest:
                continue
            v = latest[index_col]
            if pd.isna(v):
                continue
            regime[labels['key']] = (
                labels['high'] if v > threshold_high else
                labels['low']  if v < threshold_low  else
                labels['mid']
            )

        return regime

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def calculate_all(self):
        """
        Execute the full analytics pipeline in dependency order.

        Execution order:
            1. crop_profitability_ratios  — raw directional signals
            2. cost_indices               — normalised cost environment
            3. calculate_zscores          — anomaly detection on all metrics
            4. generate_signals           — filter to actionable z-score alerts
            5. detect_market_regime       — high-level cost environment label

        Returns:
            combined   (DataFrame): all ratio and index columns, date-aligned
            analytics  (dict):      structured results keyed by type, consumed
                                    by the dashboard pages and run_daily.py
        """
        logger.info("Running commodities analytics pipeline...")

        profitability = self.crop_profitability_ratios()
        indices       = self.cost_indices()

        # Concatenate only non-empty frames to avoid polluting the combined df
        frames   = [f for f in [profitability, indices] if not f.empty]
        combined = pd.concat(frames, axis=1) if frames else pd.DataFrame()

        zscores = self.calculate_zscores(combined) if not combined.empty else pd.DataFrame()
        signals = self.generate_signals(zscores)    if not zscores.empty else []
        regime  = self.detect_market_regime(indices)

        logger.info(f"  Metrics computed : {len(combined.columns)}")
        logger.info(f"  Signals detected : {len(signals)}")
        logger.info(f"  Market regime    : {regime}")

        return combined, {
            'profitability': profitability,
            'cost_indices':  indices,
            'zscores':       zscores,
            'signals':       signals,
            'regime':        regime
        }


# ── Correlation analysis ──────────────────────────────────────────────────────

class CorrelationAnalysis:
    """
    Pearson correlation between crop prices and their input costs.

    Price transmission analysis: when input costs rise, how closely do crop
    prices follow? High positive correlation suggests producers can partially
    pass through cost increases to output prices. Low or negative correlation
    indicates decoupled markets — margin pressure cannot be offset by higher
    crop prices.

    This is also relevant for forecasting: highly correlated crop-input pairs
    can use input cost series as leading or concurrent predictors.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def key_correlations(self) -> pd.DataFrame:
        """
        Compute Pearson correlation coefficients for all crop-input pairs.

        Pearson r measures linear association between two series:

            r = cov(X, Y) / (std(X) * std(Y))

        Values range from -1 (perfect inverse relationship) to +1 (perfect
        positive relationship). Both series are forward-filled before
        computation to handle mixed-frequency gaps consistently.

        Results are sorted by absolute correlation descending so the strongest
        relationships — positive or negative — surface at the top of the table.
        Returns an empty DataFrame if no valid pairs are found.
        """
        results = []
        categories = ConfigLoader.get_categories()

        crops  = categories.get('crop', [])
        inputs = categories.get('energy_input', []) + categories.get('fertilizer', [])

        for crop in crops:
            if crop not in self.df.columns:
                continue

            for inp in inputs:
                if inp not in self.df.columns:
                    continue

                corr = self.df[crop].ffill().corr(self.df[inp].ffill())

                if pd.notna(corr):
                    results.append({
                        'crop':        crop,
                        'input':       inp,
                        'correlation': round(corr, 4)
                    })

        df_corr = pd.DataFrame(results)

        if not df_corr.empty:
            df_corr['abs_corr'] = df_corr['correlation'].abs()
            df_corr = df_corr.sort_values('abs_corr', ascending=False).drop(columns='abs_corr')

        return df_corr