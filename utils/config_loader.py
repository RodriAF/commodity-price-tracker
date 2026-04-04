"""
Configuration Loader — Centralised Config Access
Ensures all system components read settings from a single source of truth:
config/commodities.json. Changes to commodity definitions, frequency parameters,
or category groupings propagate automatically to the collector, pipeline,
analytics, forecaster, and all dashboard pages without code changes.
"""

import json
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralised configuration loader with a class-level cache.

    The cache (_config) is populated on the first call to load() and reused
    on all subsequent calls within the same process. This avoids redundant
    disk I/O on every metric calculation or dashboard render cycle.

    To force a fresh read from disk (e.g. after editing the JSON file
    during a live session), call ConfigLoader.reload().
    """

    _config      = None   # Class-level cache; shared across all instances
    _config_path = 'config/commodities.json'

    @classmethod
    def load(cls) -> Dict:
        """
        Load the configuration file and return the parsed dictionary.

        Implements a simple read-through cache: returns the cached value
        immediately if available, otherwise reads from disk, caches the
        result, and returns it.

        Returns:
            Dict: Full configuration dictionary as parsed from the JSON file.

        Raises:
            FileNotFoundError: If the config file does not exist at the
                               expected path.
            json.JSONDecodeError: If the file exists but contains invalid JSON.
        """
        if cls._config is not None:
            return cls._config

        if not os.path.exists(cls._config_path):
            logger.error(f"Configuration file not found: {cls._config_path}")
            raise FileNotFoundError(f"Configuration file not found: {cls._config_path}")

        try:
            with open(cls._config_path, 'r', encoding='utf-8') as f:
                cls._config = json.load(f)

            logger.info(f"Configuration loaded: {len(cls._config.get('commodities', {}))} commodities registered")
            return cls._config

        except Exception as e:
            logger.error(f"Failed to parse configuration file: {e}")
            raise

    @classmethod
    def get_commodities(cls) -> Dict:
        """
        Return the full dictionary of registered commodities.

        Each entry maps a commodity key (e.g. 'corn', 'crude_oil') to its
        metadata: FRED series ID, display name, unit, category, and frequency.
        """
        config = cls.load()
        return config.get('commodities', {})

    @classmethod
    def get_frequency_config(cls) -> Dict:
        """
        Return the global frequency-specific parameter dictionary.

        Contains metric calculation parameters (change_periods, ma_window, etc.)
        keyed by frequency string ('daily', 'weekly', 'monthly', 'quarterly').
        These parameters ensure that moving averages and percentage changes
        are computed over time windows that are comparable across frequencies.
        """
        config = cls.load()
        return config.get('frequency_config', {})

    @classmethod
    def get_commodity_info(cls, commodity: str) -> Dict:
        """
        Return metadata for a specific commodity.

        Args:
            commodity: The commodity key as defined in commodities.json
                       (e.g. 'corn', 'crude_oil', 'urea').

        Returns:
            Dict: Commodity metadata, or an empty dict if the key is not found.
        """
        commodities = cls.get_commodities()
        return commodities.get(commodity, {})

    @classmethod
    def get_commodity_frequency(cls, commodity: str) -> str:
        """
        Return the data frequency string for a given commodity.

        Defaults to 'monthly' if the commodity is not found or the frequency
        field is absent — monthly is the most common FRED release cadence for
        agricultural commodity price indices.

        Args:
            commodity: The commodity key.

        Returns:
            str: One of 'daily', 'weekly', 'monthly', 'quarterly'.
        """
        info = cls.get_commodity_info(commodity)
        return info.get('frequency', 'monthly')

    @classmethod
    def get_metric_config(cls, frequency: str) -> Dict:
        """
        Return the metric calculation parameters for a given frequency.

        Parameters include change_periods (how many periods back for the
        percentage change calculation) and ma_window (rolling window length
        for the moving average), both calibrated to produce approximately
        equivalent lookback horizons across frequencies.

        Falls back to the 'monthly' configuration if the requested frequency
        is not present in the JSON, ensuring graceful degradation for any
        future frequency additions.

        Args:
            frequency: The data frequency string.

        Returns:
            Dict: Metric parameters for the given frequency.
        """
        freq_config = cls.get_frequency_config()
        return freq_config.get(frequency, freq_config.get('monthly'))

    @classmethod
    def get_categories(cls) -> Dict:
        """
        Build and return a mapping of category names to commodity key lists.

        Iterates over all registered commodities and groups them by their
        'category' field. Used by:
            - The collector for structured logging
            - CommoditiesAnalytics for ratio and index computation
            - CorrelationAnalysis for crop-input pair identification
            - All dashboard pages for category-based filtering

        Returns:
            Dict: {category_name: [commodity_key, ...]}
        """
        commodities = cls.get_commodities()
        categories  = {}

        for key, info in commodities.items():
            cat = info.get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(key)

        return categories

    @classmethod
    def reload(cls):
        """
        Invalidate the class-level cache and force a fresh read from disk.

        Useful during development when the JSON config is edited without
        restarting the Python process (e.g. in a running Streamlit session).
        """
        cls._config = None
        return cls.load()