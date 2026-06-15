"""
Agricultural Collector - Config-Based
Reads configuration from a centralized JSON file.
Includes robust retry mechanisms and frequency-aware imputation.
"""

from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Include the parent directory in the path for utility imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class AgriculturalCollector:
    """Data collector that retrieves commodity series using centralised configuration."""

    def __init__(self, api_key: str, days_history: int = 1095):
        """
        Initialise the collector.

        Args:
            api_key: FRED API key.
            days_history: Number of calendar days of historical data to fetch (default 1095 = 3 years).
        """
        self.fred = Fred(api_key=api_key)
        self.days_history = days_history

        # Load commodity and frequency configuration from the central JSON file
        self.commodities = ConfigLoader.get_commodities()
        self.frequency_config = ConfigLoader.get_frequency_config()

        # Define forward fill limits based on the frequency
        self.ffill_limits = {
            'daily': 3,
            'weekly': 2,
            'monthly': 2,
            'quarterly': 1
        }

        logger.info("Agricultural Collector initialised")
        logger.info(f"  Commodities configured : {len(self.commodities)}")
        logger.info(f"  History window         : {days_history} days")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_series_with_retry(self, series_id: str, start_date: str) -> pd.Series:
        """
        Fetch a series from FRED with exponential backoff retry.
        Max 3 attempts.
        """
        return self.fred.get_series(series_id, observation_start=start_date)

    def collect(self) -> tuple:
        """
        Fetch all configured commodity series from the FRED API.

        Returns:
            tuple: (Dict[str, DataFrame] with unmerged series, dict with per-series metadata)
        """
        logger.info("=" * 70)
        logger.info("DATA COLLECTION — AGRICULTURAL PRICES")
        logger.info("=" * 70)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_history)

        all_data = {}
        metadata = {}

        # Retrieve the category groupings for structured logging
        categories = ConfigLoader.get_categories()

        for category, items in sorted(categories.items()):
            logger.info(f"\n  {category.upper()}")

            for key in items:
                info = self.commodities[key]

                try:
                    # Request the series from the FRED API using the retry mechanism
                    series = self._fetch_series_with_retry(
                        info['id'],
                        start_date.strftime('%Y-%m-%d')
                    )

                    df = pd.DataFrame({
                        'date': series.index,
                        key: series.values
                    })

                    freq = info['frequency']
                    limit = self.ffill_limits.get(freq, 2) # Default fallback to 2

                    # Detect missing values prior to imputation
                    missing_mask = df[key].isna()

                    # Apply forward fill with frequency-based limits
                    df[key] = df[key].ffill(limit=limit)

                    # Mark imputed values (True if it was originally NaN but is now filled)
                    df['is_imputed'] = missing_mask & df[key].notna()

                    # Drop observations that remain empty (could not be filled)
                    df = df.dropna(subset=[key])

                    if df.empty:
                        logger.warning(f"    {info['name']}: no data returned after dropping NaNs")
                        continue

                    # Store the standalone dataframe
                    all_data[key] = df

                    # Build metadata entry, merging frequency-specific configuration
                    freq_config = self.frequency_config.get(freq, self.frequency_config.get('monthly', {}))

                    metadata[key] = {
                        'name': info['name'],
                        'unit': info['unit'],
                        'category': info['category'],
                        'frequency': freq,
                        'actual_data_points': len(df),
                        'imputed_data_points': int(df['is_imputed'].sum()),
                        **freq_config
                    }

                    logger.info(f"    {info['name']}: {len(df)} records ({freq}) - Imputed: {df['is_imputed'].sum()}")

                except Exception as e:
                    logger.warning(f"    {info['name']}: collection failed — {e}")
                    continue

        if not all_data:
            logger.error("Collection complete — no data returned for any series")
            return {}, {}

        logger.info("\n" + "=" * 70)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"  Series collected : {len(all_data)}")
        logger.info("=" * 70)

        # Returns a Dictionary of DataFrames instead of a merged DataFrame
        return all_data, metadata

    def get_categories(self):
        """Return commodities grouped by category."""
        return ConfigLoader.get_categories()

    def get_info(self, commodity: str = None):
        """
        Return configuration metadata for a specific commodity or all commodities.

        Args:
            commodity: Optional commodity key. If omitted, all metadata is returned.
        """
        if commodity:
            return ConfigLoader.get_commodity_info(commodity)
        return self.commodities