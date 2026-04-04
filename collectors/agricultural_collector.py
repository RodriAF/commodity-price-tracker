"""
Agricultural Collector - Config-Based
Reads configuration from a centralized JSON file.
"""

from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

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

        logger.info("Agricultural Collector initialised")
        logger.info(f"  Commodities configured : {len(self.commodities)}")
        logger.info(f"  History window         : {days_history} days")

    def collect(self) -> tuple:
        """
        Fetch all configured commodity series from the FRED API.

        Returns:
            tuple: (DataFrame with merged series, dict with per-series metadata)
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
                    # Request the series from the FRED API
                    series = self.fred.get_series(
                        info['id'],
                        observation_start=start_date.strftime('%Y-%m-%d')
                    )

                    df = pd.DataFrame({
                        'date': series.index,
                        key: series.values
                    })

                    # Drop observations with missing values for this series
                    df = df.dropna(subset=[key])

                    if df.empty:
                        logger.warning(f"    {info['name']}: no data returned")
                        continue

                    all_data[key] = df

                    # Build metadata entry, merging frequency-specific configuration
                    freq = info['frequency']
                    freq_config = self.frequency_config.get(freq, self.frequency_config['monthly'])

                    metadata[key] = {
                        'name': info['name'],
                        'unit': info['unit'],
                        'category': info['category'],
                        'frequency': freq,
                        'actual_data_points': len(df),
                        **freq_config
                    }

                    logger.info(f"    {info['name']}: {len(df)} records ({freq})")

                except Exception as e:
                    logger.warning(f"    {info['name']}: collection failed — {e}")
                    continue

        if not all_data:
            logger.error("Collection complete — no data returned for any series")
            return pd.DataFrame(), {}

        # Merge all series into a single DataFrame using an outer join on date
        merged = None
        for key, df in all_data.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on='date', how='outer')

        # Sort chronologically and forward-fill to handle mixed-frequency gaps
        merged = merged.sort_values('date').reset_index(drop=True)
        merged = merged.ffill()

        logger.info("\n" + "=" * 70)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"  Records     : {len(merged)}")
        logger.info(f"  Series      : {len(merged.columns) - 1}")
        logger.info(f"  Date range  : {merged['date'].min()} — {merged['date'].max()}")
        logger.info("=" * 70)

        return merged, metadata

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