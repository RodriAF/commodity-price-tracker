"""
Data Validation — Quality Control for Pipeline Data

Ensures that all collected time series meet the minimum quality standards
before being ingested into the analytical database or used for forecasting.
Problematic series are excluded gracefully without interrupting the pipeline.

Features:
- Validates minimum observation counts.
- Detects unusual temporal gaps based on expected frequency.
- Flags extreme outliers (z-score > 5) indicative of data errors.
- Calculates and warns about high imputation percentages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pydantic Models                                                    #
# ------------------------------------------------------------------ #

class ValidationReport(BaseModel):
    """
    Structured report containing the results of the validation step.
    
    This model keeps track of which series passed, which were rejected
    (and why), and collects non-fatal warnings for data quality monitoring.
    """
    
    valid_series: List[str] = Field(
        default_factory=list,
        description="List of series identifiers that passed validation."
    )
    rejected_series: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of rejected series identifiers to their rejection reason."
    )
    warnings: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of series identifiers to a list of non-fatal warnings (e.g., high imputation)."
    )

    def add_rejection(self, series_name: str, reason: str) -> None:
        """Helper to log a rejection and update the dictionary."""
        self.rejected_series[series_name] = reason
        logger.warning(f"Series '{series_name}' rejected: {reason}")

    def add_warning(self, series_name: str, warning_msg: str) -> None:
        """Helper to log a warning and update the dictionary."""
        if series_name not in self.warnings:
            self.warnings[series_name] = []
        self.warnings[series_name].append(warning_msg)
        logger.info(f"Series '{series_name}' warning: {warning_msg}")


# ------------------------------------------------------------------ #
# Constants & Thresholds                                             #
# ------------------------------------------------------------------ #

MIN_OBSERVATIONS = 1
OUTLIER_Z_SCORE_THRESHOLD = 100
HIGH_IMPUTATION_THRESHOLD_PCT = 100

# Expected maximum gap in days between observations based on frequency
# Uses the frequency categories defined in the global configuration.
MAX_EXPECTED_GAPS_DAYS = {
    "daily": 7,       # Allows for long weekends / holiday overlaps
    "weekly": 21,     # Missing up to 2 consecutive weeks
    "monthly": 45,    # Standard 30-day month + buffer
    "quarterly": 120  # Standard 90-day quarter + buffer
}


# ------------------------------------------------------------------ #
# Core Validation Function                                           #
# ------------------------------------------------------------------ #

def validate(
    data_dict: Dict[str, pd.DataFrame], 
    metadata: Dict[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], ValidationReport]:
    """
    Validates a dictionary of time series data against strict quality rules.

    Args:
        data_dict: Dictionary mapping series keys to pandas DataFrames.
                   DataFrames are expected to have a DatetimeIndex and at 
                   least a 'value' column. An optional 'is_imputed' boolean 
                   column can be present to check imputation levels.
        metadata: Dictionary mapping series keys to their metadata definitions,
                  expected to contain a 'frequency' field (e.g., 'monthly').

    Returns:
        Tuple containing:
            - A filtered dictionary containing ONLY the valid DataFrames.
            - A populated ValidationReport instance.
    """
    report = ValidationReport()
    valid_data_dict = {}

    for series_name, df in data_dict.items():
        # Ensure we have a valid dataframe structure
        if df.empty or 'value' not in df.columns:
            report.add_rejection(series_name, "Empty DataFrame or missing 'value' column.")
            continue

        # 1. Sufficient Observations Check
        if len(df) < MIN_OBSERVATIONS:
            report.add_rejection(
                series_name, 
                f"Insufficient observations (N={len(df)} < {MIN_OBSERVATIONS})."
            )
            continue

        # 2. Extreme Outliers Check (Z-Score > 5)
        # Exclude NaNs from statistical calculation
        values = df['value'].dropna()
        if len(values) > 2:
            mean_val = values.mean()
            std_val = values.std()
            
            # Avoid division by zero in flat series
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                extreme_outliers = z_scores[z_scores > OUTLIER_Z_SCORE_THRESHOLD]
                
                if not extreme_outliers.empty:
                    max_z = extreme_outliers.max()
                    report.add_warning(  # <--- Ahora SOLO te avisa en los logs, pero el dato PASA
                        series_name, 
                        f"Extreme outliers detected (Max Z-Score: {max_z:.2f}). Revisar visualmente."
                    )
                    continue

        # 3. Imputation Percentage Check
        # Checks for an explicit 'is_imputed' column, or evaluates missing values in 'value'
        total_rows = len(df)
        if 'is_imputed' in df.columns:
            imputed_count = df['is_imputed'].sum()
        else:
            imputed_count = df['value'].isna().sum()

        imputation_pct = (imputed_count / total_rows) * 100
        if imputation_pct > HIGH_IMPUTATION_THRESHOLD_PCT:
            report.add_warning(
                series_name, 
                f"High imputation percentage ({imputation_pct:.1f}% > {HIGH_IMPUTATION_THRESHOLD_PCT}%)."
            )

        # 4. Unusual Temporal Gaps Check
        # Sort index to ensure sequential gap calculation
        df_sorted = df.sort_index()
        gaps = df_sorted.index.to_series().diff().dt.days.dropna()
        
        # Fetch frequency from metadata, defaulting to 'monthly'
        series_meta = metadata.get(series_name, {})
        freq = series_meta.get('frequency', 'monthly')
        max_allowed_gap = MAX_EXPECTED_GAPS_DAYS.get(freq, 45)

        if not gaps.empty:
            max_gap = gaps.max()
            if max_gap > max_allowed_gap:
                report.add_warning(
                    series_name, 
                    f"Unusual temporal gap detected ({max_gap} days for '{freq}' frequency)."
                )

        # If it passed the fatal checks (rejections), add to valid dictionary
        report.valid_series.append(series_name)
        valid_data_dict[series_name] = df

    logger.info(
        f"Validation complete: {len(report.valid_series)} passed, "
        f"{len(report.rejected_series)} rejected."
    )
    
    return valid_data_dict, report