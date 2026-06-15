"""
Shared data-access helpers built on top of ``utils.db`` and
``utils.config_loader``.

Every router needs the same wide-format price matrix and the same
"last value / percentage change" helpers that used to live inline in
``app.py``. Centralising them here keeps the routers thin and avoids
re-implementing the same pandas pivots in four places.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import utils.db as db
from utils.config_loader import ConfigLoader


def load_wide_prices() -> pd.DataFrame:
    """
    Load all historical prices from DuckDB and pivot into a wide
    DataFrame: one ``date`` column plus one column per commodity.

    Returns an empty DataFrame if the ``prices`` table has no rows yet
    (e.g. the pipeline has never run).
    """
    df_long = db.load_prices()
    if df_long.empty:
        return pd.DataFrame()

    df_wide = df_long.pivot(index="date", columns="commodity", values="value").reset_index()
    df_wide.columns.name = None
    df_wide["date"] = pd.to_datetime(df_wide["date"])
    return df_wide.sort_values("date").reset_index(drop=True)


def load_latest_signals() -> pd.DataFrame:
    """Return only the most recent ``run_date`` rows from the ``signals`` table."""
    with db.get_connection() as conn:
        signals_df = conn.execute("SELECT * FROM signals ORDER BY run_date DESC").df()

    if signals_df.empty:
        return signals_df

    max_date = signals_df["run_date"].max()
    return signals_df[signals_df["run_date"] == max_date]


def base_commodity_columns(df_wide: pd.DataFrame) -> list[str]:
    """Return all non-date columns of a wide price/metric DataFrame."""
    return [c for c in df_wide.columns if c != "date"]


def last_valid(df_wide: pd.DataFrame, col: str) -> Optional[float]:
    """Most recent non-null value for ``col``, or ``None``."""
    if col not in df_wide.columns:
        return None
    s = df_wide[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def change_pct(df_wide: pd.DataFrame, col: str) -> Optional[float]:
    """Percentage change between the last two valid observations of ``col``."""
    if col not in df_wide.columns:
        return None
    s = df_wide[col].dropna()
    if len(s) < 2:
        return None
    return float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100)


def percentile_of(series: pd.Series, value: Optional[float]) -> Optional[float]:
    """Percentile rank of ``value`` within ``series`` (0-100), or ``None``."""
    if value is None:
        return None
    s = series.dropna()
    if s.empty or pd.isna(value):
        return None
    return float((s < value).mean() * 100)


def commodity_display_name(key: str) -> str:
    info = ConfigLoader.get_commodity_info(key)
    return info.get("name", key.replace("_", " ").title())


def commodity_unit(key: str) -> str:
    info = ConfigLoader.get_commodity_info(key)
    return info.get("unit", "")


def series_to_json(series: pd.Series) -> list:
    """Convert a pandas Series to a JSON-safe list (NaN -> None, numpy -> float)."""
    return [None if pd.isna(v) else float(v) for v in series]


def df_to_series_dict(df_wide: pd.DataFrame, columns: list[str]) -> dict:
    """Convert selected columns of a wide DataFrame into JSON-friendly lists."""
    return {col: series_to_json(df_wide[col]) for col in columns if col in df_wide.columns}
