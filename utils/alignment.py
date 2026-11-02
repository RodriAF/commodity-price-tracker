"""
Alignment helpers for mixed-frequency price data.

Storage keeps ONLY real observations: a monthly series has one row per month,
a daily series one row per business day. Several consumers (ratios between a
monthly crop and a daily energy price, correlations, charts) need all series
on a common calendar. They get it from here, explicitly and in memory, instead
of the database being filled with fabricated values.
"""

from __future__ import annotations

import pandas as pd


def align_wide_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill every numeric column of a wide price frame (rows = dates).

    The ``date`` column, if present, is left untouched. Rows are assumed to be
    sorted chronologically. Leading gaps stay NaN (nothing to carry forward).
    The input is never modified.
    """
    out = df.copy()
    cols = [c for c in out.columns if c != "date"]
    if cols:
        out[cols] = out[cols].ffill()
    return out
