"""
Analysis router — replaces ``pages/1_Analysis.py``.

Provides historical price series with a 30-period moving average and the
pipeline's stored z-scores, a Pearson correlation matrix for multi-asset
selections, descriptive statistics, and a percentile histogram for
single-asset views.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import utils.db as db
from services import prices as price_service
from utils.config_loader import ConfigLoader

router = APIRouter(tags=["analysis"])

# Matches the `time_range` select-slider options in the original page.
RANGE_DAYS = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "All": None}


def _load_analysis_frame() -> pd.DataFrame:
    """
    Build the wide analytical DataFrame: raw prices + pipeline z-scores
    (merged from the ``signals`` table, suffixed ``_zscore``) + a 30-period
    rolling moving average per commodity (suffixed ``_ma``).

    This mirrors ``load_data()`` from the original Streamlit page.
    """
    df_wide = price_service.load_wide_prices()
    if df_wide.empty:
        return df_wide

    with db.get_connection() as conn:
        df_signals = conn.execute("SELECT * FROM signals ORDER BY run_date ASC").df()

    if not df_signals.empty and "z_score" in df_signals.columns:
        try:
            z_wide = df_signals.pivot(index="run_date", columns="commodity", values="z_score").reset_index()
            z_wide = z_wide.rename(columns={"run_date": "date"})
            z_wide["date"] = pd.to_datetime(z_wide["date"])
            z_wide = z_wide.rename(columns={c: f"{c}_zscore" for c in z_wide.columns if c != "date"})
            df_wide = pd.merge(df_wide, z_wide, on="date", how="left")
        except Exception:
            # Mirrors the original page's silent fallback if the pivot fails
            # due to duplicate (run_date, commodity) pairs.
            pass

    base_cols = [c for c in df_wide.columns if c != "date" and not c.endswith("_zscore")]
    for c in base_cols:
        df_wide[f"{c}_ma"] = df_wide[c].rolling(window=30, min_periods=1).mean()

    return df_wide.ffill()


@router.get("/analysis")
def get_analysis(
    commodities: str = Query(..., description="Comma-separated commodity keys, e.g. 'corn,wheat'"),
    range: str = Query("2Y", alias="range", description="One of: 3M, 6M, 1Y, 2Y, All"),
    normalize: bool = Query(False, description="Rebase each series to 100 at the start of the window"),
):
    """
    Historical trends, moving averages, z-scores, correlation matrix,
    descriptive stats, and (for a single asset) a percentile histogram.
    """
    df = _load_analysis_frame()
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No data found in DuckDB. Ensure the pipeline has populated the 'prices' table.",
        )

    selected = [c.strip() for c in commodities.split(",") if c.strip()]
    if not selected:
        raise HTTPException(status_code=400, detail="At least one commodity must be selected.")

    base_cols = [c for c in df.columns if c != "date" and not c.endswith(("_ma", "_zscore"))]
    unknown = [c for c in selected if c not in base_cols]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown commodity key(s): {', '.join(unknown)}")

    if range not in RANGE_DAYS:
        raise HTTPException(status_code=400, detail=f"'range' must be one of {list(RANGE_DAYS)}")

    cutoff_days = RANGE_DAYS[range]
    last_date = df["date"].max()
    if cutoff_days is not None:
        df_view = df[df["date"] >= last_date - pd.Timedelta(days=cutoff_days)].copy()
    else:
        df_view = df.copy()
    df_view = df_view.reset_index(drop=True)

    # ------------------------------------------------------------ #
    # Per-asset series (price, moving average, z-score)             #
    # ------------------------------------------------------------ #
    series: dict[str, dict] = {}
    current: dict[str, dict] = {}

    for c in selected:
        base_series = df_view[c]
        non_null = df_view[c].dropna()
        base0 = float(non_null.iloc[0]) if not non_null.empty else None

        ma_col = f"{c}_ma"
        z_col = f"{c}_zscore"

        if normalize and base0:
            values = base_series / base0 * 100
            ma_values = (df_view[ma_col] / base0 * 100) if ma_col in df_view.columns else None
        else:
            values = base_series
            ma_values = df_view[ma_col] if ma_col in df_view.columns else None

        series[c] = {
            "name": price_service.commodity_display_name(c),
            "unit": price_service.commodity_unit(c),
            "values": price_service.series_to_json(values),
            "ma": price_service.series_to_json(ma_values) if ma_values is not None else None,
            "zscore": price_service.series_to_json(df_view[z_col]) if z_col in df_view.columns else None,
        }

        cur_val = float(non_null.iloc[-1]) if not non_null.empty else None
        z_val = None
        if z_col in df_view.columns:
            z_series = df_view[z_col].dropna()
            z_val = float(z_series.iloc[-1]) if not z_series.empty else None

        current[c] = {
            "value": cur_val,
            "zscore": z_val,
            "percentile": price_service.percentile_of(df[c], cur_val),
        }

    response: dict = {
        "dates": df_view["date"].dt.strftime("%Y-%m-%d").tolist(),
        "normalized": normalize,
        "range": range,
        "series": series,
        "current": current,
    }

    # ------------------------------------------------------------ #
    # Correlation matrix (2+ assets)                                 #
    # ------------------------------------------------------------ #
    if len(selected) >= 2:
        corr = df_view[selected].ffill().corr()
        response["correlation"] = {
            "keys": selected,
            "labels": [price_service.commodity_display_name(c) for c in selected],
            "matrix": [
                [None if pd.isna(v) else round(float(v), 4) for v in row] for row in corr.values
            ],
        }
    else:
        response["correlation"] = None

    # ------------------------------------------------------------ #
    # Descriptive statistics summary                                 #
    # ------------------------------------------------------------ #
    stats = []
    for c in selected:
        info = ConfigLoader.get_commodity_info(c)
        s = df_view[c].dropna()
        stats.append(
            {
                "key": c,
                "name": price_service.commodity_display_name(c),
                "frequency": info.get("frequency", "monthly").title()[:3],
                "current": current[c]["value"],
                "percentile": current[c]["percentile"],
                "mean": float(s.mean()) if not s.empty else None,
                "std": float(s.std()) if not s.empty else None,
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
            }
        )
    response["stats"] = stats

    # ------------------------------------------------------------ #
    # Percentile histogram (single-asset view only)                 #
    # ------------------------------------------------------------ #
    if len(selected) == 1:
        c = selected[0]
        full_series = df[c].dropna()
        cur_val = current[c]["value"]
        response["histogram"] = {
            "key": c,
            "name": price_service.commodity_display_name(c),
            "unit": price_service.commodity_unit(c),
            "values": [float(v) for v in full_series],
            "current": cur_val,
            "median": float(full_series.median()) if not full_series.empty else None,
            "percentile": current[c]["percentile"],
            "since_year": int(df["date"].min().year),
        }
    else:
        response["histogram"] = None

    return response
