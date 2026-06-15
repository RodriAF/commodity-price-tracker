"""
Overview router — replaces ``app.py`` (the Streamlit market overview page).

Exposes a single aggregated endpoint that powers the dashboard's KPI strip,
the price history chart, the per-category commodity cards, the active
anomaly signals list, and the commodity snapshot table.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from services import prices as price_service
from utils.config_loader import ConfigLoader

router = APIRouter(tags=["overview"])

# Human-readable labels for known categories. Anything not listed here
# falls back to a title-cased version of the category key.
CATEGORY_LABELS = {
    "energy_input": "Energy Inputs",
    "crop": "Crops",
    "fertilizer": "Fertilizers",
    "livestock": "Livestock",
    "index": "Indices",
    "economic": "Economic",
}


@router.get("/overview")
def get_overview(
    days: int = Query(
        730,
        ge=0,
        description="History window (in days) for the price chart. 0 returns full history.",
    )
):
    """
    Aggregated snapshot of the market: KPIs, category breakdown, price
    history, active anomaly signals, and the full commodity snapshot table.
    """
    df = price_service.load_wide_prices()
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No data found in DuckDB. Please run the data pipeline first.",
        )

    base_cols = price_service.base_commodity_columns(df)
    latest_signals = price_service.load_latest_signals()

    # ------------------------------------------------------------ #
    # KPI strip                                                     #
    # ------------------------------------------------------------ #
    last_date = df["date"].max()
    n_commodities = len(base_cols)
    n_signals = int(len(latest_signals))
    n_extreme = int((latest_signals["z_score"].abs() > 2).sum()) if not latest_signals.empty else 0
    n_notable = int((latest_signals["z_score"].abs() > 1).sum()) if not latest_signals.empty else 0

    # ------------------------------------------------------------ #
    # Commodities grouped by category                               #
    # ------------------------------------------------------------ #
    categories: dict[str, dict] = {}
    for c in base_cols:
        info = ConfigLoader.get_commodity_info(c)
        cat = info.get("category", "other")
        bucket = categories.setdefault(
            cat,
            {"label": CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()), "commodities": []},
        )
        bucket["commodities"].append(
            {
                "key": c,
                "name": price_service.commodity_display_name(c),
                "price": price_service.last_valid(df, c),
                "change_pct": price_service.change_pct(df, c),
            }
        )

    # ------------------------------------------------------------ #
    # Price history chart (windowed)                                #
    # ------------------------------------------------------------ #
    if days > 0:
        df_chart = df[df["date"] >= last_date - pd.Timedelta(days=days)]
    else:
        df_chart = df

    price_history = {
        "dates": df_chart["date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": price_service.df_to_series_dict(df_chart, base_cols),
    }

    # ------------------------------------------------------------ #
    # Active anomaly signals (|z| > 1)                              #
    # ------------------------------------------------------------ #
    signals = []
    if not latest_signals.empty:
        for _, row in latest_signals.iterrows():
            z = row["z_score"]
            if pd.isna(z) or abs(z) <= 1:
                continue
            signals.append(
                {
                    "key": row["commodity"],
                    "name": price_service.commodity_display_name(row["commodity"]),
                    "z_score": float(z),
                    "type": "overvalued" if z > 0 else "undervalued",
                    "level": "extreme" if abs(z) > 2 else "notable",
                }
            )
    signals.sort(key=lambda s: abs(s["z_score"]), reverse=True)

    # ------------------------------------------------------------ #
    # Commodity snapshot table                                      #
    # ------------------------------------------------------------ #
    snapshot = []
    for c in base_cols:
        info = ConfigLoader.get_commodity_info(c)

        z = None
        if not latest_signals.empty:
            match = latest_signals[latest_signals["commodity"] == c]
            if not match.empty:
                z_val = match.iloc[0]["z_score"]
                z = float(z_val) if pd.notna(z_val) else None

        if z is not None:
            signal_label = "Extreme" if abs(z) > 2 else "Notable" if abs(z) > 1 else "Normal"
        else:
            signal_label = None

        snapshot.append(
            {
                "key": c,
                "name": price_service.commodity_display_name(c),
                "category": info.get("category", "other").replace("_", " ").title(),
                "frequency": info.get("frequency", "monthly").title()[:3],
                "price": price_service.last_valid(df, c),
                "change_pct": price_service.change_pct(df, c),
                "z_score": z,
                "signal": signal_label,
            }
        )

    # Sort by |z-score| descending, unranked commodities last — mirrors the
    # original Streamlit table sort.
    snapshot.sort(key=lambda r: abs(r["z_score"]) if r["z_score"] is not None else -1, reverse=True)

    return {
        "last_update": last_date.strftime("%Y-%m-%d"),
        "n_commodities": n_commodities,
        "active_signals": n_signals,
        "extreme_signals": n_extreme,
        "notable_signals": n_notable,
        "categories": categories,
        "price_history": price_history,
        "signals": signals,
        "snapshot": snapshot,
    }
