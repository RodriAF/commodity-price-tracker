"""
Forecast router — replaces ``pages/3_Forecasting.py``.

Reads the latest run's rows from the ``forecasts`` table, reconstructs the
ensemble / individual-model dictionary exactly as the original page's
``load_forecasts()`` did, and adds the historical series, future dates,
GARCH volatility bands, and ranked model comparison needed by the chart.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import utils.db as db
from services import prices as price_service
from utils.config_loader import ConfigLoader

router = APIRouter(tags=["forecast"])

FREQ_OFFSET = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS"}
PERIOD_LABEL = {"daily": "Day", "weekly": "Week", "monthly": "Month", "quarterly": "Quarter"}


def _load_latest_forecasts() -> dict:
    """
    Load the most recent run's forecasts and reshape them into:

        {commodity: {
            "current_price": float,
            "ensemble": {predictions, confidence, avg_mape},
            "individual_models": {model_name: {predictions, metrics, method, ...}}
        }}
    """
    with db.get_connection() as conn:
        df = conn.execute("SELECT * FROM forecasts WHERE run_date = (SELECT MAX(run_date) FROM forecasts)").df()

    if df.empty:
        return {}

    prices_df = db.load_prices()
    last_prices: dict = {}
    if not prices_df.empty:
        last_prices = prices_df.sort_values("date").groupby("commodity").last()["value"].to_dict()

    forecasts: dict = {}
    for _, row in df.iterrows():
        comm = row["commodity"]
        model_name = row["model"]

        preds = json.loads(row["predictions_json"]) if isinstance(row["predictions_json"], str) else row["predictions_json"]
        metrics = json.loads(row["metrics_json"]) if isinstance(row["metrics_json"], str) else row["metrics_json"]

        forecasts.setdefault(comm, {"current_price": last_prices.get(comm, 0.0), "individual_models": {}})

        if model_name == "ensemble":
            conf_val = row.get("confidence", 0)
            if isinstance(conf_val, (int, float)):
                conf_str = "high" if conf_val >= 0.8 else "medium" if conf_val >= 0.5 else "low"
            else:
                conf_str = str(conf_val)

            forecasts[comm]["ensemble"] = {
                "predictions": preds,
                "confidence": conf_str,
                "avg_mape": (metrics or {}).get("mape", 0),
            }
        else:
            model_data = {"predictions": preds, "metrics": metrics or {}, "method": model_name}
            if model_name == "garch" and metrics and "volatility_forecast" in metrics:
                model_data["volatility_forecast"] = metrics["volatility_forecast"]
            forecasts[comm]["individual_models"][model_name] = model_data

    return forecasts


def _future_dates(last_date, frequency: str, horizon: int) -> pd.DatetimeIndex:
    """Generate forecast period dates starting after the last observed date."""
    return pd.date_range(start=last_date, periods=horizon + 1, freq=FREQ_OFFSET.get(frequency, "MS"))[1:]


@router.get("/forecast/available")
def get_available_forecasts():
    """Commodities for which a valid ensemble forecast exists — for the selector."""
    forecasts = _load_latest_forecasts()
    available = [k for k, v in forecasts.items() if v.get("ensemble") is not None]
    return {
        "commodities": [
            {"key": k, "name": price_service.commodity_display_name(k)} for k in available
        ]
    }


@router.get("/forecast")
def get_forecast(commodity: str = Query(..., description="Commodity key, e.g. 'corn'")):
    """
    Forecast summary, model comparison, GARCH volatility band, and historical
    percentile context for a single commodity.
    """
    df = price_service.load_wide_prices()
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data found. Ensure the pipeline has run.")

    forecasts = _load_latest_forecasts()
    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecast data available. Ensure the forecasting models have executed.")

    if commodity not in forecasts or forecasts[commodity].get("ensemble") is None:
        raise HTTPException(status_code=404, detail=f"No successful ensemble forecast found for '{commodity}'.")

    if commodity not in df.columns:
        raise HTTPException(status_code=404, detail=f"No historical price series found for '{commodity}'.")

    results = forecasts[commodity]
    info = ConfigLoader.get_commodity_info(commodity)
    frequency = info.get("frequency", "monthly")

    ensemble = results.get("ensemble", {})
    individual = results.get("individual_models", {})
    current_price = results.get("current_price", 0)
    horizon = len(ensemble.get("predictions", []))

    last_date = df["date"].iloc[-1]
    future_dates = _future_dates(last_date, frequency, horizon)

    series = df[commodity].dropna()
    cur_pct = price_service.percentile_of(series, current_price)

    next_p = ensemble["predictions"][0] if ensemble.get("predictions") else current_price
    next_chg = ((next_p - current_price) / current_price * 100) if current_price else None

    # ------------------------------------------------------------ #
    # Rank individual models by hold-out MAPE                       #
    # ------------------------------------------------------------ #
    valid_models = {k: v for k, v in individual.items() if "predictions" in v and "metrics" in v}
    ranked = sorted(valid_models.items(), key=lambda x: x[1]["metrics"].get("mape", 999))
    top3_methods = [v.get("method", k) for k, v in ranked[:3]]

    ranked_models = []
    for k, v in ranked:
        method = v.get("method", k)
        ranked_models.append(
            {
                "key": k,
                "method": method,
                "mape": v["metrics"].get("mape", 0),
                "mae": v["metrics"].get("mae", 0),
                "predictions": (v.get("predictions") or [])[:horizon],
                "is_top": method in top3_methods,
            }
        )

    # ------------------------------------------------------------ #
    # GARCH 90% volatility band, converted to price space            #
    # ------------------------------------------------------------ #
    garch_band = None
    garch_data = individual.get("garch", {})
    if "predictions" in garch_data and "volatility_forecast" in garch_data:
        g_preds = garch_data["predictions"][:horizon]
        g_vols = garch_data["volatility_forecast"][:horizon]
        garch_band = {
            "upper": [p * (1 + 1.645 * v) for p, v in zip(g_preds, g_vols)],
            "lower": [max(0, p * (1 - 1.645 * v)) for p, v in zip(g_preds, g_vols)],
        }

    # ------------------------------------------------------------ #
    # Historical context for the forecast histogram                 #
    # ------------------------------------------------------------ #
    final_pred = ensemble["predictions"][-1] if ensemble.get("predictions") else None
    fut_pct = price_service.percentile_of(series, final_pred) if final_pred is not None else None

    # Last ~24 months for the chart, as in the original page.
    lookback = min(len(df), 730)
    df_plot = df.tail(lookback)

    return {
        "commodity": commodity,
        "name": price_service.commodity_display_name(commodity),
        "unit": info.get("unit", ""),
        "frequency": frequency,
        "period_label": PERIOD_LABEL.get(frequency, "Period"),
        "current_price": current_price,
        "current_percentile": cur_pct,
        "confidence": ensemble.get("confidence", "low"),
        "avg_mape": ensemble.get("avg_mape", 0),
        "next_period": {"value": next_p, "change_pct": next_chg},
        "history": {
            "dates": df_plot["date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": price_service.series_to_json(df_plot[commodity]),
        },
        "last_date": last_date.strftime("%Y-%m-%d"),
        "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "ensemble": {
            "predictions": ensemble.get("predictions", []),
            "confidence": ensemble.get("confidence", "low"),
            "avg_mape": ensemble.get("avg_mape", 0),
        },
        "garch_band": garch_band,
        "ranked_models": ranked_models,
        "top_models": top3_methods,
        "histogram": (
            {
                "values": [float(v) for v in series],
                "current": current_price,
                "forecast": final_pred,
                "forecast_percentile": fut_pct,
            }
            if final_pred is not None
            else None
        ),
    }
