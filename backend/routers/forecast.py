"""
Forecast router — replaces ``pages/3_Forecasting.py``.

Reads the latest run's rows from the ``forecasts`` table and rebuilds, for one
commodity, everything the Forecasting page needs:

* the commodity's own history (its native observation calendar — never a
  forward-filled one) and the forecast dates that continue it,
* the ensemble forecast with 80% / 95% prediction intervals,
* the NAIVE "last value" baseline, so the forecast is always shown next to what
  doing nothing would have predicted,
* the rolling-origin backtest evaluation (error, skill versus naive, interval
  coverage) and the per-model comparison,
* the optional GARCH 90% volatility band (daily/weekly series).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import utils.db as db
from services import prices as price_service
from utils.config_loader import ConfigLoader

router = APIRouter(tags=["forecast"])

FREQ_OFFSET = {"daily": "B", "weekly": "W-MON", "monthly": "MS", "quarterly": "QS"}
PERIOD_LABEL = {"daily": "Day", "weekly": "Week", "monthly": "Month", "quarterly": "Quarter"}

# Observations of history drawn behind the forecast, per frequency.
HISTORY_POINTS = {"daily": 260, "weekly": 156, "monthly": 60, "quarterly": 24}

BASELINE_KEY = "naive"


def _loads(value: Any) -> dict | list | None:
    """JSON column -> Python object (the column may already be decoded)."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _confidence_label(value: Any) -> str:
    """
    Confidence is stored as text ('high' / 'medium' / 'low'). Rows written by
    older pipeline versions used a numeric score (1.0 / 0.5 / 0.0) — map those.
    """
    if value is None:
        return "low"
    text = str(value).strip().lower()
    if text in ("high", "medium", "low"):
        return text
    try:
        score = float(text)
    except ValueError:
        return "low"
    return "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"


def _num(value: Any) -> Optional[float]:
    """JSON-safe float (None for missing / non-finite)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _load_latest_forecasts() -> dict:
    """
    Load the most recent run's forecasts and reshape them into:

        {commodity: {
            "ensemble": {predictions, confidence, metrics: {...}},
            "individual_models": {model_key: {predictions, metrics, method, role, ...}}
        }}

    ``metrics`` is the decoded ``metrics_json`` (flat: error metrics next to
    extras such as the method label, intervals or series metadata).
    """
    with db.get_connection() as conn:
        df = conn.execute(
            "SELECT * FROM forecasts WHERE run_date = (SELECT MAX(run_date) FROM forecasts)"
        ).df()

    if df.empty:
        return {}

    forecasts: dict = {}
    for _, row in df.iterrows():
        comm = row["commodity"]
        model_key = row["model"]
        preds = _loads(row["predictions_json"]) or []
        metrics = _loads(row["metrics_json"]) or {}

        entry = forecasts.setdefault(comm, {"individual_models": {}})

        if model_key == "ensemble":
            entry["ensemble"] = {
                "predictions": preds,
                "confidence": _confidence_label(row.get("confidence")),
                "metrics": metrics,
            }
        else:
            entry["individual_models"][model_key] = {
                "predictions": preds,
                "metrics": metrics,
                "method": metrics.get("method", model_key),
                "role": metrics.get("role", "model"),
                "confidence": _confidence_label(row.get("confidence")),
            }

    return forecasts


def _future_dates(last_date: pd.Timestamp, frequency: str, horizon: int) -> list[str]:
    """Forecast dates continuing the commodity's own calendar (fallback for rows without metadata)."""
    idx = pd.date_range(start=last_date, periods=horizon + 1, freq=FREQ_OFFSET.get(frequency, "MS"))[1:]
    return [d.strftime("%Y-%m-%d") for d in idx]


@router.get("/forecast/available")
def get_available_forecasts():
    """Commodities for which a valid ensemble forecast exists — for the selector."""
    forecasts = _load_latest_forecasts()
    available = [k for k, v in forecasts.items() if v.get("ensemble") is not None]
    return {
        "commodities": [
            {"key": k, "name": price_service.commodity_display_name(k)} for k in sorted(available)
        ]
    }


@router.get("/forecast")
def get_forecast(commodity: str = Query(..., description="Commodity key, e.g. 'corn'")):
    """
    Forecast, intervals, naive baseline, backtest evaluation and model comparison
    for a single commodity.
    """
    # Raw (sparse) frame: each commodity keeps only its OWN observations.
    df = price_service.load_wide_prices(aligned=False)
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data found. Ensure the pipeline has run.")

    forecasts = _load_latest_forecasts()
    if not forecasts:
        raise HTTPException(
            status_code=404, detail="No forecast data available. Ensure the forecasting models have executed."
        )

    if commodity not in forecasts or forecasts[commodity].get("ensemble") is None:
        raise HTTPException(status_code=404, detail=f"No successful ensemble forecast found for '{commodity}'.")

    if commodity not in df.columns:
        raise HTTPException(status_code=404, detail=f"No historical price series found for '{commodity}'.")

    results = forecasts[commodity]
    info = ConfigLoader.get_commodity_info(commodity)
    frequency = info.get("frequency", "monthly")

    ensemble = results["ensemble"]
    ens_metrics = ensemble.get("metrics", {})
    meta = ens_metrics.get("meta") or {}
    individual = results.get("individual_models", {})

    ens_preds = ensemble.get("predictions", [])
    horizon = len(ens_preds)

    # ------------------------------------------------------------ #
    # The commodity's own history                                   #
    # ------------------------------------------------------------ #
    native = df.loc[df[commodity].notna(), ["date", commodity]]
    series = native[commodity]
    current_price = float(series.iloc[-1])
    last_date = native["date"].iloc[-1]

    future_dates = meta.get("future_dates") or _future_dates(last_date, frequency, horizon)

    cur_pct = price_service.percentile_of(series, current_price)
    next_p = ens_preds[0] if ens_preds else current_price
    next_chg = ((next_p - current_price) / current_price * 100) if current_price else None

    df_plot = native.tail(HISTORY_POINTS.get(frequency, 60))

    # ------------------------------------------------------------ #
    # Prediction intervals                                          #
    # ------------------------------------------------------------ #
    intervals = ens_metrics.get("intervals") or None
    next_interval = None
    if intervals and intervals.get("80") and horizon:
        next_interval = {
            "lower80": _num(intervals["80"]["lower"][0]),
            "upper80": _num(intervals["80"]["upper"][0]),
        }

    # ------------------------------------------------------------ #
    # Naive baseline                                                #
    # ------------------------------------------------------------ #
    baseline = None
    naive = individual.get(BASELINE_KEY)
    if naive and naive.get("predictions"):
        nm = naive["metrics"]
        baseline = {
            "method": naive["method"],
            "predictions": naive["predictions"][:horizon],
            "mae": _num(nm.get("mae")),
        }

    # ------------------------------------------------------------ #
    # Evaluation of the ensemble (rolling-origin backtest)          #
    # ------------------------------------------------------------ #
    evaluation = {
        "n_folds": ens_metrics.get("n_folds"),
        "mae": _num(ens_metrics.get("mae")),
        "mape": _num(ens_metrics.get("mape")),
        "rmae": _num(ens_metrics.get("rmae")),
        "skill_vs_naive": _num(ens_metrics.get("skill_vs_naive")),
        "directional_accuracy": _num(ens_metrics.get("directional_accuracy")),
        "coverage_80": _num(ens_metrics.get("coverage_80")),
        "coverage_95": _num(ens_metrics.get("coverage_95")),
        "naive_mae": baseline["mae"] if baseline else None,
    }

    # ------------------------------------------------------------ #
    # Model comparison (ranked by error relative to naive)          #
    # ------------------------------------------------------------ #
    top_keys = set(ens_metrics.get("top_models") or [])
    weights = ens_metrics.get("weights") or {}

    def rank_key(item):
        m = item[1]["metrics"]
        rmae = _num(m.get("rmae"))
        return rmae if rmae is not None else (_num(m.get("mape")) or 999) / 100

    scored = {
        k: v
        for k, v in individual.items()
        if v.get("role") != "volatility" and v.get("predictions") and _num(v["metrics"].get("mae")) is not None
    }
    ranked = sorted(scored.items(), key=rank_key)

    ranked_models = []
    for k, v in ranked:
        m = v["metrics"]
        ranked_models.append(
            {
                "key": k,
                "method": v["method"],
                "role": v.get("role", "model"),
                "is_baseline": k == BASELINE_KEY,
                "is_top": k in top_keys or v["method"] in top_keys,
                "weight": _num(weights.get(k)),
                "mape": _num(m.get("mape")),
                "mae": _num(m.get("mae")),
                "rmae": _num(m.get("rmae")),
                "skill_vs_naive": _num(m.get("skill_vs_naive")),
                "directional_accuracy": _num(m.get("directional_accuracy")),
                "predictions": (v.get("predictions") or [])[:horizon],
            }
        )
    top_methods = [m["method"] for m in ranked_models if m["is_top"]]

    # ------------------------------------------------------------ #
    # GARCH 90% volatility band (daily / weekly series only)        #
    # ------------------------------------------------------------ #
    garch_band = None
    garch = individual.get("garch")
    if garch and isinstance(garch["metrics"].get("band"), dict):
        band = garch["metrics"]["band"]
        garch_band = {
            "lower": band.get("lower", [])[:horizon],
            "upper": band.get("upper", [])[:horizon],
        }

    # ------------------------------------------------------------ #
    # Historical context for the forecast histogram                 #
    # ------------------------------------------------------------ #
    final_pred = ens_preds[-1] if ens_preds else None
    fut_pct = price_service.percentile_of(series, final_pred) if final_pred is not None else None

    return {
        "commodity": commodity,
        "name": price_service.commodity_display_name(commodity),
        "unit": info.get("unit", ""),
        "frequency": frequency,
        "period_label": PERIOD_LABEL.get(frequency, "Period"),
        "horizon": horizon,
        "current_price": current_price,
        "current_percentile": cur_pct,
        "confidence": ensemble.get("confidence", "low"),
        "avg_mape": _num(ens_metrics.get("avg_mape")) or 0,
        "next_period": {"value": next_p, "change_pct": next_chg, **(next_interval or {})},
        "history": {
            "dates": df_plot["date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": price_service.series_to_json(df_plot[commodity]),
        },
        "last_date": last_date.strftime("%Y-%m-%d"),
        "future_dates": future_dates,
        "ensemble": {
            "predictions": ens_preds,
            "confidence": ensemble.get("confidence", "low"),
            "avg_mape": _num(ens_metrics.get("avg_mape")) or 0,
            "top_models": top_methods,
            "weights": weights,
        },
        "intervals": intervals,
        "baseline": baseline,
        "evaluation": evaluation,
        "data_quality": {
            "n_obs": meta.get("n_obs", int(len(series))),
            "n_gap_filled": meta.get("n_gap_filled", 0),
        },
        "garch_band": garch_band,
        "ranked_models": ranked_models,
        "top_models": top_methods,
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
