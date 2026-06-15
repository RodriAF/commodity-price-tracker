"""
Ratios router — replaces ``pages/2_Ratios.py`` (Input Cost Analytics).

Recomputes cost indices, profitability ratios, z-scores, the market regime,
and a per-crop margin-pressure summary using ``CommoditiesAnalytics``, exactly
as the original page did. These derived metrics are intentionally NOT
persisted to DuckDB — they are cheap to recompute and stay consistent with
config changes.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from pipeline.calculations import CommoditiesAnalytics
from services import prices as price_service

router = APIRouter(tags=["ratios"])

# (level, display_label, description) per regime classification.
REGIME_META = {
    "high_cost": ("high", "ABOVE AVERAGE", "Energy inputs above historical norm — cost pressure on producers"),
    "low_cost": ("low", "BELOW AVERAGE", "Energy inputs below historical norm — favourable cost conditions"),
    "normal": ("normal", "NORMAL", "Energy within normal historical range"),
    "expensive": ("high", "ABOVE AVERAGE", "Fertilizer costs above historical norm"),
    "cheap": ("low", "BELOW AVERAGE", "Fertilizer costs below historical norm"),
}

INPUT_LABELS = {
    "crude_oil": "Crude Oil",
    "natural_gas": "Natural Gas",
    "diesel": "Diesel",
    "gasoline": "Gasoline",
    "urea": "Urea",
    "dap_fertilizer": "DAP",
    "phosphate": "Phosphate",
    "potash": "Potash",
}


def _input_label(name: str) -> str:
    return INPUT_LABELS.get(name, name.replace("_", " ").title())


def _crop_label(name: str) -> str:
    return name.replace("_", " ").title()


def _trend_direction(series: pd.Series, window: int = 3) -> str:
    """'up' / 'down' / 'flat' based on the slope over the last `window` points."""
    recent = series.dropna().tail(window)
    if len(recent) < 2:
        return "flat"
    slope = recent.iloc[-1] - recent.iloc[0]
    if slope > 0.5:
        return "up"
    if slope < -0.5:
        return "down"
    return "flat"


def _margin_status(z: float) -> tuple[str, str]:
    """(level, display_label) translation of a margin z-score."""
    if z < -2:
        return "pressure", "UNDER PRESSURE"
    if z < -1:
        return "elevated", "ELEVATED COSTS"
    if z > 1:
        return "normal", "STRONG MARGINS"
    return "normal", "NORMAL"


def _compute_analytics(df_wide: pd.DataFrame) -> dict:
    engine = CommoditiesAnalytics(df_wide)
    combined, analytics = engine.calculate_all()
    combined["date"] = df_wide["date"].values[: len(combined)]
    return {"combined": combined, **analytics}


@router.get("/ratios")
def get_ratios(
    pair: str | None = Query(
        None, description="Optional crop_to_input pair (e.g. 'corn_to_crude_oil') for the ratio explorer"
    )
):
    """
    Input-cost regime cards, the historical cost index, a per-crop margin
    pressure summary, the latest ratio z-scores, and (optionally) a detailed
    ratio explorer for one crop/input pair.
    """
    df_wide = price_service.load_wide_prices()
    if df_wide.empty:
        raise HTTPException(
            status_code=404,
            detail="No price data found in DuckDB. Run the pipeline first to populate the database.",
        )

    try:
        analytics = _compute_analytics(df_wide)
    except Exception as exc:  # pragma: no cover - defensive, mirrors st.error path
        raise HTTPException(status_code=500, detail=f"Analytics computation failed: {exc}") from exc

    indices = analytics.get("cost_indices", pd.DataFrame())
    profitability = analytics.get("profitability", pd.DataFrame())
    zscores = analytics.get("zscores", pd.DataFrame())
    regime = analytics.get("regime", {})

    if not indices.empty:
        indices = indices.copy()
        indices["date"] = df_wide["date"].values[: len(indices)]
    if not profitability.empty:
        profitability = profitability.copy()
        profitability["date"] = df_wide["date"].values[: len(profitability)]

    # ------------------------------------------------------------ #
    # Section 1 — Current input cost environment (regime cards)     #
    # ------------------------------------------------------------ #
    regime_out: dict[str, dict] = {}
    for regime_key, index_col, label in [
        ("energy", "energy_input_cost_index", "Energy Inputs"),
        ("fertilizer", "fertilizer_cost_index", "Fertilizers"),
    ]:
        val = regime.get(regime_key, "normal")
        level, display_label, description = REGIME_META.get(val, ("normal", val.upper().replace("_", " "), ""))

        idx_val = None
        delta_3m = None
        if not indices.empty and index_col in indices.columns:
            idx_series = indices[index_col].dropna()
            if not idx_series.empty:
                idx_val = float(idx_series.iloc[-1])
            if len(idx_series) >= 4:
                delta_3m = float(idx_series.iloc[-1] - idx_series.iloc[-4])

        regime_out[regime_key] = {
            "label": label,
            "status": val,
            "level": level,
            "display_label": display_label,
            "description": description,
            "index_value": idx_val,
            "delta_3m": delta_3m,
        }

    # ------------------------------------------------------------ #
    # Section 2 — Historical cost index                              #
    # ------------------------------------------------------------ #
    cost_index_history = None
    if not indices.empty:
        cost_index_history = {
            "dates": pd.to_datetime(indices["date"]).dt.strftime("%Y-%m-%d").tolist(),
            "series": {
                col: price_service.series_to_json(indices[col]) for col in indices.columns if col != "date"
            },
        }

    # ------------------------------------------------------------ #
    # Section 3 — Margin pressure by crop                            #
    # ------------------------------------------------------------ #
    margin_pressure = []
    if not profitability.empty and not zscores.empty:
        prof_cols = [c for c in profitability.columns if c != "date"]
        crops_seen = sorted(set(c.split("_to_")[0] for c in prof_cols))

        for crop in crops_seen:
            crop_pairs = [c for c in prof_cols if c.startswith(f"{crop}_to_")]
            z_cols = [f"{p}_zscore" for p in crop_pairs if f"{p}_zscore" in zscores.columns]
            if not z_cols:
                continue

            latest_zs = {
                col.replace("_zscore", "").replace(f"{crop}_to_", ""): float(zscores[col].dropna().iloc[-1])
                for col in z_cols
                if not zscores[col].dropna().empty
            }
            if not latest_zs:
                continue

            worst_input = min(latest_zs, key=latest_zs.get)
            worst_z = latest_zs[worst_input]

            worst_pair_col = f"{crop}_to_{worst_input}"
            trend = _trend_direction(profitability[worst_pair_col]) if worst_pair_col in profitability.columns else "flat"

            level, status_label = _margin_status(worst_z)

            margin_pressure.append(
                {
                    "crop": crop,
                    "crop_label": _crop_label(crop),
                    "status": status_label,
                    "level": level,
                    "trend": trend,
                    "worst_input": worst_input,
                    "worst_input_label": _input_label(worst_input),
                    "worst_z": worst_z,
                    "drivers": [
                        {"input": k, "label": _input_label(k), "z": v}
                        for k, v in sorted(latest_zs.items(), key=lambda x: x[1])
                    ],
                }
            )

        # Worst margin pressure (most negative z) first.
        margin_pressure.sort(key=lambda r: r["worst_z"])

    # ------------------------------------------------------------ #
    # Section 4a — Latest z-score bar chart (technical detail)      #
    # ------------------------------------------------------------ #
    zscore_bars = []
    if not profitability.empty and not zscores.empty:
        prof_cols = [c for c in profitability.columns if c != "date"]
        z_cols_avail = [f"{c}_zscore" for c in prof_cols if f"{c}_zscore" in zscores.columns]

        for col in z_cols_avail:
            s = zscores[col].dropna()
            if s.empty:
                continue
            pair_key = col.replace("_zscore", "")
            zscore_bars.append(
                {
                    "pair": pair_key,
                    "label": pair_key.replace("_to_", " / ").replace("_", " ").title(),
                    "z": float(s.iloc[-1]),
                }
            )
        zscore_bars.sort(key=lambda r: r["z"])

    # ------------------------------------------------------------ #
    # Section 4b — Ratio explorer (optional, requires ?pair=)       #
    # ------------------------------------------------------------ #
    ratio_pairs = [c for c in profitability.columns if c != "date"] if not profitability.empty else []

    ratio_explorer = None
    if pair:
        if pair not in ratio_pairs:
            raise HTTPException(status_code=400, detail=f"Unknown ratio pair '{pair}'. Available: {ratio_pairs}")

        series = profitability[pair].dropna()
        mean_val = float(series.mean()) if not series.empty else None
        std_val = float(series.std()) if not series.empty else None
        cur_val = float(series.iloc[-1]) if not series.empty else None

        z_val = 0.0
        if std_val and std_val > 0 and cur_val is not None and mean_val is not None:
            z_val = (cur_val - mean_val) / std_val

        pctile = price_service.percentile_of(series, cur_val)

        ratio_explorer = {
            "pair": pair,
            "label": pair.replace("_to_", " / ").replace("_", " ").title(),
            "dates": pd.to_datetime(profitability["date"]).dt.strftime("%Y-%m-%d").tolist(),
            "values": price_service.series_to_json(profitability[pair]),
            "mean": mean_val,
            "std": std_val,
            "current": cur_val,
            "zscore": z_val,
            "percentile": pctile,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
        }

    return {
        "regime": regime_out,
        "cost_index_history": cost_index_history,
        "margin_pressure": margin_pressure,
        "zscore_bars": zscore_bars,
        "ratio_pairs": ratio_pairs,
        "ratio_explorer": ratio_explorer,
    }
