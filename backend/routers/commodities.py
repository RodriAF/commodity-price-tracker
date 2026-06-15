"""
Commodities metadata router.

Not a 1:1 replacement of a single Streamlit page — every page reads
``ConfigLoader`` directly to build its selectors (sector dropdown, asset
multiselect, etc.). This endpoint exposes that same configuration so the
React frontend can build those controls without duplicating
``commodities.json``.
"""

from __future__ import annotations

from fastapi import APIRouter

from utils.config_loader import ConfigLoader

router = APIRouter(tags=["commodities"])


@router.get("/commodities")
def get_commodities():
    """Full commodity registry plus the category -> [commodity_key] grouping."""
    commodities = ConfigLoader.get_commodities()
    categories = ConfigLoader.get_categories()

    return {
        "commodities": [
            {
                "key": key,
                "name": info.get("name", key.replace("_", " ").title()),
                "category": info.get("category", "other"),
                "frequency": info.get("frequency", "monthly"),
                "unit": info.get("unit", ""),
            }
            for key, info in commodities.items()
        ],
        "categories": categories,
    }
