"""
FastAPI entry point for the Agricultural Commodity Tracker API.

The React dev server (Vite, default port 5173) is allowed via CORS below.
"""

from __future__ import annotations

import logging
import os
import sys

# Make `pipeline`, `utils` and `config` (one level up) importable, mirroring
# the sys.path.insert(...) pattern used by the existing Streamlit pages.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import utils.db as db
from routers import analysis, commodities, forecast, overview, ratios

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agricultural Commodity Tracker API",
    description=(
        "Decoupled FastAPI backend for the commodity tracker dashboard. "
        "Reads from the same DuckDB warehouse used by the Streamlit app and pipeline."
    ),
    version="1.0.0",
)

# ------------------------------------------------------------------ #
# CORS — local React dev servers                                      #
# ------------------------------------------------------------------ #
# Vite's default dev port is 5173; 3000 is included for CRA / other
# tooling. Both http and https are NOT needed locally, but 127.0.0.1
# is included since some browsers treat it as a different origin than
# localhost.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Startup                                                             #
# ------------------------------------------------------------------ #
@app.on_event("startup")
def on_startup() -> None:
    """The pipeline service owns schema creation; the API is read-only."""
    logger.info("Backend ready (read-only mode).")


# ------------------------------------------------------------------ #
# Error handling                                                      #
# ------------------------------------------------------------------ #
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all so unexpected errors (e.g. a missing DuckDB file, a bad
    config file) come back as a clean JSON 500 instead of a stack trace,
    while still being logged server-side for debugging.
    """
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


# ------------------------------------------------------------------ #
# Health check                                                        #
# ------------------------------------------------------------------ #
@app.get("/api/health", tags=["health"])
def health_check():
    """Lightweight readiness probe used by the frontend on boot."""
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc


# ------------------------------------------------------------------ #
# Routers                                                             #
# ------------------------------------------------------------------ #
app.include_router(overview.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(ratios.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(commodities.router, prefix="/api")
