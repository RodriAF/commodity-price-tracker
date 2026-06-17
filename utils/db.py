"""
Centralized DuckDB database operations for the agricultural commodities pipeline.

This module encapsulates all SQL queries and database interactions. No SQL
should be written outside of this file. It leverages the DuckDB Python API
and integrates seamlessly with Pandas DataFrames for bulk operations.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime
from typing import Any, Dict, Generator, Optional

import duckdb
import pandas as pd

# Assuming settings.py is located at config/settings.py
from config.settings import get_settings
import os

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Connection Management                                              #
# ------------------------------------------------------------------ #

@contextlib.contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager for DuckDB connections.

    Yields a configured DuckDB connection pointing to the path defined
    in the global settings. Automatically closes the connection upon exit.

    Usage:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    """
    settings = get_settings()
    # Reads 'DATABASE_READ_ONLY' from the environment. 
    # If it is missing, it defaults to False (so the pipeline can write).
    is_read_only = os.getenv("DATABASE_READ_ONLY", "false").lower() == "true"
    
    conn = duckdb.connect(str(settings.duckdb_path), read_only=is_read_only)
    try:
        yield conn
    finally:
        conn.close()

# ------------------------------------------------------------------ #
# DDL / Initialization                                               #
# ------------------------------------------------------------------ #

def initialize() -> None:
    """
    Initialize the DuckDB database schema.

    Creates all necessary tables for the pipeline if they do not exist.
    Primary keys are defined to enable ON CONFLICT (upsert) behaviors.
    """
    logger.info("Initializing DuckDB schema...")
    
    # SQL Statements for table creation
    create_prices_sql = """
        CREATE TABLE IF NOT EXISTS prices (
            date DATE,
            commodity VARCHAR,
            value DOUBLE,
            is_imputed BOOLEAN,
            created_at TIMESTAMP,
            PRIMARY KEY (date, commodity)
        );
    """

    create_signals_sql = """
        CREATE TABLE IF NOT EXISTS signals (
            run_date DATE,
            commodity VARCHAR,
            metric VARCHAR,
            signal_type VARCHAR,
            z_score DOUBLE,
            percentile DOUBLE,
            PRIMARY KEY (run_date, commodity, metric)
        );
    """

    create_forecasts_sql = """
        CREATE TABLE IF NOT EXISTS forecasts (
            run_date DATE,
            commodity VARCHAR,
            model VARCHAR,
            horizon INTEGER,
            predictions_json VARCHAR,
            metrics_json VARCHAR,
            confidence VARCHAR,
            PRIMARY KEY (run_date, commodity, model, horizon)
        );
    """

    create_pipeline_runs_sql = """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id VARCHAR,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status VARCHAR,
            summary_json VARCHAR,
            PRIMARY KEY (run_id)
        );
    """

    with get_connection() as conn:
        conn.execute(create_prices_sql)
        conn.execute(create_signals_sql)
        conn.execute(create_forecasts_sql)
        conn.execute(create_pipeline_runs_sql)
        
    logger.info("Database schema initialized successfully.")


# ------------------------------------------------------------------ #
# Write Operations (Upserts)                                         #
# ------------------------------------------------------------------ #

def upsert_prices(df: pd.DataFrame) -> None:
    """
    Insert or update historical prices in the database.

    Args:
        df: DataFrame containing columns matching the `prices` table:
            (date, commodity, value, is_imputed, created_at)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to upsert_prices. Skipping.")
        return

    upsert_sql = """
        INSERT INTO prices (date, commodity, value, is_imputed, created_at)
        SELECT 
            date::DATE, 
            commodity::VARCHAR, 
            value::DOUBLE, 
            is_imputed::BOOLEAN, 
            created_at::TIMESTAMP
        FROM df
        ON CONFLICT (date, commodity) DO UPDATE SET
            value = EXCLUDED.value,
            is_imputed = EXCLUDED.is_imputed,
            created_at = EXCLUDED.created_at;
    """
    
    with get_connection() as conn:
        # DuckDB automatically finds the local variable 'df' 
        # and lets us query it directly.
        conn.execute(upsert_sql)
        logger.info(f"Upserted {len(df)} rows into 'prices' table.")


def upsert_signals(df: pd.DataFrame) -> None:
    """
    Insert or update generated signals in the database.

    Args:
        df: DataFrame containing columns matching the `signals` table:
            (run_date, commodity, metric, signal_type, z_score, percentile)
    """
    if df.empty:
        return

    upsert_sql = """
        INSERT INTO signals (run_date, commodity, metric, signal_type, z_score, percentile)
        SELECT 
            run_date::DATE, 
            commodity::VARCHAR, 
            metric::VARCHAR, 
            signal_type::VARCHAR, 
            z_score::DOUBLE, 
            percentile::DOUBLE
        FROM df
        ON CONFLICT (run_date, commodity, metric) DO UPDATE SET
            signal_type = EXCLUDED.signal_type,
            z_score = EXCLUDED.z_score,
            percentile = EXCLUDED.percentile;
    """
    
    with get_connection() as conn:
        conn.execute(upsert_sql)
        logger.info(f"Upserted {len(df)} rows into 'signals' table.")


def upsert_forecasts(df: pd.DataFrame) -> None:
    """
    Insert or update forecasts in the database.

    Args:
        df: DataFrame containing columns matching the `forecasts` table:
            (run_date, commodity, model, horizon, predictions_json, metrics_json, confidence)
    """
    if df.empty:
        return

    upsert_sql = """
    INSERT INTO forecasts (run_date, commodity, model, horizon, predictions_json, metrics_json, confidence)
    SELECT 
        run_date::DATE, 
        commodity::VARCHAR, 
        model::VARCHAR, 
        horizon::INTEGER, 
        predictions_json::VARCHAR, 
        metrics_json::VARCHAR, 
        confidence::VARCHAR
    FROM df
    ON CONFLICT (run_date, commodity, model, horizon) DO UPDATE SET
        predictions_json = EXCLUDED.predictions_json,
        metrics_json = EXCLUDED.metrics_json,
        confidence = EXCLUDED.confidence;
"""
    
    with get_connection() as conn:
        conn.execute(upsert_sql)
        logger.info(f"Upserted {len(df)} rows into 'forecasts' table.")


def log_run(run_id: str, started_at: datetime, status: str, 
            completed_at: Optional[datetime] = None, 
            summary_json: Optional[str] = None) -> None:
    """
    Log or update a pipeline run metadata.

    Args:
        run_id: Unique identifier for the pipeline run.
        started_at: Timestamp when the run started.
        status: Current status (e.g., 'RUNNING', 'SUCCESS', 'FAILED').
        completed_at: Timestamp when the run finished (if applicable).
        summary_json: JSON string with metadata/stats of the run.
    """
    upsert_sql = """
        INSERT INTO pipeline_runs (run_id, started_at, completed_at, status, summary_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            completed_at = EXCLUDED.completed_at,
            status = EXCLUDED.status,
            summary_json = EXCLUDED.summary_json;
    """
    
    with get_connection() as conn:
        conn.execute(
            upsert_sql, 
            [run_id, started_at, completed_at, status, summary_json]
        )
        logger.info(f"Pipeline run '{run_id}' logged with status: {status}")


# ------------------------------------------------------------------ #
# Read Operations                                                    #
# ------------------------------------------------------------------ #

def load_prices(commodity: Optional[str] = None, 
                start_date: Optional[str | datetime] = None, 
                end_date: Optional[str | datetime] = None) -> pd.DataFrame:
    """
    Load historical prices from the database.

    Args:
        commodity: Filter by a specific commodity (optional).
        start_date: Filter for dates >= start_date (optional).
        end_date: Filter for dates <= end_date (optional).

    Returns:
        A Pandas DataFrame with the requested prices, sorted by date.
    """
    query = "SELECT * FROM prices WHERE 1=1"
    params: list[Any] = []

    if commodity:
        query += " AND commodity = ?"
        params.append(commodity)
        
    if start_date:
        query += " AND date >= ?::DATE"
        params.append(start_date)
        
    if end_date:
        query += " AND date <= ?::DATE"
        params.append(end_date)
        
    query += " ORDER BY date ASC"

    with get_connection() as conn:
        # execute() runs the query, df() fetches the result as a Pandas DataFrame
        df = conn.execute(query, params).df()
        
    return df

def check_if_schema_exists() -> bool:
    """
    Check if the primary 'pipeline_runs' table already exists in the database.

    Returns:
        True if the schema is initialized, False otherwise.
    """
    with get_connection() as conn:
        # Query DuckDB's internal information schema to verify table existence
        result = conn.execute("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'pipeline_runs'
            )
        """).fetchone()
        return result[0] if result else False