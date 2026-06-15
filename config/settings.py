"""
Centralized configuration for the agricultural commodities data pipeline.

Variables are loaded from the environment or from a .env file at the project root.
Always access settings through get_settings() to leverage the singleton cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: two levels above this file (config/settings.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Global pipeline configuration.

    Variables are resolved in this order (highest → lowest precedence):
    1. OS environment variables.
    2. .env file at the project root.
    3. Default values defined here.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # FRED_API_KEY == fred_api_key
        extra="ignore",             # ignore unknown environment variables
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ #
    # FRED API                                                             #
    # ------------------------------------------------------------------ #
    fred_api_key: str = Field(
        default="your_fred_api_key_here",
        alias="FRED_API_KEY",
        description=(
            "FRED (Federal Reserve Economic Data) API key. "
            "Get one for free at https://fred.stlouisfed.org/docs/api/api_key.html"
        ),
    )

    # ------------------------------------------------------------------ #
    # Data configuration                                                   #
    # ------------------------------------------------------------------ #
    days_history: int = Field(
        default=90,
        alias="DAYS_HISTORY",
        ge=1,
        le=3650,
        description="Number of days of history to download on each pipeline run.",
    )

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #
    data_dir: Path = Field(
        default=Path("data"),
        alias="DATA_DIR",
        description="Base directory for storing raw data, processed data, and the DuckDB database.",
    )
    
    # ------------------------------------------------------------------ #
    # Alerts & SMTP Configuration                                        #
    # ------------------------------------------------------------------ #
    alert_zscore_threshold: float = Field(
        default=2.0,
        alias="ALERT_ZSCORE_THRESHOLD",
        description="Z-score threshold to trigger an alert."
    )
    smtp_server: str | None = Field(
        default=None,
        alias="SMTP_SERVER",
        description="SMTP server address for sending emails."
    )
    smtp_port: int = Field(
        default=587,
        alias="SMTP_PORT"
    )
    smtp_username: str | None = Field(
        default=None,
        alias="SMTP_USERNAME"
    )
    smtp_password: str | None = Field(
        default=None,
        alias="SMTP_PASSWORD"
    )
    alert_email_to: str = Field(
        default="admin@example.com",
        alias="ALERT_EMAIL_TO"
    )
    alert_email_from: str = Field(
        default="pipeline@example.com",
        alias="ALERT_EMAIL_FROM"
    )

    # ------------------------------------------------------------------ #
    # Derived paths (computed, not read from .env)                        #
    # ------------------------------------------------------------------ #
    raw_dir: Path = Path("_unset_")
    processed_dir: Path = Path("_unset_")
    duckdb_path: Path = Path("_unset_")

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("fred_api_key", mode="after")
    @classmethod
    def warn_placeholder_key(cls, v: str) -> str:
        if v == "your_fred_api_key_here":
            import warnings
            warnings.warn(
                "FRED_API_KEY is not configured. "
                "The pipeline will fail when attempting to call the API. "
                "Add your key to the .env file or as an environment variable.",
                stacklevel=2,
            )
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        # If relative, anchor to the project root
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    @model_validator(mode="after")
    def build_derived_paths(self) -> Settings:
        """Build derived subdirectories and ensure they exist on disk."""
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.duckdb_path = self.data_dir / "commodities.duckdb"

        # Create directories if they don't exist (safe at import time)
        for directory in (self.raw_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return self

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @property
    def fred_api_key_is_set(self) -> bool:
        """True if the FRED key looks valid (i.e. not the placeholder value)."""
        return self.fred_api_key != "your_fred_api_key_here"

    def __repr__(self) -> str:  # mask the key in logs
        masked = f"{self.fred_api_key[:4]}****" if self.fred_api_key_is_set else "<not configured>"
        return (
            f"Settings("
            f"fred_api_key={masked!r}, "
            f"days_history={self.days_history}, "
            f"data_dir={self.data_dir}, "
            f"duckdb_path={self.duckdb_path}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the Settings singleton.

    The lru_cache decorator ensures Settings() is instantiated only once per
    process, regardless of how many times this function is called.

    Usage:
        from config.settings import get_settings

        settings = get_settings()
        api_key = settings.fred_api_key
        db_path = settings.duckdb_path

    For tests, clear the cache before changing environment variables:
        get_settings.cache_clear()
    """
    return Settings()