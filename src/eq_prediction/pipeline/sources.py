from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from pathlib import Path

from .config import DatabaseSettings, PipelineSettings
from .normalize import normalize_legacy_dataframe, normalize_usgs_features


def load_local_csv(settings: PipelineSettings, file_path: Path | None = None) -> pd.DataFrame:
    local_path = file_path or settings.local_raw_patch_path
    if not local_path.exists():
        raise FileNotFoundError(f"Local raw CSV not found: {local_path}")
    return normalize_legacy_dataframe(pd.read_csv(local_path), source="local")


def build_database_url(database: DatabaseSettings):
    from sqlalchemy.engine import URL

    if not database.available:
        raise RuntimeError(
            "Database settings are incomplete. Set DB_UNAME, DB_PASSWORD, "
            "DB_PORT, PC_IP_ADDRESS, and EQ_DB_NAME."
        )
    return URL.create(
        drivername="postgresql+psycopg",
        username=database.username,
        password=database.password,
        host=database.host,
        port=database.port,
        database=database.database,
    )


def load_db_table(settings: PipelineSettings, file_path: list[Path, Path] | None = None) -> pd.DataFrame:
    from sqlalchemy import create_engine

    db_table = file_path[0] or settings.database.raw_patch_table
    db_schema = file_path[-1] or settings.database.raw_patch_schema

    engine = create_engine(build_database_url(settings.database), pool_pre_ping=True)
    try:
        dataframe = pd.read_sql_table(
            db_table,
            engine,
            schema=db_schema,
        )
    finally:
        engine.dispose()
    return normalize_legacy_dataframe(dataframe, source="db")


def _as_usgs_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def fetch_usgs_window(
    settings: PipelineSettings,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    params = {
        "format": "geojson",
        "orderby": "time",
        "minmagnitude": settings.min_magnitude,
        "starttime": _as_usgs_time(start),
        "endtime": _as_usgs_time(end),
    }
    response = requests.get(
        settings.usgs_endpoint,
        params=params,
        timeout=settings.request_timeout_seconds,
        headers={"User-Agent": "eq-prediction-data-pipeline/0.1"},
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    if not isinstance(features, list):
        raise ValueError("USGS response did not contain a features list.")
    return features


def fetch_new_usgs_events(
    settings: PipelineSettings,
    existing_raw: pd.DataFrame | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    if end is None:
        end = now
    if start is None:
        if existing_raw is not None and not existing_raw.empty and "time" in existing_raw:
            latest = pd.to_datetime(existing_raw["time"], utc=True, errors="coerce").max()
            if pd.notna(latest):
                start = latest.to_pydatetime() - timedelta(days=settings.overlap_days)
            else:
                start = end - timedelta(days=settings.fetch_days)
        else:
            start = end - timedelta(days=settings.fetch_days)
    if start >= end:
        raise ValueError("USGS fetch start must be before end.")

    features: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        features.extend(fetch_usgs_window(settings, cursor, chunk_end))
        cursor = chunk_end
        if cursor < end:
            time.sleep(settings.request_min_interval_seconds)
    return normalize_usgs_features(features, source="usgs")
