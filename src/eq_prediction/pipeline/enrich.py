from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import PipelineSettings


KEY_MAP = {
    "nst": "num-stations-used",
    "dmin": "minimum-distance",
    "gap": "azimuthal-gap",
}

PATCH_VALUE_COLUMNS = tuple(KEY_MAP)
PATCH_INDEX_COLUMNS = ("oldindex", "old_index", "old_idx", "idx", "index", "Unnamed: 0")
ENRICHMENT_PATCH_NOT_FOUND = "enrichment patch file has not been found"


def numeric_value(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def search_detail_json(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("type") == "FeatureCollection" and data.get("features"):
        data = data["features"][0]
    products = data.get("properties", {}).get("products", {})
    found: dict[str, Any] = {}
    for category in ("origin", "phase-data"):
        for entry in products.get(category) or []:
            properties = entry.get("properties") or {}
            for key in KEY_MAP.values():
                if key in properties and key not in found:
                    found[key] = numeric_value(properties[key])
    return found


class DetailCache:
    def __init__(self, settings: PipelineSettings) -> None:
        self.cache_dir = settings.cache_dir / "detail_json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = settings.request_timeout_seconds
        self.min_interval = settings.request_min_interval_seconds
        self._next_request_at = 0.0

    def _path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def fetch(self, url: str, index: int) -> dict[str, Any]:
        path = self._path_for(url)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        wait = max(0.0, self._next_request_at - time.monotonic())
        if wait:
            time.sleep(wait)
        self._next_request_at = time.monotonic() + self.min_interval

        response = requests.get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": "eq-prediction-data-pipeline/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        payload["index"] = index
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


def _load_patch_from_database(settings: PipelineSettings) -> pd.DataFrame | None:
    if not settings.database.enrich_patch_schema or not settings.database.enrich_patch_table:
        return None
    from sqlalchemy import create_engine

    from .sources import build_database_url

    engine = create_engine(build_database_url(settings.database), pool_pre_ping=True)
    try:
        return pd.read_sql_table(
            settings.database.enrich_patch_table,
            engine,
            schema=settings.database.enrich_patch_schema,
        )
    finally:
        engine.dispose()


def load_existing_enrichment_patch(settings: PipelineSettings) -> pd.DataFrame | str:
    if settings.enrichment_patch_path.exists():
        return pd.read_csv(settings.enrichment_patch_path)

    try:
        patch_dataframe = _load_patch_from_database(settings)
    except Exception:  # noqa: BLE001 - absence of the optional patch should fall back
        return ENRICHMENT_PATCH_NOT_FOUND
    if patch_dataframe is not None:
        return patch_dataframe

    return ENRICHMENT_PATCH_NOT_FOUND


def load_saved_missing_value_patch(settings: PipelineSettings) -> pd.DataFrame | None:
    patch_dataframe = load_existing_enrichment_patch(settings)
    return patch_dataframe if isinstance(patch_dataframe, pd.DataFrame) else None


def _fill_patch_values(
    result: pd.DataFrame,
    row_mask: pd.Series,
    patch_row: pd.Series,
) -> int:
    patched_values = 0
    if not row_mask.any():
        return patched_values
    for column in PATCH_VALUE_COLUMNS:
        if column not in result.columns or column not in patch_row.index:
            continue
        value = pd.to_numeric(pd.Series([patch_row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        missing_mask = row_mask & result[column].isna()
        if missing_mask.any():
            result.loc[missing_mask, column] = value
            patched_values += int(missing_mask.sum())
    return patched_values


def patch_saved_missing_values(
    dataframe: pd.DataFrame,
    patch_dataframe: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int]:
    if dataframe.empty or patch_dataframe is None or patch_dataframe.empty:
        return dataframe.copy(), 0

    result = dataframe.copy()
    patch = patch_dataframe.copy()
    for column in PATCH_VALUE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    patched_values = 0

    if "detail" in result.columns and "detail" in patch.columns:
        result_detail = result["detail"].fillna("").astype(str).str.strip()
        for _, patch_row in patch.loc[patch["detail"].notna()].iterrows():
            detail = str(patch_row["detail"]).strip()
            if detail:
                patched_values += _fill_patch_values(result, result_detail.eq(detail), patch_row)

    if "event_id" in result.columns and "event_id" in patch.columns:
        result_event_id = result["event_id"].fillna("").astype(str)
        for _, patch_row in patch.loc[patch["event_id"].notna()].iterrows():
            event_id = str(patch_row["event_id"])
            if event_id:
                patched_values += _fill_patch_values(result, result_event_id.eq(event_id), patch_row)

    index_column = next((column for column in PATCH_INDEX_COLUMNS if column in patch.columns), None)
    if index_column is not None:
        for _, patch_row in patch.loc[patch[index_column].notna()].iterrows():
            try:
                old_index = int(patch_row[index_column])
            except (TypeError, ValueError):
                continue
            if old_index in result.index:
                row_mask = pd.Series(False, index=result.index)
                row_mask.loc[old_index] = True
                patched_values += _fill_patch_values(result, row_mask, patch_row)

    result.attrs["saved_patch_values"] = patched_values
    return result, patched_values


def enrich_missing_detail_columns(
    dataframe: pd.DataFrame,
    settings: PipelineSettings,
    fetch_missing: bool = False,
    patch_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()
    for column in KEY_MAP:
        if column not in result.columns:
            result[column] = pd.NA
    if "enrichment_status" not in result.columns:
        result["enrichment_status"] = "not_needed"

    if patch_dataframe is None:
        patch_dataframe = load_saved_missing_value_patch(settings)
    result, patched_values = patch_saved_missing_values(result, patch_dataframe)
    result.attrs["saved_patch_values"] = patched_values
        
    result.attrs["loaded_saved_patch_rows"] = (
        len(patch_dataframe) if isinstance(patch_dataframe, pd.DataFrame) else 0
    )
    if patched_values:
        result["enrichment_status"] = result["enrichment_status"].where(
            ~result[list(PATCH_VALUE_COLUMNS)].notna().any(axis=1),
            "patched_from_saved_data",
        )

    if not fetch_missing:
        return result

    if "detail" not in result.columns:
        return result

    needed = result["detail"].notna() & result[list(KEY_MAP)].isna().any(axis=1)
    # return needed
    if not needed.any():
        return result

    cache = DetailCache(settings)
    for index, row in result.loc[needed].iterrows():
        url = str(row["detail"]).strip()
        if not url:
            result.at[index, "enrichment_status"] = "missing_url"
            continue
        try:
            found = search_detail_json(cache.fetch(url, index))
            for column, json_key in KEY_MAP.items():
                if pd.isna(result.at[index, column]) and json_key in found:
                    result.at[index, column] = found[json_key]
            result.at[index, "enrichment_status"] = "success"
        except Exception as exc:  # noqa: BLE001 - record per-row enrichment failure
            result.at[index, "enrichment_status"] = f"failed:{type(exc).__name__}"
    return result
