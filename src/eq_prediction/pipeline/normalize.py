from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd


POINT_RE = re.compile(
    r"POINT\s+Z?\s*\(\s*"
    r"(?P<lon>-?\d+(?:\.\d+)?)\s+"
    r"(?P<lat>-?\d+(?:\.\d+)?)"
    r"(?:\s+(?P<depth>-?\d+(?:\.\d+)?))?"
    r"\s*\)",
    re.IGNORECASE,
)

RAW_COLUMNS = [
    "event_id",
    "time",
    "updated",
    "longitude",
    "latitude",
    "depth_km",
    "mag",
    "magType",
    "type",
    "status",
    "detail",
    "nst",
    "dmin",
    "rms",
    "gap",
    "source",
]


def parse_geo_wkt(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None or pd.isna(value):
        return None, None, None
    match = POINT_RE.search(str(value))
    if not match:
        return None, None, None
    return (
        float(match.group("lon")),
        float(match.group("lat")),
        float(match.group("depth")) if match.group("depth") is not None else None,
    )


def _fallback_event_id(row: pd.Series) -> str:
    parts = [
        str(row.get("time", "")),
        str(row.get("latitude", "")),
        str(row.get("longitude", "")),
        str(row.get("mag", "")),
        str(row.get("code", "")),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"legacy_{digest}"


def convert_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_legacy_dataframe(dataframe: pd.DataFrame, source: str = "local") -> pd.DataFrame:
    result = dataframe.copy()

    if "event_id" not in result.columns:
        if "id" in result.columns:
            result["event_id"] = result["id"]
        elif "code" in result.columns:
            result["event_id"] = result["code"]
        else:
            result["event_id"] = result.apply(_fallback_event_id, axis=1)

    missing_coords = {"longitude", "latitude"}.difference(result.columns)
    if missing_coords and "geo" in result.columns:
        parsed = result["geo"].map(parse_geo_wkt)
        result["longitude"] = parsed.map(lambda parts: parts[0])
        result["latitude"] = parsed.map(lambda parts: parts[1])
        if "depth_km" not in result.columns:
            result["depth_km"] = parsed.map(lambda parts: parts[2])

    if "depth_km" not in result.columns and "elevation" in result.columns:
        result["depth_km"] = result["elevation"]
    if "depth_km" not in result.columns and "depth" in result.columns:
        result["depth_km"] = result["depth"]
    if "type" not in result.columns and "data_type" in result.columns:
        result["type"] = result["data_type"]
    if "type" not in result.columns:
        result["type"] = "earthquake"
    if "mag" not in result.columns and "magnitudo" in result.columns:
        result["mag"] = result["magnitudo"]

    for column in ("time", "updated"):
        if column in result.columns:
            result[column] = convert_timestamp(result[column])
        else:
            result[column] = pd.NaT

    result["source"] = source
    for column in RAW_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[RAW_COLUMNS].reset_index(drop=True)


def _geometry_parts(feature: dict[str, Any]) -> tuple[Any, Any, Any]:
    coordinates = (feature.get("geometry") or {}).get("coordinates") or []
    longitude = coordinates[0] if len(coordinates) > 0 else None
    latitude = coordinates[1] if len(coordinates) > 1 else None
    depth_km = coordinates[2] if len(coordinates) > 2 else None
    return longitude, latitude, depth_km


def normalize_usgs_features(features: list[dict[str, Any]], source: str = "usgs") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        longitude, latitude, depth_km = _geometry_parts(feature)
        rows.append(
            {
                "event_id": feature.get("id") or properties.get("code"),
                "time": properties.get("time"),
                "updated": properties.get("updated"),
                "longitude": longitude,
                "latitude": latitude,
                "depth_km": depth_km,
                "mag": properties.get("mag"),
                "magType": properties.get("magType"),
                "type": properties.get("type"),
                "status": properties.get("status"),
                "detail": properties.get("detail"),
                "nst": properties.get("nst"),
                "dmin": properties.get("dmin"),
                "rms": properties.get("rms"),
                "gap": properties.get("gap"),
                "source": source,
            }
        )
    return normalize_legacy_dataframe(pd.DataFrame(rows), source=source)


def combine_raw_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(non_empty, ignore_index=True)
