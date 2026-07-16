from __future__ import annotations

import pandas as pd


NUMERIC_COLUMNS = ["longitude", "latitude", "depth_km", "mag", "nst", "dmin", "rms", "gap"]


def coerce_raw_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("time", "updated"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def validate_and_filter(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe.empty:
        return dataframe.copy(), dataframe.copy()

    result = coerce_raw_types(dataframe)
    mask = result["event_id"].notna()
    mask &= result["time"].notna()
    mask &= result["mag"].notna()
    mask &= result["longitude"].between(-180, 180)
    mask &= result["latitude"].between(-90, 90)
    if "type" in result.columns:
        mask &= result["type"].fillna("earthquake").astype(str).str.lower().eq("earthquake")

    return result.loc[mask].copy(), result.loc[~mask].copy()


def deduplicate_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    result = dataframe.copy()
    sort_columns = ["event_id"]
    if "updated" in result.columns:
        sort_columns.append("updated")
    elif "time" in result.columns:
        sort_columns.append("time")
    return (
        result.sort_values(sort_columns, na_position="first")
        .drop_duplicates(subset=["event_id"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def clean_raw_events(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid, rejected = validate_and_filter(dataframe)
    clean = deduplicate_events(valid)
    return clean, rejected.reset_index(drop=True)
