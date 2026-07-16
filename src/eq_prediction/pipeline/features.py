from __future__ import annotations

import math

import numpy as np
import pandas as pd


FEATURE_VERSION = "simple_features_v1"


MODEL_BASE_COLUMNS = [
    "event_id",
    "time",
    "mag",
    "longitude",
    "latitude",
    "depth_km",
    "dmin",
    "dmin_km",
    "rms",
    "gap",
    "nst",
]


def add_missing_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("dmin", "gap", "nst", "rms"):
        if column in result.columns:
            result[f"{column}_missing"] = result[column].isna().astype("int8")
    return result


def add_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    time = pd.to_datetime(result["time"], utc=True, errors="coerce")
    result["hour_sin"] = np.sin(2 * math.pi * time.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * math.pi * time.dt.hour / 24)
    result["dayofweek_sin"] = np.sin(2 * math.pi * time.dt.dayofweek / 7)
    result["dayofweek_cos"] = np.cos(2 * math.pi * time.dt.dayofweek / 7)
    result["month_sin"] = np.sin(2 * math.pi * time.dt.month / 12)
    result["month_cos"] = np.cos(2 * math.pi * time.dt.month / 12)
    return result


def add_mag_type_features(dataframe: pd.DataFrame, min_frequency: float = 0.003) -> pd.DataFrame:
    result = dataframe.copy()
    mag_type = result.get("magType", pd.Series(index=result.index, dtype=object))
    mag_type = mag_type.fillna("unknown").astype(str).str.lower()
    frequencies = mag_type.value_counts(normalize=True)
    allowed = set(frequencies[frequencies >= min_frequency].index)
    grouped = mag_type.where(mag_type.isin(allowed), "other")
    encoded = pd.get_dummies(grouped, prefix="magType", dtype="int8")
    return pd.concat([result, encoded], axis=1)


def add_rolling_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.sort_values("time").copy()
    indexed = result.set_index("time")
    result["eq_count_last_1d"] = indexed["mag"].rolling("1D").count().to_numpy()
    result["eq_count_last_7d"] = indexed["mag"].rolling("7D").count().to_numpy()
    result["max_mag_last_7d"] = indexed["mag"].rolling("7D").max().to_numpy()
    return result


def fill_numeric_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns
    for column in numeric_cols:
        median = result[column].median() if result[column].notna().any() else 0
        result[column] = result[column].fillna(median)
    return result


def empty_values_handler(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()


def make_model_ready(dataframe: pd.DataFrame, timeseries: bool = False) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.sort_values("time").reset_index(drop=True).copy()
    result["dmin_km"] = pd.to_numeric(result["dmin"], errors="coerce") * 111.19
    result = add_missing_indicators(result)
    result = add_mag_type_features(result)

    if timeseries:
        result = add_time_features(result)
        result = add_rolling_features(result)

    # result = fill_numeric_values(result)
    # result = 
    result["feature_version"] = FEATURE_VERSION

    passthrough = [column for column in MODEL_BASE_COLUMNS if column in result.columns]
    engineered = [
        column
        for column in result.columns
        if column.endswith("_missing")
        or column.startswith("hour_")
        or column.startswith("dayofweek_")
        or column.startswith("month_")
        or column.startswith("magType_")
        or column.startswith("eq_count_")
        or column.startswith("max_mag_")
    ]
    return result[passthrough + engineered + ["feature_version"]].reset_index(drop=True)
