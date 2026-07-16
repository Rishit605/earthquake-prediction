from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .split import DatasetSplits


TARGET_COLUMN = "mag"


@dataclass(frozen=True)
class TrainingOutputs:
    model_ready: pd.DataFrame
    splits: DatasetSplits
    prediction_input: pd.DataFrame
    scaler: dict[str, dict[str, float]]


def numeric_training_frame(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    numeric = dataframe.select_dtypes(include=["number"]).copy()
    if target_column not in numeric.columns:
        raise ValueError(f"Target column '{target_column}' is missing from model-ready data.")
    return numeric.reset_index(drop=True)


def fit_zscore_params(
    train: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> dict[str, dict[str, float]]:
    feature_columns = [column for column in train.columns if column != target_column]
    params: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        fill_value = float(train[column].median()) if train[column].notna().any() else 0.0
        filled = train[column].fillna(fill_value)
        mean = float(filled.mean())
        std = float(filled.std())
        if pd.isna(std) or std == 0:
            std = 1.0
        if pd.isna(mean):
            mean = 0.0
        params[column] = {"mean": mean, "std": std, "fill_value": fill_value}
    return params


def apply_zscore(
    dataframe: pd.DataFrame,
    params: dict[str, dict[str, float]],
) -> pd.DataFrame:
    result = dataframe.copy()
    for column, values in params.items():
        if column in result.columns:
            result[column] = result[column].fillna(values["fill_value"])
            result[column] = (result[column] - values["mean"]) / values["std"]
    return result


def build_training_outputs(
    model_ready_with_metadata: pd.DataFrame,
    splits_with_metadata: DatasetSplits,
    target_column: str = TARGET_COLUMN,
) -> TrainingOutputs:
    train = numeric_training_frame(splits_with_metadata.train, target_column)
    validation = numeric_training_frame(splits_with_metadata.validation, target_column)
    test = numeric_training_frame(splits_with_metadata.test, target_column)
    model_ready = numeric_training_frame(model_ready_with_metadata, target_column)

    scaler = fit_zscore_params(train, target_column)
    train_scaled = apply_zscore(train, scaler)
    validation_scaled = apply_zscore(validation, scaler)
    test_scaled = apply_zscore(test, scaler)
    model_ready_scaled = apply_zscore(model_ready, scaler)

    prediction_input = model_ready_scaled.tail(1).drop(columns=[target_column], errors="ignore")

    return TrainingOutputs(
        model_ready=model_ready_scaled,
        splits=DatasetSplits(
            train=train_scaled,
            validation=validation_scaled,
            test=test_scaled,
        ),
        prediction_input=prediction_input.reset_index(drop=True),
        scaler=scaler,
    )
