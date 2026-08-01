from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineSettings
from .models import PipelineResult
from .split import DatasetSplits


def _write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def save_outputs(
    *,
    settings: PipelineSettings,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    model_ready: pd.DataFrame,
    splits: DatasetSplits,
    rejected: pd.DataFrame,
    result: PipelineResult,
    prediction_input: pd.DataFrame | None = None,
    fetched_raw: pd.DataFrame | None = None,
) -> dict[str, Path]:
    paths = {
        "raw": _write_csv(raw, settings.raw_output_path),
        "clean": _write_csv(clean, settings.clean_output_path),
        "model_ready": _write_csv(model_ready, settings.model_ready_output_path),
        "train": _write_csv(splits.train, settings.train_output_path),
        "validation": _write_csv(splits.validation, settings.validation_output_path),
        "test": _write_csv(splits.test, settings.test_output_path),
        "prediction": _write_csv(
            prediction_input if prediction_input is not None else model_ready.tail(1).copy(),
            settings.prediction_output_path,
        ),
        "rejected": _write_csv(rejected, settings.rejected_output_path),
    }
    if fetched_raw is not None:
        paths["fetched_raw"] = _write_csv(fetched_raw, settings.fetched_raw_output_path)
    summary = asdict(result)
    summary["output_paths"] = {key: str(path) for key, path in paths.items()}
    settings.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.summary_output_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    paths["summary"] = settings.summary_output_path
    return paths


def load_dataset(split: str | None = None, settings: PipelineSettings | None = None) -> pd.DataFrame:
    settings = settings or PipelineSettings.from_env()
    split_paths = {
        "train": settings.train_output_path,
        "validation": settings.validation_output_path,
        "val": settings.validation_output_path,
        "test": settings.test_output_path,
    }
    path = settings.model_ready_output_path if split is None else split_paths.get(split)
    if path is None:
        raise ValueError("split must be one of: train, validation, val, test, or None.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset output not found: {path}")
    return pd.read_csv(path)


def load_prediction_input(settings: PipelineSettings | None = None) -> pd.DataFrame:
    settings = settings or PipelineSettings.from_env()
    if not settings.prediction_output_path.exists():
        raise FileNotFoundError(f"Prediction input not found: {settings.prediction_output_path}")
    return pd.read_csv(settings.prediction_output_path)


def load_run_summary(settings: PipelineSettings | None = None) -> dict[str, Any]:
    settings = settings or PipelineSettings.from_env()
    if not settings.summary_output_path.exists():
        raise FileNotFoundError(f"Run summary not found: {settings.summary_output_path}")
    return json.loads(settings.summary_output_path.read_text(encoding="utf-8"))
