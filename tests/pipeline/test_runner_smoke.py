from pathlib import Path

import pandas as pd

from eq_prediction.pipeline.config import DatabaseSettings, PipelineSettings
from eq_prediction.pipeline.runner import run_pipeline
from eq_prediction.pipeline.storage import load_dataset, load_prediction_input


def make_settings(tmp_path: Path, local_raw_path: Path) -> PipelineSettings:
    return PipelineSettings(
        root_dir=tmp_path,
        workspace_root=tmp_path.parent,
        eq_prediction_root=tmp_path.parent / "eq_prediction",
        local_raw_patch_path=local_raw_path,
        local_raw_unpatch_path=local_raw_path,
        enrichment_patch_path=tmp_path / "missing_patch.csv",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / ".cache",
        min_magnitude=2.5,
        fetch_days=7,
        overlap_days=2,
        usgs_endpoint="https://example.test/usgs",
        request_timeout_seconds=5.0,
        request_min_interval_seconds=0.0,
        database=DatabaseSettings(
            username=None,
            password=None,
            host="localhost",
            port=5432,
            database="eq_db",
            base_raw_schema="raw",
            base_raw_table="earthquakes",
            raw_patch_schema="raw",
            raw_patch_table="earthquakes",
        ),
    )


def test_end_to_end_small_local_pipeline_run(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "geo": "POINT Z (10 20 5)",
                "code": "a",
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 3.0,
                "magType": "ml",
                "type": "earthquake",
                "status": "reviewed",
                "detail": "",
                "nst": 12,
                "dmin": 0.5,
                "rms": 0.1,
                "gap": 80,
            },
            {
                "geo": "POINT Z (11 21 6)",
                "code": "b",
                "time": 1700000100000,
                "updated": 1700000101000,
                "mag": 3.5,
                "magType": "mb",
                "type": "earthquake",
                "status": "reviewed",
                "detail": "",
                "nst": 14,
                "dmin": 0.6,
                "rms": 0.2,
                "gap": 90,
            },
            {
                "geo": "POINT Z (999 21 6)",
                "code": "bad",
                "time": 1700000200000,
                "updated": 1700000201000,
                "mag": 3.5,
                "magType": "mb",
                "type": "earthquake",
            },
        ]
    ).to_csv(csv_path, index=False)
    settings = make_settings(tmp_path, csv_path)

    result = run_pipeline(source="local", fetch_new=False, save=True, settings=settings)

    assert result.status == "completed"
    assert result.rows_loaded == 3
    assert result.rows_clean == 2
    assert result.rows_rejected == 1
    assert settings.raw_output_path.exists()
    assert settings.clean_output_path.exists()
    assert settings.model_ready_output_path.exists()
    assert settings.train_output_path.exists()
    assert settings.validation_output_path.exists()
    assert settings.test_output_path.exists()
    assert settings.prediction_output_path.exists()
    assert settings.rejected_output_path.exists()
    assert settings.summary_output_path.exists()

    model_ready = load_dataset(settings=settings)
    prediction = load_prediction_input(settings=settings)

    assert len(model_ready) == 2
    assert len(prediction) == 1
    assert "time" not in model_ready.columns
    assert "event_id" not in model_ready.columns
    assert "mag" in model_ready.columns
    assert "mag" not in prediction.columns


def test_saved_enrichment_patch_does_not_replace_clean_data(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "geo": "POINT Z (10 20 5)",
                "code": "a",
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 3.0,
                "magType": "ml",
                "type": "earthquake",
                "status": "reviewed",
                "detail": "https://example.test/a",
                "nst": None,
                "dmin": None,
                "rms": 0.1,
                "gap": None,
            },
        ]
    ).to_csv(csv_path, index=False)
    settings = make_settings(tmp_path, csv_path)
    pd.DataFrame(
        [
            {
                "event_id": "a",
                "nst": 12,
                "dmin": 0.5,
                "gap": 80,
            }
        ]
    ).to_csv(settings.enrichment_patch_path, index=False)

    result = run_pipeline(source="local", fetch_new=False, save=True, settings=settings)
    clean = pd.read_csv(settings.clean_output_path)

    assert result.status == "completed"
    assert "time" in clean.columns
    assert clean.loc[0, "event_id"] == "a"
    assert clean.loc[0, "nst"] == 12
