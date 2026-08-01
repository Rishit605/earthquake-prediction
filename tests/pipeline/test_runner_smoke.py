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
    assert not settings.fetched_raw_output_path.exists()

    model_ready = load_dataset(settings=settings)
    prediction = load_prediction_input(settings=settings)

    assert len(model_ready) == 2
    assert len(prediction) == 1
    assert "time" not in model_ready.columns
    assert "event_id" not in model_ready.columns
    assert "mag" in model_ready.columns
    assert "mag" not in prediction.columns


def test_local_path_overrides_the_configured_local_source(tmp_path):
    configured_path = tmp_path / "configured.csv"
    custom_path = tmp_path / "custom.csv"
    pd.DataFrame(
        [
            {
                "geo": "POINT Z (10 20 5)",
                "code": "custom-event",
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 3.0,
                "type": "earthquake",
            }
        ]
    ).to_csv(custom_path, index=False)
    settings = make_settings(tmp_path, configured_path)

    result = run_pipeline(
        source="local", local_path=custom_path, fetch_new=False, save=True, settings=settings
    )

    raw = pd.read_csv(settings.raw_output_path)
    assert result.status == "completed"
    assert raw.loc[0, "event_id"] == "custom-event"
    assert result.stage_results[0].message == str(custom_path)


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

    result = run_pipeline(
        source="local", local_path=str(csv_path), fetch_new=False, save=True, settings=settings
    )
    clean = pd.read_csv(settings.clean_output_path)

    assert result.status == "completed"
    assert "time" in clean.columns
    assert clean.loc[0, "event_id"] == "a"
    assert clean.loc[0, "nst"] == 12


def test_fetch_new_saves_full_usgs_data_alongside_normalized_raw(tmp_path, monkeypatch):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "geo": "POINT Z (10 20 5)",
                "code": "local-event",
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 3.0,
                "type": "earthquake",
            }
        ]
    ).to_csv(csv_path, index=False)
    settings = make_settings(tmp_path, csv_path)
    features = [
        {
            "id": "usgs-event",
            "geometry": {"coordinates": [77.1, 28.6, 12.0]},
            "properties": {
                "time": 1700000200000,
                "updated": 1700000201000,
                "mag": 4.2,
                "magType": "mb",
                "type": "earthquake",
                "status": "reviewed",
                "felt": 7,
                "tsunami": 0,
                "custom_usgs_field": "preserved",
            },
        }
    ]
    monkeypatch.setattr(
        "eq_prediction.pipeline.sources.fetch_usgs_window", lambda *args: features
    )

    result = run_pipeline(
        source="local",
        local_path=str(csv_path),
        fetch_new=True,
        save=True,
        settings=settings,
        fetch_start=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        fetch_end=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
    )

    fetched_raw = pd.read_csv(settings.fetched_raw_output_path)
    normalized_raw = pd.read_csv(settings.raw_output_path)

    assert result.output_paths["fetched_raw"] == settings.fetched_raw_output_path
    assert {"felt", "tsunami", "custom_usgs_field", "event_id", "longitude", "latitude", "depth_km"}.issubset(fetched_raw.columns)
    assert fetched_raw.loc[0, "custom_usgs_field"] == "preserved"
    assert set(normalized_raw.columns) == {
        "event_id", "time", "updated", "longitude", "latitude", "depth_km", "mag",
        "magType", "type", "status", "detail", "nst", "dmin", "rms", "gap", "source",
    }
    assert "custom_usgs_field" not in normalized_raw.columns


def test_failed_usgs_fetch_does_not_save_fetched_raw_data(tmp_path, monkeypatch):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "geo": "POINT Z (10 20 5)",
                "code": "local-event",
                "time": 1700000000000,
                "updated": 1700000001000,
                "mag": 3.0,
                "type": "earthquake",
            }
        ]
    ).to_csv(csv_path, index=False)
    settings = make_settings(tmp_path, csv_path)

    def fail_fetch(*args):
        raise RuntimeError("USGS unavailable")

    monkeypatch.setattr("eq_prediction.pipeline.sources.fetch_usgs_window", fail_fetch)

    result = run_pipeline(
        source="local",
        local_path=str(csv_path),
        fetch_new=True,
        save=True,
        settings=settings,
    )

    assert result.status == "completed_with_warnings"
    assert not settings.fetched_raw_output_path.exists()
    assert "fetched_raw" not in result.output_paths
