import pandas as pd

from eq_prediction.pipeline.config import DatabaseSettings, PipelineSettings
from eq_prediction.pipeline.enrich import enrich_missing_detail_columns, patch_saved_missing_values


def test_saved_patch_matches_by_detail_without_network():
    raw = pd.DataFrame(
        [
            {
                "event_id": "a",
                "detail": "https://example.test/a",
                "nst": None,
                "dmin": None,
                "gap": None,
            }
        ]
    )
    patch = pd.DataFrame(
        [
            {
                "old_idx": 99,
                "detail": "https://example.test/a",
                "nst": 12,
                "dmin": 0.5,
                "gap": 80,
            }
        ]
    )

    result, patched_values = patch_saved_missing_values(raw, patch)

    assert patched_values == 3
    assert result.loc[0, "nst"] == 12
    assert result.loc[0, "dmin"] == 0.5
    assert result.loc[0, "gap"] == 80


def test_enrichment_without_a_saved_patch_is_safe(tmp_path):
    settings = PipelineSettings(
        root_dir=tmp_path,
        workspace_root=tmp_path.parent,
        eq_prediction_root=tmp_path.parent / "eq_prediction",
        local_raw_patch_path=tmp_path / "raw.csv",
        local_raw_unpatch_path=tmp_path / "raw.csv",
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
        ),
    )
    raw = pd.DataFrame([{"event_id": "a", "nst": None, "dmin": None, "gap": None}])

    result = enrich_missing_detail_columns(raw, settings, fetch_missing=False)

    assert result.attrs["loaded_saved_patch_rows"] == 0
    assert result.attrs["saved_patch_values"] == 0


def test_saved_patch_matches_by_legacy_oldindex():
    raw = pd.DataFrame(
        [
            {"event_id": "a", "nst": None, "dmin": None, "gap": None},
            {"event_id": "b", "nst": None, "dmin": None, "gap": None},
        ]
    )
    patch = pd.DataFrame(
        [
            {
                "oldindex": 1,
                "nst": 14,
                "dmin": 0.8,
                "gap": 95,
            }
        ]
    )

    result, patched_values = patch_saved_missing_values(raw, patch)

    assert patched_values == 3
    assert pd.isna(result.loc[0, "nst"])
    assert result.loc[1, "nst"] == 14
    assert result.loc[1, "dmin"] == 0.8
    assert result.loc[1, "gap"] == 95


def test_enrichment_applies_patch_before_network_fetch(tmp_path, monkeypatch):
    settings = PipelineSettings(
        root_dir=tmp_path,
        workspace_root=tmp_path.parent,
        eq_prediction_root=tmp_path.parent / "eq_prediction",
        local_raw_patch_path=tmp_path / "raw.csv",
        local_raw_unpatch_path=tmp_path / "raw.csv",
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
        ),
    )
    raw = pd.DataFrame(
        [
            {
                "event_id": "a",
                "detail": "https://example.test/a",
                "nst": None,
                "dmin": None,
                "gap": None,
            }
        ]
    )
    patch = pd.DataFrame(
        [
            {
                "event_id": "a",
                "nst": 12,
                "dmin": 0.5,
                "gap": 80,
            }
        ]
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("network fetch should not run after patch fills missing values")

    monkeypatch.setattr("eq_prediction.pipeline.enrich.DetailCache.fetch", fail_fetch)

    result = enrich_missing_detail_columns(
        raw,
        settings,
        fetch_missing=True,
        patch_dataframe=patch,
    )

    assert result.attrs["loaded_saved_patch_rows"] == 1
    assert result.attrs["saved_patch_values"] == 3
    assert result.loc[0, "nst"] == 12
    assert result.loc[0, "dmin"] == 0.5
    assert result.loc[0, "gap"] == 80
