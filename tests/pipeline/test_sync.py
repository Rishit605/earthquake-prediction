from pathlib import Path

import pandas as pd

from eq_prediction.pipeline.config import DatabaseSettings, PipelineSettings
from eq_prediction.pipeline import sync


def make_settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        root_dir=tmp_path,
        workspace_root=tmp_path,
        eq_prediction_root=tmp_path,
        local_raw_patch_path=tmp_path / "input.csv",
        local_raw_unpatch_path=tmp_path / "input.csv",
        enrichment_patch_path=tmp_path / "patch.csv",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / ".cache",
        min_magnitude=2.5,
        fetch_days=7,
        overlap_days=2,
        usgs_endpoint="https://example.test",
        request_timeout_seconds=1,
        request_min_interval_seconds=0,
        database=DatabaseSettings("user", "password", "localhost", 5432, "eq", "raw", "events"),
    )


def test_table_name_for_path_is_stable_and_postgres_safe():
    assert sync.table_name_for_path(Path("raw/New Events!.csv")) == "raw_new_events"
    assert sync.table_name_for_path(Path("2026/events.csv")) == "data_2026_events"


def test_offline_sync_records_pending_csvs_without_changing_them(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    source = settings.data_dir / "raw" / "events.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"event_id": ["one"], "nst": [12]}).to_csv(source, index=False)
    original = source.read_text(encoding="utf-8")
    monkeypatch.setattr(sync, "build_database_url", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))

    result = sync.sync_data(settings)

    assert result.status == "pending"
    assert result.files[0].path == "raw/events.csv"
    assert result.files[0].action == "pending"
    assert source.read_text(encoding="utf-8") == original
    assert settings.sync_manifest_path.exists()


def test_auto_conflict_requires_force_and_explicit_direction_resolves_it():
    baseline = {"checksum": "old"}
    assert sync._action_for("auto", False, True, True, "local", "database", baseline, baseline) == "conflict"
    assert sync._action_for("push", True, True, True, "local", "database", baseline, baseline) == "push"
    assert sync._action_for("pull", True, True, True, "local", "database", baseline, baseline) == "pull"
