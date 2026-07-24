from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Iterable

import pandas as pd

from .clean import clean_raw_events
from .config import PipelineSettings
from .enrich import enrich_missing_detail_columns, load_existing_enrichment_patch
from .features import make_model_ready
from .models import PipelineResult, SourceName, StageResult, utc_now
from .normalize import combine_raw_frames
from .sources import fetch_new_usgs_events, load_db_table, load_local_csv
from .split import chronological_split
from .storage import save_outputs
from .training_ready import build_training_outputs


def _stage(name: str, rows_in: int, rows_out: int, rejected: int = 0, message: str = "", code: str = ""):
    return StageResult(
        name=name,
        rows_in=rows_in,
        rows_out=rows_out,
        rejected=rejected,
        finished_at=utc_now(),
        message=message,
        code=code
    )


def _load_requested_sources(
    settings: PipelineSettings,
    source: SourceName,
    fetch_new: bool,
    fetch_start: datetime | None,
    fetch_end: datetime | None,
) -> tuple[list[pd.DataFrame], list[StageResult], list[str]]:
    frames: list[pd.DataFrame] = []
    stages: list[StageResult] = []
    errors: list[str] = []

    if source in ("local", "all"):
        local_error: str | None = None
        try:
            local = load_local_csv(settings, settings.local_raw_patch_path)
            frames.append(local)
            stages.append(_stage("load_local", 0, len(local), message=f"{settings.local_raw_patch_path}", code="L01"))
        except Exception as exc:  # noqa: BLE001 - source fallback is intentional
            local_error = f"primary local source failed: {type(exc).__name__}: {exc}"
            try:
                local = load_local_csv(settings, settings.local_raw_unpatch_path)
                frames.append(local)
                stages.append(
                    _stage(
                        "load_local",
                        0,
                        len(local),
                        message=f"{settings.local_raw_unpatch_path} (fallback)", code="L02",
                    )
                )
                errors.append(local_error)
            except Exception as fallback_exc:  # noqa: BLE001 - source fallback is intentional
                errors.append(local_error)
                errors.append(
                    f"secondary local source failed: {type(fallback_exc).__name__}: {fallback_exc}"
                )
   

    if source in ("db", "all"):
        db_error: str | None = None
        try:
            db = load_db_table(settings, [settings.database.raw_patch_table, settings.database.raw_patch_schema])
            frames.append(db)
            stages.append(
                _stage(
                    "load_db",
                    0,
                    len(db),
                    message=f"{settings.database.raw_patch_schema}.{settings.database.raw_patch_table}", code="DB01",
                )
            )
        except Exception as exc:  # noqa: BLE001 - source fallback is intentional
            db_error = f"primary db source failed: {type(exc).__name__}: {exc}"
            try:
                db = load_db_table(settings, [settings.database.base_raw_table, settings.database.base_raw_schema])
                frames.append(db)
                stages.append(
                _stage(
                    "load_db",
                    0,
                    len(db),
                    message=f"{settings.database.base_raw_schema}.{settings.database.base_raw_table}", code="DB02",
                    )
                )
                errors.append(db_error)
            except Exception as fallback_exc:  # noqa: BLE001 - source fallback is intentional
                errors.append(db_error)
                errors.append(
                    f"secondary db source failed: {type(fallback_exc).__name__}: {fallback_exc}"
                )

    if fetch_new:
        existing = combine_raw_frames(frames)
        try:
            fetched = fetch_new_usgs_events(settings, existing, fetch_start, fetch_end)
            frames.append(fetched)
            stages.append(_stage("fetch_usgs", 0, len(fetched)))
        except Exception as exc:  # noqa: BLE001 - source fallback is intentional
            errors.append(f"usgs source failed: {type(exc).__name__}: {exc}")

    return frames, stages, errors


def run_pipeline(
    source: SourceName = "all",
    fetch_new: bool = False,
    timeseries: bool = False,
    enrich_details: bool = False,
    save: bool = True,
    settings: PipelineSettings | None = None,
    fetch_start: datetime | None = None,
    fetch_end: datetime | None = None,
) -> PipelineResult:
    if source not in ("local", "db", "all"):
        raise ValueError("source must be one of: local, db, all.")

    settings = settings or PipelineSettings.from_env()
    frames, stage_results, errors = _load_requested_sources(
        settings, source, fetch_new, fetch_start, fetch_end
    )

    # print(stage_results)
    # print(type(frames))
    # print()
    # return frames

    raw = combine_raw_frames(frames)
    stage_results.append(_stage("combine_sources", 0, len(raw)))

    if raw.empty:
        result = PipelineResult(
            status="failed",
            source=source,
            fetch_new=fetch_new,
            rows_loaded=0,
            rows_clean=0,
            rows_model_ready=0,
            rows_rejected=0,
            stage_results=tuple(stage_results),
            errors=tuple(errors + ["no data loaded"]),
        )
        return result

    clean, rejected = clean_raw_events(raw)
    stage_results.append(_stage("clean", len(raw), len(clean), rejected=len(rejected)))
    # return clean, rejected, stage_results

    enrichment_patch = load_existing_enrichment_patch(settings)
    should_enrich = (
        isinstance(enrichment_patch, pd.DataFrame)
        or enrich_details
        or any(stage.code in {"L02", "DB02"} for stage in stage_results)
    )
    # return should_enrich, stage_results, enrichment_patch
    if should_enrich:
        saved_patch = enrichment_patch if isinstance(enrichment_patch, pd.DataFrame) else None
        enriched = enrich_missing_detail_columns(
            clean,
            settings,
            fetch_missing=enrich_details,
            patch_dataframe=saved_patch,
        )
        patch_count = enriched.attrs.get("saved_patch_values", 0)
        patch_rows = enriched.attrs.get("loaded_saved_patch_rows", 0)
        patch_message = (
            f"loaded_patch_rows={patch_rows}; "
            f"saved_patch_values={patch_count}; "
            f"network_fetch={'enabled' if enrich_details else 'disabled'}"
        )
        stage_results.append(
            _stage(
                "enrich",
                len(clean),
                len(enriched),
                message=patch_message,
            )
        )
    else:
        enriched = clean

    feature_engineered = make_model_ready(enriched, timeseries=timeseries)
    stage_results.append(
        _stage(
            "features",
            len(enriched),
            len(feature_engineered),
            message=f"timeseries={'enabled' if timeseries else 'disabled'}",
        )
    )

    feature_splits = chronological_split(feature_engineered)
    stage_results.append(
        _stage(
            "split",
            len(feature_engineered),
            len(feature_engineered),
            message=(
                f"train={len(feature_splits.train)}, "
                f"validation={len(feature_splits.validation)}, test={len(feature_splits.test)}"
            ),
        )
    )

    training_outputs = build_training_outputs(feature_engineered, feature_splits)
    stage_results.append(
        _stage(
            "training_ready",
            len(feature_engineered),
            len(training_outputs.model_ready),
            message="numeric_features=zscored_from_train; prediction_excludes_target",
        )
    )

    status = "completed_with_warnings" if errors else "completed"
    result = PipelineResult(
        status=status,
        source=source,
        fetch_new=fetch_new,
        rows_loaded=len(raw),
        rows_clean=len(enriched),
        rows_model_ready=len(training_outputs.model_ready),
        rows_rejected=len(rejected),
        stage_results=tuple(stage_results),
        errors=tuple(errors),
    )

    if save:
        output_paths = save_outputs(
            settings=settings,
            raw=raw,
            clean=enriched,
            model_ready=training_outputs.model_ready,
            splits=training_outputs.splits,
            prediction_input=training_outputs.prediction_input,
            rejected=rejected,
            result=result,
        )
        result = replace(result, output_paths=output_paths)
    return result
