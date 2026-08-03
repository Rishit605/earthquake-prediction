"""Bidirectional synchronization of pipeline CSV snapshots and PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import PipelineSettings
from .sources import build_database_url


SyncDirection = Literal["auto", "push", "pull"]
SYNC_SCHEMA = "pipeline_sync"
MANIFEST_TABLE = "_sync_manifest"
LOCAL_MANIFEST_NAME = ".sync_manifest.json"


@dataclass(frozen=True)
class SyncFileResult:
    path: str
    table: str
    action: str
    message: str = ""


@dataclass(frozen=True)
class SyncResult:
    status: str
    files: tuple[SyncFileResult, ...]
    errors: tuple[str, ...] = ()

    @property
    def pending(self) -> int:
        return sum(item.action in {"pending", "conflict", "failed"} for item in self.files)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_name_for_path(relative_path: Path) -> str:
    """Return a stable PostgreSQL-safe table name for a data-relative CSV path."""
    stem = relative_path.with_suffix("").as_posix().lower().replace("/", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "data"
    if name[0].isdigit():
        name = f"data_{name}"
    if len(name) > 63:
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:54]}_{suffix}"
    return name


def _csv_snapshot(path: Path) -> tuple[pd.DataFrame, str, int]:
    dataframe = pd.read_csv(path)
    # CSV formatting is not semantically meaningful; use a canonical DataFrame form.
    canonical = dataframe.to_csv(index=False, lineterminator="\n")
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return dataframe, checksum, len(dataframe)


def _load_local_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", {})
        return files if isinstance(files, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_local_manifest(path: Path, files: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"version": 1, "files": files}, indent=2), encoding="utf-8")
    temporary.replace(path)


def _manifest_record(table_name: str, checksum: str, row_count: int) -> dict[str, object]:
    return {
        "table": table_name,
        "checksum": checksum,
        "row_count": row_count,
        "synced_at": _utc_now(),
    }


def _quoted(connection, name: str) -> str:
    return connection.dialect.identifier_preparer.quote(name)


def _ensure_database(engine) -> None:
    from sqlalchemy import text

    if engine.dialect.name != "postgresql":
        raise RuntimeError("Pipeline sync requires a PostgreSQL database.")
    with engine.begin() as connection:
        schema = _quoted(connection, SYNC_SCHEMA)
        manifest = _quoted(connection, MANIFEST_TABLE)
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {schema}.{manifest} ("
                "relative_path TEXT PRIMARY KEY, table_name TEXT NOT NULL, "
                "checksum TEXT NOT NULL, row_count BIGINT NOT NULL, "
                "synced_at TIMESTAMPTZ NOT NULL)"
            )
        )


def _load_database_manifest(engine) -> dict[str, dict[str, object]]:
    from sqlalchemy import text

    query = text(
        f'SELECT relative_path, table_name, checksum, row_count, synced_at '
        f'FROM "{SYNC_SCHEMA}"."{MANIFEST_TABLE}"'
    )
    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
    return {
        row["relative_path"]: {
            "table": row["table_name"],
            "checksum": row["checksum"],
            "row_count": int(row["row_count"]),
            "synced_at": row["synced_at"].isoformat(),
        }
        for row in rows
    }


def _save_database_manifest(engine, relative_path: str, record: dict[str, object]) -> None:
    from sqlalchemy import text

    query = text(
        f'INSERT INTO "{SYNC_SCHEMA}"."{MANIFEST_TABLE}" '
        "(relative_path, table_name, checksum, row_count, synced_at) "
        "VALUES (:path, :table, :checksum, :row_count, NOW()) "
        "ON CONFLICT (relative_path) DO UPDATE SET "
        "table_name = EXCLUDED.table_name, checksum = EXCLUDED.checksum, "
        "row_count = EXCLUDED.row_count, synced_at = EXCLUDED.synced_at"
    )
    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "path": relative_path,
                "table": record["table"],
                "checksum": record["checksum"],
                "row_count": record["row_count"],
            },
        )


def _database_tables(engine) -> set[str]:
    from sqlalchemy import inspect

    return {
        table
        for table in inspect(engine).get_table_names(schema=SYNC_SCHEMA)
        if table != MANIFEST_TABLE and not table.startswith("_staging_")
    }


def _read_database_snapshot(engine, table_name: str) -> tuple[pd.DataFrame, str, int]:
    dataframe = pd.read_sql_table(table_name, engine, schema=SYNC_SCHEMA)
    canonical = dataframe.to_csv(index=False, lineterminator="\n")
    return dataframe, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), len(dataframe)


def _upload_snapshot(engine, dataframe: pd.DataFrame, table_name: str) -> None:
    from sqlalchemy import text

    suffix = uuid.uuid4().hex[:8]
    staging = f"_staging_{table_name[:44]}_{suffix}"
    dataframe.to_sql(staging, engine, schema=SYNC_SCHEMA, if_exists="fail", index=False, method="multi")
    try:
        with engine.begin() as connection:
            schema = _quoted(connection, SYNC_SCHEMA)
            table = _quoted(connection, table_name)
            staging_table = _quoted(connection, staging)
            connection.execute(text(f"DROP TABLE IF EXISTS {schema}.{table}"))
            connection.execute(text(f"ALTER TABLE {schema}.{staging_table} RENAME TO {table}"))
    except Exception:
        with engine.begin() as connection:
            connection.execute(text(f'DROP TABLE IF EXISTS "{SYNC_SCHEMA}"."{staging}"'))
        raise


def _write_local_snapshot(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    temporary.replace(path)


def _action_for(
    direction: SyncDirection,
    force: bool,
    local_exists: bool,
    database_exists: bool,
    local_checksum: str | None,
    database_checksum: str | None,
    local_record: dict[str, object] | None,
    database_record: dict[str, object] | None,
) -> str:
    if local_exists and not database_exists:
        return "push"
    if database_exists and not local_exists:
        return "pull"
    if not local_exists and not database_exists:
        return "skip"
    if local_checksum == database_checksum:
        return "matched"

    baseline = None
    if local_record and database_record and local_record.get("checksum") == database_record.get("checksum"):
        baseline = str(local_record["checksum"])
    local_changed = baseline is None or local_checksum != baseline
    database_changed = baseline is None or database_checksum != baseline
    if local_changed and not database_changed:
        return "push"
    if database_changed and not local_changed:
        return "pull"
    if force and direction in {"push", "pull"}:
        return direction
    return "conflict"


def sync_data(
    settings: PipelineSettings,
    direction: SyncDirection = "auto",
    force: bool = False,
) -> SyncResult:
    """Synchronize every CSV below ``settings.data_dir`` with PostgreSQL snapshots."""
    if direction not in {"auto", "push", "pull"}:
        raise ValueError("direction must be auto, push, or pull.")

    manifest_path = settings.sync_manifest_path
    local_manifest = _load_local_manifest(manifest_path)
    local_paths = {path.relative_to(settings.data_dir).as_posix(): path for path in settings.data_dir.rglob("*.csv")}
    files: list[SyncFileResult] = []
    try:
        from sqlalchemy import create_engine

        engine = create_engine(build_database_url(settings.database), pool_pre_ping=True)
        _ensure_database(engine)
        database_manifest = _load_database_manifest(engine)
        database_tables = _database_tables(engine)
    except Exception as exc:  # database availability must never destroy local work
        pending = [
            SyncFileResult(path, table_name_for_path(Path(path)), "pending", "database unavailable")
            for path in sorted(local_paths)
        ]
        _save_local_manifest(manifest_path, local_manifest)
        return SyncResult("pending", tuple(pending), (f"database sync unavailable: {type(exc).__name__}: {exc}",))

    try:
        database_paths = {
            path: str(record["table"])
            for path, record in database_manifest.items()
            if str(record["table"]) in database_tables
        }
        # Tables created without metadata are intentionally not guessed: they may be unrelated.
        all_paths = sorted(set(local_paths) | set(database_paths))
        for relative_path in all_paths:
            local_path = local_paths.get(relative_path)
            table_name = database_paths.get(relative_path, table_name_for_path(Path(relative_path)))
            local_frame = local_checksum = None
            database_frame = database_checksum = None
            local_rows = database_rows = 0
            if local_path is not None:
                local_frame, local_checksum, local_rows = _csv_snapshot(local_path)
            if table_name in database_tables:
                database_frame, database_checksum, database_rows = _read_database_snapshot(engine, table_name)

            action = _action_for(
                direction, force, local_path is not None, table_name in database_tables,
                local_checksum, database_checksum, local_manifest.get(relative_path), database_manifest.get(relative_path),
            )
            if action == "push":
                assert local_frame is not None and local_checksum is not None
                _upload_snapshot(engine, local_frame, table_name)
                record = _manifest_record(table_name, local_checksum, local_rows)
                local_manifest[relative_path] = record
                _save_database_manifest(engine, relative_path, record)
                files.append(SyncFileResult(relative_path, table_name, "pushed"))
            elif action == "pull":
                assert database_frame is not None and database_checksum is not None
                destination = local_path or settings.data_dir / Path(relative_path)
                _write_local_snapshot(destination, database_frame)
                record = _manifest_record(table_name, database_checksum, database_rows)
                local_manifest[relative_path] = record
                _save_database_manifest(engine, relative_path, record)
                files.append(SyncFileResult(relative_path, table_name, "pulled"))
            elif action == "matched":
                assert local_checksum is not None
                local_manifest[relative_path] = _manifest_record(table_name, local_checksum, local_rows)
                _save_database_manifest(engine, relative_path, local_manifest[relative_path])
                files.append(SyncFileResult(relative_path, table_name, "matched"))
            elif action == "conflict":
                files.append(SyncFileResult(relative_path, table_name, "conflict", "both copies changed; use push/pull with --force"))
            else:
                files.append(SyncFileResult(relative_path, table_name, "skipped"))
    except Exception as exc:
        files.append(SyncFileResult("*", "*", "failed", f"{type(exc).__name__}: {exc}"))
        _save_local_manifest(manifest_path, local_manifest)
        return SyncResult("pending", tuple(files), (f"database sync failed: {type(exc).__name__}: {exc}",))
    finally:
        engine.dispose()

    _save_local_manifest(manifest_path, local_manifest)
    status = "conflicts" if any(item.action == "conflict" for item in files) else "completed"
    return SyncResult(status, tuple(files))
