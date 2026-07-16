from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _existing_env_path(name: str, default: Path, *relative_roots: Path) -> Path:
    value = os.getenv(name)
    if value in (None, ""):
        return default

    raw_path = Path(value.strip())
    candidates = [raw_path] if raw_path.is_absolute() else [*(root / raw_path for root in relative_roots), raw_path]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return default


@dataclass(frozen=True)
class DatabaseSettings:
    username: str | None
    password: str | None
    host: str
    port: int
    database: str
    base_raw_schema: str
    base_raw_table: str
    enrich_patch_schema: str | None = None
    enrich_patch_table: str | None = None
    raw_patch_table: str | None = None
    raw_patch_schema: str | None = None
    
    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        return cls(
            username=os.getenv("DB_UNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("PC_IP_ADDRESS", "localhost"),
            port=_get_int("DB_PORT", 5432),
            database=os.getenv("EQ_DB_NAME", "eq_db"),
            base_raw_schema=os.getenv("EQ_DB_RAW_UNPATCHED_SCHEMA", "lappy_raw_data"),
            base_raw_table=os.getenv("EQ_DB_RAW_UNPATCHED_TABLE", "eq_data_updated3"),
            enrich_patch_schema=os.getenv("EQ_DB_ENRICH_PATCH_SCHEMA"),
            enrich_patch_table=os.getenv("EQ_DB_ENRICH_PATCH_TABLE"),
            raw_patch_schema=os.getenv("EQ_DB_RAW_PATCHED_SCHEMA", "final_data"),
            raw_patch_table=os.getenv("EQ_DB_RAW_PATCHED_TABLE", "eq_data_updated3_patched"),
        )

    @property
    def available(self) -> bool:
        return all([self.username, self.password, self.host, self.port, self.database])


@dataclass(frozen=True)
class PipelineSettings:
    root_dir: Path
    workspace_root: Path
    eq_prediction_root: Path
    local_raw_patch_path: Path
    local_raw_unpatch_path: Path
    enrichment_patch_path: Path
    data_dir: Path
    cache_dir: Path
    # legacy_final_dir: Path # TODO: To add a Path to the legacy finished datawith both options for local as well as database connection
    min_magnitude: float
    fetch_days: int
    overlap_days: int
    usgs_endpoint: str
    request_timeout_seconds: float
    request_min_interval_seconds: float
    database: DatabaseSettings

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        root_dir = Path(__file__).resolve().parents[3]
        env_path = root_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        workspace_root = root_dir.parent
        eq_prediction_root = root_dir

        eq_prediction_data_dir = eq_prediction_root / "data"

        default_raw = eq_prediction_data_dir / "new_raw_data" / "eq_data_updated3.csv"
        
        # default_patched_raw = eq_prediction_data_dir / "patched_raw_data" / "New_Engineered_Data.csv"
        default_patched_raw = eq_prediction_data_dir / "patched_raw_data" / "eq_data_updated3_patched.csv"

        default_encrich_patch = eq_prediction_data_dir / "enrich_patch_data" / "FinalRegressionData.csv"

        return cls(
            root_dir=root_dir,
            workspace_root=workspace_root,
            eq_prediction_root=eq_prediction_root,
            local_raw_unpatch_path=_existing_env_path(
                "EQ_PIPELINE_LOCAL_RAW_UNPATCHED_PATH",
                default_raw,
                eq_prediction_data_dir,
                eq_prediction_root,
                workspace_root,
            ),
            local_raw_patch_path=_existing_env_path(
                "EQ_PIPELINE_LOCAL_RAW_PATCHED_PATH",
                default_patched_raw,
                eq_prediction_data_dir,
                eq_prediction_root,
                workspace_root,
            ),
            enrichment_patch_path=_existing_env_path(
                "EQ_PIPELINE_LOCAL_ENRICHMENT_PATH",
                default_encrich_patch,
                eq_prediction_data_dir,
                eq_prediction_root,
                workspace_root,
            ), # Data with on the missing value patch, NOT FULL DATA
            data_dir=root_dir / "data",
            cache_dir=root_dir / ".cache",
            min_magnitude=_get_float("EQ_PIPELINE_MIN_MAGNITUDE", 2.5),
            fetch_days=_get_int("EQ_PIPELINE_FETCH_DAYS", 7),
            overlap_days=_get_int("EQ_PIPELINE_OVERLAP_DAYS", 2),
            usgs_endpoint="https://earthquake.usgs.gov/fdsnws/event/1/query.geojson",
            request_timeout_seconds=_get_float("EQ_PIPELINE_REQUEST_TIMEOUT_SECONDS", 30.0),
            request_min_interval_seconds=_get_float("EQ_PIPELINE_REQUEST_MIN_INTERVAL_SECONDS", 1.0),
            database=DatabaseSettings.from_env(),
        )

    @property
    def raw_output_path(self) -> Path:
        return self.data_dir / "raw" / "earthquakes_raw.csv"

    @property
    def clean_output_path(self) -> Path:
        return self.data_dir / "clean" / "earthquakes_clean.csv"

    @property
    def model_ready_output_path(self) -> Path:
        return self.data_dir / "model_ready" / "earthquakes_model_ready.csv"

    @property
    def train_output_path(self) -> Path:
        return self.data_dir / "splits" / "train.csv"

    @property
    def validation_output_path(self) -> Path:
        return self.data_dir / "splits" / "validation.csv"

    @property
    def test_output_path(self) -> Path:
        return self.data_dir / "splits" / "test.csv"

    @property
    def prediction_output_path(self) -> Path:
        return self.data_dir / "prediction" / "latest_prediction_input.csv"

    @property
    def rejected_output_path(self) -> Path:
        return self.data_dir / "rejected" / "rejected_raw.csv"

    @property
    def summary_output_path(self) -> Path:
        return self.data_dir / "run_summary.json"
