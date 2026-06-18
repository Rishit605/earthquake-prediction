import hashlib
import os
import re
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.schema import CreateSchema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

DB_USERNAME = os.getenv("DB_UNAME")
DB_PORT = os.getenv("DB_PORT")
DB_PASSWORD = os.getenv("DB_PASSWORD")
PC_IP_ADDRESS = os.getenv("PC_IP_ADDRESS")

DATA_DIRECTORY = PROJECT_ROOT / "data"
ENGINEERED_DATA_DIRECTORY = DATA_DIRECTORY / "engineered_data"
RAW_SCHEMA_NAME = "lappy_raw_data"
PREPPED_SCHEMA_NAME = "lappy_prepped_data"
READ_CHUNK_SIZE = 10000
WRITE_CHUNK_SIZE = 1000

def _require_env_vars():
    missing_variables = [
        name
        for name, value in {
            "DB_PORT": DB_PORT,
            "DB_USERNAME": DB_USERNAME,
            "DB_PASSWORD": DB_PASSWORD,
            "PC_IP_ADDRESS": PC_IP_ADDRESS,
        }.items()
        if not value
    ]
    if missing_variables:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing_variables)}. "
            f"Add them to {ENV_FILE}."
        )


def build_database_url():
    _require_env_vars()
    return URL.create(
        drivername="postgresql+psycopg",
        username=DB_USERNAME,
        password=DB_PASSWORD,
        host=PC_IP_ADDRESS,
        port=DB_PORT,
        database="eq_db",
    )


def create_database_engine(db_url=None):
    if db_url is None:
        db_url = build_database_url()
    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def make_table_name(csv_file_path, source_directory):
    relative_path = csv_file_path.relative_to(source_directory).with_suffix("")
    table_name = "_".join(relative_path.parts).lower()
    table_name = re.sub(r"[^a-z0-9_]+", "_", table_name).strip("_")

    if not table_name:
        raise ValueError(f"Could not create a table name for {csv_file_path}")
    if table_name[0].isdigit():
        table_name = f"data_{table_name}"
    if len(table_name) > 63:
        suffix = hashlib.sha256(table_name.encode("utf-8")).hexdigest()[:8]
        table_name = f"{table_name[:54]}_{suffix}"

    return table_name


def discover_csv_imports():
    imports = []
    table_destinations = set()

    for csv_file_path in sorted(DATA_DIRECTORY.rglob("*.csv")):
        try:
            csv_file_path.relative_to(ENGINEERED_DATA_DIRECTORY)
            schema_name = PREPPED_SCHEMA_NAME
            source_directory = ENGINEERED_DATA_DIRECTORY
        except ValueError:
            schema_name = RAW_SCHEMA_NAME
            source_directory = DATA_DIRECTORY

        table_name = make_table_name(csv_file_path, source_directory)
        destination = (schema_name, table_name)
        if destination in table_destinations:
            raise ValueError(
                f"Multiple CSV files map to {schema_name}.{table_name}. "
                "Rename one of the files to make its table name unique."
            )

        table_destinations.add(destination)
        imports.append((csv_file_path, schema_name, table_name))

    if not imports:
        raise FileNotFoundError(f"No CSV files found under {DATA_DIRECTORY}")

    return imports


def make_index_name(table_name, suffix):
    index_name = f"idx_{table_name}_{suffix}"
    if len(index_name) > 63:
        digest = hashlib.sha256(index_name.encode("utf-8")).hexdigest()[:8]
        index_name = f"{index_name[:54]}_{digest}"
    return index_name


def create_indexes(connection, schema_name, table_name, columns):
    available_columns = set(columns)
    index_definitions = [
        (("latitude", "longitude"), "lat_lon"),
        (("mag",), "mag"),
        (("magType",), "magtype"),
    ]

    preparer = connection.dialect.identifier_preparer
    quoted_schema = preparer.quote(schema_name)
    quoted_table = preparer.quote(table_name)

    for index_columns, suffix in index_definitions:
        if not set(index_columns).issubset(available_columns):
            continue

        index_name = make_index_name(table_name, suffix)
        quoted_index = preparer.quote(index_name)
        quoted_columns = ", ".join(
            preparer.quote(column_name) for column_name in index_columns
        )
        connection.execute(
            text(
                f"CREATE INDEX {quoted_index} "
                f"ON {quoted_schema}.{quoted_table} ({quoted_columns})"
            )
        )
        print(
            f"  Created index {index_name} on "
            f"{schema_name}.{table_name} ({', '.join(index_columns)})."
        )


def upload_csv(engine, csv_file_path, schema_name, table_name):
    csv_file_path = Path(csv_file_path)
    if not csv_file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    print(f"Uploading {csv_file_path} to {schema_name}.{table_name}...")
    total_rows = 0
    staging_suffix = uuid.uuid4().hex[:8]
    staging_table_name = f"_staging_{table_name[:45]}_{staging_suffix}"
    staging_created = False
    csv_columns = None

    try:
        for chunk_number, dataframe in enumerate(
            pd.read_csv(csv_file_path, chunksize=READ_CHUNK_SIZE),
            start=1,
        ):
            if csv_columns is None:
                csv_columns = list(dataframe.columns)
            with engine.begin() as connection:
                dataframe.to_sql(
                    name=staging_table_name,
                    con=connection,
                    schema=schema_name,
                    if_exists="replace" if chunk_number == 1 else "append",
                    index=False,
                    chunksize=WRITE_CHUNK_SIZE,
                    method="multi",
                )
            staging_created = True
            total_rows += len(dataframe)
            print(f"  Uploaded {total_rows:,} rows to staging...")

        if not staging_created:
            raise ValueError(f"CSV file contains no readable rows or columns: {csv_file_path}")

        with engine.begin() as connection:
            preparer = connection.dialect.identifier_preparer
            quoted_schema = preparer.quote(schema_name)
            quoted_table = preparer.quote(table_name)
            quoted_staging = preparer.quote(staging_table_name)

            connection.execute(
                text(f"DROP TABLE IF EXISTS {quoted_schema}.{quoted_table}")
            )
            connection.execute(
                text(
                    f"ALTER TABLE {quoted_schema}.{quoted_staging} "
                    f"RENAME TO {quoted_table}"
                )
            )
            create_indexes(
                connection,
                schema_name=schema_name,
                table_name=table_name,
                columns=csv_columns,
            )
        staging_created = False
    finally:
        if staging_created:
            with engine.begin() as connection:
                preparer = connection.dialect.identifier_preparer
                quoted_schema = preparer.quote(schema_name)
                quoted_staging = preparer.quote(staging_table_name)
                connection.execute(
                    text(
                        f"DROP TABLE IF EXISTS "
                        f"{quoted_schema}.{quoted_staging}"
                    )
                )

    print(f"Completed {table_name}: {total_rows:,} rows uploaded atomically.")


def fetch_table(schema_name, table_name):
    engine = create_database_engine()
    try:
        dataframe = pd.read_sql_table(
            table_name,
            engine,
            schema=schema_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {schema_name}.{table_name} from the database."
        ) from exc
    finally:
        engine.dispose()

    if dataframe.empty:
        raise ValueError(f"Table {schema_name}.{table_name} exists but contains no rows.")

    return dataframe


def fetch_engineered_data(file_name="eq_data"):
    csv_file_path = Path(file_name)
    if not csv_file_path.is_absolute():
        csv_file_path = ENGINEERED_DATA_DIRECTORY / csv_file_path
    table_name = make_table_name(csv_file_path, ENGINEERED_DATA_DIRECTORY)
    return fetch_table(PREPPED_SCHEMA_NAME, table_name)


def fetch_raw_data(file_name="eq_data_updated3.csv"):
    csv_file_path = Path(file_name)
    if not csv_file_path.is_absolute():
        csv_file_path = DATA_DIRECTORY / csv_file_path
    table_name = make_table_name(csv_file_path, DATA_DIRECTORY)
    return fetch_table(RAW_SCHEMA_NAME, table_name)


def upload_all_csv_files():
    csv_imports = discover_csv_imports()
    engine = create_database_engine()
    try:
        with engine.begin() as connection:
            for schema_name in (RAW_SCHEMA_NAME, PREPPED_SCHEMA_NAME):
                connection.execute(CreateSchema(schema_name, if_not_exists=True))
                print(f"Schema '{schema_name}' is ready.")

        for csv_file_path, schema_name, table_name in csv_imports:
            upload_csv(engine, csv_file_path, schema_name, table_name)
    finally:
        engine.dispose()

    print("All CSV files were uploaded successfully.")


if __name__ == "__main__":
    upload_all_csv_files()
