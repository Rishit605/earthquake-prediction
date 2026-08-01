# Earthquake Prediction

Predicts earthquake magnitude from USGS seismic catalogue data using a
from-scratch and scikit-learn ML pipeline. The data pipeline supports two
feature modes: **normal** row-level features (the default) and an optional
**time-series** mode that adds cyclical time and rolling-window features.
The LSTM model and training loop in `src/eq_prediction/training/training_nn.py`
and `src/eq_prediction/model/model.py` remain exploratory groundwork rather
than the default workflow.

This phase's main deliverable is the **data pipeline**: a single command
takes raw USGS data (from a live API, a local CSV, or PostgreSQL) all the
way through cleaning, missing-value enrichment, feature engineering, and a
chronological train/validation/test split, ready to feed into a model.

---

## Table of Contents

- [What's in this repo](#whats-in-this-repo)
- [Project layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Understanding the pipeline output](#understanding-the-pipeline-output)
- [Configuration reference](#configuration-reference)
- [Running with Docker](#running-with-docker)
- [Running the tests](#running-the-tests)
- [Legacy / exploratory scripts](#legacy--exploratory-scripts)
- [Known issues](#known-issues)
- [Roadmap](#roadmap)

---

## What's in this repo

- **A data pipeline** (`src/eq_prediction/pipeline/`) that fetches earthquake
  events from the USGS API, a local CSV, or a PostgreSQL database; cleans
  and deduplicates them; fills in a few commonly-missing seismic network
  fields (`nst`, `dmin`, `gap`) from cached or fetched USGS event detail
  pages; engineers normal features (missingness indicators and one-hot
  encoded magnitude type) or optional time-series features (cyclical time
  and rolling-window features); and writes a chronological
  train/validation/test split to disk.
- **From-scratch ML implementations** (`src/eq_prediction/model/`) — linear
  regression with gradient descent, a decision tree regressor, and a KNN
  imputer, all built from first principles rather than scikit-learn, as a
  learning exercise.
- **A Postgres-backed storage layer** (`src/eq_prediction/helpers/sql_data_handler.py`)
  for syncing data across machines.
- **A CLI** (`eq-pipeline`) that wraps the whole pipeline in one command.

---

## Project layout

```text
.
├── src/eq_prediction/
│   ├── pipeline/            ← the maintained, tested Phase-1 pipeline (start here)
│   │   ├── cli.py           ← `eq-pipeline` entry point
│   │   ├── config.py        ← all environment/config handling (PipelineSettings)
│   │   ├── sources.py       ← load from local CSV / Postgres / live USGS API
│   │   ├── normalize.py     ← reconciles differently-shaped inputs into one schema
│   │   ├── clean.py         ← type coercion, validity filtering, deduplication
│   │   ├── enrich.py        ← fills missing nst/dmin/gap via cached patch or USGS detail URLs
│   │   ├── features.py      ← feature engineering (time features, one-hot, missing flags)
│   │   ├── split.py         ← chronological train/validation/test split
│   │   ├── training_ready.py← z-score scaling fit on train, applied to val/test
│   │   ├── storage.py       ← writes CSVs + a JSON run summary
│   │   ├── models.py        ← StageResult / PipelineResult dataclasses
│   │   └── runner.py        ← orchestrates all of the above
│   │
│   ├── model/                ← from-scratch ML: LinearR, DecisionTreeR, KNNImpute, etc.
│   ├── training/              ← LSTM training loop (Phase 2 groundwork, not yet wired to the CLI)
│   ├── prediction/            ← inference helpers for the LSTM model
│   ├── helpers/                ← Postgres client, logging, misc utilities
│   │
│   ├── preprocessing/          ← OLDER preprocessing modules, pre-pipeline-integration
│   └── notebook/                ← notebook-extracted exploratory code
│
├── scripts/
│   ├── main3.py    ← exploratory script using the new pipeline + from-scratch models
│   ├── main2.py    ← exploratory script using the OLDER preprocessing/ modules
│   └── main1.py    ← FastAPI service serving the Phase-2 LSTM model (needs a trained checkpoint)
│
├── tests/pipeline/    ← automated tests for the pipeline package (start here to see it work)
├── data/              ← not tracked in git; the pipeline creates subfolders here on first run
├── models/             ← not tracked in git; trained model artifacts go here
├── notebooks/           ← scratch notebooks, not tracked in git
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

**If you only read one part of the source, read `src/eq_prediction/pipeline/`.**
It's the newest, most tested, most consistently structured code in the repo,
and it's the intended entry point going forward. `preprocessing/`, `notebook/`,
`scripts/main2.py`, and `scripts/main3.py` predate this pipeline or were used
to build it interactively — see [Legacy / exploratory scripts](#legacy--exploratory-scripts).

---

## Prerequisites

- **Python 3.10 or later** (3.11 recommended — that's what the Docker image uses)
- **Git**
- **PostgreSQL** — only if you want to use the database source; entirely optional
- **~2–3 GB of free disk space** — the dependency list includes PyTorch,
  XGBoost, and Comet ML (used by the Phase-2 LSTM work), so the initial
  install is sizeable even though the Phase-1 pipeline itself only needs
  pandas, numpy, requests, SQLAlchemy, and python-dotenv.

---

## Quickstart

### 1. Clone and enter the repo

```bash
git clone -b data-pipeline-integration https://github.com/Rishit605/earthquake-prediction.git
cd earthquake-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the package in editable mode (so code edits take effect
immediately) plus `pytest` for running tests. This step can take a while and
use several GB of disk space because of the PyTorch/XGBoost/Comet ML
dependencies mentioned above.

### 4. Set up your environment file

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

**Every value in `.env.example` is optional for the pipeline's default
behavior.** You only need to fill in the `DB_*` / `PC_IP_ADDRESS` /
`EQ_DB_*` values if you plan to use `--source db`. Leave everything else
blank to start.

### 5. Get some data flowing through the pipeline

You have three options — pick whichever is easiest for you.

**Option A — pull live data from the USGS API (fastest, no setup):**

```bash
eq-pipeline run --source local --fetch-new
```

`--source local` will fail with a "file not found" message internally
(there's no local CSV yet) — that's expected and harmless. The `--fetch-new`
flag then hits the public USGS earthquake API directly and fetches recent
events (last 7 days by default), which is enough data to see the full
pipeline run end-to-end. Any errors reported for the local source in the
final output can be ignored in this case.

**Option B — use a local CSV you already have:**

Place it at `data/new_raw_data/eq_data_updated3.csv` (create the folder if
it doesn't exist), then run:

```bash
eq-pipeline run --source local
```

**Option C — use a PostgreSQL database:**

Fill in the `DB_*`, `PC_IP_ADDRESS`, and `EQ_DB_*` variables in `.env`, then run:

```bash
eq-pipeline run --source db
```

### 6. Check the result

```bash
eq-pipeline status
```

This prints the JSON summary of the most recent run — row counts at each
stage, any errors, and where the output files were written.

---

## Understanding the pipeline output

A successful run writes these files under `data/` (all git-ignored, created
automatically):

| File | Contents |
|---|---|
| `data/raw/earthquakes_raw.csv` | Combined raw data from every source you requested, before cleaning |
| `data/raw/usgs_fetched_raw.csv` | Full flattened USGS response fields for newly fetched events; written only with `--fetch-new` |
| `data/clean/earthquakes_clean.csv` | After validity filtering, deduplication, and missing-value enrichment |
| `data/model_ready/earthquakes_model_ready.csv` | Fully feature-engineered, numeric-only, z-scored — ready for a model |
| `data/splits/train.csv`, `validation.csv`, `test.csv` | Chronological split of the model-ready data (70/15/15 by default) |
| `data/prediction/latest_prediction_input.csv` | The single most recent row, features only, target column dropped |
| `data/rejected/rejected_raw.csv` | Rows that failed validation (bad coordinates, missing magnitude, etc.), for auditing |
| `data/run_summary.json` | Row counts and messages for every pipeline stage — this is what `eq-pipeline status` prints |

The chronological split matters here: because you'll eventually forecast
future earthquakes from past ones, the test set is always the most recent
slice of time, never a random sample — this avoids leaking future
information into training.

---

## Configuration reference

All configuration lives in `.env` (copy it from `.env.example`) and is read
by `src/eq_prediction/pipeline/config.py`. Every variable is optional unless
noted.

| Variable | Default | Purpose |
|---|---|---|
| `DB_UNAME` | — | Postgres username (**required for `--source db`**) |
| `DB_PASSWORD` | — | Postgres password (**required for `--source db`**) |
| `DB_PORT` | `5432` | Postgres port |
| `PC_IP_ADDRESS` | `localhost` | Postgres host |
| `EQ_DB_NAME` | `eq_db` | Database name |
| `EQ_DB_RAW_UNPATCHED_SCHEMA` / `_TABLE` | `lappy_raw_data` / `eq_data_updated3` | Schema/table for raw, unpatched data |
| `EQ_DB_RAW_PATCHED_SCHEMA` / `_TABLE` | `final_data` / `eq_data_updated3_patched` | Schema/table for the enriched/patched data (preferred source) |
| `EQ_DB_ENRICH_PATCH_SCHEMA` / `_TABLE` | — | Optional table holding a saved missing-value patch |
| `EQ_PIPELINE_LOCAL_RAW_PATCHED_PATH` | `data/patched_raw_data/eq_data_updated3_patched.csv` | Local CSV path, preferred over unpatched |
| `EQ_PIPELINE_LOCAL_RAW_UNPATCHED_PATH` | `data/new_raw_data/eq_data_updated3.csv` | Local CSV path, fallback |
| `EQ_PIPELINE_LOCAL_ENRICHMENT_PATH` | `data/enrich_patch_data/FinalRegressionData.csv` | Saved missing-value patch file, avoids re-hitting the USGS API |
| `EQ_PIPELINE_MIN_MAGNITUDE` | `2.5` | Minimum magnitude when fetching from USGS |
| `EQ_PIPELINE_FETCH_DAYS` | `7` | How many days back to fetch when no prior data exists |
| `EQ_PIPELINE_OVERLAP_DAYS` | `2` | Overlap window when fetching incrementally, to avoid gaps |
| `EQ_PIPELINE_REQUEST_TIMEOUT_SECONDS` | `30` | HTTP timeout for USGS requests |
| `EQ_PIPELINE_REQUEST_MIN_INTERVAL_SECONDS` | `1` | Minimum delay between USGS requests |
| `API_KEY`, `PROJECT_NAME`, `WORKSPACE` | — | Comet ML experiment tracking, only used by `scripts/main1.py` |
| `FRONTEND_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins permitted to call `scripts/main1.py` |

### CLI reference

```bash
eq-pipeline run [--source {local,db,all}] [--fetch-new] [--timeseries] [--enrich-details]
                 [--no-save] [--fetch-start ISO_DATETIME] [--fetch-end ISO_DATETIME]
eq-pipeline status
```

- `--source` — where to load existing data from (default `all`, meaning both local and db)
- `--fetch-new` — additionally fetch new events from the live USGS API
- `--timeseries` — add cyclical time and rolling-window features. Omit it for the default row-level feature set.
- `--enrich-details` — allow network calls to USGS event-detail pages to fill missing `nst`/`dmin`/`gap` (see [Known issues](#known-issues) before using this on a fresh clone)
- `--no-save` — run the pipeline without writing any output files (useful for a dry run)
- `--fetch-start` / `--fetch-end` — explicit ISO datetimes bounding the USGS fetch window

---

## Running with Docker

```bash
docker compose build
docker compose run --rm app eq-pipeline run --source local
```

A `.env` file must exist before running `docker compose` commands (Docker
Compose reads it directly) — copy it from `.env.example` first, even if you
leave every value blank.

`data/` and `models/` are mounted as volumes, so pipeline output persists on
your host machine between container runs.

To also spin up a local Postgres instance for testing the `--source db`
path:

```bash
docker compose --profile db up -d db
```

This starts a disposable Postgres 16 container (credentials in
`docker-compose.yml`) — convenient for trying the database path without
installing Postgres locally, but not intended for storing data you care about.

---

## Running the tests

```bash
pytest
```

This runs the automated test suite under `tests/pipeline/`, which covers
normalization, cleaning, feature engineering, chronological splitting, and
two end-to-end smoke tests that run the entire pipeline against small
in-memory CSVs. No network access or database is required — network and
database code paths are tested using temporary files and monkey-patching.

---

## Legacy / exploratory scripts

`scripts/main2.py` and the `src/eq_prediction/preprocessing/` package
predate the pipeline integration — they use an older, less consistent data
loading path (`EQDataLoader`) and are kept for reference rather than as a
maintained entry point. `scripts/main3.py` is a newer exploratory script
that does use the new pipeline (`run_pipeline`, `load_dataset`), but it
still has commented-out sections and defaults to `--source db`, so treat it
as a scratch script rather than documentation of intended usage.

`scripts/main1.py` is a FastAPI service that serves predictions from the
Phase-2 LSTM model (`src/eq_prediction/model/model.py`,
`src/eq_prediction/training/training_nn.py`). It requires a trained model
checkpoint under `models/trained/` and a Comet ML API key, and isn't part of
the Phase-1 workflow this README documents.

---

## Known issues

The Docker service defaults to `--source local`, so a fresh clone needs either
a local CSV at the configured path or an explicit `--fetch-new` command.
The legacy LSTM/API path is still exploratory; see the roadmap for the work
needed to integrate it directly with pipeline outputs.

---

## Roadmap

- **Phase 2 (next):** reframe this as a time-series forecasting problem.
  `src/eq_prediction/training/training_nn.py` and
  `src/eq_prediction/model/model.py` already contain an LSTM model and
  training loop from earlier exploration — the near-term work is wiring
  the pipeline's chronological splits into that training loop directly,
  rather than treating each row as independent.
- Reconcile `preprocessing/` and `notebook/` into the new `pipeline/`
  structure, or retire them once Phase 2 supersedes the traditional-ML
  approach they were built for.
