# Earthquake Prediction

Clean project home for the earthquake prediction data pipeline, preprocessing,
training, and inference code.

This repository was populated from the existing working folders and now carries
the data pipeline inside the main `eq_prediction` package.

## Layout

```text
src/eq_prediction/pipeline/      Data pipeline code
src/eq_prediction/preprocessing/ Data preparation and feature code
src/eq_prediction/model/         Model implementations
src/eq_prediction/training/      Training code
src/eq_prediction/prediction/    Inference code
src/eq_prediction/helpers/       Shared helpers
scripts/                         Existing script entry points
tests/                           Test suite
data/                            Local data mount, ignored by Git
models/                          Local model artifacts, ignored by Git
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Copy `.env.example` to `.env` and fill only the values you need.

## Pipeline

```powershell
eq-pipeline run --source local
eq-pipeline status
```

The pipeline code came from the standalone `eq_prediction_data_pipeline`
package and is integrated as:

```python
from eq_prediction.pipeline import run_pipeline
```

By default it reads local data from `data/patched_raw_data` first, falls back to
`data/new_raw_data`, applies any saved enrichment patch from
`data/enrich_patch_data`, and writes model-ready outputs to `data/`.

## Docker

```powershell
docker compose build
docker compose run --rm app eq-pipeline run --source local
```

Data and trained models are mounted as volumes:

```text
./data   -> /app/data
./models -> /app/models
```

They are intentionally ignored by Git.

## Migration Notes

- Existing source files were copied into the new package layout.
- Runtime artifacts were not copied: CSV data, model weights, logs, caches,
  virtual environments, archives, and `__pycache__`.
- The code still contains some old imports such as `from src...` and some old
  path assumptions. Those should be cleaned in the next pass.
