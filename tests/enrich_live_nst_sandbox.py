from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from eq_prediction.pipeline.enrich import enrich_missing_detail_columns


raw = pd.read_csv(r"data/raw/usgs_fetched_raw.csv")
sample = raw.loc[raw["nst"].isna()].head(3).copy()
assert len(sample) == 3
assert sample["nst"].isna().all()

cache_dir = Path(".enrich_sandbox_cache")
cache_dir.mkdir(exist_ok=True)
settings = SimpleNamespace(
    cache_dir=cache_dir,
    request_timeout_seconds=15,
    request_min_interval_seconds=1,
)
result = enrich_missing_detail_columns(
    sample, settings, fetch_missing=True, patch_dataframe=pd.DataFrame()
)
print(
    result[["event_id", "nst", "enrichment_status"]]
    .to_string(index=False)
)
print(f"nst filled={int(result['nst'].notna().sum())}/{len(result)}")
