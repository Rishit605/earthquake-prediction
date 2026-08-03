from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

from eq_prediction.pipeline import enrich


class Response:
    def __init__(self, code, payload=None, headers=None):
        self.status_code = code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(str(self.status_code))
            error.response = self
            raise error

    def json(self):
        return self._payload


cache_dir = Path(".enrich_sandbox_cache")
cache_dir.mkdir(exist_ok=True)
settings = SimpleNamespace(
    cache_dir=cache_dir, request_timeout_seconds=5, request_min_interval_seconds=0
)
if True:
    cache = enrich.DetailCache(settings)
    original_get, original_sleep = enrich.requests.get, enrich.time.sleep
    calls, delays = [], []
    responses = [
        Response(429, headers={"Retry-After": "0"}),
        Response(429),
        Response(200, {"ok": True}),
    ]
    enrich.requests.get = lambda *args, **kwargs: (calls.append(1) or responses.pop(0))
    enrich.time.sleep = lambda seconds: delays.append(seconds)
    assert cache.fetch("https://unit-test.invalid/retry-unique", 7) == {"ok": True, "index": 7}
    assert len(calls) == 3 and delays == [0.0, 0]
    enrich.requests.get, enrich.time.sleep = original_get, original_sleep

    original_fetch = enrich.DetailCache.fetch

    def raise_404(self, url, index):
        error = requests.exceptions.HTTPError("404")
        error.response = Response(404)
        raise error

    enrich.DetailCache.fetch = raise_404
    frame = pd.DataFrame(
        {"detail": ["https://unit-test.invalid/404"], "nst": [pd.NA], "dmin": [1.0], "gap": [2.0]}
    )
    status_test = enrich.enrich_missing_detail_columns(
        frame, settings, fetch_missing=True, patch_dataframe=pd.DataFrame()
    )
    assert status_test.loc[0, "enrichment_status"] == "failed:HTTPError:404"
    enrich.DetailCache.fetch = original_fetch

raw = pd.read_csv(r"data/raw/usgs_fetched_raw.csv")
missing = raw.loc[raw["nst"].isna(), ["event_id", "detail", "nst"]]
print(f"raw rows={len(raw)}; missing nst rows={len(missing)}")
print(missing.head(3).to_string(index=False))
print("retry mock: PASS")
print("HTTP status exception handling: PASS")
