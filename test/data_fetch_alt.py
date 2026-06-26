"""Rate-limit-friendly alternative to ``feature_eng.FilterAndFill``.

The URLs in the ``detail`` column point directly to USGS event-detail JSON.
Opening one in a browser still sends an HTTP request, so a browser does not
bypass server-side rate limits.  This module instead behaves like a polite
browser: it sends browser-style headers, uses a small thread pool with global
request pacing, caches every response on disk, and never requests the same URL
twice during a run.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


KEY_MAP = {
    "nst": "num-stations-used",
    "dmin": "minimum-distance",
    "gap": "azimuthal-gap",
}


class DebugLog:
    """Write timestamped scraper progress to a file and optionally the console."""

    def __init__(
        self,
        log_path: str | Path | None = None,
        print_to_console: bool = True,
    ) -> None:
        self.path = Path(
            log_path or Path(__file__).parent / "data_fetch_alt_debug.log"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.print_to_console = print_to_console
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            if self.print_to_console:
                print(line)


def _numeric_value(value: Any) -> Any:
    """Convert numeric JSON strings to int/float, preserving other values."""
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def json_data_search(
    data: dict[str, Any],
    keys: Iterable[str] = (
        "azimuthal-gap",
        "num-stations-used",
        "minimum-distance",
    ),
    product_categories: Iterable[str] | None = ("origin", "phase-data"),
) -> dict[str, Any]:
    """Collect the same event-detail values used by ``FilterAndFill``."""
    if data.get("type") == "FeatureCollection" and data.get("features"):
        data = data["features"][0]

    products = data.get("properties", {}).get("products", {})
    categories = (
        tuple(products)
        if product_categories is None
        else tuple(product_categories)
    )
    wanted = frozenset(keys)
    result: dict[str, Any] = {}

    for category in categories:
        for entry in products.get(category) or []:
            properties = entry.get("properties") or {}
            for key in wanted:
                if key in properties and key not in result:
                    result[key] = _numeric_value(properties[key])

    return result


class CachedJSONScraper:
    """Fetch JSON links slowly and persist responses to avoid repeat requests."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 EarthquakeResearch/1.0"
    )

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        min_interval: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 5,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or Path(__file__).parent / "data" / "json_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = logger or (lambda _message: None)
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._next_request_at = 0.0
        self._url_locks_lock = threading.Lock()
        self._url_locks: dict[str, threading.Lock] = {}

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _wait_for_request_slot(self) -> None:
        """Reserve one globally paced request slot across all worker threads."""
        with self._request_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request_at - now)
            if wait:
                time.sleep(wait)
            self._next_request_at = time.monotonic() + self.min_interval

    def _defer_requests(self, delay: float) -> None:
        """Apply a shared cooldown after a rate-limit response."""
        with self._request_lock:
            self._next_request_at = max(
                self._next_request_at,
                time.monotonic() + max(0.0, delay),
            )

    def _url_lock(self, url: str) -> threading.Lock:
        """Return a stable lock so one URL cannot be fetched twice concurrently."""
        with self._url_locks_lock:
            return self._url_locks.setdefault(url, threading.Lock())

    @staticmethod
    def _retry_delay(error: HTTPError, fallback: float) -> float:
        retry_after = error.headers.get("Retry-After")
        if not retry_after:
            return fallback
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return fallback

    def fetch_json(
        self,
        url: str,
        row_label: str = "unknown row",
    ) -> dict[str, Any]:
        """Return JSON from memory/disk cache, otherwise fetch it once."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("A non-empty JSON URL is required.")

        url = url.strip()
        with self._url_lock(url):
            with self._cache_lock:
                if url in self._memory_cache:
                    self.log(
                        f"[{row_label}] Memory-cache hit; "
                        "no network request needed."
                    )
                    return self._memory_cache[url]

            cache_path = self._cache_path(url)
            if cache_path.is_file():
                self.log(f"[{row_label}] Disk-cache hit; no network request needed.")
                with cache_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                with self._cache_lock:
                    self._memory_cache[url] = data
                return data

            delay = max(2.0, self.min_interval)
            request = Request(
                url,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": (
                        "application/geo+json, application/json, text/plain, */*"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "max-age=0",
                },
            )

            for attempt in range(1, self.max_retries + 1):
                self.log(
                    f"[{row_label}] Fetch attempt "
                    f"{attempt}/{self.max_retries}: {url}"
                )
                self._wait_for_request_slot()
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        raw_data = response.read()
                    data = json.loads(raw_data.decode("utf-8-sig"))
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected a JSON object from {url}.")

                    temporary_path = cache_path.with_name(
                        f"{cache_path.stem}.{threading.get_ident()}.tmp"
                    )
                    with temporary_path.open("w", encoding="utf-8") as file:
                        json.dump(data, file, ensure_ascii=False)
                    temporary_path.replace(cache_path)
                    with self._cache_lock:
                        self._memory_cache[url] = data
                    self.log(
                        f"[{row_label}] Fetch succeeded on attempt "
                        f"{attempt}/{self.max_retries}."
                    )
                    return data
                except HTTPError as error:
                    self.log(
                        f"[{row_label}] Attempt {attempt}/{self.max_retries} "
                        f"returned HTTP {error.code}."
                    )
                    if error.code != 429 or attempt == self.max_retries:
                        raise
                    wait = self._retry_delay(error, delay)
                    self._defer_requests(wait)
                    self.log(
                        f"[{row_label}] Rate limited; shared cooldown "
                        f"set to {wait:.1f}s."
                    )
                    delay *= 2
                except URLError as error:
                    self.log(
                        f"[{row_label}] Attempt {attempt}/{self.max_retries} "
                        f"failed: {error}."
                    )
                    if attempt == self.max_retries:
                        raise
                    self._defer_requests(delay)
                    self.log(
                        f"[{row_label}] Shared retry delay set to {delay:.1f}s."
                    )
                    delay *= 2

        raise RuntimeError(f"Could not fetch JSON from {url}.")


class FilterAndFillAlt:
    """Fill ``nst``, ``dmin``, and ``gap`` from cached detail JSON links."""

    def __init__(
        self,
        scraper: CachedJSONScraper | None = None,
        log_path: str | Path | None = None,
        print_to_console: bool = True,
        max_workers: int = 3,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        self.debug_log = DebugLog(log_path, print_to_console)
        self.scraper = scraper or CachedJSONScraper(logger=self.debug_log)
        self.max_workers = max_workers
        if scraper is not None and hasattr(scraper, "log"):
            scraper.log = self.debug_log

    def filter_and_fill(
        self,
        data: pd.DataFrame,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        worker_limit = self.max_workers if max_workers is None else max_workers
        if worker_limit < 1:
            raise ValueError("max_workers must be at least 1.")

        required_columns = {"detail", *KEY_MAP}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise KeyError(
                "DataFrame is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        result = data.copy()
        valid_url = result["detail"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        needs_filling = result[list(KEY_MAP)].isna().any(axis=1)
        rows_to_fix = result.loc[valid_url & needs_filling]
        if rows_to_fix.empty:
            self.debug_log("No rows need fetching. Completed=0, Failed=0.")
            return result

        # Fetch each unique URL once. The small worker pool overlaps network
        # waiting while CachedJSONScraper globally spaces actual requests.
        updates_by_url: dict[str, dict[str, Any]] = {}
        unique_urls = rows_to_fix["detail"].str.strip().drop_duplicates()
        total = len(unique_urls)
        completed = 0
        failed = 0
        self.debug_log(
            f"Starting run: {len(rows_to_fix)} row(s), {total} unique URL(s), "
            f"max_workers={worker_limit}."
        )

        def process_url(job_number: int, url: str):
            matching_rows = rows_to_fix.index[
                rows_to_fix["detail"].str.strip().eq(url)
            ].tolist()
            row_label = f"rows/indexes {matching_rows}"
            self.debug_log(
                f"[JOB {job_number}/{total}] Worker started for {row_label}."
            )
            try:
                json_data = self.scraper.fetch_json(url, row_label=row_label)
            except TypeError as error:
                # Keep simple injected/fake scrapers compatible.
                if "row_label" not in str(error):
                    raise
                json_data = self.scraper.fetch_json(url)

            values = json_data_search(json_data)
            updates = {
                column: values[json_key]
                for column, json_key in KEY_MAP.items()
                if json_key in values
            }
            return job_number, url, row_label, updates

        worker_count = min(worker_limit, total)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="earthquake-json",
        ) as executor:
            futures = {
                executor.submit(process_url, job_number, url): (job_number, url)
                for job_number, url in enumerate(unique_urls, start=1)
            }

            for future in as_completed(futures):
                job_number, url = futures[future]
                try:
                    _, _, row_label, updates = future.result()
                    updates_by_url[url] = updates
                    completed += 1
                    found_columns = sorted(updates)
                    self.debug_log(
                        f"[JOB {job_number}/{total}] Completed {row_label}; "
                        f"found={found_columns or 'no target values'}. "
                        f"Completed={completed}, Failed={failed}, "
                        f"Remaining={total - completed - failed}."
                    )
                except (
                    HTTPError,
                    URLError,
                    ValueError,
                    json.JSONDecodeError,
                    OSError,
                ) as error:
                    failed += 1
                    updates_by_url[url] = {}
                    self.debug_log(
                        f"[JOB {job_number}/{total}] FAILED URL {url}: {error}. "
                        f"Completed={completed}, Failed={failed}, "
                        f"Remaining={total - completed - failed}."
                    )

        patch_rows = [
            updates_by_url.get(url.strip(), {})
            for url in rows_to_fix["detail"]
        ]
        patch = pd.DataFrame(patch_rows, index=rows_to_fix.index)
        self.debug_log(
            f"Run finished. Rows considered={len(rows_to_fix)}, "
            f"unique URLs={total}, Completed={completed}, Failed={failed}."
        )
        return result.fillna(patch)


# Familiar aliases for swapping this module into existing experiments.
# FilterAndFill = FilterAndFillAlt
# WebScraper = CachedJSONScraper
