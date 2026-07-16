import numpy as np
import pandas as np
import geojson
import geopandas as gpd

class FetchURLReturnJSON:
    def __init__(self, file_path=None, save=False):
        import time
        import json
        import requests
        self.time = time
        self.requests = requests
        self.json = json
        self.file_path = file_path
        self.save = save
    def fetch_json_from_url(self, url, max_retries=5):
        """
        Fetches a JSON object from a given URL with exponential backoff for 429 errors.
        """
        delay = 2  # Start with a 2-second delay
        
        for attempt in range(max_retries):
            response = self.requests.get(url)
            
            # If it's successful, return the JSON immediately
            if response.status_code == 200:
                return response.json()
                
            # If we get rate-limited (429), wait and try again
            elif response.status_code == 429:
                print(f"[429 Rate Limit] Hit limit on attempt {attempt + 1}. Waiting {delay}s...")
                self.time.sleep(delay)
                delay *= 2  # Double the wait time for the next attempt
                
            else:
                # For other errors like 404 or 500, raise the error immediately
                response.raise_for_status()
                
        # If we fall out of the loop, it means we ran out of retries
        raise Exception(f"Failed to fetch data after {max_retries} retries due to rate limiting.")
    
    # Save the fetched geojson_data to a file
    def json_data_fetch(self, url, saved=False):
        if saved:
            if self.file_path:
                with open(self.file_path, "w") as f:
                    self.json.dump(self.fetch_json_from_url(url), f, indent=4)
                return self.fetch_json_from_url(url)
            else:
                print("File path is not provided")
                from pathlib import Path
                current_file = Path(__file__).resolve()   # full path to this .py file
                current_dir = current_file.parent         # directory containing it
                file_path = current_dir / "data" / "geojson_data.json"
                with open(file_path, "w") as f:
                    self.json.dump(self.fetch_json_from_url(url), f, indent=4)
                return self.fetch_json_from_url(url)
        else:
            return self.fetch_json_from_url(url)


def json_data_search(
    data,
    keys=("azimuthal-gap", "num-stations-used", "minimum-distance"),
    product_categories=("origin", "phase-data")):
    """
    USGS event-detail GeoJSON: data['properties']['products'] maps product
    name -> list of product dicts. Each dict has nested 'properties' with
    depth, azimuthal-gap, num-stations-used, etc.

    Collects each requested key once (first occurrence), typically from origin.
    Use product_categories=None to scan all product lists.
    Converts found values to their appropriate numerical format if possible (int or float).
    """
    if keys is None:
        keys = ("depth", "azimuthal-gap", "num-stations-used", "minimum-distance")
    keys_set = frozenset(keys)

    if data.get("type") == "FeatureCollection" and data.get("features"):
        data = data["features"][0]

    products = data["properties"]["products"]
    out = {}
    categories = product_categories if product_categories is not None else tuple(products.keys())

    for cat in categories:
        for entry in products.get(cat) or []:
            props = entry.get("properties") or {}
            for k in keys_set:
                if k in props and k not in out:
                    val = props[k]
                    # Try to convert to int, then to float, else leave as is
                    try:
                        # Prefer int, but fall back to float if needed
                        if isinstance(val, str) and val.isdigit():
                            conv_val = int(val)
                        else:
                            conv_val = float(val)
                            if conv_val.is_integer():
                                conv_val = int(conv_val)
                        out[k] = conv_val
                    except Exception:
                        try:
                            out[k] = float(val)
                        except Exception:
                            out[k] = val

    return out


# Filter and fill missing data from the json data
class FilterAndFill:
    def __init__(self):
        self.json_data = FetchURLReturnJSON()
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd
        self.ThreadPoolExecutor = ThreadPoolExecutor
        self.pd = pd

    def filter_and_fill(self, data, max_workers=20):

        ddf = data.copy()
        
        key_map = {
            'nst': 'num-stations-used',
            'dmin': 'minimum-distance',
            'gap': 'azimuthal-gap'
        }
        target_cols = list(key_map.keys())

        # 1. Identify rows that have a valid URL AND at least one missing target column
        # This keeps us from making network calls for rows that don't need fixing!
        is_valid_url = ddf['detail'].apply(lambda x: isinstance(x, str) and x != "")
        needs_filling = ddf[target_cols].isna().any(axis=1)
        
        rows_to_fix = ddf[is_valid_url & needs_filling]
        
        if rows_to_fix.empty:
            return ddf
            
        urls = rows_to_fix['detail'].tolist()

        # 2. Define a clean worker function for ThreadPool
        def process_url(url):
            try:
                url_json = self.json_data.json_data_fetch(url)
                miss_data = json_data_search(url_json)
                
                # Map the json keys back to your df column names
                row_update = {}
                for col, json_key in key_map.items():
                    if json_key in miss_data:
                        row_update[col] = miss_data[json_key]
                return row_update
            except Exception:
                # If the request fails, return empty so it safely skips filling
                return {}

        # 3. Parallel fetching
        print(f"Fetching and filtering data for {len(urls)} rows in parallel...")
        with self.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_url, urls))
            
        # 4. Create the "Patch" DataFrame aligned on the original indices
        patch_df = self.pd.DataFrame(results, index=rows_to_fix.index)
        
        # 5. Patch the original DataFrame
        # fillna() matches the index and column and only overwrites NaNs!
        return ddf.fillna(patch_df)


    # With debugging checks
    def filter_and_fill_debug_new_change(self, data, max_workers=2, save=True):
        import threading

        ddf = data.copy()
        
        log_path = "filter_and_fill_debug_log.txt"
        _log_lock = threading.Lock()

        from datetime import datetime

        def log_and_print(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {msg}"
            with _log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            print(msg)

        key_map = {
            'nst': 'num-stations-used',
            'dmin': 'minimum-distance',
            'gap': 'azimuthal-gap'
        }
        target_cols = list(key_map.keys())

        # 1. Identify rows that have a valid URL AND at least one missing target column
        is_valid_url = ddf['detail'].apply(lambda x: isinstance(x, str) and x != "")
        needs_filling = ddf[target_cols].isna().any(axis=1)
        
        rows_to_fix = ddf[is_valid_url & needs_filling]
        # return rows_to_fix
        
        if rows_to_fix.empty:
            log_and_print("No rows found that need fixing and have valid URLs.")
            return ddf
            
        urls = rows_to_fix['detail'].tolist()

        # Keep both the dataframe index and a 0-based row counter for logging
        url_records = [(row_no, row_index, url) for row_no, (row_index, url) in enumerate(rows_to_fix['detail'].items())]
        # return urls

        # 2. Define worker function with added print checks
        def process_url(url_record):
            row_no, row_index, url = url_record
            try:
                log_and_print(f"[ROW {row_no} | INDEX {row_index}] Processing current URL")
                url_json = self.json_data.json_data_fetch(url)
                miss_data = json_data_search(url_json)
                
                row_update = {}
                for col, json_key in key_map.items():
                    if json_key in miss_data:
                        row_update[col] = miss_data[json_key]
                
                # --- DEBUG PRINTS ---
                if not miss_data:
                    log_and_print(f"[ROW {row_no} | INDEX {row_index}] [EMPTY JSON] The API from the row returned data, but your search function found nothing at all.")
                elif not row_update:
                    # Found data in JSON, but not the specific 3 keys we mapped
                    available_keys = list(miss_data.keys())
                    log_and_print(f"[ROW {row_no} | INDEX {row_index}] [MISSING KEYS] JSON had data, but lacked 'nst', 'dmin', or 'gap'. Found instead: {available_keys}")
                else:
                    found = list(row_update.keys())
                    log_and_print(f"[ROW {row_no} | INDEX {row_index}] [SUCCESS] Pulled {found} for URL.")
                # --------------------
                
                return row_update
                
            except Exception as e:
                # Catch timeouts, 404s, or JSON parsing errors
                log_and_print(f"[ROW {row_no} | INDEX {row_index}] [ERROR] Failed to fetch or parse URL. Reason: {e}")
                return {}

        # 3. Parallel fetching
        log_and_print(f"Starting parallel fetch for {len(urls)} rows...")
        with self.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_url, url_records))
            # print(results)
            
        # 4. Create the "Patch" DataFrame
        patch_df = self.pd.DataFrame(results, index=rows_to_fix.index)
        
        if save:
            patch_df.to_csv("patch_df.csv", index=True)
        # return patch_df
        
        # 5. Patch the original DataFrame
        # Note: If fillna() behaves weirdly due to data types, we can swap to df.update()
        return ddf.fillna(patch_df)
