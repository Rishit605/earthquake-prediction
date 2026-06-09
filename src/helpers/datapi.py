import os
import numpy as np
import pandas as pd
import geojson
import requests
import geopandas as gpd
from shapely.wkt import loads
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / f"eq_data_updated3.csv"

def generate_url(start_date: str, end_date: str, min_magnitude: float = 2.5) -> str:
    """Generate USGS earthquake data URL for given date range."""
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
    return f"{base_url}?starttime={start_date}%2000:00:00&endtime={end_date}%2000:00:00&minmagnitude={min_magnitude}&orderby=time"

def generate_url_periods(start_year: Optional[int] = None, 
                        end_year: Optional[int] = None, 
                        months_per_period: int = 6) -> Dict[str, str]:
    """
    Generate URLs for multiple periods between start and end year.
    If no dates provided, uses 5 years back from current date.
    
    Args:
        start_year: Starting year. If None, defaults to 5 years ago
        end_year: Ending year. If None, defaults to current year
        months_per_period: Number of months per data chunk
        
    Returns:
        Dictionary of period keys and their corresponding URLs
    """
    current_datetime = datetime.now()
    
    if start_year is None:
        start_year = current_datetime.year - 5
    if end_year is None:
        end_year = current_datetime.year
        
    urls = {}
    current_date = datetime(start_year, 1, 1)
    end_date = min(datetime(end_year + 1, 1, 1), current_datetime)
    
    while current_date < end_date:
        period_end = min(current_date + timedelta(days=months_per_period * 30), end_date)
        period_key = f"period_{current_date.strftime('%Y%m')}"
        
        urls[period_key] = generate_url(
            current_date.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d")
        )
        current_date = period_end
    
    return urls


def data_geo_ready(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['geo'] = dataframe['geo'].apply(loads)
    gdf = gpd.GeoDataFrame(dataframe)
    return gdf

def url_data_call(url: Optional[str] = None, save: bool = False) -> pd.DataFrame:
    """
    Fetch earthquake data either from stored CSV or from USGS API.
    
    Args:
        url: The USGS API URL to fetch data from
        stored_data: Whether to load from stored CSV file
        
    Returns:
        GeoDataFrame containing earthquake data
    """
    if save:
        try:
            df = pd.read_csv(DATA_PATH)
            return data_geo_ready(df)
        except FileNotFoundError:
            print("Stored data file not found. Fetching from API instead.")
    
    if url is None:
        # If no URL provided, get latest 4 months of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        url = generate_url(start_date.strftime("%Y-%m-%d"), 
                         end_date.strftime("%Y-%m-%d"))
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        geojson_data = geojson.loads(response.text)
        features = geojson_data['features']
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf = gdf.rename(columns={'geometry': 'geo'})
        
        return gdf
        
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def callDataFetcher(saved: bool) -> pd.DataFrame:
    # Generate URLs for the last 5 years up to current date
    data_urls = generate_url_periods(start_year=2022, end_year=2026)

    if DATA_PATH.exists() and saved:
        return url_data_call(save=saved)
    else:
        print("Data not found, in the directory.\nFetching data from API...")
        # Fetch and combine data from all periods
        earthquake_data = []
        for period, url in data_urls.items():
            print(f"Fetching data for {period}...")
            period_data = url_data_call(url)
            if not period_data.empty:
                earthquake_data.append(period_data)

        if earthquake_data:
            # Save the data to the directory
            gdf =  pd.concat(earthquake_data, ignore_index=True)
            gdf.to_csv(DATA_PATH, index=False)
            return gdf


if __name__ == "__main__":
    
    eq_Data = callDataFetcher(False)
    # print((Path.cwd().parents[1] / "data" / "eq_data_updated.csv").exists())
    # print(Path.cwd().parent[1] / "data" / "eq_data_updated.csv")
    # print(Path.cwd().parent[1] / "data" / "eq_data_updated.csv".exists())
    # print(Path.cwd().parent[1] / "data" / "eq_data_updated.csv".exists())

    # Uncomment to save to CSV
    # all_data.to_csv(r".\data\earthqukae_data.csv", index=False)

    print(f"Successfully fetched data with {len(eq_Data)} records")
    print(eq_Data)