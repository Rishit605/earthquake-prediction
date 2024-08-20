import numpy as np
import pandas as pd
import geojson
import requests
import geopandas as gpd

import torch


url1 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2023-05-01%2000:00:00&endtime=2024-01-01%2000:00:00&minmagnitude=2.5&orderby=time'
url2 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2023-01-01%2000:00:00&endtime=2023-05-01%2000:00:00&minmagnitude=2.5&orderby=time'
url3 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2022-05-01%2000:00:00&endtime=2023-01-01%2000:00:00&minmagnitude=2.5&orderby=time'

# DATA API
datas = {
    'year22': url3,
    'year23': url2,
    'year24': url1,
}


## DEFINING THE API CALLING FUNCTION
def url_data_call(URL) -> pd.DataFrame:
    response = requests.get(URL)

    if response.status_code == 200:
        # Load GeoJSON data from the response text
        geojson_data = geojson.loads(response.text)

        # Extract features from GeoJson
        features = geojson_data['features']

        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf = gdf.rename(columns={'geometry': 'geo'})

    return gdf