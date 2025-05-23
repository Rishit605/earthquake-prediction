import numpy as np
import pandas as pd
import geojson
import requests
import geopandas as gpd

import torch


url1 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2024-05-01%2000:00:00&endtime=2025-02-01%2000:00:00&minmagnitude=2.5&orderby=time'
url2 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2024-01-01%2000:00:00&endtime=2024-05-01%2000:00:00&minmagnitude=2.5&orderby=time'
url3 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2023-05-01%2000:00:00&endtime=2024-01-01%2000:00:00&minmagnitude=2.5&orderby=time'
url4 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2023-01-01%2000:00:00&endtime=2023-05-01%2000:00:00&minmagnitude=2.5&orderby=time'
url5 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2022-05-01%2000:00:00&endtime=2023-01-01%2000:00:00&minmagnitude=2.5&orderby=time'
url6 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2022-01-01%2000:00:00&endtime=2022-05-01%2000:00:00&minmagnitude=2.5&orderby=time'
url7 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2021-05-01%2000:00:00&endtime=2022-01-01%2000:00:00&minmagnitude=2.5&orderby=time'
url8 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2021-01-01%2000:00:00&endtime=2021-05-01%2000:00:00&minmagnitude=2.5&orderby=time'
url9 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2020-05-01%2000:00:00&endtime=2021-01-01%2000:00:00&minmagnitude=2.5&orderby=time'
url10 = 'https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=2020-01-01%2000:00:00&endtime=2020-05-01%2000:00:00&minmagnitude=2.5&orderby=time'


# DATA API
datas = {
    'year2020': url10,
    'year2120': url9,
    'year2121': url8,
    'year2122': url7,
    'year2222': url6,
    'year2223': url5,
    'year2323': url4,
    'year2324': url3,
    'year2424': url2,
    'year2524': url1
}


## DEFINING THE API CALLING FUNCTION
def url_data_call(stored_data: bool = False, URL = None) -> pd.DataFrame:
    if stored_data:
        # Load dataset
        df = pd.read_csv(r".\data\earthqukae_data.csv")
        gdf = df.rename(columns={'geometry': 'geo'})

        return gdf
    else:
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

    
if __name__ == "__main__":
    # print(pd.concat([url_data_call(datas[key]) for key in datas], ignore_index=True))
    all_data = pd.concat([url_data_call(datas[key], False) for key in datas], ignore_index=True)
    # all_data.to_csv(r"C:\Projs\COde\Earthquake\eq_prediction\data\fin_raw_data.csv")

    print(all_data)