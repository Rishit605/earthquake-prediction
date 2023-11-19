import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpl_toolkits.basemap import Basemap

from datetime import datetime
import time


df = pd.read_csv('database.csv')
print(df.columns)

data = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
print(data.head())


timestamp = []
for d, t in zip(data['Date'], data['Time']):
    # print(t)
    try:
        ts = datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        print(ts)
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        # print('ValueError')
        timestamp.append('ValueError')
# timeStamp = pd.Series(timestamp)
# data['Timestamp'] = timeStamp.values
# final_data = data.drop(['Date', 'Time'], axis=1)
# final_data = final_data[final_data.Timestamp != 'ValueError']
# print(final_data.head())

# m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

# longitudes = data["Longitude"].tolist()
# latitudes = data["Latitude"].tolist()
# #m = Basemap(width=12000000,height=9000000,projection='lcc',
#             #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
# x,y = m(longitudes,latitudes)

# fig = plt.figure(figsize=(12,10))
# plt.title("All affected areas")
# m.plot(x, y, "o", markersize = 2, color = 'blue')
# m.drawcoastlines()
# m.fillcontinents(color='coral',lake_color='aqua')
# m.drawmapboundary()
# m.drawcountries()
# plt.show()


# X = final_data[['Timestamp', 'Latitude', 'Longitude']]
# y = final_data[['Magnitude', 'Depth']]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)