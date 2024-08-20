import os
import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.datapi import datas, url_data_call
from model.model import Early_Stopping, ModelCheckPoint, EarthquakeModel
from preprocessing.data_preprocessing import *


def raw_data_prep() -> pd.DataFrame:
    """
    Calls and defines the data and returns a Pandas DataFrame with basic preprocssing.
    """
    df = pd.DataFrame()

    for key, values in datas.items():
        # print(f"{key} with value: {values}")
        pseudo_df = url_data_call(datas[key])

        df = pd.concat([df, pseudo_df])

    df = data_preprocessing(df, ts=True) ## This function performs basic proecprocessing with an option of Timeseries or not.
    df = imput_encode(df) ## This function encodes and imputs the input data and fills the empty values.

    return df

def event_counts_for_diff_window(dataFrame: pd.DataFrame) -> pd.DataFrame:
    new_df = dataFrame.copy()

    new_df.reset_index(inplace=True)
    new_df.sort_values('time', inplace=True)

    new_df['time_since_last_event'] = new_df['time'].diff().dt.total_seconds()
    # new_df['time_since_last_event'] = new_df['time_since_last_event'].fillna(0)
    
    # Event counts in different windows
    for window in [1, 7, 30]:
        new_df[f'events_last_{window}'] = new_df['time'].rolling(window).count()

    new_df = new_df.fillna(0)
    daily_df = new_df.resample('D', on='time').agg({
    'mag': ['count', 'max', 'mean'],
    'time_since_last_event': 'mean'
    })

    daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]
    daily_df = daily_df.reset_index()

    # # Merge daily features back to original DataFrame
    new_df = pd.merge_asof(df, daily_df, left_on='time', right_on='time', 
                   tolerance=pd.Timedelta('1D'), direction='backward')
    return new_df

def prep_D(data_Frame: pd.DataFrame):
    """
    Takes in the raw dataframe and returns a curated and scaled dataframe
    """
    df2_ffill = df.copy()
    df2_ffill = CyclicTimeTransform(df2_ffill)

    X1, Y1, target_columns = var_and_tar(df2_ffill)
    scaled_X, scaler_X = scaler_dataset(X1)
    scaled_Y, scaler_Y = scaler_dataset(Y1)

    return scaled_X, scaled_Y, scaler_X, scaler_Y


df  = raw_data_prep() # Getting the Data
df2 = df.copy()

df2 = event_counts_for_diff_window(dataFrame=df2)

scaled_X, scaled_Y, scaler_X, scaler_Y = prep_D(df2) 

# Splitting the Dataset
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(scaled_X, scaled_Y)

# Hyperparameter tuning
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],            # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],            # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                  # Whether bootstrap samples are used when building trees
}

grid_search_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")

grid_search_cv.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search_cv.best_params_
print("Best Parameters:", best_params)

with open('best_parameters_RF.txt', 'w') as text_file:
    text_file.write(str(best_params))
