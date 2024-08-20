import os
import sys
import requests

import numpy as np
import pandas as pd

import geojson
import geopandas as gpd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.datapi import url_data_call, datas


## DEFINING THE PREPROCESSING FUNCTION
def data_preprocessing(dataframe, ts=False) -> pd.DataFrame:
    data2 = dataframe.drop(columns=[
        'tz', 'url',
        'detail', 'felt', 
        'cdi', 'mmi', 'alert',
        'status', 'tsunami', 'sig',
        'net', 'code', 'ids',
        'sources', 'sources', 'nst',
        'title', 'types', 'gap', 'updated'
    ])
 
    eq_data = data2.loc[data2['type'] == 'earthquake']

    eq_data = eq_data.copy()
    eq_data.drop('type', axis=1, inplace=True)

    eq_data['time'] = pd.to_datetime(eq_data['time'], unit='ms')
    # eq_data['updated'] = pd.to_datetime(eq_data['updated'], unit='ms')

    eq_data['coords'] = eq_data['geo'].apply(lambda geom: list(geom.coords))

    x = []
    y = []
    z = []

    for coord_vals in eq_data['coords']:
        x.append(coord_vals[0][0])
        y.append(coord_vals[0][1])
        z.append(coord_vals[0][2])

    eq_data['longitude'] = x
    eq_data['latitude'] = y
    eq_data['elevation'] = z

    eq_data.drop(columns=['place', 'geo', 'coords'], inplace=True)

    if ts is True:
        eq_data.sort_values('time', inplace=True)
        eq_data = eq_data.set_index('time')
    
    return eq_data


## DATA TRRANSFORMATION
def imput_encode(data):
    df2 = data
    LE = LabelEncoder() 

    # Encode labels
    df2['magType']= LE.fit_transform(data['magType']) 

    if data.isna().any().sum() > 0:
        data_dict = data.isna().any().to_dict()

        for key, vals in data_dict.items():
            df2[key] = df2[key].ffill()
    
    return df2


def var_and_tar(dataframe):
    target_feat = ['mag', 'dmin', 'rms']

    # Split the data into input and output features
    X = dataframe
    y = dataframe[target_feat]
    return X, y, target_feat

def split_dataset(X, Y):
    # Split the data into train and test sets
    train_size = int(len(X) * 0.7)  # Use 80% for training 
    valid_size = int(len(X) * 0.15) # Use the rest 15% as validation set
    test_size = int(len(X) - train_size - valid_size)


    X_train, y_train = X[:train_size], Y[:train_size]
    X_val, y_val = X[train_size:train_size + valid_size], Y[train_size:train_size + valid_size]
    X_test, y_test = X[train_size + valid_size:], Y[train_size + valid_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def scaler_dataset(dataSet: pd.DataFrame):
    """
    Takes a DataFrame and returns a scaled and normalized DataFrame.

    Args:
        dataSet (pd.DataFrame): Input DataFrame with numerical columns.

    Returns:
        pd.DataFrame: Scaled and normalized DataFrame.
    """
    scale = MinMaxScaler(feature_range=(0,1))
    scaled_dataset = scale.fit_transform(dataSet)
    scaled_dataset = pd.DataFrame(scaled_dataset, columns=dataSet.columns, index=dataSet.index)
    return scaled_dataset, scale
    


def CyclicTimeTransform(data: pd.DataFrame) -> pd.DataFrame:    
    day = 60 * 60 * 24
    year = 365.2425 * day
    data_df = data.copy()

    data_df['Seconds'] = data_df.index.map(pd.Timestamp.timestamp)
    
    data_df['Hour sin'] = np.sin(2 * np.pi * data_df.index.hour / 24)
    data_df['Hour cos']= np.cos(2 * np.pi * data_df.index.hour / 24)

    data_df['Day sin'] = np.sin(2 * np.pi * data_df.index.day / 7)
    data_df['Day cos'] = np.cos(2 * np.pi * data_df.index.day / 7)

    data_df['Month sin'] = np.sin(2 * np.pi * data_df.index.month / 12)
    data_df['Month cos'] = np.cos(2 * np.pi * data_df.index.month / 12)

    data_df['day_of_year'] = data_df.index.dayofyear
    data_df['month'] = data_df.index.month
    data_df = data_df.drop('Seconds', axis=1)
    return data_df

def SingleStepSingleVARSampler(df, window, target_column):
    """
    For Generating SingleStep Single Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df.to_numpy()
    target_array = df[target_column].to_numpy()

    # Number of samples we can create
    num_samples = len(df) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros(num_samples)

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y

def SingleStepMultiVARSampler(df, window, target_columns):
    """
    For Generating SingleStep Multi Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df.to_numpy()
    target_array = df[target_columns].to_numpy()

    # Number of samples we can create
    num_samples = len(df) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros((num_samples, len(target_columns)))

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y

def SingleStepMultiVARS_SeperateSampler(df_X, df_Y, window, target_columns):
    """
    For Generating SingleStep Multi Target variable sequence for training.
    """

    # Convert DataFrame to NumPy array for faster operations
    features_array = df_X.to_numpy()
    target_array = df_Y[target_columns].to_numpy()

    # Number of samples we can create
    num_samples = len(df_X) - window

    # Initialize empty arrays for X and Y
    X = np.zeros((num_samples, window, features_array.shape[1]))
    Y = np.zeros((num_samples, len(target_columns)))

    for i in range(num_samples):
        X[i] = features_array[i:i+window]
        Y[i] = target_array[i + window]

    return X, Y


def Simple_create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps].values)
    return np.array(Xs), np.array(ys)