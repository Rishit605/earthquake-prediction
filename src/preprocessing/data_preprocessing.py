import os
import sys
import requests

import numpy as np
import pandas as pd

import geojson
import geopandas as gpd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# from src.helpers import url_data_call, generate_url_periods


## DEFINING THE PREPROCESSING FUNCTION
def data_preprocessing(dataframe, ts=False) -> pd.DataFrame:
    # Drop columns in a single operation
    columns_to_drop = [
        'tz', 'url', 'detail', 'felt', 'cdi', 'mmi', 'alert', 'status',
        'tsunami', 'sig', 'net', 'code', 'ids', 'sources', 'nst',
        'title', 'types', 'gap', 'updated'
    ]
    data2 = dataframe.drop(columns=columns_to_drop, errors='ignore')
    
    # Filter earthquakes using boolean indexing
    eq_data = data2[data2['type'] == 'earthquake'].copy()
    eq_data.drop('type', axis=1, inplace=True)
    
    # Convert time more efficiently
    eq_data['time'] = pd.to_datetime(eq_data['time'], unit='ms')
    
    # Extract coordinates using vectorized operations
    coords = np.array([list(geom.coords[0]) for geom in eq_data['geo']])
    eq_data['longitude'] = coords[:, 0]
    eq_data['latitude'] = coords[:, 1]
    eq_data['elevation'] = coords[:, 2]
    
    # Drop remaining columns
    eq_data.drop(columns=['place', 'geo'], inplace=True)
    
    # Sort values
    eq_data.sort_values('time', inplace=True)
    
    if ts:
        return eq_data.set_index('time')
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


def EnhancedCyclicTimeTransform(data: pd.DataFrame) -> pd.DataFrame:    
       data_df = data.copy()
       
       # Existing transformations
       data_df['Hour sin'] = np.sin(2 * np.pi * data_df.index.hour / 24)
       data_df['Hour cos'] = np.cos(2 * np.pi * data_df.index.hour / 24)
       
       data_df['Day sin'] = np.sin(2 * np.pi * data_df.index.dayofweek / 7)
       data_df['Day cos'] = np.cos(2 * np.pi * data_df.index.dayofweek / 7)
       
       data_df['Month sin'] = np.sin(2 * np.pi * data_df.index.month / 12)
       data_df['Month cos'] = np.cos(2 * np.pi * data_df.index.month / 12)
       
       # Year cycle
       data_df['Year sin'] = np.sin(2 * np.pi * data_df.index.dayofyear / 365.25)
       data_df['Year cos'] = np.cos(2 * np.pi * data_df.index.dayofyear / 365.25)
       
       # Keep original features if needed
       data_df['day_of_year'] = data_df.index.dayofyear
       data_df['month'] = data_df.index.month
       
       return data_df


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


# This function creates a spatial-temporal grid for the input dataframe.
def spatial_temp_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates spatial-temporal grid for the input DataFrame.
    
    This function takes a DataFrame with latitude and longitude, converts them to
    bins, and creates a unique identifier for each lat-lon bin combination.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'latitude' and 'longitude' columns.
    """
    # Create spatial-temporal grid
    df['time_bin'] = pd.to_datetime(df['time']).dt.to_period('D')  # Daily bins
    df['lat_bin'] = pd.cut(df['latitude'], bins=50)  # Adjust bin number as needed
    df['lon_bin'] = pd.cut(df['longitude'], bins=50)

    return df

# This function encodes the spatial-temporal grid for the input dataframe.
def encode_lat_lon_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes spatial-temporal grid information for the input DataFrame.
    
    This function takes a DataFrame with latitude and longitude bins, converts them to
    midpoints, creates a unique identifier for each lat-lon bin combination, and then
    encodes these combinations using LabelEncoder. It also applies spatial_temp_grid
    function to create time, latitude, and longitude bins.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
        pd.DataFrame: Encoded DataFrame with new columns for lat-lon bin midpoints and
                      encoded lat-lon bin combinations. Original bin columns are dropped.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()
    df_encoded = spatial_temp_grid(df=df_encoded)
    
    # Function to get the midpoint of a bin
    def get_bin_midpoint(bin_):
        if pd.isna(bin_):
            return np.nan
        left, right = bin_.left, bin_.right
        return (left + right) / 2

    # Convert lat_bin and lon_bin to their midpoints
    df_encoded['lat_bin_mid'] = df_encoded['lat_bin'].apply(get_bin_midpoint)
    df_encoded['lon_bin_mid'] = df_encoded['lon_bin'].apply(get_bin_midpoint)

    # Create unique identifiers for each lat-lon bin combination
    df_encoded['lat_lon_bin'] = df_encoded['lat_bin'].astype(str) + '_' + df_encoded['lon_bin'].astype(str)

    # Use LabelEncoder for the combined lat-lon bins
    le = LabelEncoder()
    df_encoded['lat_lon_bin_encoded'] = le.fit_transform(df_encoded['lat_lon_bin'])

    # Drop the original categorical columns and the intermediate string column
    df_encoded = df_encoded.drop(['lat_bin', 'lon_bin', 'lat_lon_bin'], axis=1)

    return df_encoded


# This function creates a new dataframe with event counts for different windows.
def event_counts_for_diff_window2(dataFrame: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new dataframe with event counts for different time windows.
    
    It performs the following operations:
    1. Sorts the input dataframe by time
    2. Calculates the time since the last event
    3. Fills NaN values in numeric columns
    4. Creates daily aggregations (count, max, mean of magnitudes, and time span)
    5. Encodes latitude and longitude bins
    6. Merges daily features back to the original dataframe
    
    The function aims to enrich the earthquake data with temporal and spatial features,
    which can be useful for further analysis or modeling of earthquake patterns.
    """
    new_df = dataFrame.copy()

    new_df =  new_df.sort_values('time')

    if 'time' in dataFrame.index.names:
        new_df = new_df.reset_index(drop=True)

    new_df['time_since_last_event'] = new_df['time'].diff().dt.total_seconds()
    
    # Fill NaN values only for non-categorical columns
    numeric_columns = new_df.select_dtypes(include=[np.number]).columns
    new_df[numeric_columns] = new_df[numeric_columns].fillna(0)

    daily_df = new_df.resample('D', on='time').agg({
        'mag': ['count', 'max', 'mean'],
        'time': lambda x: (x.max() - x.min()).total_seconds()
    })

    daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]
    daily_df = daily_df.reset_index()

    new_df = encode_lat_lon_bins(df=new_df)

    # Merge daily features back to original DataFrame
    new_df = pd.merge_asof(new_df, daily_df, left_on='time', right_on='time', 
                   tolerance=pd.Timedelta('1D'), direction='backward')
    return new_df

def rolling_windows(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates rolling window features for earthquake data.
    
    It performs the following operations:
    1. Iterates over different time windows (1, 7, and 30 days)
    2. For each window, calculates:
       - The count of earthquakes
       - The maximum magnitude
    3. Groups the data by latitude and longitude bins
    4. Merges the new features back to the original dataframe
    
    The function enhances the dataset with temporal patterns of earthquake occurrences,
    which can be valuable for analyzing trends and making predictions.
    """
    # Event counts in different windows
    for window in [1, 7, 30]:
        # Create a temporary DataFrame with the rolling calculations
        temp_df = new_df.set_index('time').groupby(['lat_bin_mid', 'lon_bin_mid'])['mag'].rolling(window).agg(['count', 'max'])
        temp_df.columns = [f'eq_count_last_{window}d', f'max_mag_last_{window}d']
        
        # Reset the index to flatten the MultiIndex
        temp_df = temp_df.reset_index()
        
        # Merge the temporary DataFrame with the original DataFrame
        new_df = pd.merge(new_df, temp_df, on=['time', 'lat_bin_mid', 'lon_bin_mid'], how='left')

    return new_df