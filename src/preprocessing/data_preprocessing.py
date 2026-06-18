from typing import List
import numpy as np
import pandas as pd
import math
import warnings


import geojson
import geopandas as gpd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from pathlib import Path
import os, sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)) if str(PROJECT_ROOT) not in sys.path else None


from src.helpers.datapi import callDataFetcher
from src.helpers.utils import find_mean, find_variance

### DATA TRANSFORMATION
# Data Preparation
class EQDataLoader:
    def __init__(self, Saved: bool = True, fill_save: bool = False) -> None:
        from notebook.data_eng_test import data_eng_test, data_eng_test_incremental
        self.save_data_flag = Saved
        self.fill_save = fill_save
        self.DATA_PATH = None
        self.data_f = self._load_raw_data(Saved)

        if fill_save:
            self.refill_data, self.DATA_PATH = self._load_engineered_data(
                "new_engineered_data"
            )
        else:
            self.refill_data = self._load_engineered_data_or_run(data_eng_test, "new_engineered_data")
            self.refill_data2 = self._load_engineered_data_or_run(data_eng_test_incremental, "new_engineered_data")
            
            if self.DATA_PATH is None:
                warnings.warn(
                    "'DATA_PATH' is currently set to None because fill_save=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _load_raw_data(saved: bool):
        try:
            from src.helpers.sql_data_handler import fetch_raw_data, fetch_table
            from src.helpers.datapi import data_geo_ready

            # dataframe = fetch_raw_data()
            dataframe = fetch_table(table_name='earthquakes_eq_data_updated3_patched', schema_name='pc_data')
            print("Loaded raw earthquake data from the database.")
            return data_geo_ready(dataframe)
        except Exception as exc:
            warnings.warn(
                f"Database raw-data load failed ({exc}). Falling back to local/API data.",
                RuntimeWarning,
                stacklevel=2,
            )
            return callDataFetcher(saved)

    @staticmethod
    def _load_engineered_data(file_name: str):
        csv_path = PROJECT_ROOT / "data" / "engineered_data" / file_name
        try:
            from src.helpers.sql_data_handler import fetch_engineered_data

            dataframe = fetch_engineered_data(file_name)
            print(f"Loaded engineered data '{file_name}' from the database.")
            return dataframe, csv_path
        except Exception as exc:
            warnings.warn(
                f"Database engineered-data load failed for '{file_name}' ({exc}). "
                "Falling back to local CSV.",
                RuntimeWarning,
                stacklevel=2,
            )
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Engineered data not found in the database or at {csv_path}."
                ) from exc
            return pd.read_csv(csv_path), csv_path

    def _load_engineered_data_or_run(self, data_fn, file_name: str):
        try:
            from src.helpers.sql_data_handler import fetch_engineered_data

            dataframe = fetch_engineered_data(file_name)
            print(f"Loaded engineered data '{file_name}' from the database.")
            return dataframe
        except Exception as exc:
            print(f"Database engineered-data load failed for '{file_name}' ({exc}). ")
            print("Falling back to local data engineering.")
            return data_fn(data=self.data_f, file_name=file_name)
    
    def coordinate_expander(self, data: pd.DataFrame):
        # Check if 'geo' column exists
        if 'geo' not in data.columns:
            raise KeyError("Input DataFrame does not contain a 'geo' column required for coordinate expansion.")
        
        # Check if 'geo' column has at least one non-null entry
        if data['geo'].isnull().all():
            raise ValueError("All entries in the 'geo' column are null. Cannot extract coordinates.")

        # Handle edge cases: filter out rows with missing or malformed 'geo'
        def valid_geom(geom):
            # Checks for shapely geometry with coords attribute and at least one coordinate
            try:
                if hasattr(geom, 'coords') and len(geom.coords) > 0:
                    coord = list(geom.coords[0])
                    # Must be at least 3D (lon, lat, elev)
                    return len(coord) == 3 and all(isinstance(x, (int, float)) for x in coord)
            except Exception:
                return False
            return False

        valid_geo_mask = data['geo'].apply(valid_geom)
        if not valid_geo_mask.any():
            raise ValueError("No valid 'geo' entries with coordinates found.")
        if not valid_geo_mask.all():
            print(f"Warning: {(~valid_geo_mask).sum()} rows have missing or invalid 'geo' geometry and will be skipped for coordinate extraction.")

        # Prepopulate new columns with NaN
        data['longitude'] = np.nan
        data['latitude'] = np.nan
        data['elevation'] = np.nan

        # Extract coordinates for valid rows only
        valid_indices = data.index[valid_geo_mask]
        coords = data.loc[valid_indices, 'geo'].apply(lambda geom: list(geom.coords[0]))
        coords_array = np.stack(coords.values)

        # Assign only to valid rows to avoid misalignment
        data.loc[valid_indices, 'longitude'] = coords_array[:, 0]
        data.loc[valid_indices, 'latitude'] = coords_array[:, 1]
        data.loc[valid_indices, 'elevation'] = coords_array[:, 2]

        # Post-check: Ensure columns have been created and filled for valid rows
        if not all(col in data.columns for col in ['longitude', 'latitude', 'elevation']):
            raise RuntimeError("Failed to create longitude, latitude, and elevation columns.")

        # Optionally print number of filled vs missing values
        n_filled = valid_geo_mask.sum()
        n_total = len(data)
        print(f"Extracted coordinates for {n_filled} out of {n_total} rows. Remaining rows contain NaN in new columns.")

    
    def refine_og_data(self, data=None):
        
        if data is None:
            if self.data_f is None or self.data_f.empty:
                raise ValueError("Original data is missing or empty. Cannot refine original data.")
            original_df = self.data_f
        else:
            original_df = data

        if self.save_data_flag and self.DATA_PATH is not None and not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Expected data file not found at {self.DATA_PATH}. Please ensure the data is saved correctly.")


        df_copy = original_df.copy()

        # Add columns for coordinates
        self.coordinate_expander(df_copy)

        # Filter valid data points
        valid_df = df_copy[df_copy['status'] == 'reviewed']

        # sort_col = 'idx' if 'idx' in valid_df.columns else ('index' if 'index' in valid_df.columns else None)
        # if sort_col:
        #     valid_df.sort_values(by=sort_col, inplace=True)
        
        filtered_df = valid_df.drop(columns=drop_rate_new(valid_df)) 
        # print(filtered_df.columns)
        
        # Setting the index for merge
        # new_df = filtered_df.reset_index()
        new_df = filtered_df.copy()
        # print(new_df)
        # return 
        if 'index' in new_df.columns:
            new_df.rename(columns={"index": "idx"}, inplace=True)

        sort_col = 'idx' if 'idx' in new_df.columns else ('index' if 'index' in new_df.columns else None)
        if sort_col:
            new_df.sort_values(by=sort_col, inplace=True)

        if 'idx' in new_df.columns:
            new_df.set_index("idx", inplace=True)
        else:
            raise KeyError("Cannot set index. Neither 'idx' nor 'index' was found in the dataframe.")

        return new_df
    
    
    def refine_refill_data(self) -> pd.DataFrame:
        recollection_df = self.refill_data
        rel_copy = recollection_df.copy()

        rel_copy.drop("Unnamed: 0", axis=1, inplace=True)
        rel_copy.rename(columns={"old_idx": "idx"}, inplace=True)
        rel_copy.set_index("idx", inplace=True)

        return rel_copy

    
    def data_prep(self, ts: bool=False) -> pd.DataFrame:

        og_data = self.refine_og_data()
        ref_df = self.refine_refill_data()

        final_data = og_data.combine_first(ref_df).loc[og_data.index]
        final_data.drop('nst', axis=1, inplace=True)
        final_data.drop('detail', axis=1, inplace=True)
        # final_data.dropna(inplace=True)
        
        if ts:
            final_data['time'] = pd.to_datetime(final_data['time'], unit='ms')
            final_data.sort_values('time')

            return final_data
        else:
            final_data.sort_values('time')
            final_data.drop('time', axis=1, inplace=True)
            return final_data

    def extra_data_prep(self, ts: bool=False, data = None) -> pd.DataFrame:
        if self.fill_save:
            return self.data_prep(ts=ts)
        
        if isinstance(data, pd.DataFrame):
            final_data = data.copy()
        else:
            final_data = self.refine_og_data()
            
        final_data.drop(columns=['nst', 'detail'], errors='ignore', inplace=True)
        final_data.sort_values('time', inplace=True)

        if ts:
            final_data['time'] = pd.to_datetime(final_data['time'], unit='ms')
            return final_data
        else:
            final_data.sort_values('time')
            final_data.drop('time', axis=1, inplace=True)
            final_data.dropna(inplace=True)
            df = final_data.reset_index(drop=True)
            return df


class ZScoreStandard:
    def __init__(self, dataframe, cols) -> None:
        self.dataframe = dataframe
        self.scale_params = {}
        self.cols = cols

    def fit_standard_Z(self):
        for col in self.cols:
            if col in self.dataframe.select_dtypes(exclude="object").columns:
                mean = find_mean(self.dataframe[col])
                std  = np.sqrt(find_variance(self.dataframe[col]))
                self.scale_params[col] = (mean, std)
                self.dataframe[col] = (self.dataframe[col] - mean) / std
        return self.dataframe

    def transform_standard_Z(self, data, _params=None):
        if self.dataframe.shape[1] != data.shape[1]:
            raise pd.errors.DataError("Dataframes miss-match, please verify if the dataframe are of the correct size.")

        scale_params = self.scale_params if _params is None else _params
        for col, (mean, std) in scale_params.items():
            if col in data.columns:
                data[col] = (data[col] - mean) / std
        return data
        

class DataPreprocessor:
    def __init__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.dataframe = dataframe

    def _log_transform(self):
        # Log-Transformation of Skewed Columns
        for col in ['dmin_km', 'elevation']:
            def safe_log(x):
                return np.log(x+1) if pd.notnull(x) and x > 0 else x
            self.dataframe[col] = self.dataframe[col].apply(safe_log)
        print("Log Transformation Used!")
        return self.dataframe

    def _root_transform(self, sq=True):
        # Square-Transformation of Skewed Columns
        for col in ['dmin_km', 'elevation']:
            def safe_root(x):
                if sq:
                    # Square root transformation
                    return np.sqrt(x) if pd.notnull(x) and x > 0 else x
                else:
                    # Cube root transformation
                    return np.cbrt(x) if pd.notnull(x) and x != 0 else x
            self.dataframe[col] = self.dataframe[col].apply(safe_root)
        transform_name = "Square" if sq else "Cube"
        print(f"{transform_name} Transformation Used!")
        return self.dataframe


    def imput_data_transform(self, transform=True, log_t=True, square=True, dmin_convert=True) -> pd.DataFrame:
        # 'dmin' columns conversion
        if dmin_convert:
            if 'dmin' not in self.dataframe.columns:
                raise KeyError("'dmin' column not found in the dataframe.")
       
            self.dataframe['dmin_km'] = self.dataframe['dmin'].apply(lambda x: x * 111.19)
            self.dataframe = self.dataframe.drop('dmin', axis=1)
 

        if transform:
            if log_t:
                return self._log_transform()
            else:
                return self._root_transform(sq=square)
            
        return self.dataframe

    def data_transform(self, transform=True, log_t=True, square=True) -> pd.DataFrame:
        # 'dmin' columns conversion
        self.dataframe['dmin_km'] = self.dataframe['dmin'].apply(lambda x: x * 111.19)
        self.dataframe = self.dataframe.drop('dmin', axis=1)

        if transform:
            if log_t:
                return self._log_transform()
            else:
                return self._root_transform(sq=square)
            
        return self.dataframe


class ExperimentalDataPreprocessor:
    def __init__(self, dataframe: pd.DataFrame, custom_cols: bool = False):
        self.dataframe = dataframe
        self.c_cols = custom_cols
        
    def data_transform(self, cols: list = None, square=False):
        # If using custom columns, validate input
        self.cols = cols if self.c_cols else None
        if self.c_cols and cols is None:
            raise ValueError("custom_cols is True, but no columns (`cols`) are specified for transformation.")

        # 'dmin' columns conversion
        if 'dmin' in self.dataframe.columns:
            self.dataframe['dmin_km'] = self.dataframe['dmin'].apply(lambda x: x * 111.19)
            self.dataframe = self.dataframe.drop('dmin', axis=1)
        elif 'dmin_km' in self.dataframe.columns:
            pass  # Already in km, do nothing
        else:
            raise KeyError("Neither 'dmin' nor 'dmin_km' column found in the dataframe.")
        

        numeric_cols = list(self.dataframe.select_dtypes(include='number').columns)

        if self.c_cols and self.cols is not None:
            cols = [col for col in self.cols if col in numeric_cols]
        else:
            cols = numeric_cols

        # Now handle transformations based on skewness
        # Compute skewness of numeric columns
        skewness = self.dataframe[cols].skew()
        for col in cols:
            if col not in self.dataframe.columns:
                continue
            sk = skewness[col]
            if sk > 1:
                self.dataframe[col] = self.dataframe[col].apply(
                    lambda x: math.log(x + 1) if pd.notnull(x) and x > 0 else x
                )
            elif 0.5 < sk <= 1:
                self.dataframe[col] = self.dataframe[col].apply(
                    lambda x: np.sqrt(x) if pd.notnull(x) and x > 0 else x
                )
            # If not, leave as is
        return self.dataframe


    def imput_data_transform(self, cols, transform=True, log_t=True, square=True, dmin_convert=True) -> pd.DataFrame:
        # 'dmin' columns conversion
        if dmin_convert:
            if 'dmin' not in self.dataframe.columns:
                raise KeyError("'dmin' column not found in the dataframe.")
       
            self.dataframe['dmin_km'] = self.dataframe['dmin'].apply(lambda x: x * 111.19)
            self.dataframe = self.dataframe.drop('dmin', axis=1)
 

        if transform:
            if log_t:
                return self.log_transform_cols(cols=cols)
            else:
                return self.root_transform_cols(cols=cols, sq=square)
            
        return self.dataframe

class BackupFunctions: 
    ## DEFINING THE PREPROCESSING FUNCTION
    def COPY_data_preprocessing(dataframe, ts=False) -> pd.DataFrame:
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

        


def drop_rate_new(data) -> List:
    """
    Returns the list of names of colums to be dropped
    """
    cols = []
    if data is None or not hasattr(data, "isnull") or not hasattr(data, "select_dtypes"):
        raise ValueError("Input argument 'data' must be a pandas DataFrame.")

    null_counts = data.isnull().sum()
    num_rows = data.shape[0]

    for q, i in null_counts.items():
        if i > num_rows * 0.3 and q not in cols:
            cols.append(q)

    # Add object dtype columns, if any
    try:
        object_columns = list(data.select_dtypes(include='object').columns)
        cols.extend(object_columns)
    except Exception as e:
        raise RuntimeError(f"Error selecting object dtype columns: {e}")

    # Add additional columns to drop only if they exist
    additional_drops = ['tsunami', 'updated', 'sig']    
    for col in additional_drops:
        if col not in cols and col in data.columns:
            cols.append(col)

    # Only attempt to remove if they exist (and only remove from 'cols' if present)
    for always_keep in ['magType', 'detail']:
        if always_keep in cols:
            cols.remove(always_keep)

    # Double-check to never drop key columns, raise if missing from DataFrame
    for must_keep in ['magType', 'detail']:
        if must_keep not in data.columns:
            raise KeyError(f"Critical column '{must_keep}' is missing from input DataFrame.")

    return cols


class DataEncoder:
    def __init__(self, data) -> None:
        self.data = data

    def lable_frequency_counter(self):
        r, c = self.data.shape

        labels_to_others = []
        if "magType" not in self.data.columns:
            raise AttributeError("Attribute not Found in the Dataframe!")
        count_ser = self.data['magType'].value_counts()
        for i, k in count_ser.items():
            occ = k / r

            if occ < 0.003:
                if "others" not in labels_to_others:
                    labels_to_others.append("others")
                else:
                    pass
            else:
                if i not in labels_to_others:
                    labels_to_others.append(i)
                else:
                    pass
        return labels_to_others

    def oneH_freqMap_Hybrid(self, labels):
        """
        Custom Hybrid encoding using Frequency mapping and One-hot Encoding.
        """
        import pandas as pd
        labels = self.lable_frequency_counter()
        label_cols = labels.copy() if 'others' in labels else labels + ['others']
        # Vectorized labels assignment (mask for unknown magTypes)
        known_mask = self.data['magType'].isin(labels)

        patch_df = pd.DataFrame(0, index=self.data.index, columns=label_cols)
        
        for lbl in labels:
            patch_df.loc[self.data['magType'] == lbl, lbl] = 1
        if 'others' in patch_df.columns:
            patch_df.loc[~known_mask, 'others'] = 1
        return patch_df
        

    def patcher(self, axis=1, ignore_index=False):
        """
        Concatenates self.data and df_other with basic error checking.

        Args:
            df_other (pd.DataFrame): DataFrame to concatenate.
            axis (int): Axis to concatenate along. Default is 0 (rows).
            ignore_index (bool): Whether to ignore the index.

        Returns:
            pd.DataFrame: Concatenated DataFrame.

        Raises:
            ValueError: If df_other is not a DataFrame or columns do not align (for axis=0).
        """
        import pandas as pd

        df_other = self.oneH_freqMap_Hybrid(self.lable_frequency_counter())

        if not isinstance(df_other, pd.DataFrame):
            raise ValueError(f"df_other must be a pandas DataFrame, got {type(df_other)}")

        if axis == 0:
            # For row-wise concat, columns must match
            mismatched_cols = set(self.data.columns) ^ set(df_other.columns)
            if mismatched_cols:
                raise ValueError(f"Columns do not match for concatenation along axis 0. Mismatched: {mismatched_cols}")
        elif axis == 1:
            # For column-wise concat, indexes must align
            if not self.data.index.equals(df_other.index):
                raise ValueError("Indexes do not align for concatenation along axis 1.")

        try:
            result = pd.concat([self.data, df_other], axis=axis, ignore_index=ignore_index)
            return result
        except Exception as e:
            raise RuntimeError(f"Concatenation failed: {e}")


def imput_encode(data): # Currently not in use
    df2 = data
    LE = LabelEncoder() 

    # Encode labels
    df2['magType']= LE.fit_transform(data['magType']) 

    if data.isna().any().sum() > 0:
        data_dict = data.isna().any().to_dict()

        for key, vals in data_dict.items():
            df2[key] = df2[key].ffill()
    
    return df2


class Data_Sets:
    @staticmethod
    def split_dataset(
        data,
        size=0.7
    ):
        """
        Splits the dataset into train, validation, and test sets.

        Args:
            data (pd.DataFrame): The dataset to split.
            size (float): Fraction of data to use for training (between 0 and 1).

        Returns:
            (train, val, test): Tuple of DataFrames.
        """
        n = len(data)
        if not (0 < size < 1):
            raise ValueError("size must be between 0 and 1.")

        train_end = int(n * size)
        val_end = train_end + int((n - train_end) / 2)

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        return train, val, test
   

    
    @staticmethod
    def split_dataset_xy(
        df: pd.DataFrame,
        target_col: str = 'mag',
        train_frac=0.7,
        validation_flag=False,
        test_flag=False,
        set_split=True
    ):
        """
        Splits a DataFrame into (X, y) training, validation, or test sets
        by reusing split_dataset() for boundary logic.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is missing or empty.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is not in the DataFrame.")
        if not (0 < train_frac < 1):
            raise ValueError("train_frac must be between 0 and 1.")

        if set_split:
            # Reuse existing split logic (no duplicate index calculations)
            train_df, val_df, test_df = Data_Sets.split_dataset(df, size=train_frac)

            # Select which split to use
            if validation_flag:
                chosen_df = val_df
            elif test_flag:
                chosen_df = test_df
            else:
                chosen_df = train_df  # default train
        else:
            chosen_df = df

        X = chosen_df.drop(columns=[target_col])
        y = chosen_df[target_col]
        return X, y

    
    @staticmethod
    def stratify_split(df, shuffle: bool = True, strat_col: str = 'gap_missing'):  
        from sklearn.model_selection import train_test_split

        data = df.copy()
        data["gap_missing"] = data["gap"].isna().astype(int)

        # If shuffle is False, stratify must also be None or shuffle must be True when using stratify
        if not shuffle:
            # If shuffle is False, stratify must be None for sklearn's train_test_split
            if strat_col is not None:
                raise ValueError("Stratified split requires shuffle=True in train_test_split. Please set shuffle=True if stratifying.")
            else:
                train_df, pred_df = train_test_split(
                    data,
                    test_size=0.2,
                    random_state=42,
                    shuffle=False,
                    stratify=None
                )
        else:
            train_df, pred_df = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=data[strat_col] if strat_col is not None else None  # stratify only if provided
            )
        return train_df, pred_df


class DataScaler:
    def cus_Scaler():
        pass


    @staticmethod
    def lib_Scaler(dataSet: pd.DataFrame):
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


## Temporal and Advanced Feature Engineering
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
def spatial_temp_grid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates spatial-temporal grid for the input DataFrame.
    
    This function takes a DataFrame with latitude and longitude, converts them to
    bins, and creates a unique identifier for each lat-lon bin combination.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'latitude' and 'longitude' columns.
    """
    # Create spatial-temporal grid
    data['time_bin'] = pd.to_datetime(data['time']).dt.to_period('D')  # Daily bins
    data['lat_bin'] = pd.cut(data['latitude'], bins=50)  # Adjust bin number as needed
    data['lon_bin'] = pd.cut(data['longitude'], bins=50)

    return data

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
def event_counts_for_diff_window2(dataFrame: pd.DataFrame, filler = "mean") -> pd.DataFrame:
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
        new_df = new_df.reset_index(drop=False)

    new_df['time_since_last_event'] = new_df['time'].diff().dt.total_seconds()

    numeric_columns = new_df.select_dtypes(include=[np.number]).columns
    if filler == "mean":
        new_df[numeric_columns] = new_df[numeric_columns].fillna(new_df[numeric_columns].mean())
    else:
        # Fill NaN values only for non-categorical columns
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
